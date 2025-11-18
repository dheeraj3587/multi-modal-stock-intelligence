import argparse
import os
import logging
import mlflow
import numpy as np
import pandas as pd
from datetime import datetime
import optuna

from models.growth_scorer import GrowthScorer
from backend.utils.growth_data import (
    load_fundamentals,
    load_technical_indicators,
    load_prices,
    merge_growth_features,
    compute_forward_returns,
    engineer_growth_features,
    engineer_growth_features,
    split_growth_data,
    split_by_stocks,
    temporal_subsplit,
    validate_growth_features
)
from backend.utils.growth_metrics import compute_all_growth_metrics
from backend.utils.model_utils import set_seed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train Growth Scorer")
    
    # Data
    parser.add_argument('--fundamentals-dir', type=str, default='data/raw/fundamentals')
    parser.add_argument('--technical-dir', type=str, default='data/processed')
    parser.add_argument('--price-dir', type=str, default='data/raw/prices')
    parser.add_argument('--horizon-days', type=int, default=60)
    
    # Model
    parser.add_argument('--model-type', type=str, default='random_forest', choices=['random_forest', 'gradient_boosting'])
    parser.add_argument('--n-estimators', type=int, default=100)
    parser.add_argument('--max-depth', type=int, default=10)
    parser.add_argument('--min-samples-split', type=int, default=5)
    parser.add_argument('--learning-rate', type=float, default=0.1, help='Learning rate for gradient boosting')
    parser.add_argument('--seed', type=int, default=42)
    
    # Hyperparameter tuning
    parser.add_argument('--tune', action='store_true', help='Enable Optuna hyperparameter tuning')
    parser.add_argument('--num-trials', type=int, default=30, help='Number of Optuna trials')
    
    # Output
    parser.add_argument('--checkpoint-dir', type=str, default='models/checkpoints')
    parser.add_argument('--experiment-name', type=str, default='growth_scoring_phase4')
    parser.add_argument('--cross-stock-split', action='store_true', default=False, help='Use cross-stock split for generalization testing (80/20 by tickers, stratified by sector)')
    
    return parser.parse_args()

def flatten_metrics(metrics_dict):
    """Flatten nested metrics for MLflow logging."""
    flat = {}
    for key, value in metrics_dict.items():
        if isinstance(value, dict):
            # Flatten nested dicts (e.g., top_k_precision)
            for sub_key, sub_value in value.items():
                flat_key = f"{key}_{sub_key}"
                if isinstance(sub_value, (int, float, np.number)):
                    flat[flat_key] = float(sub_value)
        elif isinstance(value, (int, float, np.number)):
            flat[key] = float(value)
    return flat

def optuna_objective(trial, args, X_train, y_train, X_val, y_val, feature_names, scalers):
    """Optuna objective function for hyperparameter tuning."""
    # Suggest hyperparameters based on model type
    model_type = args.model_type
    n_estimators = trial.suggest_int('n_estimators', 50, 300, step=50)
    max_depth = trial.suggest_int('max_depth', 5, 30)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    
    params = {
        'model_type': model_type,
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'random_state': args.seed
    }
    
    if model_type == 'gradient_boosting':
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
        params['learning_rate'] = learning_rate
    
    # Create and train model
    model = GrowthScorer(**params)
    model.fit(X_train, y_train, feature_names=feature_names, scalers=scalers)
    
    # Evaluate on validation set
    val_preds = model.predict(X_val)
    metrics = compute_all_growth_metrics(val_preds, y_val, benchmark_return=0.0)
    
    # Return Spearman correlation as optimization target
    return metrics['correlation']

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # MLflow
    mlflow.set_experiment(args.experiment_name)
    
    with mlflow.start_run():
        mlflow.log_params(vars(args))
        
        # Load Data
        logger.info("Loading data...")
        
        # Load price data
        price_df = load_prices(args.price_dir)
        if price_df.empty:
            logger.error("No price data found. Cannot proceed without price data.")
            return
        
        logger.info(f"Loaded {len(price_df)} price records")
        
        # Load fundamentals
        fund_df = load_fundamentals(args.fundamentals_dir)
        if fund_df.empty:
            logger.warning("No fundamentals data found. Will proceed with technical indicators only.")
        else:
            logger.info(f"Loaded {len(fund_df)} fundamental records")
        
        # Load or compute technical indicators
        tech_df = load_technical_indicators(args.technical_dir, price_df=price_df)
        if tech_df.empty:
            logger.warning("No technical indicators found. Computing from price data...")
            tech_df = load_technical_indicators(args.technical_dir, price_df=price_df)
        
        logger.info(f"Loaded {len(tech_df)} technical indicator records")
        
        # Compute forward returns (target variable)
        logger.info(f"Computing {args.horizon_days}-day forward returns...")
        price_df['fwd_return'] = compute_forward_returns(price_df, args.horizon_days)
        
        # Merge features
        if not fund_df.empty:
            logger.info("Merging fundamentals, technicals, and prices...")
            merged_df = merge_growth_features(fund_df, tech_df, price_df)
        else:
            # Use only technical indicators
            logger.info("Using technical indicators only...")
            merged_df = tech_df.merge(
                price_df[['ticker', 'date', 'fwd_return']], 
                on=['ticker', 'date'], 
                how='inner'
            )
        
        if merged_df.empty:
            logger.error("Merged dataset is empty. Check data alignment.")
            return
        
        # Drop rows with missing target
        merged_df = merged_df.dropna(subset=['fwd_return'])
        logger.info(f"Merged dataset size: {len(merged_df)} records")
        
        # Split data
        logger.info("Splitting data...")
        
        if args.cross_stock_split:
            logger.info("Using cross-stock split (stratified by sector)...")
            if len(merged_df['ticker'].unique()) < 20:
                logger.warning('Few tickers (<20); stratification may be unstable')
                
            train_full, test_df = split_by_stocks(merged_df, train_ratio=0.8, stratify_by='sector')
            train_df, val_df = temporal_subsplit(train_full, val_ratio=0.125) # 70/10/20 effective
            
            mlflow.log_param('split_type', 'cross_stock')
        else:
            logger.info("Using temporal split...")
            train_df, val_df, test_df = split_growth_data(merged_df)
            mlflow.log_param('split_type', 'temporal')
            
        logger.info(f"Split sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        # Extract targets before feature engineering
        y_train = train_df['fwd_return'].values
        y_val = val_df['fwd_return'].values
        y_test = test_df['fwd_return'].values
        
        # Engineer Features
        logger.info("Engineering features...")
        train_engineered, feature_names, scalers = engineer_growth_features(
            train_df, is_train=True
        )
        val_engineered, _, _ = engineer_growth_features(
            val_df, is_train=False, scalers=scalers
        )
        test_engineered, _, _ = engineer_growth_features(
            test_df, is_train=False, scalers=scalers
        )
        
        # Extract feature matrices
        X_train = train_engineered[feature_names].values
        X_val = val_engineered[feature_names].values
        X_test = test_engineered[feature_names].values
        
        logger.info(f"Feature matrix shape: {X_train.shape}, Features: {len(feature_names)}")
        
        # Validate
        validate_growth_features(X_train, y_train)
        validate_growth_features(X_val, y_val)
        
        # Hyperparameter tuning with Optuna
        if args.tune:
            logger.info(f"Starting Optuna hyperparameter tuning with {args.num_trials} trials...")
            
            study = optuna.create_study(
                direction='maximize',
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)
            )
            
            study.optimize(
                lambda trial: optuna_objective(trial, args, X_train, y_train, X_val, y_val, feature_names, scalers),
                n_trials=args.num_trials,
                show_progress_bar=True
            )
            
            logger.info("Optuna tuning complete!")
            logger.info(f"Best trial: {study.best_trial.number}")
            logger.info(f"Best validation Spearman: {study.best_value:.4f}")
            logger.info(f"Best hyperparameters: {study.best_params}")
            
            # Update args with best hyperparameters
            args.n_estimators = study.best_params['n_estimators']
            args.max_depth = study.best_params['max_depth']
            args.min_samples_split = study.best_params['min_samples_split']
            if 'learning_rate' in study.best_params:
                args.learning_rate = study.best_params['learning_rate']
            
            # Log best params to MLflow
            mlflow.log_params({f'best_{k}': v for k, v in study.best_params.items()})
            mlflow.log_metric('best_val_spearman', study.best_value)
        
        # Model with final hyperparameters
        model_params = {
            'model_type': args.model_type,
            'n_estimators': args.n_estimators,
            'max_depth': args.max_depth,
            'min_samples_split': args.min_samples_split,
            'random_state': args.seed
        }
        if args.model_type == 'gradient_boosting':
            model_params['learning_rate'] = args.learning_rate
            
        model = GrowthScorer(**model_params)
        
        # Train
        logger.info("Training final model...")
        model.fit(X_train, y_train, feature_names=feature_names, scalers=scalers)
        
        # Evaluate
        logger.info("Evaluating on validation set...")
        val_preds = model.predict(X_val)
        
        # Use 0 as benchmark (market-neutral)
        metrics = compute_all_growth_metrics(val_preds, y_val, benchmark_return=0.0)
        
        # Flatten metrics for MLflow
        flat_metrics = flatten_metrics(metrics)
        
        logger.info(f"Validation Spearman Correlation: {metrics.get('correlation', 0):.4f} (p={metrics.get('spearman_p_value', 1.0):.4f})")
        logger.info(f"Validation Excess Return: {metrics.get('excess_return', 0):.4f} (p={metrics.get('excess_p_value', 1.0):.4f})")
        if 'top_k_precision' in metrics and 10 in metrics['top_k_precision']:
            logger.info(f"Top-10 Precision: {metrics['top_k_precision'][10]:.4f}")
        
        # Log flattened metrics to MLflow
        mlflow.log_metrics(flat_metrics)
        
        # If cross-stock split, compute and log sector metrics on test set (simulating OOS)
        if args.cross_stock_split and 'sector' in test_df.columns:
            logger.info("Computing per-sector metrics on test set...")
            test_preds = model.predict(X_test)
            sector_metrics = {}
            sector_spearmans = []
            
            for sector in test_df['sector'].unique():
                mask = test_df['sector'] == sector
                if mask.sum() < 5: continue # Skip small sectors
                
                sec_metrics = compute_all_growth_metrics(test_preds[mask], y_test[mask], 0.0)
                sector_metrics[sector] = sec_metrics
                sector_spearmans.append(sec_metrics.get('correlation', 0))
                
                mlflow.log_metric(f'sector_{sector}_spearman', sec_metrics.get('correlation', 0))
            
            if sector_spearmans:
                mean_sector_spearman = np.mean(sector_spearmans)
                logger.info(f'Mean sector Spearman: {mean_sector_spearman:.3f}')
                mlflow.log_metric('mean_sector_spearman', mean_sector_spearman)
        
        # Feature Importance
        importances = model.get_feature_importances()
        logger.info("Top 10 Features:")
        logger.info("\n" + str(importances.head(10)))
        
        # Save importance as artifact
        importance_path = os.path.join(args.checkpoint_dir, 'feature_importances.csv')
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        importances.to_csv(importance_path, index=False)
        mlflow.log_artifact(importance_path)
        
        # Save Checkpoint
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(args.checkpoint_dir, f"growth_scorer_{timestamp}.pkl")
        model.save_checkpoint(path, {
            'metrics': metrics, 
            'args': vars(args),
            'feature_names': feature_names,
            'num_features': len(feature_names),
            'split_type': 'cross_stock' if args.cross_stock_split else 'temporal',
            'num_train_tickers': len(train_df['ticker'].unique()),
            'sector_distribution': train_df['sector'].value_counts(normalize=True).to_dict() if 'sector' in train_df.columns else {}
        })
        logger.info(f"Saved model to {path}")
        
        mlflow.log_artifact(path)

if __name__ == "__main__":
    main()
