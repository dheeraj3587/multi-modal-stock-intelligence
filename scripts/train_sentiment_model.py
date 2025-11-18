import argparse
import os
import logging
import torch
import mlflow
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from datetime import datetime
import optuna

from models.sentiment_classifier import FinBERTSentimentClassifier
from backend.utils.sentiment_data import (
    load_news_articles, 
    load_social_sentiment, 
    create_labeled_dataset, 
    split_sentiment_data, 
    SentimentDataset,
    validate_sentiment_labels
)
from backend.utils.sentiment_metrics import compute_all_sentiment_metrics
from backend.utils.model_utils import set_seed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train FinBERT Sentiment Classifier")
    
    # Data
    parser.add_argument('--data-dir', type=str, default='data/raw/news', help='Path to news data')
    parser.add_argument('--social-dir', type=str, default='data/raw/social', help='Path to social data')
    parser.add_argument('--labeling-strategy', type=str, default='price_change', choices=['price_change', 'manual', 'hybrid'])
    parser.add_argument('--price-dir', type=str, default='data/raw/prices', help='Path to price data for labeling')
    parser.add_argument('--labeled-dataset', type=str, default=None, help='Path to external labeled dataset CSV (for manual/hybrid strategy)')
    
    # Model
    parser.add_argument('--model-name', type=str, default='ProsusAI/finbert')
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--freeze-bert', action='store_true', default=False, help='Freeze BERT encoder weights during training')
    
    # Training
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--max-epochs', type=int, default=10)
    parser.add_argument('--early-stopping-patience', type=int, default=3)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--seed', type=int, default=42)
    
    # Hyperparameter tuning
    parser.add_argument('--tune', action='store_true', help='Enable Optuna hyperparameter tuning')
    parser.add_argument('--num-trials', type=int, default=20, help='Number of Optuna trials')
    

    
    # Output
    parser.add_argument('--checkpoint-dir', type=str, default='models/checkpoints')
    parser.add_argument('--experiment-name', type=str, default='sentiment_phase4')
    
    return parser.parse_args()

def get_device(device_str):
    if device_str == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_str)

def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    criterion = torch.nn.CrossEntropyLoss()
    
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)
        
        optimizer.zero_grad()
        
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
    avg_loss = total_loss / len(dataloader)
    metrics = compute_all_sentiment_metrics(np.array(all_labels), np.array(all_preds))
    metrics['loss'] = avg_loss
    return metrics

def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    avg_loss = total_loss / len(dataloader)
    metrics = compute_all_sentiment_metrics(np.array(all_labels), np.array(all_preds))
    metrics['loss'] = avg_loss
    return metrics

def optuna_objective(trial, args, train_df, val_df, tokenizer, device):
    """Optuna objective function for hyperparameter tuning."""
    # Suggest hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 5e-5, log=True)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    freeze_bert = trial.suggest_categorical('freeze_bert', [True, False])
    
    # Create datasets with suggested batch size
    train_dataset = SentimentDataset(train_df['text'].tolist(), train_df['label'].tolist(), tokenizer, args.max_length)
    val_dataset = SentimentDataset(val_df['text'].tolist(), val_df['label'].tolist(), tokenizer, args.max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Create model with suggested hyperparameters
    model = FinBERTSentimentClassifier(
        model_name=args.model_name,
        freeze_bert=freeze_bert,
        dropout=dropout
    )
    model.to(device)
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * args.max_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)
    
    best_f1 = 0
    patience_counter = 0
    
    for epoch in range(args.max_epochs):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_metrics = validate(model, val_loader, device)
        
        # Report to Optuna
        trial.report(val_metrics['f1']['macro'], epoch)
        
        # Handle pruning
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        # Early stopping
        if val_metrics['f1']['macro'] > best_f1:
            best_f1 = val_metrics['f1']['macro']
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.early_stopping_patience:
                break
    
    return best_f1

def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load Data
    logger.info("Loading data...")
    news_df = load_news_articles(args.data_dir)
    
    if news_df.empty:
        logger.error("No news data found. Please provide news articles.")
        return
    
    logger.info(f"Loaded {len(news_df)} news articles")
    
    # Load social sentiment data if directory is provided
    social_df = pd.DataFrame()
    if args.social_dir and os.path.exists(args.social_dir):
        logger.info(f"Loading social sentiment data from {args.social_dir}...")
        social_df = load_social_sentiment(args.social_dir)
        if not social_df.empty:
            logger.info(f"Loaded {len(social_df)} social sentiment records")
        else:
            logger.warning(f"No social sentiment data found in {args.social_dir}")
    
    # Label Data based on strategy
    if args.labeling_strategy == 'price_change':
        # Load price data for labeling
        from backend.utils.sentiment_data import load_prices_for_labeling
        
        price_df = load_prices_for_labeling(args.price_dir)
        if price_df.empty:
            logger.error("No price data found. Cannot use price_change labeling strategy without price data.")
            return
        
        logger.info(f"Loaded {len(price_df)} price records")
        logger.info("Generating labels based on forward price returns...")
        
        df = create_labeled_dataset(news_df, price_df, args.labeling_strategy)
        
        if df.empty:
            logger.error("No labeled data after merging news with prices. Check date alignment.")
            return
        
        logger.info(f"Generated {len(df)} labeled samples")
        
    elif args.labeling_strategy == 'manual':
        # Load from external labeled dataset if provided, otherwise check for sentiment_label column
        if args.labeled_dataset:
            logger.info(f"Loading external labeled dataset from {args.labeled_dataset}...")
            labeled_df = pd.read_csv(args.labeled_dataset)
            
            # Merge with news_df by text or use directly if contains text column
            if 'text' in labeled_df.columns and 'label' in labeled_df.columns:
                df = labeled_df[['text', 'label']].copy()
                # Add ticker and published_at if available
                if 'ticker' not in df.columns:
                    df['ticker'] = 'UNKNOWN'
                if 'published_at' not in df.columns:
                    df['published_at'] = pd.Timestamp.now()
            else:
                logger.error("Labeled dataset must contain 'text' and 'label' columns")
                return
        else:
            # Merge social sentiment data with news_df if available
            combined_df = news_df.copy()
            if not social_df.empty and 'sentiment_label' in social_df.columns:
                logger.info("Merging social sentiment data with news articles...")
                # Concatenate social data with news data
                # Ensure both have required columns
                if 'text' not in social_df.columns:
                    logger.warning("Social data missing 'text' column, skipping social data merge")
                else:
                    # Add missing columns to social_df if needed
                    if 'ticker' not in social_df.columns:
                        social_df['ticker'] = 'UNKNOWN'
                    if 'published_at' not in social_df.columns:
                        social_df['published_at'] = pd.Timestamp.now()
                    # Concatenate
                    combined_df = pd.concat([combined_df, social_df], ignore_index=True)
                    logger.info(f"Combined dataset size: {len(combined_df)} (news: {len(news_df)}, social: {len(social_df)})")
            
            # Check for sentiment_label column in combined data
            if 'sentiment_label' in combined_df.columns or 'label' in combined_df.columns:
                df = create_labeled_dataset(combined_df, None, args.labeling_strategy)
            else:
                logger.error("Manual labeling requires either --labeled-dataset or 'sentiment_label'/'label' column in the data (from news or social sources).")
                return
        
    elif args.labeling_strategy == 'hybrid':
        # Hybrid approach: use manual labels where available, price_change otherwise
        logger.info("Using hybrid labeling strategy...")
        
        # Load price data
        from backend.utils.sentiment_data import load_prices_for_labeling
        price_df = load_prices_for_labeling(args.price_dir)
        
        if price_df.empty:
            logger.error("No price data found. Cannot use hybrid labeling strategy without price data.")
            return
        
        # Combine news and social data
        combined_df = news_df.copy()
        if not social_df.empty and 'sentiment_label' in social_df.columns:
            logger.info("Merging social sentiment data with news articles...")
            if 'text' not in social_df.columns:
                logger.warning("Social data missing 'text' column, skipping social data merge")
            else:
                # Add missing columns to social_df if needed
                if 'ticker' not in social_df.columns:
                    social_df['ticker'] = 'UNKNOWN'
                if 'published_at' not in social_df.columns:
                    social_df['published_at'] = pd.Timestamp.now()
                # Concatenate
                combined_df = pd.concat([combined_df, social_df], ignore_index=True)
                logger.info(f"Combined dataset size: {len(combined_df)} (news: {len(news_df)}, social: {len(social_df)})")
        
        # Check for external labeled dataset
        if args.labeled_dataset:
            logger.info(f"Loading external labeled dataset from {args.labeled_dataset}...")
            labeled_df = pd.read_csv(args.labeled_dataset)
            if 'text' in labeled_df.columns and 'label' in labeled_df.columns:
                # Add to combined_df with sentiment_label
                if 'sentiment_label' not in labeled_df.columns:
                    # Map label to sentiment_label format
                    label_to_sentiment = {0: 'Bullish', 1: 'Neutral', 2: 'Bearish'}
                    labeled_df['sentiment_label'] = labeled_df['label'].map(label_to_sentiment)
                # Ensure required columns
                if 'ticker' not in labeled_df.columns:
                    labeled_df['ticker'] = 'UNKNOWN'
                if 'published_at' not in labeled_df.columns:
                    labeled_df['published_at'] = pd.Timestamp.now()
                combined_df = pd.concat([combined_df, labeled_df], ignore_index=True)
        
        # Separate rows with manual labels from unlabeled rows
        if 'sentiment_label' in combined_df.columns:
            manual_labeled = combined_df[combined_df['sentiment_label'].notna()].copy()
            unlabeled = combined_df[combined_df['sentiment_label'].isna()].copy()
            
            logger.info(f"Manual labeled rows: {len(manual_labeled)}, Unlabeled rows: {len(unlabeled)}")
            
            # Create datasets separately
            df_manual = create_labeled_dataset(manual_labeled, None, 'manual') if not manual_labeled.empty else pd.DataFrame()
            df_price = create_labeled_dataset(unlabeled, price_df, 'price_change') if not unlabeled.empty else pd.DataFrame()
            
            # Combine
            if not df_manual.empty and not df_price.empty:
                df = pd.concat([df_manual, df_price], ignore_index=True)
            elif not df_manual.empty:
                df = df_manual
            elif not df_price.empty:
                df = df_price
            else:
                logger.error("No labeled data generated from hybrid strategy")
                return
        else:
            # Fall back to price-based if no manual labels available
            logger.warning("No manual labels found (from social data or external dataset), falling back to price-based labeling")
            df = create_labeled_dataset(combined_df, price_df, 'price_change')
    
    else:
        logger.error(f"Unknown labeling strategy: {args.labeling_strategy}")
        return
        
    validate_sentiment_labels(df)
    
    # Split
    train_df, val_df, test_df = split_sentiment_data(df)
    logger.info(f"Split sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # MLflow
    mlflow.set_experiment(args.experiment_name)
    
    # Hyperparameter tuning with Optuna
    if args.tune:
        logger.info(f"Starting Optuna hyperparameter tuning with {args.num_trials} trials...")
        
        study = optuna.create_study(
            direction='maximize',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
        )
        
        study.optimize(
            lambda trial: optuna_objective(trial, args, train_df, val_df, tokenizer, device),
            n_trials=args.num_trials,
            show_progress_bar=True
        )
        
        logger.info("Optuna tuning complete!")
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best validation F1: {study.best_value:.4f}")
        logger.info(f"Best hyperparameters: {study.best_params}")
        
        # Update args with best hyperparameters
        args.learning_rate = study.best_params['learning_rate']
        args.batch_size = study.best_params['batch_size']
        args.freeze_bert = study.best_params['freeze_bert']
        dropout = study.best_params['dropout']
        
        # Log best params to MLflow
        with mlflow.start_run(run_name='optuna_best'):
            mlflow.log_params(study.best_params)
            mlflow.log_metric('best_val_f1', study.best_value)
    else:
        dropout = 0.3  # default dropout
    
    # Datasets with final hyperparameters
    train_dataset = SentimentDataset(train_df['text'].tolist(), train_df['label'].tolist(), tokenizer, args.max_length)
    val_dataset = SentimentDataset(val_df['text'].tolist(), val_df['label'].tolist(), tokenizer, args.max_length)
    test_dataset = SentimentDataset(test_df['text'].tolist(), test_df['label'].tolist(), tokenizer, args.max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    with mlflow.start_run():
        mlflow.log_params(vars(args))
        
        # Model (use dropout from tuning if available)
        model = FinBERTSentimentClassifier(
            model_name=args.model_name,
            freeze_bert=args.freeze_bert,
            dropout=dropout if args.tune else 0.3
        )
        model.to(device)
        
        # Optimizer
        optimizer = AdamW(model.parameters(), lr=args.learning_rate)
        total_steps = len(train_loader) * args.max_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)
        
        best_f1 = 0
        patience_counter = 0
        best_checkpoint_path = None
        
        for epoch in range(args.max_epochs):
            logger.info(f"Epoch {epoch+1}/{args.max_epochs}")
            
            train_metrics = train_epoch(model, train_loader, optimizer, scheduler, device)
            val_metrics = validate(model, val_loader, device)
            
            logger.info(f"Train Loss: {train_metrics['loss']:.4f}, Precision: {train_metrics.get('precision', {}).get('macro', 0):.4f}, Recall: {train_metrics.get('recall', {}).get('macro', 0):.4f}, F1: {train_metrics['f1']['macro']:.4f}")
            logger.info(f"Val Loss: {val_metrics['loss']:.4f}, Precision: {val_metrics.get('precision', {}).get('macro', 0):.4f}, Recall: {val_metrics.get('recall', {}).get('macro', 0):.4f}, F1: {val_metrics['f1']['macro']:.4f}")
            
            mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items() if isinstance(v, (int, float))}, step=epoch)
            mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items() if isinstance(v, (int, float))}, step=epoch)
            
            # Early stopping
            if val_metrics['f1']['macro'] > best_f1:
                best_f1 = val_metrics['f1']['macro']
                patience_counter = 0
                # Save checkpoint
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                best_checkpoint_path = os.path.join(args.checkpoint_dir, f"finbert_sentiment_{timestamp}.pth")
                model.save_checkpoint(best_checkpoint_path, {'epoch': epoch, 'val_f1': best_f1, 'args': vars(args)})
                logger.info(f"Saved best model to {best_checkpoint_path}")
            else:
                patience_counter += 1
                if patience_counter >= args.early_stopping_patience:
                    logger.info("Early stopping triggered")
                    break
        
        # Test evaluation
        logger.info("Evaluating on test set...")
        if best_checkpoint_path and os.path.exists(best_checkpoint_path):
            # Load best checkpoint
            logger.info(f"Loading best checkpoint from {best_checkpoint_path}")
            from models.sentiment_classifier import FinBERTSentimentClassifier
            model, _ = FinBERTSentimentClassifier.load_checkpoint(best_checkpoint_path, device=device)
        
        test_metrics = validate(model, test_loader, device)
        
        logger.info("=" * 50)
        logger.info("TEST SET RESULTS:")
        logger.info(f"Test Loss: {test_metrics['loss']:.4f}")
        logger.info(f"Test Precision (macro): {test_metrics.get('precision', {}).get('macro', 0):.4f}")
        logger.info(f"Test Recall (macro): {test_metrics.get('recall', {}).get('macro', 0):.4f}")
        logger.info(f"Test F1 (macro): {test_metrics['f1']['macro']:.4f}")
        logger.info(f"Test Weighted F1: {test_metrics.get('weighted_f1', 0):.4f}")
        
        if 'confusion_matrix' in test_metrics:
            cm = test_metrics['confusion_matrix']
            logger.info(f"Confusion Matrix Threshold Passed: {cm.get('threshold_passed', False)}")
            logger.info(f"Max Off-Diagonal: {cm.get('max_off_diagonal', 0):.4f}")
        
        logger.info("=" * 50)
        
        # Log test metrics to MLflow
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items() if isinstance(v, (int, float))})
        mlflow.log_metric("test_precision_macro", test_metrics.get('precision', {}).get('macro', 0))
        mlflow.log_metric("test_recall_macro", test_metrics.get('recall', {}).get('macro', 0))
        mlflow.log_metric("test_f1_macro", test_metrics['f1']['macro'])
        mlflow.log_metric("test_weighted_f1", test_metrics.get('weighted_f1', 0))
                    
if __name__ == "__main__":
    main()
