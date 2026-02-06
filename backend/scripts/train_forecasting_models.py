#!/usr/bin/env python3
"""
Unified training script for LSTM, GRU, and Transformer forecasting models.

Features:
- CLI-driven model selection and hyperparameter configuration
- MLflow experiment tracking
- Optuna hyperparameter tuning
- Early stopping and checkpointing
- Reproducibility via seeding

Aligned with docs/metrics_and_evaluation.md Section 4.1 (walk-forward validation)
and Section 8.2 (reproducibility via seeding and MLflow tracking).

Usage:
    # Basic training
    python scripts/train_forecasting_models.py --model-type lstm --ticker RELIANCE.NS
    
    # With hyperparameter tuning
    python scripts/train_forecasting_models.py --model-type transformer --ticker RELIANCE.NS --tune --num-trials 50
    
    # Custom config
    python scripts/train_forecasting_models.py --model-type gru --ticker RELIANCE.NS --config-file models/configs/gru_custom.json
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.utils.config import config
from backend.utils.logger import get_logger
from backend.utils.model_utils import set_seed, get_device, count_model_parameters
from backend.utils.dataset import create_dataloaders
from backend.utils.metrics import mean_absolute_error
from models import (
    LSTMForecaster, GRUForecaster, TransformerForecaster,
    LSTMConfig, GRUConfig, TransformerConfig
)

# Optional imports
try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    warnings.warn("MLflow not available. Experiment tracking disabled.")

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    warnings.warn("Optuna not available. Hyperparameter tuning disabled.")


logger = get_logger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train forecasting models with MLflow tracking and Optuna tuning"
    )
    
    # Model selection
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["lstm", "gru", "transformer"],
        help="Type of model to train"
    )
    
    # Data configuration
    parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Ticker symbol (e.g., RELIANCE.NS)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Base directory for processed data"
    )
    
    # Hyperparameters
    parser.add_argument(
        "--config-file",
        type=str,
        help="Path to JSON config file (overrides defaults)"
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Maximum training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for training"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate"
    )
    
    # Optuna tuning
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Enable Optuna hyperparameter tuning"
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=50,
        help="Number of Optuna trials"
    )
    
    # System configuration
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Device for training (auto-detects if 'auto')"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="models/checkpoints",
        help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="forecasting_phase3",
        help="MLflow experiment name"
    )
    
    return parser.parse_args()


def find_latest_data_dir(base_dir: Path, ticker: str) -> Path:
    """
    Find the most recent processed data directory for ticker.
    
    Args:
        base_dir: Base processed data directory
        ticker: Ticker symbol
        
    Returns:
        Path to latest data directory
        
    Raises:
        FileNotFoundError: If no processed data found
    """
    ticker_dir = base_dir / ticker
    
    if not ticker_dir.exists():
        raise FileNotFoundError(
            f"No processed data found for ticker {ticker} in {base_dir}"
        )
    
    # Find timestamped subdirectories
    subdirs = [d for d in ticker_dir.iterdir() if d.is_dir()]
    
    if not subdirs:
        raise FileNotFoundError(
            f"No timestamped data directories found in {ticker_dir}"
        )
    
    # Sort by modification time
    subdirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    
    latest = subdirs[0]
    logger.info(f"Using latest processed data: {latest}")
    
    return latest


def load_model_config(model_type: str, config_file: Optional[str], input_dim: int, args) -> Any:
    """
    Load model configuration from file or defaults.
    
    Args:
        model_type: Type of model ('lstm', 'gru', 'transformer')
        config_file: Optional path to JSON config
        input_dim: Number of input features
        args: Command-line arguments
        
    Returns:
        Configuration object (LSTMConfig, GRUConfig, or TransformerConfig)
    """
    if config_file:
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        config_dict['input_dim'] = input_dim  # Override from metadata
    else:
        config_dict = {'input_dim': input_dim}
    
    # Override with CLI arguments
    if args.learning_rate is not None:
        config_dict['learning_rate'] = args.learning_rate
    if args.batch_size is not None:
        config_dict['batch_size'] = args.batch_size
    if args.max_epochs is not None:
        config_dict['max_epochs'] = args.max_epochs
    
    # Create config object
    if model_type == 'lstm':
        return LSTMConfig.from_dict(config_dict) if 'from_dict' in dir(LSTMConfig) else LSTMConfig(**config_dict)
    elif model_type == 'gru':
        return GRUConfig.from_dict(config_dict) if 'from_dict' in dir(GRUConfig) else GRUConfig(**config_dict)
    else:  # transformer
        return TransformerConfig.from_dict(config_dict) if 'from_dict' in dir(TransformerConfig) else TransformerConfig(**config_dict)


def create_model(model_type: str, model_config: Any) -> nn.Module:
    """
    Create model instance from configuration.
    
    Args:
        model_type: Type of model
        model_config: Configuration object
        
    Returns:
        Model instance
    """
    config_dict = model_config.to_dict()
    
    if model_type == 'lstm':
        return LSTMForecaster(**config_dict)
    elif model_type == 'gru':
        return GRUForecaster(**config_dict)
    else:  # transformer
        return TransformerForecaster(**config_dict)


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for features, targets in tqdm(train_loader, desc="Training", leave=False):
        features, targets = features.to(device), targets.to(device)
        
        optimizer.zero_grad()
        predictions = model(features)
        
        # Reshape targets if necessary (ensure [batch, horizon])
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)
        
        # Ensure predictions and targets have same shape
        if predictions.shape != targets.shape:
            # If single-step prediction but multi-step target, or vice versa
            if predictions.shape[1] == 1 and targets.shape[1] > 1:
                targets = targets[:, 0:1]
            elif predictions.shape[1] > 1 and targets.shape[1] == 1:
                predictions = predictions[:, 0:1]
        
        loss = criterion(predictions, targets)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for features, targets in tqdm(val_loader, desc="Validating", leave=False):
            features, targets = features.to(device), targets.to(device)
            
            predictions = model(features)
            
            # Reshape targets if necessary (ensure [batch, horizon])
            if targets.dim() == 1:
                targets = targets.unsqueeze(1)
            
            # Ensure predictions and targets have same shape
            if predictions.shape != targets.shape:
                if predictions.shape[1] == 1 and targets.shape[1] > 1:
                    targets = targets[:, 0:1]
                elif predictions.shape[1] > 1 and targets.shape[1] == 1:
                    predictions = predictions[:, 0:1]
            
            loss = criterion(predictions, targets)
            total_loss += loss.item()
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    # Compute MAE
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)
    
    # For multi-step, compute MAE on first step
    if all_predictions.ndim > 1 and all_predictions.shape[1] > 1:
        mae = mean_absolute_error(all_targets, all_predictions[:, 0])
    else:
        mae = mean_absolute_error(all_targets.flatten(), all_predictions.flatten())
    
    return total_loss / len(val_loader), mae


def train_model(model, dataloaders, model_config, device, checkpoint_dir, ticker, args):
    """
    Main training loop with early stopping and checkpointing.
    
    Returns:
        Dict with training history and best metrics
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=model_config.learning_rate)
    criterion = nn.MSELoss()
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training state
    best_val_loss = float('inf')
    best_val_mae = float('inf')
    patience_counter = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': [],
        'learning_rate': []
    }
    
    logger.info(f"Starting training for {model_config.max_epochs} epochs")
    logger.info(f"Model parameters: {count_model_parameters(model):,}")
    
    for epoch in range(model_config.max_epochs):
        # Train
        train_loss = train_epoch(model, dataloaders['train'], optimizer, criterion, device)
        
        # Validate
        val_loss, val_mae = validate(model, dataloaders['val'], criterion, device)
        
        # Update scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        history['learning_rate'].append(current_lr)
        
        # Log metrics
        logger.info(
            f"Epoch {epoch+1}/{model_config.max_epochs} - "
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
            f"Val MAE: {val_mae:.4f}, LR: {current_lr:.6f}"
        )
        
        if MLFLOW_AVAILABLE:
            mlflow.log_metrics({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_mae': val_mae,
                'learning_rate': current_lr
            }, step=epoch)
        
        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_mae = val_mae
            patience_counter = 0
            
            # Save checkpoint
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = checkpoint_dir / f"{args.model_type}_{ticker}_{timestamp}.pth"
            
            metadata = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_mae': val_mae,
                'hyperparameters': model_config.to_dict(),
                'ticker': ticker,
                'timestamp': timestamp
            }
            
            model.save_checkpoint(str(checkpoint_path), metadata, optimizer)
            logger.info(f"Saved checkpoint: {checkpoint_path}")
            
            if MLFLOW_AVAILABLE:
                mlflow.log_artifact(str(checkpoint_path))
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= model_config.early_stopping_patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    return {
        'history': history,
        'best_val_loss': best_val_loss,
        'best_val_mae': best_val_mae,
        'final_epoch': epoch + 1
    }


def objective(trial, model_type, data_dir, ticker, device, args, input_dim):
    """Optuna objective function for hyperparameter tuning."""
    
    # Suggest hyperparameters
    if model_type in ['lstm', 'gru']:
        suggested_config = {
            'input_dim': input_dim,
            'hidden_dim': trial.suggest_int('hidden_dim', 64, 256, step=32),
            'num_layers': trial.suggest_int('num_layers', 1, 3),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-2),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
            'forecast_horizon': 7,
            'max_epochs': 50,  # Reduced for tuning
            'early_stopping_patience': 5
        }
        config_cls = LSTMConfig if model_type == 'lstm' else GRUConfig
    else:  # transformer
        suggested_config = {
            'input_dim': input_dim,
            'd_model': trial.suggest_categorical('d_model', [64, 128, 256]),
            'nhead': trial.suggest_categorical('nhead', [2, 4, 8]),
            'num_layers': trial.suggest_int('num_layers', 1, 3),
            'dim_feedforward': trial.suggest_categorical('dim_feedforward', [256, 512, 1024]),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'patch_len': trial.suggest_categorical('patch_len', [6, 10, 12, 15, 20]),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-2),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
            'forecast_horizon': 7,
            'max_epochs': 50,
            'early_stopping_patience': 5
        }
        config_cls = TransformerConfig
    
    # Validate config
    model_config = config_cls(**suggested_config)
    try:
        model_config.validate()
    except ValueError as e:
        logger.warning(f"Invalid hyperparameters: {e}")
        return float('inf')
    
    # Create model and dataloaders
    model = create_model(model_type, model_config).to(device)
    dataloaders = create_dataloaders(
        data_dir, 
        batch_size=model_config.batch_size,
        num_workers=0  # Disable multiprocessing for Optuna
    )
    
    # Train
    results = train_model(
        model, dataloaders, model_config, device,
        args.checkpoint_dir, ticker, args
    )
    
    return results['best_val_mae']


def main():
    """Main training workflow."""
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    logger.info(f"Set random seed to {args.seed}")
    
    # Determine device
    if args.device == 'auto':
        device = get_device()
    else:
        device = args.device
    logger.info(f"Using device: {device}")
    
    # Find processed data
    data_dir = find_latest_data_dir(Path(args.data_dir), args.ticker)
    
    # Load metadata
    metadata_path = data_dir / "metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    input_dim = len(metadata.get('feature_names', []))
    logger.info(f"Input dimension: {input_dim}")
    
    # Hyperparameter tuning mode
    if args.tune:
        if not OPTUNA_AVAILABLE:
            logger.error("Optuna not available. Cannot run hyperparameter tuning.")
            return
        
        # Start MLflow run only after checks
        if MLFLOW_AVAILABLE:
            mlflow.set_experiment(args.experiment_name)
            mlflow.start_run()
            mlflow.log_params({
                'model_type': args.model_type,
                'ticker': args.ticker,
                'seed': args.seed,
                'device': device,
                'mode': 'tuning'
            })
        
        logger.info(f"Starting Optuna hyperparameter tuning with {args.num_trials} trials")
        
        study = optuna.create_study(direction='minimize')
        # study.set_user_attr('input_dim', input_dim) # Not needed with closure
        
        study.optimize(
            lambda trial: objective(trial, args.model_type, data_dir, args.ticker, device, args, input_dim),
            n_trials=args.num_trials
        )
        
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best validation MAE: {study.best_value:.4f}")
        logger.info(f"Best hyperparameters: {study.best_params}")
        
        if MLFLOW_AVAILABLE:
            mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
            mlflow.log_metric("best_val_mae", study.best_value)
        
        # Train final model with best config
        best_config_dict = {**study.best_params, 'input_dim': input_dim}
        # Add missing fields
        best_config_dict.update({
            'forecast_horizon': 7,
            'max_epochs': args.max_epochs,
            'early_stopping_patience': 10
        })
        
        if args.model_type == 'lstm':
            model_config = LSTMConfig(**best_config_dict)
        elif args.model_type == 'gru':
            model_config = GRUConfig(**best_config_dict)
        else:
            model_config = TransformerConfig(**best_config_dict)
    else:
        # Start MLflow run for standard training
        if MLFLOW_AVAILABLE:
            mlflow.set_experiment(args.experiment_name)
            mlflow.start_run()
            mlflow.log_params({
                'model_type': args.model_type,
                'ticker': args.ticker,
                'seed': args.seed,
                'device': device,
                'mode': 'training'
            })

        # Load config
        model_config = load_model_config(args.model_type, args.config_file, input_dim, args)
        model_config.validate()
    
    # Create model
    model = create_model(args.model_type, model_config).to(device)
    logger.info(f"Created {args.model_type.upper()} model with {count_model_parameters(model):,} parameters")
    
    # Create dataloaders
    dataloaders = create_dataloaders(
        data_dir,
        batch_size=model_config.batch_size,
        num_workers=4
    )
    
    logger.info(f"Train batches: {len(dataloaders['train'])}, Val batches: {len(dataloaders['val'])}")
    
    # Train
    results = train_model(model, dataloaders, model_config, device, args.checkpoint_dir, args.ticker, args)
    
    # Log final results
    logger.info(f"Training completed!")
    logger.info(f"Best validation loss: {results['best_val_loss']:.4f}")
    logger.info(f"Final epoch: {results['final_epoch']}")
    
    if MLFLOW_AVAILABLE:
        mlflow.log_metric("final_best_val_loss", results['best_val_loss'])
        mlflow.end_run()
    
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Model Type: {args.model_type.upper()}")
    print(f"Ticker: {args.ticker}")
    print(f"Best Validation Loss: {results['best_val_loss']:.4f}")
    print(f"Total Epochs: {results['final_epoch']}")
    print(f"Checkpoints saved to: {args.checkpoint_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
