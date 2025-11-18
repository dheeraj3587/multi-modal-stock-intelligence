"""
PyTorch Dataset for loading preprocessed time-series data.

Loads Phase 2 outputs (train/val/test .npy files) from 
data/processed/<TICKER>/YYYY-MM-DD_HHMMSS/ directories.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from pathlib import Path
from typing import Tuple, Dict


class ForecastingDataset(Dataset):
    """
    Dataset for loading preprocessed time-series features and targets.
    
    Loads outputs from scripts/feature_engineering.py:
    - {split}_features.npy: shape [samples, 60, num_features]
    - {split}_targets.npy: shape [samples]
    - metadata.json: scaler params, split boundaries, feature names
    
    Usage:
        dataset = ForecastingDataset(
            data_dir='data/processed/RELIANCE.NS/2024-01-01_120000',
            split='train',
            forecast_horizon=7
        )
        features, targets = dataset[0]  # Returns tensors
    """
    
    def __init__(
        self,
        data_dir: Path,
        split: str,
        forecast_horizon: int = 7
    ):
        """
        Initialize dataset.
        
        Args:
            data_dir (Path): Path to processed ticker directory
            split (str): One of 'train', 'val', or 'test'
            forecast_horizon (int): Number of forecast steps (default: 7)
            
        Raises:
            FileNotFoundError: If required .npy files don't exist
            ValueError: If data validation fails
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.forecast_horizon = forecast_horizon
        
        # Load features and targets
        features_path = self.data_dir / f"{split}_features.npy"
        targets_path = self.data_dir / f"{split}_targets.npy"
        metadata_path = self.data_dir / "metadata.json"
        
        if not features_path.exists():
            raise FileNotFoundError(f"Features not found: {features_path}")
        if not targets_path.exists():
            raise FileNotFoundError(f"Targets not found: {targets_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        self.features = np.load(features_path)
        self.targets = np.load(targets_path)
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Validate data
        self._validate()
        
    def _validate(self):
        """
        Validate loaded data against metadata and check for issues.
        
        Raises:
            ValueError: If validation fails
        """
        # Check features shape [samples, 60, num_features]
        if len(self.features.shape) != 3:
            raise ValueError(
                f"Expected 3D features array, got shape {self.features.shape}"
            )
        
        samples, lookback, num_features = self.features.shape
        
        if lookback != 60:
            raise ValueError(
                f"Expected lookback_window=60, got {lookback}"
            )
        
        # Check targets shape [samples]
        if len(self.targets.shape) != 1:
            raise ValueError(
                f"Expected 1D targets array, got shape {self.targets.shape}"
            )
        
        if len(self.targets) != samples:
            raise ValueError(
                f"Features ({samples} samples) and targets ({len(self.targets)} samples) mismatch"
            )
        
        # Check for NaN values
        if np.isnan(self.features).any():
            raise ValueError("Features contain NaN values")
        if np.isnan(self.targets).any():
            raise ValueError("Targets contain NaN values")
        
        # Validate against metadata if available
        if 'window_counts' in self.metadata and self.split in self.metadata['window_counts']:
            expected_count = self.metadata['window_counts'][self.split]
            if samples != expected_count:
                raise ValueError(
                    f"Sample count ({samples}) doesn't match metadata ({expected_count})"
                )
    
    def __len__(self) -> int:
        """Return number of samples."""
        # Ensure we have enough data for the forecast horizon
        return len(self.features) - self.forecast_horizon + 1
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx (int): Sample index
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (features, targets)
                - features: [60, num_features]
                - targets: [forecast_horizon]
        """
        features = torch.FloatTensor(self.features[idx])
        
        # Slice targets for multi-step forecasting
        # targets[idx] corresponds to the target for the window ending at idx
        # We want targets from idx to idx + forecast_horizon
        target_slice = self.targets[idx : idx + self.forecast_horizon]
        target = torch.FloatTensor(target_slice)
        
        return features, target


def create_dataloaders(
    data_dir: Path,
    batch_size: int = 32,
    num_workers: int = 4,
    forecast_horizon: int = 7
) -> Dict[str, DataLoader]:
    """
    Factory function to create DataLoaders for all splits.
    
    Args:
        data_dir (Path): Path to processed ticker directory
        batch_size (int): Batch size for DataLoaders (default: 32)
        num_workers (int): Number of workers for data loading (default: 4)
        forecast_horizon (int): Number of forecast steps (default: 7)
        
    Returns:
        Dict[str, DataLoader]: Dictionary with keys 'train', 'val', 'test'
        
    Usage:
        dataloaders = create_dataloaders(
            'data/processed/RELIANCE.NS/2024-01-01_120000',
            batch_size=32
        )
        for batch in dataloaders['train']:
            features, targets = batch
    """
    dataloaders = {}
    
    for split in ['train', 'val', 'test']:
        dataset = ForecastingDataset(
            data_dir=data_dir,
            split=split,
            forecast_horizon=forecast_horizon
        )
        
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),  # Shuffle only training data
            num_workers=num_workers,
            pin_memory=True
        )
    
    return dataloaders
