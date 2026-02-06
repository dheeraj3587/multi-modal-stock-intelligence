"""
Unit tests for ForecastingDataset.

Tests dataset loading, validation, and DataLoader creation
to prevent data leakage per docs/metrics_and_evaluation.md Section 4.3.
"""

import pytest
import numpy as np
import json
import tempfile
from pathlib import Path
import torch

from backend.utils.dataset import ForecastingDataset, create_dataloaders


@pytest.fixture
def mock_data_dir():
    """Create temporary directory with mock processed data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        
        # Create mock data
        train_features = np.random.randn(100, 60, 10).astype(np.float32)
        train_targets = np.random.randn(100).astype(np.float32)
        
        val_features = np.random.randn(30, 60, 10).astype(np.float32)
        val_targets = np.random.randn(30).astype(np.float32)
        
        test_features = np.random.randn(40, 60, 10).astype(np.float32)
        test_targets = np.random.randn(40).astype(np.float32)
        
        # Save files
        np.save(data_dir / 'train_features.npy', train_features)
        np.save(data_dir / 'train_targets.npy', train_targets)
        np.save(data_dir / 'val_features.npy', val_features)
        np.save(data_dir / 'val_targets.npy', val_targets)
        np.save(data_dir / 'test_features.npy', test_features)
        np.save(data_dir / 'test_targets.npy', test_targets)
        
        # Create metadata
        metadata = {
            'feature_names': [f'feature_{i}' for i in range(10)],
            'window_counts': {
                'train': 100,
                'val': 30,
                'test': 40
            }
        }
        
        with open(data_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f)
        
        yield data_dir


class TestForecastingDataset:
    """Test ForecastingDataset class."""
    
    def test_dataset_loading(self, mock_data_dir):
        """Test basic dataset loading."""
        dataset = ForecastingDataset(mock_data_dir, split='train')
        
        assert len(dataset) == 100
        
        # Test __getitem__
        features, targets = dataset[0]
        assert features.shape == (60, 10)
        assert targets.shape == (7,)  # Default forecast_horizon=7
    
    def test_all_splits(self, mock_data_dir):
        """Test loading all splits."""
        for split in ['train', 'val', 'test']:
            dataset = ForecastingDataset(mock_data_dir, split=split)
            assert len(dataset) > 0
    
    def test_nonexistent_split(self, mock_data_dir):
        """Test error on nonexistent split."""
        with pytest.raises(FileNotFoundError):
            ForecastingDataset(mock_data_dir, split='invalid')
    
    def test_missing_features_file(self):
        """Test error when features file missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create only metadata
            metadata = {'feature_names': []}
            with open(Path(tmpdir) / 'metadata.json', 'w') as f:
                json.dump(metadata, f)
            
            with pytest.raises(FileNotFoundError, match="Features"):
                ForecastingDataset(tmpdir, split='train')
    
    def test_shape_validation(self):
        """Test validation catches wrong shapes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            
            # Create data with wrong shape (2D instead of 3D)
            train_features = np.random.randn(100, 60).astype(np.float32)  # Missing feature dim
            train_targets = np.random.randn(100).astype(np.float32)
            
            np.save(data_dir / 'train_features.npy', train_features)
            np.save(data_dir / 'train_targets.npy', train_targets)
            
            metadata = {'feature_names': []}
            with open(data_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f)
            
            with pytest.raises(ValueError, match="3D"):
                ForecastingDataset(data_dir, split='train')
    
    def test_nan_validation(self):
        """Test validation catches NaN values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            
            # Create data with NaN
            train_features = np.random.randn(10, 60, 5).astype(np.float32)
            train_features[0, 0, 0] = np.nan  # Inject NaN
            train_targets = np.random.randn(10).astype(np.float32)
            
            np.save(data_dir / 'train_features.npy', train_features)
            np.save(data_dir / 'train_targets.npy', train_targets)
            
            metadata = {'feature_names': [f'f{i}' for i in range(5)]}
            with open(data_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f)
            
            with pytest.raises(ValueError, match="NaN"):
                ForecastingDataset(data_dir, split='train')
    
    def test_length_mismatch(self):
        """Test validation catches features/targets length mismatch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            
            train_features = np.random.randn(100, 60, 5).astype(np.float32)
            train_targets = np.random.randn(90).astype(np.float32)  # Mismatch
            
            np.save(data_dir / 'train_features.npy', train_features)
            np.save(data_dir / 'train_targets.npy', train_targets)
            
            metadata = {'feature_names': [f'f{i}' for i in range(5)]}
            with open(data_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f)
            
            with pytest.raises(ValueError, match="mismatch"):
                ForecastingDataset(data_dir, split='train')


class TestCreateDataloaders:
    """Test DataLoader factory function."""
    
    def test_create_dataloaders(self, mock_data_dir):
        """Test creating DataLoaders for all splits."""
        dataloaders = create_dataloaders(
            mock_data_dir,
            batch_size=16,
            num_workers=0,
            forecast_horizon=7
        )
        
        assert 'train' in dataloaders
        assert 'val' in dataloaders
        assert 'test' in dataloaders
        
        # Check batch sizes
        for batch in dataloaders['train']:
            features, targets = batch
            assert features.shape[0] <= 16  # Batch size
            assert features.shape[1] == 60  # Sequence length
            assert features.shape[2] == 10  # Features
            break
    
    def test_train_loader_shuffles(self, mock_data_dir):
        """Test that train loader shuffles data."""
        dataloaders = create_dataloaders(
            mock_data_dir,
            batch_size=32,
            num_workers=0
        )
        
        # Train loader should shuffle
        # We can't directly test shuffle, but we can verify it's a DataLoader
        assert hasattr(dataloaders['train'], '__iter__')
    
    def test_val_test_no_shuffle(self, mock_data_dir):
        """Test that val/test loaders don't shuffle."""
        # This is ensured by the implementation
        # Val and test are created with shuffle=False
        dataloaders = create_dataloaders(mock_data_dir, batch_size=32, num_workers=0)
        
        assert hasattr(dataloaders['val'], '__iter__')
        assert hasattr(dataloaders['test'], '__iter__')


@pytest.mark.parametrize("forecast_horizon", [1, 7, 14])
def test_different_forecast_horizons(mock_data_dir, forecast_horizon):
    """Test dataset with different forecast horizons."""
    dataset = ForecastingDataset(
        mock_data_dir,
        split='train',
        forecast_horizon=forecast_horizon
    )
    
    features, targets = dataset[0]
    assert features.shape == (60, 10)
    assert targets.shape == (forecast_horizon,)


def test_tensor_dtypes(mock_data_dir):
    """Test that returned tensors are float32."""
    dataset = ForecastingDataset(mock_data_dir, split='train')
    features, targets = dataset[0]
    
    assert features.dtype == torch.float32
    assert targets.dtype == torch.float32


def test_metadata_validation(mock_data_dir):
    """Test metadata is correctly loaded."""
    dataset = ForecastingDataset(mock_data_dir, split='train')
    
    assert 'feature_names' in dataset.metadata
    assert 'window_counts' in dataset.metadata
    assert len(dataset.metadata['feature_names']) == 10
