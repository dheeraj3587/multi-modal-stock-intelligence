"""
Unit tests for forecasting models.

Tests model architectures, checkpoint functionality, and configuration
validation per docs/metrics_and_evaluation.md Section 6.3.
"""

import pytest
import torch
import tempfile
from pathlib import Path

from models import (
    LSTMForecaster, GRUForecaster, TransformerForecaster,
    LSTMConfig, GRUConfig, TransformerConfig
)


class TestLSTMForecaster:
    """Test LSTM model."""
    
    def test_forward_pass_shape(self):
        """Test output shape of forward pass."""
        model = LSTMForecaster(input_dim=10, hidden_dim=128, forecast_horizon=7)
        
        # Input: [batch=4, seq=60, features=10]
        input_tensor = torch.randn(4, 60, 10)
        output = model(input_tensor)
        
        # Expected output: [batch=4, forecast_horizon=7]
        assert output.shape == (4, 7)
    
    def test_model_type(self):
        """Test model type property."""
        model = LSTMForecaster(input_dim=10)
        assert model.model_type == 'lstm'
    
    def test_parameter_count(self):
        """Test reasonable parameter count."""
        model = LSTMForecaster(input_dim=10, hidden_dim=128, num_layers=2)
        params = model.count_parameters()
        
        # Should be in reasonable range (roughly 100k-500k for default config)
        assert 10000 < params < 1000000
    
    @pytest.mark.parametrize("hidden_dim", [64, 128, 256])
    def test_different_hidden_dims(self, hidden_dim):
        """Test model with different hidden dimensions."""
        model = LSTMForecaster(input_dim=10, hidden_dim=hidden_dim)
        input_tensor = torch.randn(2, 60, 10)
        output = model(input_tensor)
        
        assert output.shape == (2, 7)


class TestGRUForecaster:
    """Test GRU model."""
    
    def test_forward_pass_shape(self):
        """Test output shape of forward pass."""
        model = GRUForecaster(input_dim=10, hidden_dim=128, forecast_horizon=7)
        
        input_tensor = torch.randn(4, 60, 10)
        output = model(input_tensor)
        
        assert output.shape == (4, 7)
    
    def test_model_type(self):
        """Test model type property."""
        model = GRUForecaster(input_dim=10)
        assert model.model_type == 'gru'
    
    def test_fewer_params_than_lstm(self):
        """Test that GRU has fewer parameters than LSTM."""
        lstm_model = LSTMForecaster(input_dim=10, hidden_dim=128, num_layers=2)
        gru_model = GRUForecaster(input_dim=10, hidden_dim=128, num_layers=2)
        
        lstm_params = lstm_model.count_parameters()
        gru_params = gru_model.count_parameters()
        
        # GRU typically has ~75% of LSTM parameters
        assert gru_params < lstm_params


class TestTransformerForecaster:
    """Test Transformer model."""
    
    def test_forward_pass_shape(self):
        """Test output shape of forward pass."""
        model = TransformerForecaster(
            input_dim=10, 
            d_model=128, 
            nhead=4, 
            patch_len=12,
            forecast_horizon=7
        )
        
        input_tensor = torch.randn(4, 60, 10)
        output = model(input_tensor)
        
        assert output.shape == (4, 7)
    
    def test_model_type(self):
        """Test model type property."""
        model = TransformerForecaster(input_dim=10)
        assert model.model_type == 'transformer'
    
    @pytest.mark.parametrize("patch_len", [6, 10, 12, 15, 20, 30])
    def test_different_patch_lengths(self, patch_len):
        """Test model with different patch lengths that divide 60."""
        model = TransformerForecaster(input_dim=10, patch_len=patch_len)
        input_tensor = torch.randn(2, 60, 10)
        output = model(input_tensor)
        
        assert output.shape == (2, 7)
    
    def test_invalid_patch_length(self):
        """Test that invalid patch length raises error during validation."""
        # Patch length that doesn't divide 60
        config = TransformerConfig(input_dim=10, patch_len=7)
        
        with pytest.raises(ValueError, match="patch_len"):
            config.validate()


class TestCheckpointFunctionality:
    """Test checkpoint save/load for all models."""
    
    @pytest.mark.parametrize("model_class", [LSTMForecaster, GRUForecaster, TransformerForecaster])
    def test_save_and_load_checkpoint(self, model_class):
        """Test checkpoint save and load preserves weights."""
        model = model_class(input_dim=10)
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "test_checkpoint.pth"
            
            # Save checkpoint
            metadata = {'epoch': 10, 'loss': 0.5}
            model.save_checkpoint(str(checkpoint_path), metadata)
            
            # Verify files created
            assert checkpoint_path.exists()
            assert checkpoint_path.with_suffix('.json').exists()
            
            # Create new model and load checkpoint
            model2 = model_class(input_dim=10)
            loaded_metadata = model2.load_checkpoint(str(checkpoint_path))
            
            # Verify metadata
            assert loaded_metadata['epoch'] == 10
            assert loaded_metadata['loss'] == 0.5
            
            # Verify weights match
            for p1, p2 in zip(model.parameters(), model2.parameters()):
                assert torch.allclose(p1, p2)
    
    def test_checkpoint_not_found(self):
        """Test error when checkpoint doesn't exist."""
        model = LSTMForecaster(input_dim=10)
        
        with pytest.raises(FileNotFoundError):
            model.load_checkpoint("nonexistent.pth")


class TestModelConfigs:
    """Test model configuration classes."""
    
    def test_lstm_config_serialization(self):
        """Test LSTMConfig to_dict and from_dict."""
        config = LSTMConfig(
            input_dim=20,
            hidden_dim=128,
            num_layers=2,
            dropout=0.2
        )
        
        config_dict = config.to_dict()
        config2 = LSTMConfig.from_dict(config_dict)
        
        assert config.input_dim == config2.input_dim
        assert config.hidden_dim == config2.hidden_dim
        assert config.num_layers == config2.num_layers
        assert config.dropout == config2.dropout
    
    def test_lstm_config_validation_valid(self):
        """Test validation with valid config."""
        config = LSTMConfig(input_dim=10, hidden_dim=128, dropout=0.2)
        config.validate()  # Should not raise
    
    def test_lstm_config_validation_invalid_dropout(self):
        """Test validation catches invalid dropout."""
        config = LSTMConfig(input_dim=10, dropout=1.5)
        
        with pytest.raises(ValueError, match="dropout"):
            config.validate()
    
    def test_lstm_config_validation_invalid_hidden_dim(self):
        """Test validation catches invalid hidden_dim."""
        config = LSTMConfig(input_dim=10, hidden_dim=-10)
        
        with pytest.raises(ValueError, match="hidden_dim"):
            config.validate()
    
    def test_transformer_config_nhead_validation(self):
        """Test Transformer config validates nhead divides d_model."""
        config = TransformerConfig(input_dim=10, d_model=128, nhead=3)  # 128 not divisible by 3
        
        with pytest.raises(ValueError, match="d_model.*nhead"):
            config.validate()


class TestInvalidInputs:
    """Test models handle invalid inputs gracefully."""
    
    def test_lstm_wrong_input_shape(self):
        """Test LSTM with 2D input raises error."""
        model = LSTMForecaster(input_dim=10)
        
        # 2D input instead of 3D
        input_tensor = torch.randn(4, 10)
        
        with pytest.raises(Exception):  # Will raise RuntimeError from PyTorch
            model(input_tensor)
    
    def test_lstm_wrong_feature_dim(self):
        """Test LSTM with wrong number of features."""
        model = LSTMForecaster(input_dim=10)
        
        # Input has 20 features instead of 10
        input_tensor = torch.randn(4, 60, 20)
        
        with pytest.raises(Exception):
            model(input_tensor)


@pytest.mark.parametrize("num_layers", [1, 2, 3])
def test_different_num_layers(num_layers):
    """Test models with different number of layers."""
    model = LSTMForecaster(input_dim=10, num_layers=num_layers)
    input_tensor = torch.randn(2, 60, 10)
    output = model(input_tensor)
    
    assert output.shape == (2, 7)
