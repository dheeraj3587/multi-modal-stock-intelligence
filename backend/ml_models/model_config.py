"""
Configuration dataclasses for forecasting models.

Provides structured hyperparameter management with validation,
serialization, and default values aligned with evaluation targets.
"""

from dataclasses import dataclass, asdict, field
from typing import Dict, Any


class ModelConfig:
    """
    Base configuration class for all forecasting models.
    
    Provides common hyperparameters and shared functionality (serialization,
    validation) that all model-specific configs can inherit from.
    
    Note: This is NOT a dataclass itself to avoid field ordering issues.
    Subclasses should be dataclasses and include these common fields.
    """
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ModelConfig':
        """Create config from dictionary."""
        return cls(**d)
    
    def validate(self):
        """
        Validate common hyperparameter constraints.
        
        Subclasses should call super().validate() and add their own checks.
        
        Raises:
            ValueError: If constraints are violated
        """
        if hasattr(self, 'learning_rate') and self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if hasattr(self, 'batch_size') and self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if hasattr(self, 'max_epochs') and self.max_epochs < 1:
            raise ValueError(f"max_epochs must be >= 1, got {self.max_epochs}")
        if hasattr(self, 'early_stopping_patience') and self.early_stopping_patience < 1:
            raise ValueError(f"early_stopping_patience must be >= 1, got {self.early_stopping_patience}")
        if hasattr(self, 'forecast_horizon') and self.forecast_horizon < 1:
            raise ValueError(f"forecast_horizon must be >= 1, got {self.forecast_horizon}")


@dataclass
class LSTMConfig(ModelConfig):
    """
    Configuration for LSTM forecaster.
    
    Default values align with docs/metrics_and_evaluation.md Section 1.1 targets:
    - MAE reduction ≥15% vs ARIMA baseline
    - Directional Accuracy ≥55% (1-day)
    
    Hyperparameters are tunable via Optuna in scripts/train_forecasting_models.py.
    
    Fields:
        input_dim: Number of input features (required)
        hidden_dim: Hidden layer dimension
        num_layers: Number of LSTM layers
        dropout: Dropout rate for regularization
        forecast_horizon: Number of days to forecast ahead
        learning_rate: Optimizer learning rate
        batch_size: Training batch size
        max_epochs: Maximum training epochs
        early_stopping_patience: Epochs to wait before early stopping
    """
    
    input_dim: int
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    forecast_horizon: int = 7
    learning_rate: float = 0.001
    batch_size: int = 32
    max_epochs: int = 100
    early_stopping_patience: int = 10
    
    def validate(self):
        """
        Validate LSTM-specific hyperparameter constraints.
        
        Raises:
            ValueError: If constraints are violated
        """
        super().validate()  # Validate common fields
        
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {self.hidden_dim}")
        if not 0 <= self.dropout <= 1:
            raise ValueError(f"dropout must be in [0, 1], got {self.dropout}")
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {self.num_layers}")


@dataclass
class GRUConfig(ModelConfig):
    """
    Configuration for GRU forecaster.
    
    Identical structure to LSTMConfig to enable direct comparison
    in ablation studies.
    
    Fields:
        input_dim: Number of input features (required)
        hidden_dim: Hidden layer dimension
        num_layers: Number of GRU layers
        dropout: Dropout rate for regularization
        forecast_horizon: Number of days to forecast ahead
        learning_rate: Optimizer learning rate
        batch_size: Training batch size
        max_epochs: Maximum training epochs
        early_stopping_patience: Epochs to wait before early stopping
    """
    
    input_dim: int
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    forecast_horizon: int = 7
    learning_rate: float = 0.001
    batch_size: int = 32
    max_epochs: int = 100
    early_stopping_patience: int = 10
    
    def validate(self):
        """
        Validate GRU-specific hyperparameter constraints.
        
        Raises:
            ValueError: If constraints are violated
        """
        super().validate()  # Validate common fields
        
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {self.hidden_dim}")
        if not 0 <= self.dropout <= 1:
            raise ValueError(f"dropout must be in [0, 1], got {self.dropout}")
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {self.num_layers}")


@dataclass
class TransformerConfig(ModelConfig):
    """
    Configuration for Transformer forecaster (PatchTST architecture).
    
    Fields:
        input_dim: Number of input features (required)
        d_model: Embedding dimension
        nhead: Number of attention heads (must divide d_model)
        num_layers: Number of transformer layers
        dim_feedforward: FFN dimension
        dropout: Dropout rate for regularization
        patch_len: Patch length (should divide lookback_window=60 evenly)
        forecast_horizon: Number of days to forecast ahead
        learning_rate: Optimizer learning rate
        batch_size: Training batch size
        max_epochs: Maximum training epochs
        early_stopping_patience: Epochs to wait before early stopping
    
    Recommended patch_len values: {6, 10, 12, 15, 20, 30}
    """
    
    input_dim: int
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 512
    dropout: float = 0.2
    patch_len: int = 12
    forecast_horizon: int = 7
    learning_rate: float = 0.001
    batch_size: int = 32
    max_epochs: int = 100
    early_stopping_patience: int = 10
    
    def validate(self):
        """
        Validate Transformer-specific hyperparameter constraints.
        
        Raises:
            ValueError: If constraints are violated
        """
        super().validate()  # Validate common fields
        
        if self.d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {self.d_model}")
        if self.d_model % self.nhead != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by nhead ({self.nhead})")
        if not 0 <= self.dropout <= 1:
            raise ValueError(f"dropout must be in [0, 1], got {self.dropout}")
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {self.num_layers}")
        if 60 % self.patch_len != 0:
            raise ValueError(f"patch_len ({self.patch_len}) must divide lookback_window (60)")
