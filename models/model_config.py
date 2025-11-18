"""
Configuration dataclasses for forecasting models.

Provides structured hyperparameter management with validation,
serialization, and default values aligned with evaluation targets.
"""

from dataclasses import dataclass, asdict, field
from typing import Dict, Any


@dataclass
class LSTMConfig:
    """
    Configuration for LSTM forecaster.
    
    Default values align with docs/metrics_and_evaluation.md Section 1.1 targets:
    - MAE reduction ≥15% vs ARIMA baseline
    - Directional Accuracy ≥55% (1-day)
    
    Hyperparameters are tunable via Optuna in scripts/train_forecasting_models.py.
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'LSTMConfig':
        """Create config from dictionary."""
        return cls(**d)
    
    def validate(self):
        """
        Validate hyperparameter constraints.
        
        Raises:
            ValueError: If constraints are violated
        """
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {self.hidden_dim}")
        if not 0 <= self.dropout <= 1:
            raise ValueError(f"dropout must be in [0, 1], got {self.dropout}")
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {self.num_layers}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")


@dataclass
class GRUConfig:
    """
    Configuration for GRU forecaster.
    
    Identical structure to LSTMConfig to enable direct comparison
    in ablation studies.
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'GRUConfig':
        """Create config from dictionary."""
        return cls(**d)
    
    def validate(self):
        """
        Validate hyperparameter constraints.
        
        Raises:
            ValueError: If constraints are violated
        """
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {self.hidden_dim}")
        if not 0 <= self.dropout <= 1:
            raise ValueError(f"dropout must be in [0, 1], got {self.dropout}")
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {self.num_layers}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")


@dataclass
class TransformerConfig:
    """
    Configuration for Transformer forecaster (PatchTST architecture).
    
    Additional hyperparameters:
    - d_model: Embedding dimension
    - nhead: Number of attention heads (must divide d_model)
    - dim_feedforward: FFN dimension
    - patch_len: Patch length (should divide lookback_window=60 evenly)
    
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TransformerConfig':
        """Create config from dictionary."""
        return cls(**d)
    
    def validate(self):
        """
        Validate hyperparameter constraints.
        
        Raises:
            ValueError: If constraints are violated
        """
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
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
