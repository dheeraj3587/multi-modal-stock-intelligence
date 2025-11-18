"""
Models package for Phase 3: Time-Series Forecasting.

This package contains PyTorch forecasting models (LSTM, GRU, Transformer)
with a shared base class for consistent API.

Usage Example:
    from models import LSTMForecaster, LSTMConfig
    
    config = LSTMConfig(input_dim=20, hidden_dim=128)
    model = LSTMForecaster(**config.to_dict())
    predictions = model(input_tensor)
"""

from .base_forecaster import BaseForecaster
from .lstm_forecaster import LSTMForecaster
from .gru_forecaster import GRUForecaster
from .transformer_forecaster import TransformerForecaster
from .model_config import LSTMConfig, GRUConfig, TransformerConfig

__all__ = [
    'BaseForecaster',
    'LSTMForecaster',
    'GRUForecaster',
    'TransformerForecaster',
    'LSTMConfig',
    'GRUConfig',
    'TransformerConfig',
]
