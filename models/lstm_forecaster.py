"""
LSTM-based forecasting model for time-series prediction.

This module implements a 2-layer LSTM architecture for multi-step stock price forecasting.
"""

import torch
import torch.nn as nn
from .base_forecaster import BaseForecaster


class LSTMForecaster(BaseForecaster):
    """
    LSTM-based time-series forecasting model.
    
    Architecture:
    - 2-layer LSTM with configurable hidden dimensions
    - Dropout for regularization
    - Fully connected layer for multi-step prediction
    
    Rationale:
    - LSTM chosen for capturing long-term dependencies in 60-day sequences
    - Multi-step output (forecast_horizon=7) for direct prediction
    - Aligned with docs/metrics_and_evaluation.md Section 1.1 targets
      (MAE reduction ≥15% vs ARIMA baseline)
    
    Input shape: [batch, 60, features]
    Output shape: [batch, 7] (multi-step predictions)
    
    Usage:
        model = LSTMForecaster(input_dim=20, hidden_dim=128)
        predictions = model(input_tensor)  # [batch, 7]
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        forecast_horizon: int = 7
    ):
        """
        Initialize LSTM forecaster.
        
        Args:
            input_dim (int): Number of input features (from metadata.json)
            hidden_dim (int): LSTM hidden dimension (default: 128)
            num_layers (int): Number of LSTM layers (default: 2)
            dropout (float): Dropout rate (default: 0.2)
            forecast_horizon (int): Number of steps to forecast (default: 7 from .env.template)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.forecast_horizon = forecast_horizon
        
        # LSTM layer with batch_first=True for [batch, seq, features] input
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layer for multi-step prediction
        self.fc = nn.Linear(hidden_dim, forecast_horizon)
        
    @property
    def model_type(self) -> str:
        """Return model type identifier."""
        return 'lstm'
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LSTM.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch, 60, features]
            
        Returns:
            torch.Tensor: Predictions of shape [batch, forecast_horizon]
        """
        # Pass through LSTM: output shape [batch, seq_len, hidden_dim]
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Extract final hidden state: [batch, hidden_dim]
        final_hidden = lstm_out[:, -1, :]
        
        # Project to forecast_horizon outputs: [batch, forecast_horizon]
        predictions = self.fc(final_hidden)
        
        return predictions
