"""
Transformer-based forecasting model using PatchTST architecture.

This module implements a patch-based Transformer for time-series forecasting,
designed to capture long-range dependencies more effectively than RNNs.
"""

import torch
import torch.nn as nn
import math
from .base_forecaster import BaseForecaster


class TransformerForecaster(BaseForecaster):
    """
    PatchTST-based time-series forecasting model.
    
    Architecture:
    1. Patching layer: Convert [batch, 60, features] to [batch, num_patches, patch_len*features]
    2. Linear projection to d_model dimension
    3. Learnable positional encoding
    4. TransformerEncoder with multi-head self-attention
    5. Global average pooling over patch dimension
    6. Linear head projecting to forecast_horizon outputs
    
    Rationale:
    - Patch-based approach reduces sequence length for efficient self-attention
    - Captures long-range dependencies better than LSTM/GRU
    - Proven effective on long-horizon forecasting (Nie et al. 2023)
    - patch_len should divide lookback_window evenly (e.g., 12 divides 60)
    
    Input shape: [batch, 60, features]
    Output shape: [batch, 7] (multi-step predictions)
    
    Hyperparameter Tuning Recommendations:
    - patch_len: {6, 10, 12, 15, 20, 30} (divisors of 60)
    - d_model: {64, 128, 256}
    - nhead: {2, 4, 8} (must divide d_model)
    
    Usage:
        model = TransformerForecaster(input_dim=20, d_model=128, patch_len=12)
        predictions = model(input_tensor)  # [batch, 7]
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.2,
        patch_len: int = 12,
        forecast_horizon: int = 7
    ):
        """
        Initialize Transformer forecaster.
        
        Args:
            input_dim (int): Number of input features
            d_model (int): Embedding dimension (default: 128)
            nhead (int): Number of attention heads (default: 4, must divide d_model)
            num_layers (int): Number of transformer encoder layers (default: 2)
            dim_feedforward (int): Dimension of FFN (default: 512)
            dropout (float): Dropout rate (default: 0.2)
            patch_len (int): Length of each patch (default: 12 for 60-day sequences)
            forecast_horizon (int): Number of steps to forecast (default: 7)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout
        self.patch_len = patch_len
        self.forecast_horizon = forecast_horizon
        
        # Calculate number of patches (assuming lookback_window=60)
        self.lookback_window = 60
        self.num_patches = self.lookback_window // patch_len
        
        # Patching: project each patch to d_model
        self.patch_embedding = nn.Linear(patch_len * input_dim, d_model)
        
        # Learnable positional encoding
        self.positional_encoding = nn.Parameter(
            torch.randn(1, self.num_patches, d_model)
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output head
        self.fc = nn.Linear(d_model, forecast_horizon)
        
        self.dropout = nn.Dropout(dropout)
        
    @property
    def model_type(self) -> str:
        """Return model type identifier."""
        return 'transformer'
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Transformer.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch, 60, features]
            
        Returns:
            torch.Tensor: Predictions of shape [batch, forecast_horizon]
        """
        batch_size = x.shape[0]
        
        # Patching: reshape to [batch, num_patches, patch_len, features]
        x = x.view(batch_size, self.num_patches, self.patch_len, self.input_dim)
        
        # Flatten patches: [batch, num_patches, patch_len * features]
        x = x.view(batch_size, self.num_patches, -1)
        
        # Project to d_model: [batch, num_patches, d_model]
        x = self.patch_embedding(x)
        x = self.dropout(x)
        
        # Add positional encoding
        x = x + self.positional_encoding
        
        # Pass through transformer encoder: [batch, num_patches, d_model]
        x = self.transformer_encoder(x)
        
        # Global average pooling over patches: [batch, d_model]
        x = x.mean(dim=1)
        
        # Project to forecast_horizon: [batch, forecast_horizon]
        predictions = self.fc(x)
        
        return predictions
