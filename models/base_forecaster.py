"""
Base forecaster class for all time-series forecasting models.

This module defines an abstract base class that provides a consistent interface
for LSTM, GRU, and Transformer forecasting models.
"""

import torch
import torch.nn as nn
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np


class BaseForecaster(nn.Module, ABC):
    """
    Abstract base class for all forecasting models.
    
    Provides shared functionality for:
    - Forward pass interface (abstract method)
    - Checkpoint saving/loading with metadata
    - Parameter counting
    - Inference utilities with optional inverse scaling
    
    Usage Example:
        class LSTMForecaster(BaseForecaster):
            @property
            def model_type(self):
                return 'lstm'
                
            def forward(self, x):
                # Implementation
                pass
    """
    
    def __init__(self):
        super().__init__()
        
    @property
    @abstractmethod
    def model_type(self) -> str:
        """
        Return string identifier for the model type.
        
        Returns:
            str: Model type identifier (e.g., 'lstm', 'gru', 'transformer')
        """
        pass
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch, lookback, features]
                             where lookback is typically 60 days
            
        Returns:
            torch.Tensor: Output tensor of shape [batch, forecast_horizon]
                         where forecast_horizon is typically 7 days
        """
        pass
        
    def save_checkpoint(
        self, 
        path: str, 
        metadata: Optional[Dict[str, Any]] = None, 
        optimizer: Optional[torch.optim.Optimizer] = None
    ):
        """
        Save model checkpoint including state dict and metadata.
        
        Checkpoint format:
        - .pth file: PyTorch state dict + optimizer state + metadata
        - .json file: Human-readable metadata for inspection
        
        Args:
            path (str): Path to save the checkpoint (.pth file)
            metadata (Dict[str, Any], optional): Additional metadata to save
                (e.g., hyperparameters, training history, split info)
            optimizer (torch.optim.Optimizer, optional): Optimizer state for resuming training
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'metadata': metadata or {},
            'model_type': self.model_type
        }
        
        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            
        torch.save(checkpoint, path)
        
        # Save metadata as separate JSON for easy inspection
        if metadata:
            json_path = path.with_suffix('.json')
            serializable_metadata = self._make_serializable(metadata)
            with open(json_path, 'w') as f:
                json.dump(serializable_metadata, f, indent=4)
                
    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """
        Load model weights from checkpoint.
        
        Args:
            path (str): Path to the checkpoint file
            
        Returns:
            Dict[str, Any]: Metadata dictionary from the checkpoint
            
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {path}")
            
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        
        self.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint.get('metadata', {})
        
    def count_parameters(self) -> int:
        """
        Return total number of trainable parameters.
        
        Returns:
            int: Total trainable parameter count
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
    def predict(self, x: torch.Tensor, scaler=None) -> np.ndarray:
        """
        Perform inference with optional inverse scaling.
        
        Args:
            x (torch.Tensor): Input tensor [batch, lookback, features]
            scaler: Optional scaler object with inverse_transform method
                   (from backend.utils.preprocessing)
            
        Returns:
            np.ndarray: Predictions (inverse transformed if scaler provided)
        """
        self.eval()
        with torch.no_grad():
            output = self(x)
            
        output_np = output.cpu().numpy()
        
        if scaler:
            # Inverse transform predictions to original scale
            # Note: Implementation depends on scaler structure from preprocessing.py
            try:
                return scaler.inverse_transform(output_np)
            except Exception:
                # Fallback if scaler doesn't match expected interface
                return output_np
            
        return output_np
        
    def _make_serializable(self, obj):
        """
        Helper to convert objects to JSON-serializable format.
        
        Args:
            obj: Object to serialize
            
        Returns:
            JSON-serializable version of obj
        """
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return str(obj)
