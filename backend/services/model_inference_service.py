"""
Model Inference Service for Stock Price Prediction

Loads trained LSTM/GRU/Transformer models and provides prediction API.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

logger = logging.getLogger(__name__)

# Try to import torch, but make it optional for API startup
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. ML models will not be loaded.")


class ModelInferenceService:
    """
    Service for loading and using trained forecasting models.
    
    Provides:
    - Model loading from checkpoints
    - Price prediction for stocks
    - Confidence estimation
    """
    
    def __init__(self, models_dir: str = None):
        """
        Initialize the inference service.
        
        Args:
            models_dir: Directory containing model checkpoints
        """
        self.models_dir = Path(models_dir) if models_dir else Path(__file__).parent.parent.parent.parent / "data" / "models"
        self.models: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Dict] = {}
        self.scalers: Dict[str, Any] = {}
        
        if TORCH_AVAILABLE:
            self._load_models()
        else:
            logger.warning("PyTorch not available. Running in fallback mode.")
    
    def _load_models(self):
        """Load all available model checkpoints."""
        # Ensure directory exists to suppress warnings
        if not self.models_dir.exists():
            try:
                self.models_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.warning(f"Could not create models directory: {e}")
                
        if not self.models_dir.exists():
            logger.warning(f"Models directory not found: {self.models_dir}")
            return
        
        # Look for model checkpoints
        checkpoint_files = list(self.models_dir.glob("*.pth"))
        
        if not checkpoint_files:
            logger.warning("No model checkpoints found.")
            return
        
        for checkpoint_path in checkpoint_files:
            try:
                self._load_model_checkpoint(checkpoint_path)
            except Exception as e:
                logger.error(f"Failed to load model from {checkpoint_path}: {e}")
    
    def _load_model_checkpoint(self, checkpoint_path: Path):
        """Load a single model checkpoint."""
        from backend.ml_models import LSTMForecaster, GRUForecaster, TransformerForecaster
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        
        model_type = checkpoint.get('model_type', 'unknown')
        metadata = checkpoint.get('metadata', {})
        
        # Get model configuration
        input_dim = metadata.get('input_dim', 20)
        hidden_dim = metadata.get('hidden_dim', 128)
        num_layers = metadata.get('num_layers', 2)
        dropout = metadata.get('dropout', 0.2)
        forecast_horizon = metadata.get('forecast_horizon', 7)
        
        # Create model instance
        if model_type == 'lstm':
            model = LSTMForecaster(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                forecast_horizon=forecast_horizon
            )
        elif model_type == 'gru':
            model = GRUForecaster(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                forecast_horizon=forecast_horizon
            )
        elif model_type == 'transformer':
            model = TransformerForecaster(
                input_dim=input_dim,
                d_model=metadata.get('d_model', 128),
                nhead=metadata.get('nhead', 8),
                num_layers=num_layers,
                dropout=dropout,
                forecast_horizon=forecast_horizon
            )
        else:
            logger.warning(f"Unknown model type: {model_type}")
            return
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Store model
        model_name = checkpoint_path.stem
        self.models[model_name] = model
        self.model_metadata[model_name] = metadata
        
        logger.info(f"Loaded {model_type} model: {model_name}")
    

    def predict_price(
        self, 
        symbol: str, 
        current_price: float, 
        price_history: List[float] = None,
        sentiment_score: float = 0.0
    ) -> Optional[Dict]:
        """
        Predict future price for a stock.
        
        Args:
            symbol: Stock symbol
            current_price: Current stock price
            price_history: Historical price data (optional)
            sentiment_score: Sentiment score from -1 to 1
            
        Returns:
            Dict with predicted_price, confidence, short_term_outlook, long_term_outlook
            OR None if prediction cannot be made.
        """
        if not TORCH_AVAILABLE or not self.models:
            return None
        
        try:
            # Use the first available model (or implement model selection logic)
            model_name = list(self.models.keys())[0]
            model = self.models[model_name]
            metadata = self.model_metadata[model_name]
            
            # Prepare input data
            lookback = metadata.get('lookback', 60)
            input_dim = metadata.get('input_dim', 20)
            
            # Check sufficient history
            if not price_history or len(price_history) < lookback:
                # Insufficient data for prediction
                return None

            # Create feature vector
            features = self._prepare_features(price_history[-lookback:], sentiment_score, input_dim)
            
            # Convert to tensor
            input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            
            # Make prediction
            with torch.no_grad():
                prediction = model(input_tensor)
            
            # Convert prediction to price
            predicted_change = prediction[0][0].item()  # First day prediction
            predicted_price = current_price * (1 + predicted_change)
            
            # Calculate confidence based on model variance
            forecast_horizon = metadata.get('forecast_horizon', 7)
            predictions = prediction[0].numpy()
            
            # Confidence decreases with time
            confidence = self._calculate_confidence(predictions, sentiment_score)
            
            # Determine outlook
            short_term, long_term, recommendation = self._determine_outlook(
                current_price, predicted_price, predictions, sentiment_score
            )
            
            return {
                "predicted_price": round(predicted_price, 2),
                "forecast_confidence": round(confidence, 1),
                "short_term_outlook": short_term,
                "long_term_outlook": long_term,
                "recommendation": recommendation,
                "model_used": model_name
            }
            
        except Exception as e:
            logger.error(f"Prediction failed for {symbol}: {e}")
            return None
    
    def _prepare_features(
        self, 
        price_history: List[float], 
        sentiment_score: float,
        input_dim: int
    ) -> np.ndarray:
        """Prepare feature matrix from price history."""
        lookback = len(price_history)
        
        # Normalize prices
        prices = np.array(price_history)
        normalized_prices = (prices - prices.mean()) / (prices.std() + 1e-8)
        
        # Create feature matrix
        features = np.zeros((lookback, input_dim))
        
        # Price features
        features[:, 0] = normalized_prices
        
        # Add technical indicators
        if lookback >= 10:
            # Moving averages
            for i in range(lookback):
                start = max(0, i - 5)
                features[i, 1] = normalized_prices[start:i+1].mean() if i > 0 else normalized_prices[0]
                start = max(0, i - 10)
                features[i, 2] = normalized_prices[start:i+1].mean() if i > 0 else normalized_prices[0]
            
            # Price momentum
            features[1:, 3] = np.diff(normalized_prices)
            
            # Volatility
            for i in range(lookback):
                start = max(0, i - 5)
                features[i, 4] = normalized_prices[start:i+1].std() if i > start else 0
        
        # Sentiment feature (repeated for all timesteps)
        features[:, 5] = sentiment_score
        
        return features
    
    def _calculate_confidence(self, predictions: np.ndarray, sentiment_score: float) -> float:
        """Calculate prediction confidence."""
        # Base confidence from prediction variance
        pred_std = np.std(predictions)
        confidence = max(30, 95 - pred_std * 100)
        
        # Adjust by sentiment strength
        sentiment_boost = abs(sentiment_score) * 10
        confidence = min(95, confidence + sentiment_boost)
        
        return confidence
    
    def _determine_outlook(
        self, 
        current_price: float, 
        predicted_price: float,
        predictions: np.ndarray,
        sentiment_score: float
    ) -> Tuple[str, str, str]:
        """Determine market outlook and recommendation."""
        price_change = (predicted_price - current_price) / current_price
        
        # Short-term outlook (1-3 days)
        if price_change > 0.03 or sentiment_score > 0.3:
            short_term = "bullish"
        elif price_change < -0.03 or sentiment_score < -0.3:
            short_term = "bearish"
        else:
            short_term = "neutral"
        
        # Long-term outlook (7+ days)
        avg_prediction = np.mean(predictions)
        if avg_prediction > 0.05 or sentiment_score > 0.5:
            long_term = "bullish"
        elif avg_prediction < -0.05 or sentiment_score < -0.5:
            long_term = "bearish"
        else:
            long_term = "neutral"
        
        # Recommendation
        if short_term == "bullish" and long_term == "bullish":
            recommendation = "buy"
        elif short_term == "bearish" and long_term == "bearish":
            recommendation = "sell"
        else:
            recommendation = "hold"
        
        return short_term, long_term, recommendation


# Singleton instance
model_inference_service = ModelInferenceService()
