"""
Baseline forecasting models for comparison.

Implements traditional forecasting methods per docs/metrics_and_evaluation.md Section 7.3:
- ARIMA
- Simple Moving Average
- Exponential Smoothing

Target: Deep learning models should achieve ≥15% MAE reduction vs these baselines.
"""

import numpy as np
from typing import Optional
import warnings

# Try importing statsmodels, but make it optional
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("statsmodels not available. ARIMA and ExponentialSmoothing baselines disabled.")


class ARIMABaseline:
    """
    ARIMA baseline model.
    
    Default order (1,1,1) provides reasonable baseline for stock prices.
    Fallback to moving average if ARIMA convergence fails.
    """
    
    def __init__(self, order: tuple = (1, 1, 1)):
        """
        Initialize ARIMA baseline.
        
        Args:
            order (tuple): ARIMA order (p, d, q). Default (1,1,1)
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels required for ARIMABaseline")
        
        self.order = order
        self.model = None
        self.model_fit = None
        
    def fit(self, y_train: np.ndarray):
        """
        Fit ARIMA model on training data.
        
        Args:
            y_train (np.ndarray): 1D array of historical prices
        """
        try:
            self.model = ARIMA(y_train, order=self.order)
            self.model_fit = self.model.fit()
        except Exception as e:
            warnings.warn(f"ARIMA fitting failed: {e}. Using last value as fallback.")
            self.model_fit = None
            self.last_value = y_train[-1]
    
    def predict(self, horizon: int) -> np.ndarray:
        """
        Generate forecasts for next `horizon` steps.
        
        Args:
            horizon (int): Number of steps to forecast
            
        Returns:
            np.ndarray: Predictions of shape [horizon]
        """
        if self.model_fit is None:
            # Fallback: return last value
            return np.full(horizon, self.last_value)
        
        try:
            forecast = self.model_fit.forecast(steps=horizon)
            return np.array(forecast)
        except Exception:
            # Fallback on prediction failure
            return np.full(horizon, self.last_value)


class MovingAverageBaseline:
    """
    Simple Moving Average baseline.
    
    Predicts next value as average of last `window` values.
    """
    
    def __init__(self, window: int = 20):
        """
        Initialize Moving Average baseline.
        
        Args:
            window (int): Window size for averaging (default: 20 days)
        """
        self.window = window
        self.history = None
        
    def fit(self, y_train: np.ndarray):
        """
        Store training data.
        
        Args:
            y_train (np.ndarray): 1D array of historical prices
        """
        self.history = y_train
        
    def predict(self, horizon: int) -> np.ndarray:
        """
        Generate forecasts using moving average.
        
        Args:
            horizon (int): Number of steps to forecast
            
        Returns:
            np.ndarray: Predictions of shape [horizon]
        """
        if self.history is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Compute moving average of last `window` values
        ma_value = np.mean(self.history[-self.window:])
        
        # Return constant prediction (simple approach)
        return np.full(horizon, ma_value)


class ExponentialSmoothingBaseline:
    """
    Exponential Smoothing baseline.
    
    Uses Holt's linear trend method (trend='add', seasonal=None).
    """
    
    def __init__(self):
        """Initialize Exponential Smoothing baseline."""
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels required for ExponentialSmoothingBaseline")
        
        self.model = None
        self.model_fit = None
        
    def fit(self, y_train: np.ndarray):
        """
        Fit Exponential Smoothing model.
        
        Args:
            y_train (np.ndarray): 1D array of historical prices
        """
        try:
            self.model = ExponentialSmoothing(
                y_train,
                trend='add',
                seasonal=None
            )
            self.model_fit = self.model.fit()
        except Exception as e:
            warnings.warn(f"Exponential Smoothing fitting failed: {e}")
            self.model_fit = None
            self.last_value = y_train[-1]
    
    def predict(self, horizon: int) -> np.ndarray:
        """
        Generate forecasts.
        
        Args:
            horizon (int): Number of steps to forecast
            
        Returns:
            np.ndarray: Predictions of shape [horizon]
        """
        if self.model_fit is None:
            # Fallback
            return np.full(horizon, self.last_value)
        
        try:
            forecast = self.model_fit.forecast(steps=horizon)
            return np.array(forecast)
        except Exception:
            return np.full(horizon, self.last_value)


def evaluate_baseline(
    baseline,
    y_train: np.ndarray,
    y_test: np.ndarray,
    horizon: int
) -> dict:
    """
    Evaluate baseline model on test data.
    
    Args:
        baseline: Baseline model instance (ARIMA, MA, or ES)
        y_train (np.ndarray): Training data
        y_test (np.ndarray): Test data
        horizon (int): Forecast horizon
        
    Returns:
        dict: Dictionary with 'mae' and 'rmse' metrics
    """
    from .metrics import mean_absolute_error, root_mean_squared_error
    
    # Fit on training data
    baseline.fit(y_train)
    
    # Generate predictions
    predictions = baseline.predict(horizon)
    
    # Compute metrics (compare with first `horizon` test values)
    y_test_subset = y_test[:horizon]
    
    mae = mean_absolute_error(y_test_subset, predictions)
    rmse = root_mean_squared_error(y_test_subset, predictions)
    
    return {
        'mae': mae,
        'rmse': rmse
    }
