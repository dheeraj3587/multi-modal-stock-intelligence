"""
Evaluation metrics for forecasting models.

Implements metrics aligned with docs/metrics_and_evaluation.md specifications:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- Directional Accuracy
- Sharpe Ratio
- Maximum Drawdown
"""

import numpy as np
from typing import Dict, Optional


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Absolute Error.
    
    Formula (docs Section 1.1):
        MAE = (1/n) * Σ|y_true - y_pred|
    
    Args:
        y_true (np.ndarray): Ground truth values
        y_pred (np.ndarray): Predicted values
        
    Returns:
        float: MAE value
        
    Raises:
        ValueError: If arrays have different shapes or contain NaN
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    
    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        raise ValueError("Arrays contain NaN values")
    
    if len(y_true) == 0:
        raise ValueError("Empty arrays provided")
    
    return np.mean(np.abs(y_true - y_pred))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Root Mean Squared Error.
    
    Formula (docs Section 1.1):
        RMSE = sqrt((1/n) * Σ(y_true - y_pred)²)
    
    Args:
        y_true (np.ndarray): Ground truth values
        y_pred (np.ndarray): Predicted values
        
    Returns:
        float: RMSE value
        
    Raises:
        ValueError: If arrays have different shapes or contain NaN
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    
    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        raise ValueError("Arrays contain NaN values")
    
    if len(y_true) == 0:
        raise ValueError("Empty arrays provided")
    
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Absolute Percentage Error with zero-division handling.
    
    Formula:
        MAPE = (100/n) * Σ|y_true - y_pred| / |y_true|
    
    Args:
        y_true (np.ndarray): Ground truth values
        y_pred (np.ndarray): Predicted values
        
    Returns:
        float: MAPE value (percentage)
        
    Raises:
        ValueError: If arrays have different shapes or contain NaN
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    
    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        raise ValueError("Arrays contain NaN values")
    
    if len(y_true) == 0:
        raise ValueError("Empty arrays provided")
    
    # Handle zero-division: exclude samples where y_true is zero
    mask = y_true != 0
    if not mask.any():
        return np.nan  # All true values are zero
    
    return 100.0 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray, lag: int = 1) -> float:
    """
    Compute Directional Accuracy.
    
    Formula (docs Section 1.1):
        DA = (1/n) * Σ I(sign(y_true[t] - y_true[t-lag]) == sign(y_pred[t] - y_true[t-lag]))
    
    Where I is the indicator function.
    
    Args:
        y_true (np.ndarray): Ground truth values (prices)
        y_pred (np.ndarray): Predicted values (prices)
        lag (int): Lag for computing direction (1, 3, or 7 days)
        
    Returns:
        float: Directional accuracy (0 to 1)
        
    Raises:
        ValueError: If arrays have different shapes or insufficient length
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    
    if len(y_true) <= lag:
        raise ValueError(f"Insufficient data: need > {lag} samples, got {len(y_true)}")
    
    # Compute actual and predicted directions
    actual_direction = np.sign(y_true[lag:] - y_true[:-lag])
    predicted_direction = np.sign(y_pred[lag:] - y_true[:-lag])
    
    # Count matches
    matches = (actual_direction == predicted_direction).astype(float)
    
    return np.mean(matches)


def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.072) -> float:
    """
    Compute annualized Sharpe ratio.
    
    Formula (docs Section 1.2):
        Sharpe = (mean_return - risk_free_rate) / std_return * sqrt(252)
    
    Args:
        returns (np.ndarray): Daily returns
        risk_free_rate (float): Annual risk-free rate (default: 0.072 for Indian G-Sec 7.2%)
        
    Returns:
        float: Annualized Sharpe ratio
        
    Raises:
        ValueError: If returns array is empty or contains NaN
    """
    returns = np.asarray(returns)
    
    if len(returns) == 0:
        raise ValueError("Empty returns array")
    
    if np.isnan(returns).any():
        raise ValueError("Returns contain NaN values")
    
    # Handle zero standard deviation
    std_return = np.std(returns)
    if std_return == 0:
        return 0.0
    
    # Convert annual risk-free rate to daily
    daily_rf = risk_free_rate / 252
    
    # Compute Sharpe ratio
    mean_return = np.mean(returns)
    sharpe = (mean_return - daily_rf) / std_return * np.sqrt(252)
    
    return sharpe


def maximum_drawdown(portfolio_values: np.ndarray) -> float:
    """
    Compute Maximum Drawdown.
    
    Formula (docs Section 1.3):
        MDD = max((peak - trough) / peak)
    
    Args:
        portfolio_values (np.ndarray): Time series of portfolio values
        
    Returns:
        float: Maximum drawdown (0 to 1, where 1 = 100% loss)
        
    Raises:
        ValueError: If array is empty or contains NaN
    """
    portfolio_values = np.asarray(portfolio_values)
    
    if len(portfolio_values) == 0:
        raise ValueError("Empty portfolio values array")
    
    if np.isnan(portfolio_values).any():
        raise ValueError("Portfolio values contain NaN")
    
    # Compute running maximum
    running_max = np.maximum.accumulate(portfolio_values)
    
    # Compute drawdown at each point
    drawdown = (running_max - portfolio_values) / running_max
    
    # Return maximum drawdown
    return np.max(drawdown)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    returns: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.
    
    Args:
        y_true (np.ndarray): Ground truth values
        y_pred (np.ndarray): Predicted values
        returns (np.ndarray, optional): Returns for Sharpe ratio computation
        
    Returns:
        Dict[str, float]: Dictionary with all metrics
    """
    metrics = {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': root_mean_squared_error(y_true, y_pred),
        'mape': mean_absolute_percentage_error(y_true, y_pred),
        'da_1day': directional_accuracy(y_true, y_pred, lag=1),
        'da_3day': directional_accuracy(y_true, y_pred, lag=3) if len(y_true) > 3 else np.nan,
        'da_7day': directional_accuracy(y_true, y_pred, lag=7) if len(y_true) > 7 else np.nan,
    }
    
    if returns is not None:
        metrics['sharpe_ratio'] = sharpe_ratio(returns)
    
    return metrics
