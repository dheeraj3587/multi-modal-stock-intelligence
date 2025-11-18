"""
Trading simulation utilities for backtesting forecasting models.

Implements portfolio simulation and performance metrics computation
per docs/metrics_and_evaluation.md Section 1.2.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from .metrics import sharpe_ratio, maximum_drawdown


class TradingSimulator:
    """
    Trading simulator for backtesting forecasting models.
    
    Implements long-only strategy:
    - Buy if predicted return > 0
    - Sell (or hold cash) otherwise
    
    Target: Sharpe ratio ≥1.0 (docs Section 1.2)
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        transaction_cost: float = 0.001
    ):
        """
        Initialize trading simulator.
        
        Args:
            initial_capital (float): Starting portfolio value (default: 100,000)
            transaction_cost (float): Cost per trade as fraction (default: 0.001 = 0.1%)
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        
    def simulate_long_only(
        self,
        predictions: np.ndarray,
        actual_prices: np.ndarray,
        dates: Optional[List] = None
    ) -> Dict:
        """
        Execute long-only trading strategy.
        
        Strategy:
        - If predicted_return > 0: hold stock
        - Else: hold cash
        
        Args:
            predictions (np.ndarray): Predicted prices [T]
            actual_prices (np.ndarray): Actual prices [T]
            dates (List, optional): Date labels for tracking
            
        Returns:
            Dict with keys:
                - 'portfolio_values': Portfolio value over time
                - 'returns': Daily returns
                - 'trades': List of trade signals (1=long, 0=cash)
                
        Raises:
            ValueError: If predictions and prices have different lengths
        """
        if len(predictions) != len(actual_prices):
            raise ValueError(
                f"Length mismatch: predictions ({len(predictions)}) "
                f"vs actual_prices ({len(actual_prices)})"
            )
        
        T = len(predictions)
        portfolio_values = np.zeros(T)
        returns = np.zeros(T - 1)
        trades = np.zeros(T)
        
        # Initialize
        portfolio_values[0] = self.initial_capital
        position = 0  # 0 = cash, 1 = long
        
        for t in range(1, T):
            # Predict return
            predicted_return = (predictions[t] - actual_prices[t - 1]) / actual_prices[t - 1]
            actual_return = (actual_prices[t] - actual_prices[t - 1]) / actual_prices[t - 1]
            
            # Trading decision
            if predicted_return > 0:
                # Go long (or stay long)
                if position == 0:
                    # Entering position: pay transaction cost
                    portfolio_values[t] = portfolio_values[t - 1] * (1 - self.transaction_cost) * (1 + actual_return)
                    position = 1
                else:
                    # Holding position
                    portfolio_values[t] = portfolio_values[t - 1] * (1 + actual_return)
                trades[t] = 1
            else:
                # Hold cash
                if position == 1:
                    # Exiting position: pay transaction cost
                    portfolio_values[t] = portfolio_values[t - 1] * (1 - self.transaction_cost)
                    position = 0
                else:
                    # Staying in cash
                    portfolio_values[t] = portfolio_values[t - 1]
                trades[t] = 0
            
            # Compute return
            returns[t - 1] = (portfolio_values[t] - portfolio_values[t - 1]) / portfolio_values[t - 1]
        
        return {
            'portfolio_values': portfolio_values,
            'returns': returns,
            'trades': trades
        }
    
    def compute_performance_metrics(
        self,
        portfolio_values: np.ndarray,
        returns: np.ndarray,
        risk_free_rate: float = 0.072
    ) -> Dict[str, float]:
        """
        Compute trading performance metrics.
        
        Metrics (docs Section 1.2):
        - Sharpe ratio (target ≥1.0)
        - Maximum Drawdown
        - Cumulative return
        - Annualized return
        - Win rate
        
        Args:
            portfolio_values (np.ndarray): Portfolio value time series
            returns (np.ndarray): Daily returns
            risk_free_rate (float): Annual risk-free rate (default: 0.072)
            
        Returns:
            Dict[str, float]: Performance metrics
        """
        # Sharpe ratio
        sharpe = sharpe_ratio(returns, risk_free_rate)
        
        # Maximum Drawdown
        mdd = maximum_drawdown(portfolio_values)
        
        # Cumulative return
        cumulative_return = (portfolio_values[-1] - self.initial_capital) / self.initial_capital
        
        # Annualized return
        num_days = len(returns)
        annualized_return = (1 + cumulative_return) ** (252 / num_days) - 1
        
        # Win rate
        win_rate = np.sum(returns > 0) / len(returns) if len(returns) > 0 else 0
        
        return {
            'sharpe_ratio': sharpe,
            'max_drawdown': mdd,
            'cumulative_return': cumulative_return,
            'annualized_return': annualized_return,
            'win_rate': win_rate,
            'final_portfolio_value': portfolio_values[-1]
        }


def backtest_model(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    scaler,
    simulator: TradingSimulator,
    device: str = 'cpu'
) -> Dict:
    """
    Backtest forecasting model with walk-forward simulation.
    
    Workflow:
    1. Generate predictions for test set
    2. Inverse-transform to original price scale
    3. Compute returns from predictions
    4. Execute trading strategy
    5. Aggregate results
    
    Args:
        model (torch.nn.Module): Trained forecasting model
        test_loader (DataLoader): Test data loader
        scaler: Scaler object from backend.utils.preprocessing
        simulator (TradingSimulator): Trading simulator instance
        device (str): Device for model inference
        
    Returns:
        Dict: Backtest results with metrics and simulation data
        
    Usage:
        from backend.utils.trading_sim import TradingSimulator, backtest_model
        
        simulator = TradingSimulator(initial_capital=100000)
        results = backtest_model(model, test_loader, scaler, simulator)
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    """
    model.eval()
    model = model.to(device)
    
    all_predictions = []
    all_actuals = []
    
    with torch.no_grad():
        for features, targets in test_loader:
            features = features.to(device)
            
            # Generate predictions
            preds = model(features)
            
            # Store (assuming single-step for simplicity)
            all_predictions.append(preds.cpu().numpy())
            all_actuals.append(targets.cpu().numpy())
    
    # Concatenate
    predictions = np.concatenate(all_predictions, axis=0)
    actuals = np.concatenate(all_actuals, axis=0)
    
    # For multi-step, take first step or reshape as needed
    if predictions.ndim > 1:
        predictions = predictions[:, 0]  # Take first forecast step
    if actuals.ndim > 1:
        actuals = actuals[:, 0]
    
    # Inverse transform if scaler provided
    if scaler is not None:
        # Reshape to 2D [N, 1] for scaler
        predictions_2d = predictions.reshape(-1, 1)
        actuals_2d = actuals.reshape(-1, 1)
        
        # Inverse transform
        # Note: This assumes scaler was fitted on target column or is compatible with 1D input
        # If scaler is multivariate, this might need adjustment based on implementation details
        try:
            predictions = scaler.inverse_transform(predictions_2d).flatten()
            actuals = scaler.inverse_transform(actuals_2d).flatten()
        except ValueError:
            # Fallback for multivariate scaler where we might need dummy features
            # This is a simplification; ideally we'd have the full feature set to inverse transform
            # For now, we assume the user provides a scaler capable of transforming the target
            pass
    
    # Simulate trading
    sim_results = simulator.simulate_long_only(predictions, actuals)
    
    # Compute performance metrics
    metrics = simulator.compute_performance_metrics(
        sim_results['portfolio_values'],
        sim_results['returns']
    )
    
    return {
        **metrics,
        **sim_results
    }
