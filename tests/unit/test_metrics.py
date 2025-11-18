"""
Unit tests for evaluation metrics.

Tests metrics functions from backend/utils/metrics.py against known values
and edge cases, ensuring correctness per docs/metrics_and_evaluation.md.
"""

import pytest
import numpy as np
from backend.utils.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    mean_absolute_percentage_error,
    directional_accuracy,
    sharpe_ratio,
    maximum_drawdown,
    compute_all_metrics
)


class TestMeanAbsoluteError:
    """Test MAE computation."""
    
    def test_basic_mae(self):
        """Test MAE with known values."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.5, 2.5, 2.5, 4.5, 5.5])
        
        expected_mae = 0.5
        assert pytest.approx(mean_absolute_error(y_true, y_pred)) == expected_mae
    
    def test_perfect_prediction(self):
        """Test MAE when predictions are perfect."""
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 3])
        
        assert mean_absolute_error(y_true, y_pred) == 0.0
    
    def test_shape_mismatch(self):
        """Test error on shape mismatch."""
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2])
        
        with pytest.raises(ValueError, match="Shape mismatch"):
            mean_absolute_error(y_true, y_pred)
    
    def test_nan_values(self):
        """Test error on NaN values."""
        y_true = np.array([1, 2, np.nan])
        y_pred = np.array([1, 2, 3])
        
        with pytest.raises(ValueError, match="NaN"):
            mean_absolute_error(y_true, y_pred)
    
    def test_empty_arrays(self):
        """Test error on empty arrays."""
        with pytest.raises(ValueError, match="Empty"):
            mean_absolute_error(np.array([]), np.array([]))


class TestRootMeanSquaredError:
    """Test RMSE computation."""
    
    def test_basic_rmse(self):
        """Test RMSE with known values."""
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 3])
        
        assert root_mean_squared_error(y_true, y_pred) == 0.0
    
    def test_rmse_calculation(self):
        """Test RMSE formula."""
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 1, 1])
        
        expected_rmse = 1.0
        assert pytest.approx(root_mean_squared_error(y_true, y_pred)) == expected_rmse


class TestMeanAbsolutePercentageError:
    """Test MAPE computation."""
    
    def test_basic_mape(self):
        """Test MAPE with known values."""
        y_true = np.array([100, 200, 300])
        y_pred = np.array([110, 190, 310])
        
        # Expected: (10/100 + 10/200 + 10/300) / 3 * 100 = 5%
        expected_mape = (10.0 + 5.0 + 3.333) / 3
        assert pytest.approx(mean_absolute_percentage_error(y_true, y_pred), rel=0.01) == expected_mape
    
    def test_zero_division_handling(self):
        """Test handling of zero true values."""
        y_true = np.array([0, 100, 200])
        y_pred = np.array([10, 110, 190])
        
        # Should exclude the zero value
        result = mean_absolute_percentage_error(y_true, y_pred)
        assert not np.isnan(result)
    
    def test_all_zeros(self):
        """Test when all true values are zero."""
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 2, 3])
        
        result = mean_absolute_percentage_error(y_true, y_pred)
        assert np.isnan(result)


class TestDirectionalAccuracy:
    """Test Directional Accuracy computation."""
    
    def test_perfect_direction(self):
        """Test when all directions match."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        
        da = directional_accuracy(y_true, y_pred, lag=1)
        assert da == 1.0
    
    def test_partial_match(self):
        """Test with partial direction match."""
        # True: up, up, down, up
        y_true = np.array([1, 2, 3, 2, 3])
        # Pred: up, down, down, up (2/4 = 0.5)
        y_pred = np.array([1.1, 1.9, 2.9, 1.9, 3.1])
        
        da = directional_accuracy(y_true, y_pred, lag=1)
        assert pytest.approx(da, abs=0.1) == 0.5
    
    def test_insufficient_data(self):
        """Test error with insufficient data."""
        y_true = np.array([1, 2])
        y_pred = np.array([1, 2])
        
        with pytest.raises(ValueError, match="Insufficient"):
            directional_accuracy(y_true, y_pred, lag=3)


class TestSharpeRatio:
    """Test Sharpe ratio computation."""
    
    def test_positive_returns(self):
        """Test Sharpe with positive returns."""
        returns = np.array([0.01] * 252)  # 1% daily return for a year
        
        sharpe = sharpe_ratio(returns, risk_free_rate=0.0)
        assert sharpe > 0
    
    def test_zero_std(self):
        """Test Sharpe with zero standard deviation."""
        returns = np.array([0.01] * 10)  # Constant returns
        
        sharpe = sharpe_ratio(returns)
        assert sharpe == 0.0
    
    def test_empty_returns(self):
        """Test error on empty returns."""
        with pytest.raises(ValueError, match="Empty"):
            sharpe_ratio(np.array([]))


class TestMaximumDrawdown:
    """Test Maximum Drawdown computation."""
    
    def test_no_drawdown(self):
        """Test when portfolio only increases."""
        portfolio_values = np.array([100, 110, 120, 130])
        
        mdd = maximum_drawdown(portfolio_values)
        assert mdd == 0.0
    
    def test_known_drawdown(self):
        """Test with known drawdown."""
        # Peak at 100, trough at 70, drawdown = 30%
        portfolio_values = np.array([100, 110, 100, 70, 80])
        
        expected_mdd = 0.36363636  # (110-70)/110
        assert pytest.approx(maximum_drawdown(portfolio_values), rel=0.01) == expected_mdd
    
    def test_complete_loss(self):
        """Test maximum possible drawdown."""
        portfolio_values = np.array([100, 0])
        
        assert maximum_drawdown(portfolio_values) == 1.0


class TestComputeAllMetrics:
    """Test comprehensive metrics computation."""
    
    def test_all_metrics_returned(self):
        """Test that all expected metrics are returned."""
        y_true = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        y_pred = np.array([1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1])
        
        metrics = compute_all_metrics(y_true, y_pred)
        
        expected_keys = ['mae', 'rmse', 'mape', 'da_1day', 'da_3day', 'da_7day']
        for key in expected_keys:
            assert key in metrics
            assert not np.isnan(metrics[key]) or key in ['da_3day', 'da_7day']
    
    def test_with_returns(self):
        """Test metrics with returns for Sharpe ratio."""
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 3])
        returns = np.array([0.01, 0.02, 0.01])
        
        metrics = compute_all_metrics(y_true, y_pred, returns=returns)
        
        assert 'sharpe_ratio' in metrics
        assert not np.isnan(metrics['sharpe_ratio'])
