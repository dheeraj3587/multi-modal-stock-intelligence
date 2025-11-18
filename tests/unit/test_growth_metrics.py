import pytest
import numpy as np
from backend.utils.growth_metrics import (
    spearman_correlation, top_k_precision, compute_excess_return, decile_analysis
)

class TestGrowthMetrics:
    def test_spearman_correlation(self):
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 2, 3, 4, 5])
        res = spearman_correlation(x, y)
        assert res['correlation'] == 1.0
        
        y_neg = np.array([5, 4, 3, 2, 1])
        res_neg = spearman_correlation(x, y_neg)
        assert res_neg['correlation'] == -1.0
        
    def test_top_k_precision(self):
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        returns = np.array([0.1, 0.05, 0.02, -0.01, -0.05])
        benchmark = 0.0
        
        # Top 2: 0.9, 0.8 -> returns 0.1, 0.05. Both > 0. Precision 1.0
        res = top_k_precision(scores, returns, benchmark, k_values=[2])
        assert res[2] == 1.0
        
    def test_excess_return(self):
        scores = np.array([0.9, 0.1])
        returns = np.array([0.1, 0.0])
        benchmark = 0.0
        
        # Top 1 (50% of 2 is 1)
        res = compute_excess_return(scores, returns, benchmark, top_decile=True)
        # Top 1 is 0.9 -> return 0.1. Excess 0.1.
        assert res['excess_return'] == 0.1
        
    def test_decile_analysis(self):
        scores = np.linspace(0, 1, 100)
        returns = np.linspace(0, 1, 100) # Perfect correlation
        
        df = decile_analysis(scores, returns, num_deciles=10)
        assert len(df) == 10
        # Decile 1 (highest scores) should have highest returns
        assert df.loc[1, 'mean'] > df.loc[10, 'mean']
