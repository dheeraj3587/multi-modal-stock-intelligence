import pytest
import numpy as np
import pandas as pd
from backend.utils.sentiment_metrics import (
    precision_score, recall_score, f1_score, 
    compute_confusion_matrix, sentiment_price_correlation
)

class TestSentimentMetrics:
    def test_perfect_scores(self):
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 1, 2])
        
        assert precision_score(y_true, y_pred)['macro'] == 1.0
        assert recall_score(y_true, y_pred)['macro'] == 1.0
        assert f1_score(y_true, y_pred)['macro'] == 1.0
        
    def test_confusion_matrix(self):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 2, 2, 0])
        
        cm = compute_confusion_matrix(y_true, y_pred)
        assert np.array(cm['matrix']).shape == (3, 3)
        assert cm['normalized_matrix'][0][0] == 0.5 # 1 correct out of 2
        
    def test_sentiment_price_correlation(self):
        # Perfect correlation
        sent = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        ret = np.array([0.1, 0.2, 0.3, 0.4, 0.5]) # No lag correlation if aligned? 
        # Function shifts returns backwards. 
        # sent[t] vs ret[t+1]. 
        # If ret is shifted by -1, ret[0] becomes ret[1].
        # So sent[0] aligns with ret[1].
        
        # Create data where ret[t+1] = sent[t]
        sent = pd.Series([1, 2, 3, 4, 5])
        ret = pd.Series([0, 1, 2, 3, 4, 5]) # ret[1]=1, sent[0]=1
        
        res = sentiment_price_correlation(sent, ret, lags=[1])
        assert res['lag_1']['correlation'] > 0.99
        
    def test_input_validation(self):
        with pytest.raises(ValueError):
            precision_score(np.array([1]), np.array([1, 2]))
