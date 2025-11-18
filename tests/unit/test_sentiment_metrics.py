import pytest
import numpy as np
import pandas as pd
from backend.utils.sentiment_metrics import (
    precision_score, recall_score, f1_score, 
    compute_confusion_matrix, sentiment_price_correlation,
    compare_to_baseline
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
    
    def test_compare_to_baseline_f1(self):
        # Model performs better than baseline
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred_model = np.array([0, 0, 1, 1, 2, 2])  # Perfect
        y_pred_baseline = np.array([0, 1, 1, 2, 2, 0])  # Some errors
        
        result = compare_to_baseline(y_true, y_pred_model, y_pred_baseline, metric='f1')
        
        assert 'model_score' in result
        assert 'baseline_score' in result
        assert 'improvement_pct' in result
        assert 'p_value' in result
        assert 'test_stat' in result
        
        assert result['model_score'] == 1.0
        assert result['baseline_score'] < 1.0
        assert result['improvement_pct'] > 0
        assert 0 <= result['p_value'] <= 1
        assert result['test_stat'] >= 0
    
    def test_compare_to_baseline_accuracy(self):
        # Test with accuracy metric
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred_model = np.array([0, 0, 1, 1, 2, 2])
        y_pred_baseline = np.array([0, 1, 1, 2, 2, 0])
        
        result = compare_to_baseline(y_true, y_pred_model, y_pred_baseline, metric='accuracy')
        
        assert result['model_score'] == 1.0
        assert result['baseline_score'] < 1.0
        assert 'p_value' in result
        assert 'test_stat' in result
    
    def test_compare_to_baseline_no_discordant_pairs(self):
        # When model and baseline are identical, no discordant pairs
        y_true = np.array([0, 1, 2])
        y_pred_model = np.array([0, 1, 2])
        y_pred_baseline = np.array([0, 1, 2])
        
        result = compare_to_baseline(y_true, y_pred_model, y_pred_baseline, metric='f1')
        
        assert result['model_score'] == result['baseline_score']
        assert result['improvement_pct'] == 0.0
        assert result['p_value'] == 1.0  # No significant difference
        assert result['test_stat'] == 0.0
    
    def test_compare_to_baseline_synthetic_data(self):
        # Create synthetic data where model is significantly better
        np.random.seed(42)
        y_true = np.random.randint(0, 3, size=100)
        y_pred_baseline = y_true.copy()
        # Baseline gets 30% wrong
        baseline_errors = np.random.choice(100, size=30, replace=False)
        y_pred_baseline[baseline_errors] = np.random.randint(0, 3, size=30)
        
        y_pred_model = y_true.copy()
        # Model gets only 10% wrong
        model_errors = np.random.choice(100, size=10, replace=False)
        y_pred_model[model_errors] = np.random.randint(0, 3, size=10)
        
        result = compare_to_baseline(y_true, y_pred_model, y_pred_baseline, metric='f1')
        
        assert result['model_score'] > result['baseline_score']
        assert result['improvement_pct'] > 0
        assert 'p_value' in result
        assert 0 <= result['p_value'] <= 1
        assert result['test_stat'] >= 0
