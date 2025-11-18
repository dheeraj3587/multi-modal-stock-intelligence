import numpy as np
import pandas as pd
from sklearn.metrics import precision_score as sk_precision
from sklearn.metrics import recall_score as sk_recall
from sklearn.metrics import f1_score as sk_f1
from sklearn.metrics import confusion_matrix, classification_report as sk_report
from scipy.stats import pearsonr, ttest_rel, chi2
from typing import Dict, Any, List, Optional, Union, Tuple

def validate_inputs(y_true: np.ndarray, y_pred: np.ndarray):
    """Validate input arrays."""
    if len(y_true) != len(y_pred):
        raise ValueError(f"Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}")
    if len(y_true) == 0:
        raise ValueError("Empty input arrays")

def precision_score(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'macro') -> Dict[str, float]:
    """
    Compute per-class and average precision.
    Target: per-class >= 0.75, macro >= 0.78
    """
    validate_inputs(y_true, y_pred)
    macro = sk_precision(y_true, y_pred, average='macro', zero_division=0)
    per_class = sk_precision(y_true, y_pred, average=None, zero_division=0)
    
    return {
        'macro': float(macro),
        'per_class': per_class.tolist()
    }

def recall_score(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'macro') -> Dict[str, float]:
    """
    Compute per-class and average recall.
    Target: per-class >= 0.75, macro >= 0.78
    """
    validate_inputs(y_true, y_pred)
    macro = sk_recall(y_true, y_pred, average='macro', zero_division=0)
    per_class = sk_recall(y_true, y_pred, average=None, zero_division=0)
    
    return {
        'macro': float(macro),
        'per_class': per_class.tolist()
    }

def f1_score(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'macro') -> Dict[str, float]:
    """
    Compute per-class and average F1 score.
    Target: macro F1 >= 0.80
    """
    validate_inputs(y_true, y_pred)
    macro = sk_f1(y_true, y_pred, average='macro', zero_division=0)
    per_class = sk_f1(y_true, y_pred, average=None, zero_division=0)
    
    return {
        'macro': float(macro),
        'per_class': per_class.tolist()
    }

def weighted_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute weighted F1 score to handle class imbalance.
    Target: >= 0.82
    """
    validate_inputs(y_true, y_pred)
    return float(sk_f1(y_true, y_pred, average='weighted', zero_division=0))

def compute_confusion_matrix(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    class_names: List[str] = ['Positive', 'Neutral', 'Negative']
) -> Dict[str, Any]:
    """
    Compute raw and normalized confusion matrix.
    Target: No off-diagonal cell > 15% of class total.
    """
    validate_inputs(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize by row (true class)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm) # Handle division by zero
    
    # Check off-diagonal threshold
    off_diagonal_mask = ~np.eye(cm_norm.shape[0], dtype=bool)
    max_off_diagonal = np.max(cm_norm[off_diagonal_mask]) if np.any(off_diagonal_mask) else 0.0
    
    return {
        'matrix': cm.tolist(),
        'normalized_matrix': cm_norm.tolist(),
        'max_off_diagonal': float(max_off_diagonal),
        'threshold_passed': float(max_off_diagonal) <= 0.15,
        'class_names': class_names
    }

def classification_report(y_true: np.ndarray, y_pred: np.ndarray, class_names: Optional[List[str]] = None) -> str:
    """Return formatted classification report."""
    validate_inputs(y_true, y_pred)
    return sk_report(y_true, y_pred, target_names=class_names, zero_division=0)

def sentiment_price_correlation(
    sentiment_scores: Union[np.ndarray, pd.Series], 
    price_returns: Union[np.ndarray, pd.Series], 
    lags: List[int] = [1, 2, 3, 5, 7]
) -> Dict[str, Dict[str, float]]:
    """
    Compute lagged correlation between sentiment scores and forward price returns.
    Target: |rho| >= 0.15 for at least one lag (p < 0.05).
    
    Args:
        sentiment_scores: Continuous sentiment scores (e.g., confidence of positive class - confidence of negative class)
                          or just model confidence.
        price_returns: Daily price returns.
        lags: List of forward lags to check.
    """
    if isinstance(sentiment_scores, np.ndarray):
        sentiment_scores = pd.Series(sentiment_scores)
    if isinstance(price_returns, np.ndarray):
        price_returns = pd.Series(price_returns)
        
    if len(sentiment_scores) != len(price_returns):
        # Try to align if indices match, otherwise raise error
        if not (sentiment_scores.index.equals(price_returns.index)):
             raise ValueError("Length mismatch and indices do not align")
    
    results = {}
    
    for lag in lags:
        # Shift returns backwards to align current sentiment with future return
        # sentiment[t] vs returns[t+lag]
        # returns.shift(-lag) moves t+lag to t
        shifted_returns = price_returns.shift(-lag)
        
        # Drop NaNs created by shift
        valid_mask = ~np.isnan(shifted_returns) & ~np.isnan(sentiment_scores)
        
        if valid_mask.sum() < 2:
            results[f'lag_{lag}'] = {'correlation': 0.0, 'p_value': 1.0}
            continue
            
        corr, p_val = pearsonr(sentiment_scores[valid_mask], shifted_returns[valid_mask])
        results[f'lag_{lag}'] = {'correlation': float(corr), 'p_value': float(p_val)}
        
    return results

def compare_to_baseline(
    y_true: np.ndarray, 
    y_pred_model: np.ndarray, 
    y_pred_baseline: np.ndarray, 
    metric: str = 'f1'
) -> Dict[str, float]:
    """
    Compare model performance to baseline using McNemar's test for statistical significance.
    
    For classification, we compute overall correctness for model and baseline per sample,
    then apply McNemar's test on the 2×2 contingency table to derive a p-value.
    
    Returns:
        Dictionary with model_score, baseline_score, improvement_pct, p_value, and test_stat.
    """
    validate_inputs(y_true, y_pred_model)
    validate_inputs(y_true, y_pred_baseline)
    
    if metric == 'f1':
        score_model = sk_f1(y_true, y_pred_model, average='macro', zero_division=0)
        score_baseline = sk_f1(y_true, y_pred_baseline, average='macro', zero_division=0)
    elif metric == 'accuracy':
        score_model = np.mean(y_true == y_pred_model)
        score_baseline = np.mean(y_true == y_pred_baseline)
    else:
        raise ValueError(f"Unsupported metric: {metric}")
        
    improvement = (score_model - score_baseline) / score_baseline if score_baseline != 0 else 0.0
    
    # McNemar's test: compute correctness for each sample
    model_correct = (y_pred_model == y_true)
    baseline_correct = (y_pred_baseline == y_true)
    
    # Contingency table:
    # b = model correct, baseline incorrect
    # c = model incorrect, baseline correct
    b = int(np.logical_and(model_correct, ~baseline_correct).sum())
    c = int(np.logical_and(~model_correct, baseline_correct).sum())
    
    # McNemar's test statistic: (|b - c| - 1)^2 / (b + c)
    # with continuity correction
    if b + c > 0:
        test_stat = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = float(chi2.sf(test_stat, df=1))
    else:
        # No discordant pairs, cannot compute test
        test_stat = 0.0
        p_value = 1.0
    
    return {
        'model_score': float(score_model),
        'baseline_score': float(score_baseline),
        'improvement_pct': float(improvement * 100),
        'p_value': p_value,
        'test_stat': float(test_stat)
    }

def compute_all_sentiment_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    sentiment_scores: Optional[np.ndarray] = None, 
    price_returns: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Compute all sentiment metrics."""
    metrics = {}
    metrics['precision'] = precision_score(y_true, y_pred)
    metrics['recall'] = recall_score(y_true, y_pred)
    metrics['f1'] = f1_score(y_true, y_pred)
    metrics['weighted_f1'] = weighted_f1_score(y_true, y_pred)
    metrics['confusion_matrix'] = compute_confusion_matrix(y_true, y_pred)
    
    if sentiment_scores is not None and price_returns is not None:
        metrics['price_correlation'] = sentiment_price_correlation(sentiment_scores, price_returns)
        
    return metrics
