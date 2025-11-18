import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_1samp, ttest_rel
from typing import Dict, Any, List, Optional, Union, Tuple

def validate_growth_inputs(growth_scores: np.ndarray, realized_returns: np.ndarray):
    """Validate input arrays."""
    if len(growth_scores) != len(realized_returns):
        raise ValueError(f"Shape mismatch: scores {growth_scores.shape}, returns {realized_returns.shape}")
    if len(growth_scores) == 0:
        raise ValueError("Empty input arrays")
    if np.isnan(growth_scores).any() or np.isnan(realized_returns).any():
        raise ValueError("Inputs contain NaN values")

def spearman_correlation(growth_scores: np.ndarray, realized_returns: np.ndarray) -> Dict[str, float]:
    """
    Compute Spearman rank correlation.
    Target: rho >= 0.30 (p < 0.01)
    """
    validate_growth_inputs(growth_scores, realized_returns)
    corr, p_val = spearmanr(growth_scores, realized_returns)
    return {
        'correlation': float(corr),
        'spearman_p_value': float(p_val)
    }

def top_k_precision(
    growth_scores: np.ndarray, 
    realized_returns: np.ndarray, 
    benchmark_return: Union[float, np.ndarray], 
    k_values: List[int] = [10, 20, 50]
) -> Dict[int, float]:
    """
    Compute Top-K Precision: % of top-k stocks that outperformed benchmark.
    Target: Top-10 >= 70%
    """
    validate_growth_inputs(growth_scores, realized_returns)
    
    # Create DataFrame for sorting
    df = pd.DataFrame({
        'score': growth_scores,
        'return': realized_returns
    })
    
    # Handle benchmark alignment
    if isinstance(benchmark_return, np.ndarray):
        if len(benchmark_return) != len(realized_returns):
             raise ValueError("Benchmark return length mismatch")
        df['benchmark'] = benchmark_return
    else:
        df['benchmark'] = benchmark_return
        
    # Sort by score descending
    df = df.sort_values('score', ascending=False)
    
    results = {}
    for k in k_values:
        if k > len(df):
            continue
            
        top_k = df.iloc[:k]
        outperformed = (top_k['return'] > top_k['benchmark']).sum()
        precision = outperformed / k
        results[k] = float(precision)
        
    return results

def compute_excess_return(
    growth_scores: np.ndarray, 
    realized_returns: np.ndarray, 
    benchmark_return: Union[float, np.ndarray], 
    top_decile: bool = True
) -> Dict[str, float]:
    """
    Compute excess return of top decile vs benchmark.
    Target: Excess return >= 3%
    """
    validate_growth_inputs(growth_scores, realized_returns)
    
    df = pd.DataFrame({
        'score': growth_scores,
        'return': realized_returns
    })
    
    if isinstance(benchmark_return, np.ndarray):
        df['benchmark'] = benchmark_return
    else:
        df['benchmark'] = benchmark_return
        
    # Select top decile
    n = len(df)
    k = max(1, int(n * 0.1)) if top_decile else n
    
    df = df.sort_values('score', ascending=False)
    top_k = df.iloc[:k]
    
    # Excess returns for top k stocks
    excess_returns = top_k['return'] - top_k['benchmark']
    mean_excess = excess_returns.mean()
    
    # t-test (H0: mean excess return = 0)
    t_stat, p_val = ttest_1samp(excess_returns, 0)
    
    return {
        'excess_return': float(mean_excess),
        'excess_p_value': float(p_val),
        'mean_top_return': float(top_k['return'].mean())
    }

def information_ratio(portfolio_returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
    """
    Compute Information Ratio.
    Target: IR >= 0.5
    """
    if len(portfolio_returns) != len(benchmark_returns):
        raise ValueError("Length mismatch")
        
    active_returns = portfolio_returns - benchmark_returns
    tracking_error = np.std(active_returns)
    
    if tracking_error == 0:
        return 0.0
        
    return float(np.mean(active_returns) / tracking_error)

def decile_analysis(growth_scores: np.ndarray, realized_returns: np.ndarray, num_deciles: int = 10) -> pd.DataFrame:
    """
    Compute mean return for each decile.
    """
    validate_growth_inputs(growth_scores, realized_returns)
    
    df = pd.DataFrame({
        'score': growth_scores,
        'return': realized_returns
    })
    
    # Create deciles (qcut labels 0..9, where 0 is lowest score)
    # We want 1 to be highest score, so we reverse labels or sort logic
    # pd.qcut assigns bins based on value. Higher score -> higher bin index.
    # So bin 9 is highest scores.
    try:
        df['decile'] = pd.qcut(df['score'], num_deciles, labels=False, duplicates='drop')
    except ValueError:
        # Fallback if not enough unique values
        return pd.DataFrame()
        
    # Group by decile
    stats = df.groupby('decile')['return'].agg(['mean', 'std', 'count'])
    
    # Rename index to 1..10 where 1 is highest score
    # Current: 0 (lowest) to 9 (highest)
    # Map: 9->1, 8->2, ..., 0->10
    stats.index = [num_deciles - i for i in stats.index]
    stats = stats.sort_index()
    
    return stats

def hit_rate(growth_scores: np.ndarray, realized_returns: np.ndarray, threshold: float = 0.0, k_values: List[int] = [10, 20, 50]) -> Dict[int, float]:
    """
    Compute hit rate (% positive returns) for top-k stocks.
    """
    validate_growth_inputs(growth_scores, realized_returns)
    
    df = pd.DataFrame({
        'score': growth_scores,
        'return': realized_returns
    })
    df = df.sort_values('score', ascending=False)
    
    results = {}
    for k in k_values:
        if k > len(df):
            continue
        top_k = df.iloc[:k]
        hits = (top_k['return'] > threshold).sum()
        results[k] = float(hits / k)
        
    return results

def rank_ic(growth_scores: np.ndarray, realized_returns: np.ndarray) -> Dict[str, float]:
    """
    Compute Rank IC (Information Coefficient).
    Same as Spearman correlation but often used in time-series context.
    """
    return spearman_correlation(growth_scores, realized_returns)

def compare_to_baseline(
    growth_scores_model: np.ndarray, 
    growth_scores_baseline: np.ndarray, 
    realized_returns: np.ndarray
) -> Dict[str, float]:
    """
    Compare model Spearman correlation to baseline.
    """
    validate_growth_inputs(growth_scores_model, realized_returns)
    validate_growth_inputs(growth_scores_baseline, realized_returns)
    
    corr_model, _ = spearmanr(growth_scores_model, realized_returns)
    corr_baseline, _ = spearmanr(growth_scores_baseline, realized_returns)
    
    improvement = (corr_model - corr_baseline) / abs(corr_baseline) if corr_baseline != 0 else 0.0
    
    return {
        'model_ic': float(corr_model),
        'baseline_ic': float(corr_baseline),
        'improvement_pct': float(improvement * 100)
    }

def compute_all_growth_metrics(
    growth_scores: np.ndarray, 
    realized_returns: np.ndarray, 
    benchmark_return: Union[float, np.ndarray]
) -> Dict[str, Any]:
    """Compute all growth metrics."""
    metrics = {}
    metrics.update(spearman_correlation(growth_scores, realized_returns))
    metrics['top_k_precision'] = top_k_precision(growth_scores, realized_returns, benchmark_return)
    metrics.update(compute_excess_return(growth_scores, realized_returns, benchmark_return))
    
    # Decile analysis summary (spread)
    deciles = decile_analysis(growth_scores, realized_returns)
    if not deciles.empty:
        metrics['decile_spread'] = float(deciles.loc[1, 'mean'] - deciles.loc[10, 'mean'])
        
    return metrics
