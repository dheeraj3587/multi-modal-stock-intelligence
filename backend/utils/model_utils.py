"""
Model management and inference utilities.

Provides functions for checkpoint loading, parameter counting,
latency measurement, device selection, and reproducibility.
"""

import torch
import numpy as np
import random
import time
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import glob


def load_latest_checkpoint(
    checkpoint_dir: str,
    model_type: str,
    ticker: str
) -> Tuple[Path, Dict[str, Any]]:
    """
    Load the most recent checkpoint for given model type and ticker.
    
    Scans models/checkpoints/ for files matching pattern:
    {model_type}_{ticker}_*.pth
    
    Args:
        checkpoint_dir (str): Directory containing checkpoints
        model_type (str): Model type ('lstm', 'gru', or 'transformer')
        ticker (str): Ticker symbol (e.g., 'RELIANCE.NS')
        
    Returns:
        Tuple[Path, Dict]: (checkpoint_path, metadata)
        
    Raises:
        FileNotFoundError: If no checkpoints found
        
    Usage:
        from models import LSTMForecaster
        
        path, metadata = load_latest_checkpoint('models/checkpoints', 'lstm', 'RELIANCE.NS')
        model = LSTMForecaster(input_dim=metadata['input_dim'])
        model.load_checkpoint(path)
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    # Find all matching checkpoints
    pattern = f"{model_type}_{ticker}_*.pth"
    matches = list(checkpoint_dir.glob(pattern))
    
    if not matches:
        raise FileNotFoundError(
            f"No checkpoints found in {checkpoint_dir} matching pattern {pattern}"
        )
    
    # Sort by modification time (most recent first)
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    latest = matches[0]
    
    # Load metadata if available
    metadata_path = latest.with_suffix('.json')
    metadata = {}
    if metadata_path.exists():
        import json
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    return latest, metadata


def count_model_parameters(model: torch.nn.Module) -> int:
    """
    Count total trainable parameters in model.
    
    Args:
        model (torch.nn.Module): PyTorch model
        
    Returns:
        int: Total trainable parameter count
        
    Usage:
        model = LSTMForecaster(input_dim=20)
        params = count_model_parameters(model)
        print(f"Model has {params:,} trainable parameters")
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_inference_latency(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    num_runs: int = 100
) -> Tuple[float, float, float]:
    """
    Measure model inference latency.
    
    Performs `num_runs` forward passes and computes statistics.
    Validates against ≤300ms p95 threshold (docs Section 6.1).
    
    Args:
        model (torch.nn.Module): Model to benchmark
        input_tensor (torch.Tensor): Sample input tensor
        num_runs (int): Number of runs for benchmarking (default: 100)
        
    Returns:
        Tuple[float, float, float]: (mean_ms, p95_ms, p99_ms)
        
    Usage:
        model.eval()
        sample_input = torch.randn(1, 60, 20)
        mean, p95, p99 = measure_inference_latency(model, sample_input)
        print(f"P95 latency: {p95:.2f}ms (target: ≤300ms)")
    """
    model.eval()
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    latencies = []
    
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = model(input_tensor)
        
        # Benchmark
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(input_tensor)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms
    
    latencies = np.array(latencies)
    
    mean_ms = np.mean(latencies)
    p95_ms = np.percentile(latencies, 95)
    p99_ms = np.percentile(latencies, 99)
    
    return mean_ms, p95_ms, p99_ms


def get_device() -> str:
    """
    Determine available device (CUDA or CPU).
    
    Returns:
        str: 'cuda' if available, else 'cpu'
        
    Usage:
        device = get_device()
        model = model.to(device)
    """
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def set_seed(seed: int):
    """
    Set random seeds for reproducibility.
    
    Sets seeds for:
    - PyTorch (CPU and CUDA)
    - NumPy
    - Python's random module
    
    Aligned with docs Section 8.2 (reproducibility requirements).
    
    Args:
        seed (int): Random seed value
        
    Usage:
        set_seed(42)  # Ensure reproducible results
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # For additional reproducibility (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
