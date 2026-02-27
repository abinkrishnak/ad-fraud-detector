"""
Helper Utilities
Common functions used across the project
"""

import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
from loguru import logger


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed value
        
    Why this matters:
        ML models use randomness (data shuffling, weight initialization).
        Setting seed ensures same results every time (reproducibility).
        
    Example:
        >>> set_seed(42)
        >>> # Now random operations will be consistent
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # If using deep learning libraries:
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    
    logger.debug(f"Random seed set to: {seed}")


def ensure_dir(path: str) -> Path:
    """
    Create directory if it doesn't exist
    
    Args:
        path: Directory path to create
        
    Returns:
        Path object
        
    Example:
        >>> ensure_dir("models/saved")
        >>> # Directory created: models/saved/
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def memory_usage(df: pd.DataFrame) -> str:
    """
    Calculate memory usage of DataFrame
    
    Args:
        df: Pandas DataFrame
        
    Returns:
        Formatted string with memory usage
        
    Example:
        >>> memory_usage(df)
        "45.2 MB"
    """
    mem_bytes = df.memory_usage(deep=True).sum()
    
    if mem_bytes < 1024:
        return f"{mem_bytes} bytes"
    elif mem_bytes < 1024 ** 2:
        return f"{mem_bytes / 1024:.2f} KB"
    elif mem_bytes < 1024 ** 3:
        return f"{mem_bytes / (1024 ** 2):.2f} MB"
    else:
        return f"{mem_bytes / (1024 ** 3):.2f} GB"


def print_dataset_info(df: pd.DataFrame, name: str = "Dataset"):
    """
    Print useful information about dataset
    
    Args:
        df: Pandas DataFrame
        name: Dataset name for display
        
    Example:
        >>> print_dataset_info(train_df, "Training Set")
        
        === Training Set Info ===
        Shape: (80000, 8)
        Memory: 6.1 MB
        Columns: ['ip', 'app', 'device', ...]
        Missing values: 120 (0.2%)
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"{name} Information")
    logger.info(f"{'='*60}")
    logger.info(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    logger.info(f"Memory usage: {memory_usage(df)}")
    logger.info(f"Columns: {list(df.columns)}")
    
    # Missing values
    missing = df.isnull().sum().sum()
    if missing > 0:
        missing_pct = (missing / df.size) * 100
        logger.warning(f"Missing values: {missing:,} ({missing_pct:.2f}%)")
    else:
        logger.success("No missing values ✓")
    
    # If classification task, show class distribution
    if 'is_attributed' in df.columns:
        fraud_count = df['is_attributed'].sum()
        fraud_pct = (fraud_count / len(df)) * 100
        logger.info(f"\nClass distribution:")
        logger.info(f"  Real clicks: {len(df) - fraud_count:,} ({100-fraud_pct:.2f}%)")
        logger.info(f"  Fraud clicks: {fraud_count:,} ({fraud_pct:.2f}%)")
    
    logger.info(f"{'='*60}\n")


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
        
    Example:
        >>> format_time(125.5)
        "2m 5.5s"
        >>> format_time(3665)
        "1h 1m 5s"
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {mins}m {secs:.0f}s"


def calculate_class_weights(y: np.ndarray, cost_ratio: float = 400) -> dict:
    """
    Calculate class weights for imbalanced classification
    
    Args:
        y: Target labels (0 or 1)
        cost_ratio: How much more expensive is missing fraud vs false positive
        
    Returns:
        Dictionary with class weights {0: weight_0, 1: weight_1}
        
    Example:
        >>> y = np.array([0, 0, 0, 1])  # 75% class 0, 25% class 1
        >>> calculate_class_weights(y, cost_ratio=400)
        {0: 1.0, 1: 400.0}
        
    Why this matters:
        In ad fraud, fraud is rare (0.2%) but expensive to miss.
        Class weights tell model: "Pay 400x more attention to fraud!"
    """
    n_samples = len(y)
    n_classes = len(np.unique(y))
    
    # Count each class
    counts = np.bincount(y)
    
    # Calculate weights (inverse frequency)
    weights = n_samples / (n_classes * counts)
    
    # Apply cost ratio to minority class (fraud = class 1)
    weights[1] *= cost_ratio
    
    class_weights = {i: weight for i, weight in enumerate(weights)}
    
    logger.info(f"Class weights calculated: {class_weights}")
    return class_weights


def train_test_split_temporal(
    df: pd.DataFrame,
    time_column: str,
    test_size: float = 0.2,
    val_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data based on time (important for time-series)
    
    Args:
        df: DataFrame with time column
        time_column: Name of time column
        test_size: Fraction for test set
        val_size: Fraction for validation set
        
    Returns:
        Tuple of (train_df, val_df, test_df)
        
    Why temporal split?
        In real world, you train on PAST data, predict FUTURE data.
        Random split would leak future into training (cheating!).
        
    Example:
        >>> train, val, test = train_test_split_temporal(df, 'click_time')
        >>> # Train: oldest 60%, Val: middle 20%, Test: newest 20%
    """
    # Sort by time
    df_sorted = df.sort_values(time_column).reset_index(drop=True)
    
    n = len(df_sorted)
    
    # Calculate split points
    test_idx = int(n * (1 - test_size))
    val_idx = int(n * (1 - test_size - val_size))
    
    train_df = df_sorted[:val_idx].copy()
    val_df = df_sorted[val_idx:test_idx].copy()
    test_df = df_sorted[test_idx:].copy()
    
    logger.info(f"Temporal split:")
    logger.info(f"  Train: {len(train_df):,} samples ({len(train_df)/n*100:.1f}%)")
    logger.info(f"  Val:   {len(val_df):,} samples ({len(val_df)/n*100:.1f}%)")
    logger.info(f"  Test:  {len(test_df):,} samples ({len(test_df)/n*100:.1f}%)")
    
    return train_df, val_df, test_df