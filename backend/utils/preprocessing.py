"""
Preprocessing utilities for feature engineering workflows.

Provides reusable helpers for temporal train/validation/test splits, scaling,
leakage validation, and scaler metadata persistence. Designed to align with
the strict data leakage prevention rules described in docs/metrics_and_evaluation.md
(Sections 4.3 and 4.4).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Tuple, Literal, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from backend.utils.logger import get_logger


logger = get_logger(__name__)


SplitName = Literal["train", "val", "test"]


@dataclass(frozen=True)
class SplitInfo:
    """Dataclass describing split boundaries and metadata."""

    start_idx: int
    end_idx: int
    size: int
    start_date: str
    end_date: str

    def to_dict(self) -> Dict[str, str | int]:
        """Serialize split info to built-in types."""
        return {
            "start_idx": self.start_idx,
            "end_idx": self.end_idx,
            "size": self.size,
            "start_date": self.start_date,
            "end_date": self.end_date,
        }


def _validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float, tolerance: float = 0.01) -> None:
    """Ensure split ratios are valid and sum to ~1.0."""
    ratios = [("train", train_ratio), ("val", val_ratio), ("test", test_ratio)]
    for name, value in ratios:
        if value <= 0 or value >= 1:
            raise ValueError(f"{name}_ratio must be between 0 and 1 (exclusive). Got {value}.")
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > tolerance:
        raise ValueError(
            f"Split ratios must sum to 1.0 +/- {tolerance}. Got {total:.4f} (train={train_ratio}, val={val_ratio}, test={test_ratio})."
        )


def temporal_train_test_split(
    df: pd.DataFrame,
    date_column: str = "date",
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
) -> Tuple[pd.DataFrame, Dict[SplitName, SplitInfo]]:
    """
    Split a time-series DataFrame into chronological train/val/test segments.

    Args:
        df: Input DataFrame containing a sortable datetime column.
        date_column: Name of the datetime column for ordering (default 'date').
        train_ratio: Proportion allocated to training (default 0.6 per docs Section 4.4).
        val_ratio: Proportion allocated to validation (default 0.2).
        test_ratio: Proportion allocated to testing (default 0.2).

    Returns:
        Tuple of (sorted_df, split_info_dict). `sorted_df` is the chronologically
        ordered DataFrame with integer index, and `split_info_dict` maps split
        names to SplitInfo objects containing index boundaries and date metadata.

    Raises:
        ValueError: If ratios are invalid, column missing, or insufficient rows.
    """
    if date_column not in df.columns:
        raise ValueError(f"Column '{date_column}' not found in DataFrame.")

    if df[date_column].isnull().any():
        raise ValueError(f"Column '{date_column}' contains null values; cannot perform temporal split.")

    _validate_ratios(train_ratio, val_ratio, test_ratio)

    sorted_df = df.sort_values(date_column).reset_index(drop=True)
    total_rows = len(sorted_df)

    if total_rows < 30:
        raise ValueError(
            "Insufficient rows for 60/20/20 split. "
            "Provide at least 30 chronological records to satisfy docs Section 4.4."
        )

    train_end = int(np.floor(total_rows * train_ratio))
    val_end = train_end + int(np.floor(total_rows * val_ratio))
    test_end = total_rows

    # Ensure each split has at least 1 sample
    if train_end == 0 or val_end - train_end == 0 or test_end - val_end == 0:
        raise ValueError(
            f"Split produced empty segment(s). "
            f"Check ratios train={train_ratio}, val={val_ratio}, test={test_ratio} for dataset of size {total_rows}."
        )

    def _build_split(name: SplitName, start_idx: int, end_idx: int) -> SplitInfo:
        start_date = pd.to_datetime(sorted_df.loc[start_idx, date_column]).isoformat()
        end_date = pd.to_datetime(sorted_df.loc[end_idx - 1, date_column]).isoformat()
        return SplitInfo(
            start_idx=start_idx,
            end_idx=end_idx,
            size=end_idx - start_idx,
            start_date=start_date,
            end_date=end_date,
        )

    splits: Dict[SplitName, SplitInfo] = {
        "train": _build_split("train", 0, train_end),
        "val": _build_split("val", train_end, val_end),
        "test": _build_split("test", val_end, test_end),
    }

    logger.info(
        "Temporal split completed | total_records={total} train={train} val={val} test={test} "
        "| train_range={train_range} val_range={val_range} test_range={test_range}",
        total=total_rows,
        train=splits["train"].size,
        val=splits["val"].size,
        test=splits["test"].size,
        train_range=f"{splits['train'].start_date} → {splits['train'].end_date}",
        val_range=f"{splits['val'].start_date} → {splits['val'].end_date}",
        test_range=f"{splits['test'].start_date} → {splits['test'].end_date}",
    )

    return sorted_df, splits


def fit_scaler(
    train_data: np.ndarray | pd.DataFrame,
    scaler_type: Literal["minmax", "standard"] = "minmax",
    feature_range: Tuple[float, float] = (0.0, 1.0),
) -> MinMaxScaler | StandardScaler:
    """
    Fit a scaler on training data only, preventing leakage per docs Section 4.3.

    Args:
        train_data: Training features as numpy array or DataFrame.
        scaler_type: 'minmax' (default) or 'standard'.
        feature_range: MinMaxScaler feature range (ignored for StandardScaler).

    Returns:
        Fitted scaler instance.

    Raises:
        ValueError: For invalid scaler type or empty data.
    """
    if isinstance(train_data, pd.DataFrame):
        values = train_data.values
    else:
        values = np.asarray(train_data)

    if values.size == 0:
        raise ValueError("Training data is empty; cannot fit scaler.")

    scaler: MinMaxScaler | StandardScaler
    if scaler_type == "minmax":
        scaler = MinMaxScaler(feature_range=feature_range, clip=True)
    elif scaler_type == "standard":
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unsupported scaler_type '{scaler_type}'. Expected 'minmax' or 'standard'.")

    scaler.fit(values)
    logger.info("Scaler fitted on training data | scaler_type={scaler_type} shape={shape}", scaler_type=scaler_type, shape=values.shape)
    return scaler


def transform_with_scaler(
    scaler: MinMaxScaler | StandardScaler,
    data: np.ndarray | pd.DataFrame,
) -> np.ndarray:
    """
    Transform data with a pre-fitted scaler.

    Args:
        scaler: Pre-fitted MinMaxScaler or StandardScaler.
        data: Features to transform.

    Returns:
        Scaled numpy array.

    Raises:
        ValueError: If scaler is unfitted or feature dimensions mismatch.
    """
    if scaler is None:
        raise ValueError("Scaler instance is required.")

    if isinstance(data, pd.DataFrame):
        values = data.values
    else:
        values = np.asarray(data)

    if not hasattr(scaler, "scale_") and not hasattr(scaler, "data_min_"):
        raise ValueError("Scaler appears to be unfitted. Fit on training data before transforming.")

    if values.shape[1] != scaler.n_features_in_:
        raise ValueError(
            f"Feature mismatch when applying scaler. "
            f"Scaler trained on {scaler.n_features_in_} features but received {values.shape[1]}."
        )

    return scaler.transform(values)


def save_scaler_metadata(
    scaler: MinMaxScaler | StandardScaler,
    filepath: str | Path,
) -> Dict[str, object]:
    """
    Persist scaler parameters for reproducibility.

    Stores scaler type, initialization params, and fitted statistics (min, max,
    mean, scale). Metadata is JSON-serializable to integrate with timestamped
    processed data directories.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    params: Dict[str, object] = {
        "scaler_type": scaler.__class__.__name__,
        "n_features_in": getattr(scaler, "n_features_in_", None),
    }

    if isinstance(scaler, MinMaxScaler):
        params.update(
            {
                "feature_range": scaler.feature_range,
                "data_min": scaler.data_min_.tolist(),
                "data_max": scaler.data_max_.tolist(),
                "data_range": scaler.data_range_.tolist(),
                "min": scaler.min_.tolist(),
                "scale": scaler.scale_.tolist(),
            }
        )
    elif isinstance(scaler, StandardScaler):
        params.update(
            {
                "mean": scaler.mean_.tolist(),
                "scale": scaler.scale_.tolist(),
                "var": scaler.var_.tolist(),
            }
        )
    else:
        raise ValueError(f"Unsupported scaler instance: {type(scaler).__name__}")

    with path.open("w") as f:
        json.dump(params, f, indent=2)

    logger.info("Scaler metadata saved | path={path}", path=str(path))
    return params


def load_scaler_metadata(filepath: str | Path) -> MinMaxScaler | StandardScaler:
    """
    Reconstruct a scaler from previously saved metadata.

    Args:
        filepath: Path to scaler metadata JSON.

    Returns:
        Scaler instance with restored statistics.

    Raises:
        FileNotFoundError: If metadata file missing.
        ValueError: If scaler type unsupported or metadata invalid.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Scaler metadata file not found: {path}")

    with path.open() as f:
        metadata = json.load(f)

    scaler_type = metadata.get("scaler_type")

    if scaler_type == MinMaxScaler.__name__:
        scaler = MinMaxScaler(feature_range=tuple(metadata.get("feature_range", (0.0, 1.0))))
        scaler.data_min_ = np.array(metadata["data_min"])
        scaler.data_max_ = np.array(metadata["data_max"])
        scaler.data_range_ = np.array(metadata["data_range"])
        scaler.min_ = np.array(metadata["min"])
        scaler.scale_ = np.array(metadata["scale"])
        scaler.n_features_in_ = metadata["n_features_in"]
    elif scaler_type == StandardScaler.__name__:
        scaler = StandardScaler()
        scaler.mean_ = np.array(metadata["mean"])
        scaler.scale_ = np.array(metadata["scale"])
        scaler.var_ = np.array(metadata["var"])
        scaler.n_features_in_ = metadata["n_features_in"]
    else:
        raise ValueError(f"Unsupported scaler_type '{scaler_type}' in metadata.")

    return scaler


def validate_no_leakage(
    sorted_df: pd.DataFrame,
    splits: Dict[SplitName, SplitInfo],
    date_column: str = "date",
) -> None:
    """
    Validate that validation/test samples occur strictly after training samples.

    Args:
        sorted_df: Chronologically ordered DataFrame used for splitting.
        splits: Output from `temporal_train_test_split`.
        date_column: Name of datetime column.

    Raises:
        ValueError: If leakage detected (overlapping indices or out-of-order dates).
    """
    if date_column not in sorted_df.columns:
        raise ValueError(f"Column '{date_column}' not found for leakage validation.")

    train_end_idx = splits["train"].end_idx - 1
    val_start_idx = splits["val"].start_idx
    val_end_idx = splits["val"].end_idx - 1
    test_start_idx = splits["test"].start_idx

    train_end_date = pd.to_datetime(sorted_df.loc[train_end_idx, date_column])
    val_start_date = pd.to_datetime(sorted_df.loc[val_start_idx, date_column])
    val_end_date = pd.to_datetime(sorted_df.loc[val_end_idx, date_column])
    test_start_date = pd.to_datetime(sorted_df.loc[test_start_idx, date_column])

    if train_end_idx >= val_start_idx:
        raise ValueError("Train split overlaps with validation split indices.")
    if val_end_idx >= splits["test"].start_idx:
        raise ValueError("Validation split overlaps with test split indices.")
    if not (train_end_date < val_start_date <= val_end_date < test_start_date or train_end_date < val_start_date < test_start_date):
        raise ValueError(
            "Temporal ordering violated. Ensure train < val < test chronologically per docs Section 4.3."
        )

    logger.info(
        "Leakage validation passed | train_end={train_end} val_start={val_start} test_start={test_start}",
        train_end=train_end_date.isoformat(),
        val_start=val_start_date.isoformat(),
        test_start=test_start_date.isoformat(),
    )

