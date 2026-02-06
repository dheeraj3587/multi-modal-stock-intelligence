#!/usr/bin/env python3
"""
Feature engineering pipeline for technical indicators and windowed sequences.

This script reads raw OHLCV data, computes pandas-ta indicators, performs
temporal 60/20/20 splits, scales features without leakage, and saves
split-aware NumPy tensors plus metadata for downstream LSTM/GRU models,
as specified in docs/metrics_and_evaluation.md (Sections 4.2–4.4).
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Ensure repository root is available for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.utils.logger import get_logger
from backend.utils.config import config
from backend.utils.file_operations import (
    create_timestamped_directory,
    load_existing_data,
    save_json,
    validate_data_quality,
)
from backend.utils.preprocessing import (
    temporal_train_test_split,
    fit_scaler,
    transform_with_scaler,
    save_scaler_metadata,
    validate_no_leakage,
)


logger = get_logger(__name__)

REQUIRED_PRICE_COLUMNS = ["date", "open", "high", "low", "close", "volume"]
BOLLINGER_SETTINGS = {"length": 20, "std": 2.0}
EMA_LENGTHS = [20, 50, 200]
RSI_LENGTH = 14
MACD_SETTINGS = (12, 26, 9)
ATR_LENGTH = 14


def sanitize_ticker(ticker: str) -> str:
    """Convert ticker to filesystem-friendly format."""
    return ticker.replace(".", "_").replace(" ", "_").upper()


def parse_indicator_list(indicator_arg: str | Sequence[str]) -> List[str]:
    """Normalize indicator list from CLI or config."""
    if isinstance(indicator_arg, str):
        indicators = [item.strip() for item in indicator_arg.split(",") if item.strip()]
    else:
        indicators = list(indicator_arg)
    if not indicators:
        raise ValueError("At least one technical indicator must be specified.")
    return indicators


def find_latest_price_file(ticker: str, input_dir: Path) -> Path:
    """Locate the most recent raw price file for the ticker."""
    ticker_file = sanitize_ticker(ticker)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    candidate_dirs = sorted(
        [p for p in input_dir.iterdir() if p.is_dir()],
        key=lambda p: p.name,
        reverse=True,
    )
    for date_dir in candidate_dirs:
        candidate = date_dir / f"{ticker_file}.csv"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No price file found for ticker '{ticker}' in {input_dir}. "
        "Ensure Phase 1 ingestion has been executed."
    )


def load_price_data(ticker: str, input_dir: Path) -> pd.DataFrame:
    """Load the latest raw OHLCV data for a ticker."""
    price_path = find_latest_price_file(ticker, input_dir)
    df = load_existing_data(price_path, file_format="csv")
    if df is None or df.empty:
        raise ValueError(f"Loaded empty DataFrame from {price_path}")

    df = df.copy()
    df.columns = [col.lower() for col in df.columns]
    if "date" not in df.columns:
        raise ValueError(f"'date' column missing in raw data {price_path}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    is_valid, errors = validate_data_quality(df, REQUIRED_PRICE_COLUMNS, timestamp_column="date")
    if not is_valid:
        raise ValueError(f"Data quality validation failed for {ticker}: {errors}")

    logger.info(
        "Loaded price data | ticker={ticker} rows={rows} range={start}->{end}",
        ticker=ticker,
        rows=len(df),
        start=df["date"].min().date(),
        end=df["date"].max().date(),
    )
    return df


def compute_technical_indicators(df: pd.DataFrame, indicators: Iterable[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Compute configured technical indicators using pandas-ta.

    Returns:
        Tuple of (DataFrame with indicators, indicator column names).
    """
    indicators_upper = {ind.strip().upper() for ind in indicators}
    work_df = df.copy()
    indicator_columns: List[str] = []

    if "RSI" in indicators_upper:
        col = "rsi_14"
        work_df[col] = ta.rsi(work_df["close"], length=RSI_LENGTH)
        indicator_columns.append(col)

    if "MACD" in indicators_upper:
        fast, slow, signal = MACD_SETTINGS
        macd = ta.macd(work_df["close"], fast=fast, slow=slow, signal=signal)
        rename_map = {
            f"MACD_{fast}_{slow}_{signal}": "macd_line",
            f"MACDh_{fast}_{slow}_{signal}": "macd_hist",
            f"MACDs_{fast}_{slow}_{signal}": "macd_signal",
        }
        macd = macd.rename(columns=rename_map)
        work_df = work_df.join(macd)
        indicator_columns.extend(rename_map.values())

    if "EMA" in indicators_upper:
        for length in EMA_LENGTHS:
            col = f"ema_{length}"
            work_df[col] = ta.ema(work_df["close"], length=length)
            indicator_columns.append(col)

    if "BB" in indicators_upper or "BOLLINGER" in indicators_upper:
        bbands = ta.bbands(work_df["close"], length=BOLLINGER_SETTINGS["length"], std=BOLLINGER_SETTINGS["std"])
        rename_map = {
            f"BBL_{BOLLINGER_SETTINGS['length']}_{BOLLINGER_SETTINGS['std']}": "bb_lower_20",
            f"BBM_{BOLLINGER_SETTINGS['length']}_{BOLLINGER_SETTINGS['std']}": "bb_middle_20",
            f"BBU_{BOLLINGER_SETTINGS['length']}_{BOLLINGER_SETTINGS['std']}": "bb_upper_20",
        }
        bbands = bbands.rename(columns=rename_map)
        work_df = work_df.join(bbands)
        indicator_columns.extend(rename_map.values())

    if "ATR" in indicators_upper:
        col = "atr_14"
        work_df[col] = ta.atr(high=work_df["high"], low=work_df["low"], close=work_df["close"], length=ATR_LENGTH)
        indicator_columns.append(col)

    if not indicator_columns:
        raise ValueError("No indicators computed. Check --indicators input.")

    work_df[indicator_columns] = work_df[indicator_columns].ffill()
    before_drop = len(work_df)
    work_df = work_df.dropna(subset=indicator_columns).reset_index(drop=True)
    dropped = before_drop - len(work_df)
    if dropped > 0:
        logger.warning(
            "Dropped rows with insufficient indicator history | dropped_rows={dropped}",
            dropped=dropped,
        )

    return work_df, indicator_columns


def create_windowed_sequences(features: np.ndarray, targets: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    """Convert 2D features into 3D windowed sequences suitable for LSTM/GRU."""
    if features.shape[0] <= lookback:
        raise ValueError(
            f"Not enough rows ({features.shape[0]}) to create windows with lookback={lookback}."
        )

    windows: List[np.ndarray] = []
    y: List[float] = []
    for idx in range(lookback, features.shape[0]):
        windows.append(features[idx - lookback : idx])
        y.append(targets[idx])

    return np.stack(windows), np.asarray(y)


def split_and_scale(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    lookback: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    scaler_type: str,
) -> Tuple[
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, Dict[str, int]],
    Dict[str, Dict[str, str]],
    List[str],
    MinMaxScaler | StandardScaler,
    Dict[str, Dict[str, int]],
]:
    """
    Split dataframe chronologically, scale features, and create windowed sequences.

    Returns:
        Tuple containing:
            feature_tensors: Dict of split name -> 3D numpy array
            target_arrays: Dict of split name -> 1D numpy array
            split_indices: Dict with index metadata for each split
            split_date_ranges: Dict with ISO date ranges per split
            feature_names: Ordered list of feature column names
            scaler: Fitted scaler instance (MinMaxScaler or StandardScaler)
            window_counts: Dict summarizing samples/timesteps/features per split
    """
    sorted_df, splits = temporal_train_test_split(
        df,
        date_column="date",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    validate_no_leakage(sorted_df, splits)

    train_slice = sorted_df.iloc[splits["train"].start_idx : splits["train"].end_idx].reset_index(drop=True)
    scaler = fit_scaler(train_slice[feature_cols], scaler_type=scaler_type)

    scaled_features: Dict[str, np.ndarray] = {}
    split_targets: Dict[str, np.ndarray] = {}
    for name, split in splits.items():
        subset = sorted_df.iloc[split.start_idx : split.end_idx].reset_index(drop=True)
        scaled_features[name] = transform_with_scaler(scaler, subset[feature_cols])
        split_targets[name] = subset["close"].to_numpy(dtype=float)

    split_tensors: Dict[str, np.ndarray] = {}
    split_y: Dict[str, np.ndarray] = {}
    window_counts: Dict[str, Dict[str, int]] = {}
    for name in ["train", "val", "test"]:
        if scaled_features[name].shape[0] <= lookback:
            raise ValueError(
                f"{name} split does not have enough rows ({scaled_features[name].shape[0]}) "
                f"for lookback window {lookback}. "
                "Fetch more historical data to satisfy docs Section 4.4."
            )
        tensors, targets = create_windowed_sequences(scaled_features[name], split_targets[name], lookback)
        split_tensors[name] = tensors.astype(np.float32)
        split_y[name] = targets.astype(np.float32)
        window_counts[name] = {
            "samples": int(tensors.shape[0]),
            "timesteps": int(tensors.shape[1]),
            "features": int(tensors.shape[2]),
        }

    split_indices = {
        name: {
            "start_idx": split.start_idx,
            "end_idx": split.end_idx,
            "size": split.size,
        }
        for name, split in splits.items()
    }
    split_ranges = {
        name: {
            "start_date": split.start_date,
            "end_date": split.end_date,
        }
        for name, split in splits.items()
    }

    return split_tensors, split_y, split_indices, split_ranges, list(feature_cols), scaler, window_counts


def save_numpy_array(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array, allow_pickle=False)


def build_metadata(
    ticker: str,
    indicator_columns: Sequence[str],
    feature_columns: Sequence[str],
    split_info: Dict[str, Dict[str, int]],
    split_ranges: Dict[str, Dict[str, str]],
    window_counts: Dict[str, Dict[str, int]],
    lookback: int,
    scaler_type: str,
    scaler_metadata: Dict[str, object],
    scaler_path: Path,
) -> Dict[str, object]:
    """Assemble metadata dictionary for this processing run."""
    return {
        "ticker": ticker,
        "generated_at": datetime.utcnow().isoformat(),
        "lookback_window": lookback,
        "indicators": list(indicator_columns),
        "feature_columns": list(feature_columns),
        "split_indices": split_info,
        "split_date_ranges": split_ranges,
        "window_counts": window_counts,
        "scaler": {
            "type": scaler_type,
            "parameters": scaler_metadata,
            "path": str(scaler_path),
        },
    }


def process_ticker(
    ticker: str,
    input_dir: Path,
    output_dir: Path,
    lookback: int,
    indicators: Sequence[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    scaler_type: str,
) -> Path:
    """Process a single ticker end-to-end."""
    logger.info("Starting feature engineering | ticker={ticker}", ticker=ticker)
    raw_df = load_price_data(ticker, input_dir)
    enriched_df, indicator_columns = compute_technical_indicators(raw_df, indicators)

    if len(enriched_df) < max(lookback + 5, 260):
        raise ValueError(
            f"{ticker}: insufficient history ({len(enriched_df)} rows). "
            f"Need at least {max(lookback + 5, 260)} rows for EMA-200 + lookback window."
        )

    base_features = [col for col in ["open", "high", "low", "close", "volume"] if col in enriched_df.columns]
    feature_columns = base_features + list(indicator_columns)

    (
        split_tensors,
        split_targets,
        split_info,
        split_ranges,
        feature_names,
        scaler,
        window_counts,
    ) = split_and_scale(
        enriched_df,
        feature_columns,
        lookback=lookback,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        scaler_type=scaler_type,
    )

    processed_dir = create_timestamped_directory(output_dir / sanitize_ticker(ticker))
    scaler_metadata_path = processed_dir / "scaler_metadata.json"
    scaler_metadata = save_scaler_metadata(scaler, scaler_metadata_path)

    for split_name, tensor in split_tensors.items():
        save_numpy_array(tensor, processed_dir / f"{split_name}_features.npy")
        save_numpy_array(split_targets[split_name], processed_dir / f"{split_name}_targets.npy")

    metadata = build_metadata(
        ticker=ticker,
        indicator_columns=indicator_columns,
        feature_columns=feature_names,
        split_info=split_info,
        split_ranges=split_ranges,
        window_counts=window_counts,
        lookback=lookback,
        scaler_type=scaler_type,
        scaler_metadata=scaler_metadata,
        scaler_path=scaler_metadata_path,
    )
    save_json(metadata, processed_dir / "metadata.json")

    logger.info(
        "Feature engineering completed | ticker={ticker} train_samples={train} val_samples={val} test_samples={test}",
        ticker=ticker,
        train=window_counts["train"]["samples"],
        val=window_counts["val"]["samples"],
        test=window_counts["test"]["samples"],
    )
    return processed_dir


def run_batch(
    ticker_file: Path,
    input_dir: Path,
    output_dir: Path,
    lookback: int,
    indicators: Sequence[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    scaler_type: str,
) -> None:
    """Process multiple tickers listed in a CSV file with a 'ticker' column."""
    df = pd.read_csv(ticker_file)
    if "ticker" not in df.columns:
        raise ValueError("Ticker file must contain a 'ticker' column.")

    success = 0
    failures: List[str] = []
    for ticker in df["ticker"].dropna().unique():
        try:
            process_ticker(
                ticker=ticker,
                input_dir=input_dir,
                output_dir=output_dir,
                lookback=lookback,
                indicators=indicators,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                scaler_type=scaler_type,
            )
            success += 1
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to process ticker | ticker={ticker} error={error}", ticker=ticker, error=str(exc))
            failures.append(ticker)

    logger.info(
        "Batch feature engineering finished | success={success} failures={failures}",
        success=success,
        failures=",".join(failures) if failures else "none",
    )


def resolve_default_input_dir(source: str | None) -> Path:
    """Return default raw data directory (prices)."""
    return Path("data/raw/prices")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute technical indicators and windowed sequences for stock price data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ticker", type=str, help="Ticker symbol to process (e.g., RELIANCE.NS)")
    parser.add_argument("--ticker-file", type=Path, help="CSV file with a 'ticker' column (used with --batch-mode)")
    parser.add_argument("--batch-mode", action="store_true", help="Enable batch processing using --ticker-file")
    parser.add_argument("--input-dir", type=Path, default=resolve_default_input_dir(None), help="Raw prices directory")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"), help="Processed data directory")
    parser.add_argument("--lookback-window", type=int, default=config.lookback_window, help="Sequence length in days")
    parser.add_argument(
        "--indicators",
        type=str,
        default=",".join(config.technical_indicators),
        help="Comma-separated indicator list (e.g., RSI,MACD,EMA,BB,ATR)",
    )
    parser.add_argument("--train-split", type=float, default=config.train_split_ratio, help="Training split ratio")
    parser.add_argument("--val-split", type=float, default=config.val_split_ratio, help="Validation split ratio")
    parser.add_argument("--test-split", type=float, default=config.test_split_ratio, help="Test split ratio")
    parser.add_argument(
        "--scaler-type",
        type=str,
        default=config.scaler_type,
        choices=["minmax", "standard"],
        help="Scaler strategy for features",
    )

    args = parser.parse_args()

    if args.batch_mode:
        if not args.ticker_file:
            parser.error("--batch-mode requires --ticker-file")
    else:
        if not args.ticker:
            parser.error("--ticker is required unless using --batch-mode")

    total = args.train_split + args.val_split + args.test_split
    if abs(total - 1.0) > 0.01:
        parser.error(f"Train/val/test ratios must sum to 1.0 (current total={total:.2f})")

    indicators = parse_indicator_list(args.indicators)

    if args.batch_mode:
        run_batch(
            ticker_file=args.ticker_file,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            lookback=args.lookback_window,
            indicators=indicators,
            train_ratio=args.train_split,
            val_ratio=args.val_split,
            test_ratio=args.test_split,
            scaler_type=args.scaler_type,
        )
    else:
        process_ticker(
            ticker=args.ticker,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            lookback=args.lookback_window,
            indicators=indicators,
            train_ratio=args.train_split,
            val_ratio=args.val_split,
            test_ratio=args.test_split,
            scaler_type=args.scaler_type,
        )


if __name__ == "__main__":
    main()

