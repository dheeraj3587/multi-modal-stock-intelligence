from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.feature_engineering import (
    compute_technical_indicators,
    create_windowed_sequences,
    split_and_scale,
    build_metadata,
    parse_indicator_list,
)


@pytest.fixture
def ohlcv_df() -> pd.DataFrame:
    periods = 320
    dates = pd.date_range("2020-01-01", periods=periods, freq="D")
    prices = np.linspace(100, 200, num=periods)
    data = {
        "date": dates,
        "open": prices + np.random.default_rng(42).normal(0, 1, periods),
        "high": prices + 2,
        "low": prices - 2,
        "close": prices,
        "volume": np.linspace(1e6, 5e6, num=periods),
    }
    return pd.DataFrame(data)


def test_compute_indicators_includes_expected_columns(ohlcv_df):
    df, indicator_cols = compute_technical_indicators(
        ohlcv_df,
        indicators=["RSI", "MACD", "EMA", "BB", "ATR"],
    )
    assert "rsi_14" in indicator_cols
    assert df["rsi_14"].between(0, 100).all()
    assert {"macd_line", "macd_hist", "macd_signal"}.issubset(df.columns)
    assert {"ema_20", "ema_50", "ema_200"}.issubset(df.columns)
    assert {"bb_lower_20", "bb_middle_20", "bb_upper_20"}.issubset(df.columns)
    assert (df["atr_14"] >= 0).all()


def test_create_windowed_sequences_shape():
    features = np.random.rand(100, 5)
    targets = np.random.rand(100)
    windows, y = create_windowed_sequences(features, targets, lookback=60)
    assert windows.shape == (40, 60, 5)
    assert y.shape == (40,)


def test_create_windowed_sequences_insufficient_data():
    features = np.random.rand(50, 3)
    targets = np.random.rand(50)
    with pytest.raises(ValueError):
        create_windowed_sequences(features, targets, lookback=60)


def test_split_and_scale_generates_windows(ohlcv_df):
    df, indicator_cols = compute_technical_indicators(
        ohlcv_df,
        indicators=["RSI", "MACD", "EMA", "BB", "ATR"],
    )
    base_features = ["open", "high", "low", "close", "volume"]
    feature_cols = base_features + indicator_cols
    split_tensors, split_targets, split_info, split_ranges, feature_names, scaler, window_counts = split_and_scale(
        df,
        feature_cols=feature_cols,
        lookback=60,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        scaler_type="minmax",
    )
    assert "train" in split_tensors and "val" in split_tensors and "test" in split_tensors
    assert split_tensors["train"].shape[1] == 60
    assert split_tensors["train"].shape[2] == len(feature_cols)
    assert split_targets["train"].shape[0] == split_tensors["train"].shape[0]
    assert feature_names == feature_cols
    assert scaler.n_features_in_ == len(feature_cols)
    assert window_counts["train"]["samples"] == split_tensors["train"].shape[0]
    assert pd.Timestamp(split_ranges["train"]["end_date"]) <= pd.Timestamp(split_ranges["val"]["start_date"])


def test_build_metadata_contains_expected_fields(tmp_path: Path):
    feature_cols = ["open", "close"]
    indicator_cols = ["rsi_14"]
    split_info = {
        "train": {"start_idx": 0, "end_idx": 100, "size": 100},
        "val": {"start_idx": 100, "end_idx": 150, "size": 50},
        "test": {"start_idx": 150, "end_idx": 200, "size": 50},
    }
    split_ranges = {
        "train": {"start_date": "2020-01-01", "end_date": "2020-04-09"},
        "val": {"start_date": "2020-04-10", "end_date": "2020-05-30"},
        "test": {"start_date": "2020-05-31", "end_date": "2020-07-20"},
    }
    scaler_metadata = {"scaler_type": "MinMaxScaler", "n_features_in": 2}
    scaler_path = tmp_path / "scaler.json"
    scaler_path.write_text("{}")
    metadata = build_metadata(
        ticker="RELIANCE.NS",
        indicator_columns=indicator_cols,
        feature_columns=feature_cols,
        split_info=split_info,
        split_ranges=split_ranges,
        window_counts={"train": {"samples": 10, "timesteps": 60, "features": 3}},
        lookback=60,
        scaler_type="minmax",
        scaler_metadata=scaler_metadata,
        scaler_path=scaler_path,
    )
    assert metadata["ticker"] == "RELIANCE.NS"
    assert metadata["split_indices"]["train"]["size"] == 100
    assert metadata["split_date_ranges"]["val"]["start_date"] == "2020-04-10"
    assert metadata["scaler"]["path"] == str(scaler_path)


def test_parse_indicator_list_handles_commas():
    parsed = parse_indicator_list("RSI, MACD ,EMA")
    assert parsed == ["RSI", "MACD", "EMA"]

