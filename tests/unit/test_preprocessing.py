import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend.utils.preprocessing import (
    temporal_train_test_split,
    fit_scaler,
    transform_with_scaler,
    save_scaler_metadata,
    load_scaler_metadata,
    validate_no_leakage,
    SplitInfo,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=1000, freq="D")
    values = np.linspace(0, 100, num=1000)
    return pd.DataFrame({"date": dates, "feature": values, "feature_b": values[::-1]})


def test_temporal_split_sizes(sample_df):
    sorted_df, splits = temporal_train_test_split(sample_df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    assert len(sorted_df) == 1000
    assert splits["train"].size == 600
    assert splits["val"].size == 200
    assert splits["test"].size == 200
    assert pd.Timestamp(splits["train"].end_date) < pd.Timestamp(splits["val"].start_date)


def test_temporal_split_insufficient_rows():
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=20, freq="D"), "value": range(20)})
    with pytest.raises(ValueError):
        temporal_train_test_split(df)


@pytest.mark.parametrize("ratios", [(0.7, 0.15, 0.15), (0.5, 0.25, 0.25)])
def test_temporal_split_custom_ratios(sample_df, ratios):
    train, val, test = ratios
    _, splits = temporal_train_test_split(sample_df, train_ratio=train, val_ratio=val, test_ratio=test)
    total = sum(split.size for split in splits.values())
    assert total == len(sample_df)
    assert splits["train"].size == int(np.floor(len(sample_df) * train))


def test_fit_scaler_minmax(sample_df):
    scaler = fit_scaler(sample_df[["feature", "feature_b"]], scaler_type="minmax")
    transformed = transform_with_scaler(scaler, sample_df[["feature", "feature_b"]])
    assert transformed.min() >= 0.0
    assert transformed.max() <= 1.0


def test_fit_scaler_standard(sample_df):
    scaler = fit_scaler(sample_df[["feature"]], scaler_type="standard")
    transformed = transform_with_scaler(scaler, sample_df[["feature"]])
    assert np.isclose(transformed.mean(), 0.0, atol=1e-6)
    assert np.isclose(transformed.std(), 1.0, atol=1e-3)


def test_fit_scaler_invalid_type(sample_df):
    with pytest.raises(ValueError):
        fit_scaler(sample_df[["feature"]], scaler_type="robust")  # type: ignore[arg-type]


def test_transform_unfitted_scaler(sample_df):
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    with pytest.raises(ValueError):
        transform_with_scaler(scaler, sample_df[["feature"]])


def test_transform_feature_mismatch(sample_df):
    scaler = fit_scaler(sample_df[["feature", "feature_b"]], scaler_type="minmax")
    with pytest.raises(ValueError):
        transform_with_scaler(scaler, sample_df[["feature"]])


def test_save_and_load_scaler_metadata(tmp_path: Path, sample_df):
    scaler = fit_scaler(sample_df[["feature"]], scaler_type="minmax")
    metadata_path = tmp_path / "scaler.json"
    saved = save_scaler_metadata(scaler, metadata_path)
    assert metadata_path.exists()
    assert saved["n_features_in"] == 1

    loaded = load_scaler_metadata(metadata_path)
    transformed = transform_with_scaler(loaded, sample_df[["feature"]])
    assert transformed.shape == (1000, 1)


def test_load_scaler_metadata_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_scaler_metadata(tmp_path / "missing.json")


def test_load_scaler_metadata_corrupted(tmp_path: Path):
    metadata_path = tmp_path / "bad.json"
    metadata_path.write_text("{ invalid json }")
    with pytest.raises(json.JSONDecodeError):
        load_scaler_metadata(metadata_path)


def test_validate_no_leakage_success(sample_df):
    sorted_df, splits = temporal_train_test_split(sample_df)
    validate_no_leakage(sorted_df, splits)


def test_validate_no_leakage_overlap(sample_df):
    sorted_df, splits = temporal_train_test_split(sample_df)
    # Introduce overlap by shifting validation start into train range
    splits["val"] = SplitInfo(
        start_idx=splits["train"].end_idx - 1,
        end_idx=splits["val"].end_idx,
        size=splits["val"].size,
        start_date=splits["train"].end_date,
        end_date=splits["val"].end_date,
    )
    with pytest.raises(ValueError):
        validate_no_leakage(sorted_df, splits)


def test_validate_no_leakage_missing_date(sample_df):
    sorted_df, splits = temporal_train_test_split(sample_df)
    sorted_df = sorted_df.drop(columns=["date"])
    with pytest.raises(ValueError):
        validate_no_leakage(sorted_df, splits)


def test_minmax_scaler_constant_feature_roundtrip(tmp_path: Path):
    """Test that MinMaxScaler save/load preserves scale_ for constant features (no divide by zero)."""
    # Create data with one constant feature and one varying feature
    data = pd.DataFrame({
        "constant": [5.0] * 100,  # Constant feature - would cause divide by zero
        "varying": np.linspace(0, 100, 100),
    })
    
    # Fit scaler on training data
    scaler = fit_scaler(data, scaler_type="minmax")
    
    # Verify scale_ has been computed (may have inf for constant feature)
    assert hasattr(scaler, "scale_")
    assert scaler.scale_.shape[0] == 2
    
    # Transform with original scaler
    original_transformed = transform_with_scaler(scaler, data)
    
    # Save and load metadata
    metadata_path = tmp_path / "scaler_constant.json"
    save_scaler_metadata(scaler, metadata_path)
    loaded_scaler = load_scaler_metadata(metadata_path)
    
    # Verify loaded scaler has the same scale_ values (including any inf/finite values)
    assert hasattr(loaded_scaler, "scale_")
    np.testing.assert_array_equal(scaler.scale_, loaded_scaler.scale_, 
                                   err_msg="scale_ not preserved through save/load")
    
    # Verify that all scale_ values are finite or properly preserved
    # (sklearn handles inf in scale_ internally)
    for i, (orig_scale, loaded_scale) in enumerate(zip(scaler.scale_, loaded_scaler.scale_)):
        if np.isfinite(orig_scale):
            assert np.isfinite(loaded_scale), f"Feature {i} scale became non-finite after loading"
            assert np.isclose(orig_scale, loaded_scale), f"Feature {i} scale changed after loading"
        else:
            # If original was inf, loaded should also be inf (preserving the exact value)
            assert orig_scale == loaded_scale, f"Feature {i} inf/nan scale not preserved"
    
    # Transform with loaded scaler should produce identical results
    loaded_transformed = transform_with_scaler(loaded_scaler, data)
    np.testing.assert_array_almost_equal(
        original_transformed, 
        loaded_transformed,
        decimal=6,
        err_msg="Transformation results differ between original and loaded scaler"
    )

