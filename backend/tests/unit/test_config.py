"""
Unit tests for backend.utils.config module.

Tests configuration parsing and validation, including support for both
JSON and comma-separated formats for technical indicators.
"""

import os
from unittest.mock import patch

import pytest

from backend.utils.config import Config, ConfigError


class TestConfigTechnicalIndicators:
    """Test suite for technical_indicators property parsing."""

    def test_technical_indicators_comma_separated_default(self):
        """Test default comma-separated format."""
        config = Config()
        with patch.dict(os.environ, {"TECHNICAL_INDICATORS": "RSI,MACD,EMA"}):
            config_new = Config()
            indicators = config_new.technical_indicators
            assert indicators == ["RSI", "MACD", "EMA"]

    def test_technical_indicators_comma_separated_with_spaces(self):
        """Test comma-separated format with whitespace."""
        with patch.dict(os.environ, {"TECHNICAL_INDICATORS": " RSI , MACD , EMA "}):
            config = Config()
            indicators = config.technical_indicators
            assert indicators == ["RSI", "MACD", "EMA"]

    def test_technical_indicators_json_array_format(self):
        """Test legacy JSON array format."""
        with patch.dict(os.environ, {"TECHNICAL_INDICATORS": '["RSI", "MACD", "EMA"]'}):
            config = Config()
            indicators = config.technical_indicators
            assert indicators == ["RSI", "MACD", "EMA"]

    def test_technical_indicators_json_with_lowercase(self):
        """Test JSON format with lowercase indicators (should be normalized to uppercase)."""
        with patch.dict(os.environ, {"TECHNICAL_INDICATORS": '["rsi", "macd", "ema"]'}):
            config = Config()
            indicators = config.technical_indicators
            assert indicators == ["RSI", "MACD", "EMA"]

    def test_technical_indicators_json_with_spaces(self):
        """Test JSON format with whitespace in strings."""
        with patch.dict(os.environ, {"TECHNICAL_INDICATORS": '["  RSI  ", " MACD", "EMA "]'}):
            config = Config()
            indicators = config.technical_indicators
            assert indicators == ["RSI", "MACD", "EMA"]

    def test_technical_indicators_comma_separated_lowercase(self):
        """Test comma-separated format with lowercase (should normalize to uppercase)."""
        with patch.dict(os.environ, {"TECHNICAL_INDICATORS": "rsi,macd,ema"}):
            config = Config()
            indicators = config.technical_indicators
            assert indicators == ["RSI", "MACD", "EMA"]

    def test_technical_indicators_empty_string(self):
        """Test empty string returns empty list."""
        with patch.dict(os.environ, {"TECHNICAL_INDICATORS": ""}):
            config = Config()
            indicators = config.technical_indicators
            assert indicators == []

    def test_technical_indicators_single_value(self):
        """Test single indicator."""
        with patch.dict(os.environ, {"TECHNICAL_INDICATORS": "RSI"}):
            config = Config()
            indicators = config.technical_indicators
            assert indicators == ["RSI"]

    def test_technical_indicators_json_empty_array(self):
        """Test JSON empty array."""
        with patch.dict(os.environ, {"TECHNICAL_INDICATORS": "[]"}):
            config = Config()
            indicators = config.technical_indicators
            assert indicators == []

    def test_technical_indicators_mixed_case_comma(self):
        """Test mixed case in comma-separated format."""
        with patch.dict(os.environ, {"TECHNICAL_INDICATORS": "RsI,MaCd,EmA,bb,atr"}):
            config = Config()
            indicators = config.technical_indicators
            assert indicators == ["RSI", "MACD", "EMA", "BB", "ATR"]

    def test_technical_indicators_default_fallback(self):
        """Test default value when env var not set."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config()
            indicators = config.technical_indicators
            # Should return default indicators
            assert "SMA" in indicators
            assert "RSI" in indicators
            assert len(indicators) > 0


class TestConfigSplitRatios:
    """Test suite for split ratio validation."""

    def test_validate_split_ratios_valid(self):
        """Test valid split ratios."""
        with patch.dict(os.environ, {
            "TRAIN_SPLIT_RATIO": "0.6",
            "VAL_SPLIT_RATIO": "0.2",
            "TEST_SPLIT_RATIO": "0.2"
        }):
            config = Config()
            train, val, test = config.validate_split_ratios()
            assert train == 0.6
            assert val == 0.2
            assert test == 0.2

    def test_validate_split_ratios_invalid_sum(self):
        """Test that ratios must sum to 1.0."""
        with patch.dict(os.environ, {
            "TRAIN_SPLIT_RATIO": "0.5",
            "VAL_SPLIT_RATIO": "0.3",
            "TEST_SPLIT_RATIO": "0.3"
        }):
            config = Config()
            with pytest.raises(ConfigError, match="must sum to 1.0"):
                config.validate_split_ratios()


class TestConfigScalerType:
    """Test suite for scaler_type validation."""

    def test_scaler_type_minmax(self):
        """Test minmax scaler type."""
        with patch.dict(os.environ, {"SCALER_TYPE": "minmax"}):
            config = Config()
            assert config.scaler_type == "minmax"

    def test_scaler_type_standard(self):
        """Test standard scaler type."""
        with patch.dict(os.environ, {"SCALER_TYPE": "standard"}):
            config = Config()
            assert config.scaler_type == "standard"

    def test_scaler_type_uppercase(self):
        """Test scaler type with uppercase (should normalize to lowercase)."""
        with patch.dict(os.environ, {"SCALER_TYPE": "MINMAX"}):
            config = Config()
            assert config.scaler_type == "minmax"

    def test_scaler_type_invalid(self):
        """Test invalid scaler type raises ConfigError."""
        with patch.dict(os.environ, {"SCALER_TYPE": "robust"}):
            config = Config()
            with pytest.raises(ConfigError, match="must be 'minmax' or 'standard'"):
                _ = config.scaler_type


