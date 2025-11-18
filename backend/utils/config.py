"""
Configuration management module for environment variables and application settings.

Loads and validates environment variables from .env file using python-dotenv.
Provides centralized access to API keys, paths, and feature engineering parameters.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv


class ConfigError(Exception):
    """Raised when configuration validation fails."""
    pass


class Config:
    """
    Configuration class with properties for API keys, paths, and parameters.
    Loads from environment variables and provides validation.
    """
    
    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize configuration by loading environment variables.
        
        Args:
            env_file: Path to .env file. If None, searches for .env in current and parent directories.
        """
        # Load environment variables from .env file
        if env_file:
            load_dotenv(env_file)
        else:
            # Search for .env file in current and parent directories
            current_dir = Path.cwd()
            env_path = current_dir / '.env'
            if not env_path.exists():
                # Try parent directories (for scripts/ subdirectory)
                parent_env = current_dir.parent / '.env'
                if parent_env.exists():
                    env_path = parent_env
            
            if env_path.exists():
                load_dotenv(env_path)
    
    # API Keys
    @property
    def newsapi_key(self) -> str:
        """NewsAPI key for news data fetching."""
        key = os.getenv("NEWSAPI_KEY", "")
        if not key:
            raise ConfigError("NEWSAPI_KEY not found in environment variables. Please set it in .env file.")
        return key
    
    @property
    def alpha_vantage_key(self) -> Optional[str]:
        """Alpha Vantage API key (optional, for fundamentals)."""
        return os.getenv("ALPHA_VANTAGE_KEY", "")
    
    @property
    def stocktwits_access_token(self) -> str:
        """StockTwits access token for social sentiment data."""
        token = os.getenv("STOCKTWITS_ACCESS_TOKEN", "")
        if not token:
            raise ConfigError("STOCKTWITS_ACCESS_TOKEN not found in environment variables. Please set it in .env file.")
        return token
    
    @property
    def world_bank_api_key(self) -> Optional[str]:
        """World Bank API key (optional, API is public)."""
        return os.getenv("WORLD_BANK_API_KEY", "")
    
    @property
    def upstox_api_key(self) -> Optional[str]:
        """Upstox API key (optional, for future integration)."""
        return os.getenv("UPSTOX_API_KEY", "")
    
    @property
    def upstox_api_secret(self) -> Optional[str]:
        """Upstox API secret (optional, for future integration)."""
        return os.getenv("UPSTOX_API_SECRET", "")
    
    # Database URLs
    @property
    def database_url(self) -> str:
        """Database connection URL."""
        return os.getenv("DATABASE_URL", "sqlite:///data/processed/stock_data.db")
    
    # File Paths
    @property
    def log_file(self) -> str:
        """Path to application log file."""
        return os.getenv("LOG_FILE", "logs/app.log")
    
    @property
    def log_level(self) -> str:
        """Logging level (DEBUG/INFO/WARNING/ERROR)."""
        return os.getenv("LOG_LEVEL", "INFO").upper()
    
    @property
    def model_checkpoint_dir(self) -> Path:
        """Directory for model checkpoints."""
        path = os.getenv("MODEL_CHECKPOINT_DIR", "models/checkpoints")
        return Path(path)
    
    # Feature Engineering Parameters
    @property
    def lookback_window(self) -> int:
        """Number of historical days to use for features."""
        return int(os.getenv("LOOKBACK_WINDOW", "60"))
    
    @property
    def forecast_horizon(self) -> int:
        """Number of days ahead to forecast."""
        return int(os.getenv("FORECAST_HORIZON", "30"))
    
    @property
    def technical_indicators(self) -> list:
        """List of technical indicators to compute."""
        indicators = os.getenv("TECHNICAL_INDICATORS", "SMA,EMA,RSI,MACD,BB,ATR")
        return [ind.strip() for ind in indicators.split(",") if ind.strip()]

    @property
    def scaler_type(self) -> str:
        """
        Default scaler type for feature pipelines.

        Returns 'minmax' or 'standard' (default minmax) aligned with docs Section 4.4.
        """
        scaler = os.getenv("SCALER_TYPE", "minmax").lower()
        if scaler not in {"minmax", "standard"}:
            raise ConfigError("SCALER_TYPE must be 'minmax' or 'standard'.")
        return scaler

    @property
    def train_split_ratio(self) -> float:
        """Training split proportion (default 0.6 per docs Section 4.4)."""
        value = float(os.getenv("TRAIN_SPLIT_RATIO", "0.6"))
        if not 0 < value < 1:
            raise ConfigError("TRAIN_SPLIT_RATIO must be between 0 and 1.")
        return value

    @property
    def val_split_ratio(self) -> float:
        """Validation split proportion (default 0.2)."""
        value = float(os.getenv("VAL_SPLIT_RATIO", "0.2"))
        if not 0 < value < 1:
            raise ConfigError("VAL_SPLIT_RATIO must be between 0 and 1.")
        return value

    @property
    def test_split_ratio(self) -> float:
        """Test split proportion (default 0.2)."""
        value = float(os.getenv("TEST_SPLIT_RATIO", "0.2"))
        if not 0 < value < 1:
            raise ConfigError("TEST_SPLIT_RATIO must be between 0 and 1.")
        return value

    def validate_split_ratios(self, tolerance: float = 0.01) -> tuple[float, float, float]:
        """
        Validate that train/val/test ratios sum to 1 within tolerance.

        Args:
            tolerance: Acceptable floating point tolerance (default 0.01).

        Returns:
            Tuple of validated ratios (train, val, test).
        """
        train = self.train_split_ratio
        val = self.val_split_ratio
        test = self.test_split_ratio
        total = train + val + test
        if abs(total - 1.0) > tolerance:
            raise ConfigError(
                f"Train/val/test ratios must sum to 1.0 +/- {tolerance}. Current total: {total:.4f}"
            )
        return train, val, test

    @property
    def finbert_model_name(self) -> str:
        """Default FinBERT checkpoint for embeddings."""
        return os.getenv("FINBERT_MODEL_NAME", "ProsusAI/finbert")

    @property
    def finbert_batch_size(self) -> int:
        """Default batch size for FinBERT inference (default 16)."""
        value = int(os.getenv("FINBERT_BATCH_SIZE", "16"))
        if value <= 0:
            raise ConfigError("FINBERT_BATCH_SIZE must be positive.")
        return value

    @property
    def finbert_max_length(self) -> int:
        """Maximum token length for FinBERT inputs (default 512)."""
        value = int(os.getenv("FINBERT_MAX_LENGTH", "512"))
        if not 1 <= value <= 512:
            raise ConfigError("FINBERT_MAX_LENGTH must be between 1 and 512.")
        return value
    
    # Helper Methods
    def get_data_dir(self, subdir: str = "") -> Path:
        """
        Get path to data directory or subdirectory.
        
        Args:
            subdir: Subdirectory name (e.g., 'raw', 'processed', 'external').
            
        Returns:
            Path object to the data directory.
        """
        base_dir = Path(os.getenv("DATA_DIR", "data"))
        if subdir:
            return base_dir / subdir
        return base_dir
    
    def get_raw_data_dir(self, source: str = "") -> Path:
        """
        Get path to raw data directory for a specific source.
        
        Args:
            source: Data source name (e.g., 'prices', 'news', 'social', 'fundamentals', 'macro').
            
        Returns:
            Path object to the raw data directory.
        """
        raw_dir = self.get_data_dir("raw")
        if source:
            return raw_dir / source
        return raw_dir
    
    def get_processed_data_dir(self) -> Path:
        """Get path to processed data directory."""
        return self.get_data_dir("processed")
    
    def get_external_data_dir(self) -> Path:
        """Get path to external data directory."""
        return self.get_data_dir("external")
    
    def validate_required_keys(self, keys: list):
        """
        Validate that required API keys are present.
        
        Args:
            keys: List of required key names (e.g., ['newsapi_key', 'stocktwits_access_token']).
            
        Raises:
            ConfigError: If any required key is missing.
        """
        for key in keys:
            try:
                value = getattr(self, key)
                if not value:
                    raise ConfigError(f"Required configuration '{key}' is not set.")
            except AttributeError:
                raise ConfigError(f"Unknown configuration key: '{key}'")


# Global configuration instance
config = Config()
