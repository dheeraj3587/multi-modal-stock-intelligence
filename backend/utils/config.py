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
        return [ind.strip() for ind in indicators.split(",")]
    
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
