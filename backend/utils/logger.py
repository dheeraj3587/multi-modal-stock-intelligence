"""
Centralized logging configuration module using loguru.

Provides structured logging with rotation, retention, and severity levels.
Configurable via LOG_LEVEL and LOG_FILE environment variables.
"""

import os
import sys
from pathlib import Path
from loguru import logger


def setup_logger(log_level: str = None, log_file: str = None):
    """
    Configure the global logger with console and file handlers.
    
    Args:
        log_level: Logging level (DEBUG/INFO/WARNING/ERROR). Defaults to env LOG_LEVEL or INFO.
        log_file: Path to log file. Defaults to env LOG_FILE or logs/app.log.
    """
    # Remove default handler
    logger.remove()
    
    # Get configuration from environment or use defaults
    level = log_level or os.getenv("LOG_LEVEL", "INFO").upper()
    log_path = log_file or os.getenv("LOG_FILE", "logs/app.log")
    
    # Ensure logs directory exists
    log_dir = Path(log_path).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Console handler with color-coded output
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True,
    )
    
    # File handler with rotation and retention
    logger.add(
        log_path,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=level,
        rotation="10 MB",  # Rotate when file reaches 10 MB
        retention="30 days",  # Keep logs for 30 days
        compression="zip",  # Compress rotated logs
    )
    
    # JSON file handler for production (structured logs)
    json_log_path = str(Path(log_path).with_suffix('.json'))
    logger.add(
        json_log_path,
        format="{message}",
        level=level,
        rotation="10 MB",
        retention="30 days",
        serialize=True,  # Output as JSON
    )
    
    return logger


def get_logger(module_name: str = None):
    """
    Get a logger instance with contextual information.
    
    Args:
        module_name: Name of the module requesting the logger.
        
    Returns:
        Configured logger instance with module context.
    """
    if module_name:
        return logger.bind(module=module_name)
    return logger


# Initialize logger on module import
setup_logger()
