"""
File and directory operation utilities for data storage.

Provides functions for timestamp-based directory creation, data serialization,
file I/O operations, and data quality validation.
"""

import json
import fcntl
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Union
import pandas as pd


class FileOperationError(Exception):
    """Raised when file operations fail."""
    pass


def create_timestamped_directory(base_path: Union[str, Path], date: Optional[datetime] = None) -> Path:
    """
    Create a date-partitioned directory for data storage.
    
    Args:
        base_path: Base directory path (e.g., 'data/raw/prices').
        date: Date for partitioning. If None, uses current date.
        
    Returns:
        Path object to the created timestamped directory.
        
    Example:
        create_timestamped_directory('data/raw/prices', datetime(2024, 11, 18))
        # Returns Path('data/raw/prices/2024-11-18/')
    """
    if date is None:
        date = datetime.now()
    
    # Format date as YYYY-MM-DD
    date_str = date.strftime("%Y-%m-%d")
    
    # Create full path
    dir_path = Path(base_path) / date_str
    
    # Create directory with parents if they don't exist
    dir_path.mkdir(parents=True, exist_ok=True)
    
    return dir_path


def save_dataframe_to_csv(
    df: pd.DataFrame,
    filepath: Union[str, Path],
    append_mode: bool = False,
    index: bool = True
) -> None:
    """
    Save a pandas DataFrame to CSV with error handling.
    
    Args:
        df: DataFrame to save.
        filepath: Destination file path.
        append_mode: If True, append to existing file. If False, overwrite.
        index: Whether to write row index to CSV.
        
    Raises:
        FileOperationError: If save operation fails.
    """
    try:
        filepath = Path(filepath)
        
        # Ensure parent directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine write mode
        mode = 'a' if append_mode else 'w'
        
        # Write CSV with file locking to prevent concurrent write conflicts
        with open(filepath, mode, newline='') as f:
            # Acquire exclusive lock
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                # Write header only if not appending or file is new
                header = not append_mode or not filepath.exists()
                df.to_csv(f, index=index, header=header)
            finally:
                # Release lock
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                
    except Exception as e:
        raise FileOperationError(f"Failed to save DataFrame to {filepath}: {str(e)}")


def save_json(data: Union[dict, list], filepath: Union[str, Path], indent: int = 2) -> None:
    """
    Save data to JSON file with error handling.
    
    Args:
        data: Dictionary or list to serialize as JSON.
        filepath: Destination file path.
        indent: Number of spaces for JSON indentation (default 2).
        
    Raises:
        FileOperationError: If save operation fails.
    """
    try:
        filepath = Path(filepath)
        
        # Ensure parent directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Write JSON with file locking
        with open(filepath, 'w') as f:
            # Acquire exclusive lock
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(data, f, indent=indent, ensure_ascii=False)
            finally:
                # Release lock
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                
    except Exception as e:
        raise FileOperationError(f"Failed to save JSON to {filepath}: {str(e)}")


def load_existing_data(filepath: Union[str, Path], file_format: str = 'csv') -> Optional[pd.DataFrame]:
    """
    Load existing data file if it exists.
    
    Args:
        filepath: Path to data file.
        file_format: File format ('csv' or 'json').
        
    Returns:
        DataFrame if file exists and is readable, None otherwise.
    """
    try:
        filepath = Path(filepath)
        
        if not filepath.exists():
            return None
        
        if file_format == 'csv':
            return pd.read_csv(filepath)
        elif file_format == 'json':
            with open(filepath, 'r') as f:
                data = json.load(f)
                # Try to convert to DataFrame if data is list of dicts
                if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                    return pd.DataFrame(data)
                return data
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
            
    except Exception as e:
        # Return None if file can't be loaded
        return None


def validate_data_quality(
    df: pd.DataFrame,
    required_columns: List[str],
    check_duplicates: bool = True,
    timestamp_column: Optional[str] = None
) -> tuple[bool, List[str]]:
    """
    Validate data quality before saving.
    
    Args:
        df: DataFrame to validate.
        required_columns: List of column names that must be present.
        check_duplicates: If True, check for duplicate timestamps.
        timestamp_column: Name of timestamp column for duplicate checking.
        
    Returns:
        Tuple of (is_valid, error_messages).
        is_valid is True if all checks pass, False otherwise.
        error_messages is a list of validation error descriptions.
    """
    errors = []
    
    # Check if DataFrame is empty
    if df.empty:
        errors.append("DataFrame is empty")
        return False, errors
    
    # Check for required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {', '.join(missing_columns)}")
    
    # Check for missing values in required columns
    for col in required_columns:
        if col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                errors.append(f"Column '{col}' has {missing_count} missing values")
    
    # Check for duplicate timestamps
    if check_duplicates and timestamp_column and timestamp_column in df.columns:
        duplicate_count = df[timestamp_column].duplicated().sum()
        if duplicate_count > 0:
            errors.append(f"Found {duplicate_count} duplicate timestamps in '{timestamp_column}'")
    
    is_valid = len(errors) == 0
    return is_valid, errors


def get_latest_file(directory: Union[str, Path], pattern: str = "*") -> Optional[Path]:
    """
    Get the most recently modified file in a directory matching a pattern.
    
    Args:
        directory: Directory to search.
        pattern: Glob pattern for file matching (default '*' matches all files).
        
    Returns:
        Path to the latest file, or None if no files found.
    """
    try:
        directory = Path(directory)
        
        if not directory.exists():
            return None
        
        files = list(directory.glob(pattern))
        
        if not files:
            return None
        
        # Return file with latest modification time
        return max(files, key=lambda f: f.stat().st_mtime)
        
    except Exception:
        return None


def ensure_directory_exists(directory: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Directory path to ensure exists.
        
    Returns:
        Path object to the directory.
    """
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path
