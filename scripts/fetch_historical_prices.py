#!/usr/bin/env python3
"""
Historical price data fetching script using yfinance.

Fetches OHLCV (Open, High, Low, Close, Volume) data for stocks with support
for batch processing, incremental updates, and date-partitioned storage.
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import yfinance as yf

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.utils.logger import get_logger
from backend.utils.file_operations import (
    create_timestamped_directory,
    save_dataframe_to_csv,
    validate_data_quality,
    get_latest_file,
    load_existing_data
)


logger = get_logger(__name__)


def fetch_historical_prices(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str = '1d'
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data for a ticker using yfinance.
    
    Args:
        ticker: Stock symbol (e.g., 'RELIANCE.NS' for NSE stocks).
        start_date: Start date in YYYY-MM-DD format.
        end_date: End date in YYYY-MM-DD format.
        interval: Data interval (1d, 1wk, 1mo).
        
    Returns:
        DataFrame with OHLCV data.
        
    Raises:
        Exception: If data fetch fails.
    """
    logger.info(f"Fetching {ticker} data from {start_date} to {end_date} (interval: {interval})")
    
    try:
        # Create ticker object
        stock = yf.Ticker(ticker)
        
        # Fetch historical data
        df = stock.history(start=start_date, end=end_date, interval=interval)
        
        if df.empty:
            raise ValueError(f"No data returned for ticker {ticker}")
        
        # Reset index to make Date a column
        df.reset_index(inplace=True)
        
        # Rename columns for consistency
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]
        
        # Add ticker column
        df['ticker'] = ticker
        
        logger.info(f"Successfully fetched {len(df)} records for {ticker}")
        
        return df
        
    except Exception as e:
        logger.error(f"Failed to fetch data for {ticker}: {str(e)}")
        raise


def save_price_data(df: pd.DataFrame, ticker: str, output_dir: Path) -> Path:
    """
    Save price data to date-partitioned directory.
    
    Args:
        df: DataFrame with price data.
        ticker: Stock ticker symbol.
        output_dir: Base output directory.
        
    Returns:
        Path to saved file.
    """
    # Create timestamped directory
    date_dir = create_timestamped_directory(output_dir)
    
    # Create filename with ticker
    filename = f"{ticker.replace('.', '_')}.csv"
    filepath = date_dir / filename
    
    # Validate data quality
    required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    is_valid, errors = validate_data_quality(df, required_columns, check_duplicates=True, timestamp_column='date')
    
    if not is_valid:
        logger.warning(f"Data quality issues for {ticker}:")
        for error in errors:
            logger.warning(f"  - {error}")
    
    # Save to CSV
    save_dataframe_to_csv(df, filepath, append_mode=False, index=False)
    logger.info(f"Saved {len(df)} records to {filepath}")
    
    return filepath


def fetch_from_ticker_file(ticker_file: Path, start_date: str, end_date: str, interval: str, output_dir: Path):
    """
    Fetch historical prices for multiple tickers from a CSV file.
    
    Args:
        ticker_file: Path to CSV file with 'ticker' column.
        start_date: Start date in YYYY-MM-DD format.
        end_date: End date in YYYY-MM-DD format.
        interval: Data interval (1d, 1wk, 1mo).
        output_dir: Output directory for data.
    """
    logger.info(f"Reading tickers from {ticker_file}")
    
    # Read ticker file
    tickers_df = pd.read_csv(ticker_file)
    
    if 'ticker' not in tickers_df.columns:
        raise ValueError("Ticker file must have a 'ticker' column")
    
    tickers = tickers_df['ticker'].tolist()
    logger.info(f"Found {len(tickers)} tickers to process")
    
    # Process each ticker
    success_count = 0
    failed_tickers = []
    
    for ticker in tickers:
        try:
            df = fetch_historical_prices(ticker, start_date, end_date, interval)
            save_price_data(df, ticker, output_dir)
            success_count += 1
        except Exception as e:
            logger.error(f"Failed to process {ticker}: {str(e)}")
            failed_tickers.append(ticker)
    
    # Log summary
    logger.info(f"Batch processing complete: {success_count}/{len(tickers)} succeeded")
    if failed_tickers:
        logger.warning(f"Failed tickers: {', '.join(failed_tickers)}")


def fetch_incremental(ticker: str, interval: str, output_dir: Path):
    """
    Fetch only new data since the last run (incremental update).
    
    Args:
        ticker: Stock ticker symbol.
        interval: Data interval (1d, 1wk, 1mo).
        output_dir: Output directory for data.
    """
    logger.info(f"Fetching incremental data for {ticker}")
    
    # Find latest existing file
    latest_file = get_latest_file(output_dir, f"*/{ticker.replace('.', '_')}.csv")
    
    if latest_file is None:
        logger.warning(f"No existing data found for {ticker}. Fetching last 365 days.")
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    else:
        # Load existing data to find latest date
        existing_df = load_existing_data(latest_file, file_format='csv')
        
        if existing_df is not None and 'date' in existing_df.columns:
            # Parse date column
            existing_df['date'] = pd.to_datetime(existing_df['date'])
            latest_date = existing_df['date'].max()
            
            # Start from day after latest date
            start_date = (latest_date + timedelta(days=1)).strftime("%Y-%m-%d")
            logger.info(f"Latest data for {ticker}: {latest_date.strftime('%Y-%m-%d')}. Fetching from {start_date}.")
        else:
            logger.warning(f"Could not parse existing data for {ticker}. Fetching last 365 days.")
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    
    # Fetch new data
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    try:
        df = fetch_historical_prices(ticker, start_date, end_date, interval)
        
        if df.empty:
            logger.info(f"No new data available for {ticker}")
        else:
            save_price_data(df, ticker, output_dir)
    except Exception as e:
        logger.error(f"Incremental fetch failed for {ticker}: {str(e)}")


def log_summary_statistics(df: pd.DataFrame, ticker: str):
    """
    Log summary statistics for verification.
    
    Args:
        df: DataFrame with price data.
        ticker: Stock ticker symbol.
    """
    logger.info(f"Summary statistics for {ticker}:")
    logger.info(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    logger.info(f"  Records: {len(df)}")
    logger.info(f"  Price range: {df['close'].min():.2f} to {df['close'].max():.2f}")
    logger.info(f"  Average close: {df['close'].mean():.2f}")
    logger.info(f"  Total volume: {df['volume'].sum():,.0f}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Fetch historical stock price data using yfinance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch single ticker
  python fetch_historical_prices.py --ticker RELIANCE.NS --start 2020-01-01 --end 2024-12-31
  
  # Fetch multiple tickers from file
  python fetch_historical_prices.py --ticker-file tickers.csv --start 2020-01-01 --end 2024-12-31
  
  # Incremental update
  python fetch_historical_prices.py --ticker RELIANCE.NS --incremental
        """
    )
    
    parser.add_argument('--ticker', type=str, help='Stock ticker symbol (e.g., RELIANCE.NS)')
    parser.add_argument('--ticker-file', type=Path, help='CSV file with ticker symbols')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--interval', type=str, default='1d', 
                       choices=['1d', '1wk', '1mo'],
                       help='Data interval (default: 1d)')
    parser.add_argument('--output-dir', type=Path, default='data/raw/prices',
                       help='Output directory (default: data/raw/prices)')
    parser.add_argument('--incremental', action='store_true',
                       help='Fetch only new data since last run')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.ticker and not args.ticker_file:
        parser.error("Either --ticker or --ticker-file must be specified")
    
    if args.incremental and not args.ticker:
        parser.error("--incremental requires --ticker (single ticker mode)")
    
    if not args.incremental and (not args.start or not args.end):
        parser.error("--start and --end are required unless using --incremental")
    
    logger.info("Starting historical price data fetch")
    logger.info(f"Output directory: {args.output_dir}")
    
    try:
        if args.incremental:
            # Incremental mode
            fetch_incremental(args.ticker, args.interval, args.output_dir)
        
        elif args.ticker_file:
            # Batch mode
            fetch_from_ticker_file(args.ticker_file, args.start, args.end, args.interval, args.output_dir)
        
        else:
            # Single ticker mode
            df = fetch_historical_prices(args.ticker, args.start, args.end, args.interval)
            save_price_data(df, args.ticker, args.output_dir)
            log_summary_statistics(df, args.ticker)
        
        logger.info("Historical price data fetch completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to fetch historical prices: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
