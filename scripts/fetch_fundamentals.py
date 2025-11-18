#!/usr/bin/env python3
"""
Company fundamentals data fetching script.

Fetches fundamental metrics (P/E ratio, market cap, revenue, earnings, etc.)
using Yahoo Finance via yfinance and optionally Alpha Vantage API.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import pandas as pd
import yfinance as yf

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.utils.logger import get_logger
from backend.utils.config import Config
from backend.utils.api_helpers import make_api_request, retry_with_backoff, handle_api_error
from backend.utils.file_operations import create_timestamped_directory, save_json


logger = get_logger(__name__)
config = Config()


def fetch_fundamentals_yahoo(ticker: str) -> Dict:
    """
    Fetch fundamental data using Yahoo Finance via yfinance.
    
    Args:
        ticker: Stock ticker symbol.
        
    Returns:
        Dictionary with fundamental metrics.
    """
    logger.info(f"Fetching Yahoo Finance fundamentals for {ticker}")
    
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Extract key fundamental metrics
        fundamentals = {
            'ticker': ticker,
            'fetch_date': datetime.now().isoformat(),
            'source': 'yahoo_finance',
            
            # Valuation metrics
            'market_cap': info.get('marketCap'),
            'enterprise_value': info.get('enterpriseValue'),
            'trailing_pe': info.get('trailingPE'),
            'forward_pe': info.get('forwardPE'),
            'peg_ratio': info.get('pegRatio'),
            'price_to_book': info.get('priceToBook'),
            'price_to_sales': info.get('priceToSalesTrailing12Months'),
            'enterprise_to_revenue': info.get('enterpriseToRevenue'),
            'enterprise_to_ebitda': info.get('enterpriseToEbitda'),
            
            # Financial health
            'total_debt': info.get('totalDebt'),
            'total_cash': info.get('totalCash'),
            'debt_to_equity': info.get('debtToEquity'),
            'current_ratio': info.get('currentRatio'),
            'quick_ratio': info.get('quickRatio'),
            
            # Profitability
            'profit_margins': info.get('profitMargins'),
            'operating_margins': info.get('operatingMargins'),
            'gross_margins': info.get('grossMargins'),
            'return_on_assets': info.get('returnOnAssets'),
            'return_on_equity': info.get('returnOnEquity'),
            'ebitda': info.get('ebitda'),
            
            # Growth metrics
            'revenue_growth': info.get('revenueGrowth'),
            'earnings_growth': info.get('earningsGrowth'),
            'earnings_quarterly_growth': info.get('earningsQuarterlyGrowth'),
            
            # Revenue and earnings
            'total_revenue': info.get('totalRevenue'),
            'revenue_per_share': info.get('revenuePerShare'),
            'earnings_per_share': info.get('trailingEps'),
            'forward_eps': info.get('forwardEps'),
            
            # Dividend metrics
            'dividend_rate': info.get('dividendRate'),
            'dividend_yield': info.get('dividendYield'),
            'payout_ratio': info.get('payoutRatio'),
            
            # Company info
            'sector': info.get('sector'),
            'industry': info.get('industry'),
            'country': info.get('country'),
            'employees': info.get('fullTimeEmployees'),
            'website': info.get('website'),
            'business_summary': info.get('longBusinessSummary'),
        }
        
        logger.info(f"Successfully fetched Yahoo Finance fundamentals for {ticker}")
        return fundamentals
        
    except Exception as e:
        logger.error(f"Failed to fetch Yahoo Finance fundamentals for {ticker}: {str(e)}")
        raise


@retry_with_backoff(max_retries=3)
def fetch_fundamentals_alphavantage(ticker: str, api_key: str) -> Dict:
    """
    Fetch fundamental data using Alpha Vantage API.
    
    Args:
        ticker: Stock ticker symbol.
        api_key: Alpha Vantage API key.
        
    Returns:
        Dictionary with fundamental metrics.
    """
    logger.info(f"Fetching Alpha Vantage fundamentals for {ticker}")
    
    try:
        # Alpha Vantage OVERVIEW endpoint
        url = "https://www.alphavantage.co/query"
        params = {
            'function': 'OVERVIEW',
            'symbol': ticker,
            'apikey': api_key
        }
        
        response = make_api_request(url, params=params)
        data = response.json()
        
        # Check for API error
        if 'Error Message' in data:
            raise ValueError(f"Alpha Vantage error: {data['Error Message']}")
        
        if 'Note' in data:
            raise ValueError(f"Alpha Vantage rate limit: {data['Note']}")
        
        # Extract and normalize key metrics
        fundamentals = {
            'ticker': ticker,
            'fetch_date': datetime.now().isoformat(),
            'source': 'alpha_vantage',
            
            # Company info
            'name': data.get('Name'),
            'exchange': data.get('Exchange'),
            'currency': data.get('Currency'),
            'sector': data.get('Sector'),
            'industry': data.get('Industry'),
            'country': data.get('Country'),
            'description': data.get('Description'),
            
            # Valuation metrics
            'market_cap': _safe_float(data.get('MarketCapitalization')),
            'trailing_pe': _safe_float(data.get('TrailingPE')),
            'forward_pe': _safe_float(data.get('ForwardPE')),
            'peg_ratio': _safe_float(data.get('PEGRatio')),
            'price_to_book': _safe_float(data.get('PriceToBookRatio')),
            'price_to_sales': _safe_float(data.get('PriceToSalesRatioTTM')),
            'enterprise_value': _safe_float(data.get('EnterpriseValue')),
            'enterprise_to_revenue': _safe_float(data.get('EVToRevenue')),
            'enterprise_to_ebitda': _safe_float(data.get('EVToEBITDA')),
            
            # Profitability
            'profit_margin': _safe_float(data.get('ProfitMargin')),
            'operating_margin': _safe_float(data.get('OperatingMarginTTM')),
            'return_on_assets': _safe_float(data.get('ReturnOnAssetsTTM')),
            'return_on_equity': _safe_float(data.get('ReturnOnEquityTTM')),
            'ebitda': _safe_float(data.get('EBITDA')),
            
            # Financial health
            'debt_to_equity': _safe_float(data.get('DebtToEquity')),
            'current_ratio': _safe_float(data.get('CurrentRatio')),
            'quick_ratio': _safe_float(data.get('QuickRatio')),
            
            # Growth
            'revenue_growth_yoy': _safe_float(data.get('QuarterlyRevenueGrowthYOY')),
            'earnings_growth_yoy': _safe_float(data.get('QuarterlyEarningsGrowthYOY')),
            
            # Earnings and revenue
            'revenue_per_share': _safe_float(data.get('RevenuePerShareTTM')),
            'earnings_per_share': _safe_float(data.get('EPS')),
            'diluted_eps': _safe_float(data.get('DilutedEPSTTM')),
            
            # Dividend
            'dividend_per_share': _safe_float(data.get('DividendPerShare')),
            'dividend_yield': _safe_float(data.get('DividendYield')),
            'payout_ratio': _safe_float(data.get('PayoutRatio')),
            
            # Analyst targets
            'analyst_target_price': _safe_float(data.get('AnalystTargetPrice')),
            
            # Additional metrics
            '52_week_high': _safe_float(data.get('52WeekHigh')),
            '52_week_low': _safe_float(data.get('52WeekLow')),
            '50_day_ma': _safe_float(data.get('50DayMovingAverage')),
            '200_day_ma': _safe_float(data.get('200DayMovingAverage')),
            'shares_outstanding': _safe_float(data.get('SharesOutstanding')),
        }
        
        logger.info(f"Successfully fetched Alpha Vantage fundamentals for {ticker}")
        return fundamentals
        
    except Exception as e:
        logger.error(f"Failed to fetch Alpha Vantage fundamentals for {ticker}: {str(e)}")
        raise


def _safe_float(value) -> Optional[float]:
    """Safely convert string to float, returning None if conversion fails."""
    if value is None or value == 'None' or value == '':
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def fetch_quarterly_financials(ticker: str) -> Dict:
    """
    Fetch quarterly financial statements (income statement, balance sheet, cash flow).
    
    Args:
        ticker: Stock ticker symbol.
        
    Returns:
        Dictionary with quarterly financial data.
    """
    logger.info(f"Fetching quarterly financials for {ticker}")
    
    try:
        stock = yf.Ticker(ticker)
        
        financials = {
            'ticker': ticker,
            'fetch_date': datetime.now().isoformat(),
            'quarterly_income_statement': stock.quarterly_income_stmt.to_dict() if hasattr(stock, 'quarterly_income_stmt') and stock.quarterly_income_stmt is not None else {},
            'quarterly_balance_sheet': stock.quarterly_balance_sheet.to_dict() if hasattr(stock, 'quarterly_balance_sheet') and stock.quarterly_balance_sheet is not None else {},
            'quarterly_cash_flow': stock.quarterly_cashflow.to_dict() if hasattr(stock, 'quarterly_cashflow') and stock.quarterly_cashflow is not None else {},
        }
        
        # Convert timestamp keys to strings for JSON serialization
        for key in ['quarterly_income_statement', 'quarterly_balance_sheet', 'quarterly_cash_flow']:
            if financials[key]:
                financials[key] = {str(k): v for k, v in financials[key].items()}
        
        logger.info(f"Successfully fetched quarterly financials for {ticker}")
        return financials
        
    except Exception as e:
        logger.error(f"Failed to fetch quarterly financials for {ticker}: {str(e)}")
        return {'ticker': ticker, 'error': str(e)}


def validate_fundamentals(fundamentals: Dict, ticker: str) -> bool:
    """
    Validate that critical fundamental fields are present.
    
    Args:
        fundamentals: Dictionary with fundamental data.
        ticker: Stock ticker symbol.
        
    Returns:
        True if validation passes, False otherwise.
    """
    critical_fields = ['market_cap', 'trailing_pe']
    missing_fields = []
    
    for field in critical_fields:
        if field not in fundamentals or fundamentals[field] is None:
            missing_fields.append(field)
    
    if missing_fields:
        logger.warning(f"Missing critical fundamental data for {ticker}: {', '.join(missing_fields)}")
        return False
    
    return True


def save_fundamentals(fundamentals: Dict, ticker: str, output_dir: Path) -> Path:
    """
    Save fundamentals data to date-partitioned directory.
    
    Args:
        fundamentals: Dictionary with fundamental data.
        ticker: Stock ticker symbol.
        output_dir: Base output directory.
        
    Returns:
        Path to saved file.
    """
    # Create timestamped directory
    date_dir = create_timestamped_directory(output_dir)
    
    # Create filename with ticker
    filename = f"{ticker.replace('.', '_')}.json"
    filepath = date_dir / filename
    
    # Save to JSON
    save_json(fundamentals, filepath)
    logger.info(f"Saved fundamentals to {filepath}")
    
    return filepath


def process_ticker_file(ticker_file: Path, source: str, output_dir: Path, include_quarterly: bool):
    """
    Fetch fundamentals for multiple tickers from a CSV file.
    
    Args:
        ticker_file: Path to CSV file with 'ticker' column.
        source: Data source ('yahoo' or 'alphavantage').
        output_dir: Output directory for data.
        include_quarterly: Whether to include quarterly financial statements.
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
            # Fetch fundamentals based on source
            if source == 'yahoo':
                fundamentals = fetch_fundamentals_yahoo(ticker)
            elif source == 'alphavantage':
                if not config.alpha_vantage_key:
                    raise ValueError("Alpha Vantage API key not configured")
                fundamentals = fetch_fundamentals_alphavantage(ticker, config.alpha_vantage_key)
            else:
                raise ValueError(f"Unknown source: {source}")
            
            # Optionally fetch quarterly financials
            if include_quarterly and source == 'yahoo':
                quarterly_data = fetch_quarterly_financials(ticker)
                fundamentals['quarterly_financials'] = quarterly_data
            
            # Validate and save
            validate_fundamentals(fundamentals, ticker)
            save_fundamentals(fundamentals, ticker, output_dir)
            success_count += 1
            
        except Exception as e:
            logger.error(f"Failed to process {ticker}: {str(e)}")
            failed_tickers.append(ticker)
    
    # Log summary
    logger.info(f"Batch processing complete: {success_count}/{len(tickers)} succeeded")
    if failed_tickers:
        logger.warning(f"Failed tickers: {', '.join(failed_tickers)}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Fetch company fundamental data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch from Yahoo Finance
  python fetch_fundamentals.py --ticker RELIANCE.NS --source yahoo
  
  # Fetch from Alpha Vantage
  python fetch_fundamentals.py --ticker RELIANCE.NS --source alphavantage
  
  # Batch processing with quarterly financials
  python fetch_fundamentals.py --ticker-file tickers.csv --source yahoo --quarterly
        """
    )
    
    parser.add_argument('--ticker', type=str, help='Stock ticker symbol')
    parser.add_argument('--ticker-file', type=Path, help='CSV file with ticker symbols')
    parser.add_argument('--source', type=str, default='yahoo',
                       choices=['yahoo', 'alphavantage'],
                       help='Data source (default: yahoo)')
    parser.add_argument('--output-dir', type=Path, default='data/raw/fundamentals',
                       help='Output directory (default: data/raw/fundamentals)')
    parser.add_argument('--quarterly', action='store_true',
                       help='Include quarterly financial statements (Yahoo only)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.ticker and not args.ticker_file:
        parser.error("Either --ticker or --ticker-file must be specified")
    
    logger.info("Starting fundamentals data fetch")
    logger.info(f"Source: {args.source}")
    logger.info(f"Output directory: {args.output_dir}")
    
    try:
        if args.ticker_file:
            # Batch mode
            process_ticker_file(args.ticker_file, args.source, args.output_dir, args.quarterly)
        
        else:
            # Single ticker mode
            if args.source == 'yahoo':
                fundamentals = fetch_fundamentals_yahoo(args.ticker)
            elif args.source == 'alphavantage':
                if not config.alpha_vantage_key:
                    raise ValueError("Alpha Vantage API key not configured in .env file")
                fundamentals = fetch_fundamentals_alphavantage(args.ticker, config.alpha_vantage_key)
            
            # Optionally fetch quarterly financials
            if args.quarterly and args.source == 'yahoo':
                quarterly_data = fetch_quarterly_financials(args.ticker)
                fundamentals['quarterly_financials'] = quarterly_data
            
            # Validate and save
            validate_fundamentals(fundamentals, args.ticker)
            save_fundamentals(fundamentals, args.ticker, args.output_dir)
        
        logger.info("Fundamentals data fetch completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to fetch fundamentals: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
