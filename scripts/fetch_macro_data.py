#!/usr/bin/env python3
"""
Macroeconomic data fetching script using World Bank API.

Fetches macroeconomic indicators (GDP, inflation, interest rates, etc.)
with support for multiple countries and indicator presets.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.utils.logger import get_logger
from backend.utils.api_helpers import make_api_request, retry_with_backoff, handle_api_error
from backend.utils.file_operations import (
    create_timestamped_directory,
    save_dataframe_to_csv,
    save_json
)


logger = get_logger(__name__)


# Indicator presets for common use cases
INDICATOR_PRESETS = {
    'growth': [
        'NY.GDP.MKTP.KD.ZG',  # GDP growth (annual %)
        'NY.GDP.PCAP.KD.ZG',  # GDP per capita growth (annual %)
        'NE.GDI.TOTL.ZS',     # Gross capital formation (% of GDP)
    ],
    'inflation': [
        'FP.CPI.TOTL.ZG',     # Inflation, consumer prices (annual %)
        'FP.WPI.TOTL',        # Wholesale price index
    ],
    'monetary': [
        'FR.INR.RINR',        # Real interest rate (%)
        'FR.INR.LEND',        # Lending interest rate (%)
        'FM.LBL.BMNY.ZG',     # Broad money growth (annual %)
    ],
    'trade': [
        'NE.EXP.GNFS.ZS',     # Exports of goods and services (% of GDP)
        'NE.IMP.GNFS.ZS',     # Imports of goods and services (% of GDP)
        'BN.CAB.XOKA.GD.ZS',  # Current account balance (% of GDP)
    ],
    'labor': [
        'SL.UEM.TOTL.ZS',     # Unemployment, total (% of total labor force)
        'SL.TLF.TOTL.IN',     # Labor force, total
    ],
    'full': [
        # Combined set of most important indicators
        'NY.GDP.MKTP.KD.ZG',  # GDP growth
        'FP.CPI.TOTL.ZG',     # Inflation
        'FR.INR.RINR',        # Real interest rate
        'NE.EXP.GNFS.ZS',     # Exports (% of GDP)
        'NE.IMP.GNFS.ZS',     # Imports (% of GDP)
        'SL.UEM.TOTL.ZS',     # Unemployment
        'NY.GDP.PCAP.CD',     # GDP per capita (current US$)
        'FI.RES.TOTL.CD',     # Total reserves (current US$)
        'GC.DOD.TOTL.GD.ZS',  # Central government debt (% of GDP)
        'BN.CAB.XOKA.GD.ZS',  # Current account balance (% of GDP)
    ]
}


def fetch_world_bank_indicator(
    country: str,
    indicator: str,
    start_year: int,
    end_year: int
) -> List[Dict]:
    """
    Fetch a single indicator from World Bank API.
    
    Args:
        country: ISO country code (e.g., 'IND' for India).
        indicator: World Bank indicator code (e.g., 'NY.GDP.MKTP.KD.ZG').
        start_year: Start year for data.
        end_year: End year for data.
        
    Returns:
        List of dictionaries with year, value, and metadata.
    """
    logger.info(f"Fetching indicator {indicator} for {country} ({start_year}-{end_year})")
    
    try:
        # World Bank API endpoint
        url = f"https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"
        
        params = {
            'date': f"{start_year}:{end_year}",
            'format': 'json',
            'per_page': 1000  # Max results per page
        }
        
        # Make API request with retry
        @retry_with_backoff(max_retries=3)
        def _fetch():
            response = make_api_request(url, params=params, timeout=30, rate_limiter=None)
            return response.json()
        
        data = _fetch()
        
        # World Bank API returns [metadata, data] array
        if not isinstance(data, list) or len(data) < 2:
            logger.warning(f"Unexpected response format for {indicator}")
            return []
        
        metadata = data[0]
        records = data[1]
        
        if not records:
            logger.warning(f"No data available for indicator {indicator}")
            return []
        
        # Extract and normalize data
        normalized_records = []
        for record in records:
            if record.get('value') is not None:  # Skip null values
                normalized_records.append({
                    'country_code': record.get('country', {}).get('id'),
                    'country_name': record.get('country', {}).get('value'),
                    'indicator_code': record.get('indicator', {}).get('id'),
                    'indicator_name': record.get('indicator', {}).get('value'),
                    'year': int(record.get('date', 0)),
                    'value': float(record.get('value')),
                })
        
        logger.info(f"Fetched {len(normalized_records)} data points for {indicator}")
        return normalized_records
        
    except Exception as e:
        logger.error(f"Failed to fetch indicator {indicator} for {country}: {str(e)}")
        return []


def fetch_multiple_indicators(
    country: str,
    indicators: List[str],
    start_year: int,
    end_year: int
) -> pd.DataFrame:
    """
    Fetch multiple indicators and combine into a DataFrame.
    
    Args:
        country: ISO country code.
        indicators: List of indicator codes.
        start_year: Start year.
        end_year: End year.
        
    Returns:
        DataFrame with all indicators.
    """
    all_records = []
    
    for indicator in indicators:
        records = fetch_world_bank_indicator(country, indicator, start_year, end_year)
        all_records.extend(records)
    
    if not all_records:
        logger.warning(f"No data fetched for any indicators for {country}")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_records)
    logger.info(f"Combined data: {len(df)} total records for {len(indicators)} indicators")
    
    return df


def fetch_multiple_countries(
    countries: List[str],
    indicators: List[str],
    start_year: int,
    end_year: int
) -> pd.DataFrame:
    """
    Fetch indicators for multiple countries.
    
    Args:
        countries: List of ISO country codes.
        indicators: List of indicator codes.
        start_year: Start year.
        end_year: End year.
        
    Returns:
        DataFrame with data for all countries.
    """
    all_data = []
    
    for country in countries:
        logger.info(f"Processing country: {country}")
        df = fetch_multiple_indicators(country, indicators, start_year, end_year)
        if not df.empty:
            all_data.append(df)
    
    if not all_data:
        logger.warning("No data fetched for any countries")
        return pd.DataFrame()
    
    combined_df = pd.concat(all_data, ignore_index=True)
    logger.info(f"Combined data for {len(countries)} countries: {len(combined_df)} total records")
    
    return combined_df


def calculate_summary_statistics(df: pd.DataFrame, country: str) -> Dict:
    """
    Calculate summary statistics for fetched data.
    
    Args:
        df: DataFrame with indicator data.
        country: Country code.
        
    Returns:
        Dictionary with summary statistics by indicator.
    """
    summary = {}
    
    for indicator_code in df['indicator_code'].unique():
        indicator_df = df[df['indicator_code'] == indicator_code]
        indicator_name = indicator_df['indicator_name'].iloc[0]
        
        values = indicator_df['value']
        years = indicator_df['year']
        
        summary[indicator_code] = {
            'indicator_name': indicator_name,
            'data_points': len(indicator_df),
            'year_range': f"{years.min()}-{years.max()}",
            'mean': round(values.mean(), 2),
            'std': round(values.std(), 2),
            'min': round(values.min(), 2),
            'max': round(values.max(), 2),
            'latest_year': int(years.max()),
            'latest_value': round(indicator_df[indicator_df['year'] == years.max()]['value'].iloc[0], 2),
        }
    
    return summary


def save_macro_data(df: pd.DataFrame, country: str, output_dir: Path, save_format: str = 'csv') -> Path:
    """
    Save macroeconomic data to date-partitioned directory.
    
    Args:
        df: DataFrame with macro data.
        country: Country code.
        output_dir: Base output directory.
        save_format: Output format ('csv' or 'json').
        
    Returns:
        Path to saved file.
    """
    # Create timestamped directory
    date_dir = create_timestamped_directory(output_dir)
    
    # Create filename
    if save_format == 'csv':
        filename = f"{country}_macro.csv"
        filepath = date_dir / filename
        save_dataframe_to_csv(df, filepath, index=False)
    else:
        filename = f"{country}_macro.json"
        filepath = date_dir / filename
        save_json(df.to_dict(orient='records'), filepath)
    
    logger.info(f"Saved {len(df)} records to {filepath}")
    
    return filepath


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Fetch macroeconomic indicators from World Bank API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch GDP growth and inflation for India
  python fetch_macro_data.py --country IND --indicators NY.GDP.MKTP.KD.ZG,FP.CPI.TOTL.ZG
  
  # Use preset indicator set
  python fetch_macro_data.py --country IND --preset full
  
  # Fetch for multiple countries
  python fetch_macro_data.py --country IND,USA,CHN --preset growth --start-year 2010
  
  # Full indicator list with custom date range
  python fetch_macro_data.py --country IND --preset full --start-year 2000 --end-year 2024
  
Available presets:
  - growth: GDP and investment indicators
  - inflation: Inflation and price indices
  - monetary: Interest rates and money supply
  - trade: Exports, imports, and current account
  - labor: Unemployment and labor force
  - full: Comprehensive set of 10+ key indicators
        """
    )
    
    parser.add_argument('--country', type=str, default='IND',
                       help='ISO country code(s), comma-separated for multiple (default: IND)')
    parser.add_argument('--indicators', type=str,
                       help='Comma-separated World Bank indicator codes')
    parser.add_argument('--preset', type=str, choices=list(INDICATOR_PRESETS.keys()),
                       help='Use preset indicator set (growth, inflation, monetary, trade, labor, full)')
    parser.add_argument('--start-year', type=int, default=2010,
                       help='Start year (default: 2010)')
    parser.add_argument('--end-year', type=int, default=datetime.now().year,
                       help='End year (default: current year)')
    parser.add_argument('--output-dir', type=Path, default='data/raw/macro',
                       help='Output directory (default: data/raw/macro)')
    parser.add_argument('--format', type=str, default='csv', choices=['csv', 'json'],
                       help='Output format (default: csv)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.indicators and not args.preset:
        parser.error("Either --indicators or --preset must be specified")
    
    # Determine indicator list
    if args.preset:
        indicators = INDICATOR_PRESETS[args.preset]
        logger.info(f"Using preset '{args.preset}' with {len(indicators)} indicators")
    else:
        indicators = [ind.strip() for ind in args.indicators.split(',')]
    
    # Parse country list
    countries = [c.strip() for c in args.country.split(',')]
    
    logger.info("Starting macroeconomic data fetch")
    logger.info(f"Countries: {', '.join(countries)}")
    logger.info(f"Indicators: {len(indicators)}")
    logger.info(f"Year range: {args.start_year}-{args.end_year}")
    logger.info(f"Output directory: {args.output_dir}")
    
    try:
        # Fetch data
        if len(countries) == 1:
            df = fetch_multiple_indicators(countries[0], indicators, args.start_year, args.end_year)
        else:
            df = fetch_multiple_countries(countries, indicators, args.start_year, args.end_year)
        
        if df.empty:
            logger.warning("No data fetched")
            return
        
        # Calculate and log summary statistics
        for country in countries:
            country_df = df[df['country_code'] == country] if len(countries) > 1 else df
            
            if not country_df.empty:
                summary = calculate_summary_statistics(country_df, country)
                
                logger.info(f"\nSummary statistics for {country}:")
                for ind_code, stats in summary.items():
                    logger.info(f"  {stats['indicator_name']}:")
                    logger.info(f"    Range: {stats['year_range']}, Data points: {stats['data_points']}")
                    logger.info(f"    Mean: {stats['mean']}, Latest ({stats['latest_year']}): {stats['latest_value']}")
        
        # Save data
        if len(countries) == 1:
            save_macro_data(df, countries[0], args.output_dir, args.format)
        else:
            # Save combined file for multiple countries
            save_macro_data(df, '_'.join(countries), args.output_dir, args.format)
        
        logger.info("Macroeconomic data fetch completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to fetch macroeconomic data: {str(e)}")
        handle_api_error(e, context=f"countries={','.join(countries)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
