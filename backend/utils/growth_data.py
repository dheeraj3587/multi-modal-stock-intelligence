import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def load_fundamentals(
    data_dir: str, 
    tickers: Optional[List[str]] = None, 
    start_date: Optional[str] = None, 
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Load fundamentals from raw data directory.
    Structure: data_dir/{YYYY-MM-DD}/*.json
    """
    data_path = Path(data_dir)
    all_data = []
    
    for date_dir in data_path.iterdir():
        if not date_dir.is_dir(): continue
        date_str = date_dir.name
        
        if start_date and date_str < start_date: continue
        if end_date and date_str > end_date: continue
        
        for file_path in date_dir.glob("*.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Handle list or dict
                items = data if isinstance(data, list) else [data]
                
                for item in items:
                    ticker = item.get('symbol') or item.get('ticker')
                    if tickers and ticker not in tickers: continue
                    
                    # Extract key metrics
                    metrics = {
                        'date': date_str,
                        'ticker': ticker,
                        'trailing_pe': item.get('trailingPE'),
                        'forward_pe': item.get('forwardPE'),
                        'price_to_book': item.get('priceToBook'),
                        'debt_to_equity': item.get('debtToEquity'),
                        'return_on_equity': item.get('returnOnEquity'),
                        'return_on_assets': item.get('returnOnAssets'),
                        'profit_margins': item.get('profitMargins'),
                        'revenue_growth': item.get('revenueGrowth'),
                        'sector': item.get('sector'),
                        'market_cap': item.get('marketCap')
                    }
                    all_data.append(metrics)
            except Exception:
                continue
                
    df = pd.DataFrame(all_data)
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['ticker', 'date'])
        # Forward fill within ticker to handle quarterly updates
        df = df.groupby('ticker').ffill()
        
    return df

def load_prices(
    data_dir: str,
    tickers: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Load price data from CSV files.
    Expected structure: data_dir/{ticker}.csv or data_dir/prices.csv
    """
    data_path = Path(data_dir)
    all_prices = []
    
    # Try loading from individual ticker CSV files
    if tickers:
        for ticker in tickers:
            ticker_file = data_path / f"{ticker}.csv"
            if ticker_file.exists():
                df = pd.read_csv(ticker_file)
                if 'ticker' not in df.columns:
                    df['ticker'] = ticker
                all_prices.append(df)
    
    # Try loading from consolidated prices.csv
    consolidated_file = data_path / "prices.csv"
    if consolidated_file.exists():
        df = pd.read_csv(consolidated_file)
        if tickers:
            df = df[df['ticker'].isin(tickers)]
        all_prices.append(df)
    
    if not all_prices:
        return pd.DataFrame()
    
    df = pd.concat(all_prices, ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])
    
    # Normalize column names to lowercase
    df.columns = df.columns.str.lower()
    
    if start_date:
        df = df[df['date'] >= start_date]
    if end_date:
        df = df[df['date'] <= end_date]
    
    return df.sort_values(['ticker', 'date'])


def load_technical_indicators(
    data_dir: str, 
    tickers: Optional[List[str]] = None,
    price_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Load or compute technical indicators.
    
    If price_df is provided, compute indicators directly.
    Otherwise, try to load from processed files.
    
    Returns DataFrame with columns: ticker, date, momentum_5d, momentum_20d, 
    volatility_20d, rsi_14, macd, macd_signal, sma_20, sma_50
    """
    # Try to load from processed CSV if it exists
    processed_path = Path(data_dir) / 'technical_indicators.csv'
    if processed_path.exists():
        df = pd.read_csv(processed_path)
        df['date'] = pd.to_datetime(df['date'])
        if tickers:
            df = df[df['ticker'].isin(tickers)]
        return df
    
    # If no processed file and no price_df, return empty
    if price_df is None:
        return pd.DataFrame()
    
    # Compute indicators from price data
    df = price_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['ticker', 'date'])
    
    all_features = []
    
    for ticker in df['ticker'].unique():
        if tickers and ticker not in tickers:
            continue
            
        ticker_df = df[df['ticker'] == ticker].copy()
        
        # Momentum (returns over different periods)
        ticker_df['momentum_5d'] = ticker_df['close'].pct_change(periods=5)
        ticker_df['momentum_20d'] = ticker_df['close'].pct_change(periods=20)
        
        # Volatility (rolling standard deviation of returns)
        returns = ticker_df['close'].pct_change()
        ticker_df['volatility_20d'] = returns.rolling(window=20).std()
        
        # RSI (Relative Strength Index)
        delta = ticker_df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, 1e-10)
        ticker_df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = ticker_df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = ticker_df['close'].ewm(span=26, adjust=False).mean()
        ticker_df['macd'] = ema_12 - ema_26
        ticker_df['macd_signal'] = ticker_df['macd'].ewm(span=9, adjust=False).mean()
        
        # Moving averages
        ticker_df['sma_20'] = ticker_df['close'].rolling(window=20).mean()
        ticker_df['sma_50'] = ticker_df['close'].rolling(window=50).mean()
        
        # Distance from moving averages (normalized)
        ticker_df['dist_sma_20'] = (ticker_df['close'] - ticker_df['sma_20']) / ticker_df['sma_20']
        ticker_df['dist_sma_50'] = (ticker_df['close'] - ticker_df['sma_50']) / ticker_df['sma_50']
        
        all_features.append(ticker_df)
    
    if not all_features:
        return pd.DataFrame()
    
    result = pd.concat(all_features, ignore_index=True)
    
    # Select only the indicator columns
    indicator_cols = ['ticker', 'date', 'momentum_5d', 'momentum_20d', 'volatility_20d',
                      'rsi_14', 'macd', 'macd_signal', 'dist_sma_20', 'dist_sma_50']
    
    return result[indicator_cols]

def compute_forward_returns(price_df: pd.DataFrame, horizon_days: int = 60) -> pd.Series:
    """
    Compute forward returns: (Price(t+h) - Price(t)) / Price(t)
    """
    df = price_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['ticker', 'date'])
    
    # Shift close price back by horizon_days (rows)
    # Assuming daily data and approx 1 row per day. 
    # Ideally use merge_asof with dates.
    # Simplified: use shift assuming sorted daily data
    
    indexer = df.groupby('ticker')['close']
    df['fwd_price'] = indexer.shift(-horizon_days)
    
    fwd_returns = (df['fwd_price'] - df['close']) / df['close']
    return fwd_returns

def merge_growth_features(
    fundamentals_df: pd.DataFrame, 
    technical_df: pd.DataFrame, 
    price_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge features with 45-day lag for fundamentals.
    
    Fundamentals are lagged by 45 days to simulate reporting delay.
    Technical indicators and prices are aligned on the same date.
    """
    if fundamentals_df.empty or technical_df.empty or price_df.empty:
        return pd.DataFrame()
    
    # Add 45 days to fundamental dates to simulate availability lag
    fund_lagged = fundamentals_df.copy()
    fund_lagged['date'] = pd.to_datetime(fund_lagged['date'])
    fund_lagged['date_avail'] = fund_lagged['date'] + timedelta(days=45)
    
    # Prepare technical and price data
    tech_df = technical_df.copy()
    tech_df['date'] = pd.to_datetime(tech_df['date'])
    
    price_df = price_df.copy()
    price_df['date'] = pd.to_datetime(price_df['date'])
    
    # For each fundamental record, find the nearest future date in technical/price data
    # that is on or after date_avail
    merged_records = []
    
    for ticker in fund_lagged['ticker'].unique():
        fund_ticker = fund_lagged[fund_lagged['ticker'] == ticker].sort_values('date_avail')
        tech_ticker = tech_df[tech_df['ticker'] == ticker].sort_values('date')
        price_ticker = price_df[price_df['ticker'] == ticker].sort_values('date')
        
        if tech_ticker.empty or price_ticker.empty:
            continue
        
        # Use merge_asof to align fundamentals with the nearest future technical/price date
        fund_ticker = fund_ticker.rename(columns={'date_avail': 'date'})
        
        # Merge with technical indicators
        merged = pd.merge_asof(
            fund_ticker.sort_values('date'),
            tech_ticker.sort_values('date'),
            on='date',
            by='ticker',
            direction='forward',
            suffixes=('', '_tech')
        )
        
        # Merge with price data
        merged = pd.merge_asof(
            merged.sort_values('date'),
            price_ticker[['ticker', 'date', 'close']].sort_values('date'),
            on='date',
            by='ticker',
            direction='forward',
            suffixes=('', '_price')
        )
        
        merged_records.append(merged)
    
    if not merged_records:
        return pd.DataFrame()
    
    result = pd.concat(merged_records, ignore_index=True)
    
    # Drop rows with missing values (no matching future date found)
    result = result.dropna(subset=['close'])
    
    return result

def engineer_growth_features(
    df: pd.DataFrame, 
    is_train: bool = True, 
    scalers: Optional[Dict] = None,
    sector_medians: Optional[Dict] = None
) -> Tuple[pd.DataFrame, List[str], Dict]:
    """
    Normalize features and encode categoricals.
    
    - Imputes missing fundamentals with sector medians (computed on training data)
    - Applies separate scalers for fundamental and technical features
    - Encodes sector as dummy variables
    """
    if scalers is None:
        scalers = {}
    if sector_medians is None:
        sector_medians = {}
    
    df = df.copy()
    
    # Identify feature types
    fundamental_cols = ['trailing_pe', 'forward_pe', 'price_to_book', 'debt_to_equity',
                       'return_on_equity', 'return_on_assets', 'profit_margins', 
                       'revenue_growth', 'market_cap']
    
    technical_cols = ['momentum_5d', 'momentum_20d', 'volatility_20d', 'rsi_14',
                      'macd', 'macd_signal', 'dist_sma_20', 'dist_sma_50']
    
    # Filter to existing columns
    fundamental_cols = [c for c in fundamental_cols if c in df.columns]
    technical_cols = [c for c in technical_cols if c in df.columns]
    
    # Sector median imputation for fundamentals
    if is_train and 'sector' in df.columns:
        # Compute sector medians on training data
        for col in fundamental_cols:
            sector_medians[col] = df.groupby('sector')[col].median().to_dict()
        
        # Fill missing with sector median
        for col in fundamental_cols:
            df[col] = df.apply(
                lambda row: sector_medians[col].get(row['sector'], df[col].median()) 
                if pd.isna(row[col]) else row[col],
                axis=1
            )
    elif not is_train and sector_medians:
        # Use training sector medians
        for col in fundamental_cols:
            if col in sector_medians:
                df[col] = df.apply(
                    lambda row: sector_medians[col].get(row['sector'], df[col].median()) 
                    if pd.isna(row[col]) else row[col],
                    axis=1
                )
    
    # Fill remaining NaNs with column median
    for col in fundamental_cols + technical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    # Encode sector
    if 'sector' in df.columns:
        df = pd.get_dummies(df, columns=['sector'], prefix='sector', drop_first=True)
    
    # Scale features
    if is_train:
        # Separate scalers for different feature types
        if fundamental_cols:
            fund_scaler = StandardScaler()
            df[fundamental_cols] = fund_scaler.fit_transform(df[fundamental_cols])
            scalers['fundamental'] = fund_scaler
        
        if technical_cols:
            tech_scaler = StandardScaler()
            df[technical_cols] = tech_scaler.fit_transform(df[technical_cols])
            scalers['technical'] = tech_scaler
        
        scalers['sector_medians'] = sector_medians
    else:
        # Apply training scalers
        if 'fundamental' in scalers and fundamental_cols:
            df[fundamental_cols] = scalers['fundamental'].transform(df[fundamental_cols])
        
        if 'technical' in scalers and technical_cols:
            df[technical_cols] = scalers['technical'].transform(df[technical_cols])
    
    # Get feature names (exclude date, ticker, target, original date column if present)
    exclude_cols = ['ticker', 'date', 'target', 'close', 'fwd_price', 'date_avail']
    feature_names = [c for c in df.columns if c not in exclude_cols]
    
    return df, feature_names, scalers

def split_growth_data(
    df: pd.DataFrame, 
    train_ratio: float = 0.6, 
    val_ratio: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Temporal split."""
    dates = df['date'].sort_values().unique()
    n = len(dates)
    train_end = dates[int(n * train_ratio)]
    val_end = dates[int(n * (train_ratio + val_ratio))]
    
    train = df[df['date'] <= train_end]
    val = df[(df['date'] > train_end) & (df['date'] <= val_end)]
    test = df[df['date'] > val_end]
    
    return train, val, test

def split_by_stocks(
    df: pd.DataFrame, 
    train_ratio: float = 0.8, 
    stratify_by: str = 'sector'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data by tickers for cross-stock generalization testing.
    
    Args:
        df: DataFrame containing 'ticker' and optionally 'sector' columns
        train_ratio: Proportion of tickers to use for training
        stratify_by: Column to stratify by (usually 'sector')
        
    Returns:
        Tuple of (train_df, test_df)
    """
    tickers = df['ticker'].unique()
    
    if stratify_by in df.columns:
        from sklearn.model_selection import train_test_split
        # Get one entry per ticker to determine stratification label
        ticker_strat = df.groupby('ticker')[stratify_by].first()
        
        try:
            train_tickers, test_tickers = train_test_split(
                tickers, 
                train_size=train_ratio, 
                stratify=ticker_strat, 
                random_state=42
            )
        except ValueError:
            # Fallback if stratification fails (e.g. too few samples per class)
            train_tickers, test_tickers = train_test_split(
                tickers, 
                train_size=train_ratio, 
                random_state=42
            )
    else:
        # Random split if no stratification column
        np.random.seed(42)
        shuffled_tickers = np.random.permutation(tickers)
        split_idx = int(len(tickers) * train_ratio)
        train_tickers = shuffled_tickers[:split_idx]
        test_tickers = shuffled_tickers[split_idx:]
        
    train_df = df[df['ticker'].isin(train_tickers)].sort_values(['ticker', 'date'])
    test_df = df[df['ticker'].isin(test_tickers)].sort_values(['ticker', 'date'])
    
    return train_df, test_df

def temporal_subsplit(
    train_df: pd.DataFrame, 
    val_ratio: float = 0.125
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a temporal validation split from the training set.
    
    Args:
        train_df: Training DataFrame
        val_ratio: Fraction of time range to use for validation (from the end)
        
    Returns:
        Tuple of (new_train_df, val_df)
    """
    dates = train_df['date'].sort_values().unique()
    val_start_idx = int(len(dates) * (1 - val_ratio))
    val_start_date = dates[val_start_idx]
    
    new_train_df = train_df[train_df['date'] <= val_start_date]
    val_df = train_df[train_df['date'] > val_start_date]
    
    return new_train_df, val_df

def validate_growth_features(X: np.ndarray, y: np.ndarray):
    """Validate features and target."""
    if np.isnan(X).any():
        raise ValueError("NaN in features")
    if np.isnan(y).any():
        raise ValueError("NaN in target")
        
def save_growth_dataset(X: np.ndarray, y: np.ndarray, feature_names: List[str], output_dir: str):
    """Save processed dataset."""
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'X.npy'), X)
    np.save(os.path.join(output_dir, 'y.npy'), y)
    with open(os.path.join(output_dir, 'features.json'), 'w') as f:
        json.dump(feature_names, f)
