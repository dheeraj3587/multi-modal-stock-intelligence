import pytest
import pandas as pd
import numpy as np
from backend.utils.growth_data import split_by_stocks, temporal_subsplit

def test_split_by_stocks():
    # Create sample data with 200 rows: 100 for ticker A (Tech), 100 for ticker B (Finance)
    # Actually we need more tickers to test stratification properly.
    # Let's create 20 tickers: 10 Tech, 10 Finance.
    
    tickers = [f'TICKER_{i}' for i in range(20)]
    sectors = ['Tech'] * 10 + ['Finance'] * 10
    
    data = []
    dates = pd.date_range('2020-01-01', periods=10, freq='D')
    
    for ticker, sector in zip(tickers, sectors):
        for date in dates:
            data.append({
                'ticker': ticker,
                'sector': sector,
                'date': date,
                'value': np.random.random()
            })
            
    df = pd.DataFrame(data)
    
    # Split 80/20
    train_df, test_df = split_by_stocks(df, train_ratio=0.8, stratify_by='sector')
    
    # Check disjoint tickers
    train_tickers = set(train_df['ticker'])
    test_tickers = set(test_df['ticker'])
    assert len(train_tickers & test_tickers) == 0
    
    # Check ratios (approximate due to small sample)
    assert len(train_tickers) == 16
    assert len(test_tickers) == 4
    
    # Check stratification
    train_sectors = train_df.groupby('ticker')['sector'].first().value_counts()
    test_sectors = test_df.groupby('ticker')['sector'].first().value_counts()
    
    # Should be roughly balanced
    assert train_sectors['Tech'] == 8
    assert train_sectors['Finance'] == 8
    assert test_sectors['Tech'] == 2
    assert test_sectors['Finance'] == 2

def test_temporal_subsplit():
    # Create sample data for a single ticker over 100 days
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    df = pd.DataFrame({
        'ticker': ['A'] * 100,
        'date': dates,
        'value': range(100)
    })
    
    # Split last 20% for validation
    train_sub, val_df = temporal_subsplit(df, val_ratio=0.2)
    
    assert len(train_sub) == 80
    assert len(val_df) == 20
    
    # Check temporal order
    assert train_sub['date'].max() < val_df['date'].min()
