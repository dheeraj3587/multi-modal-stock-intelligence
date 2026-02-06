import pytest
import pandas as pd
import numpy as np
import os
import json
from backend.utils.sentiment_data import load_news_articles, create_labeled_dataset, split_sentiment_data

@pytest.fixture
def mock_news_dir(tmp_path):
    date_dir = tmp_path / "2023-01-01"
    date_dir.mkdir()
    
    data = [{"title": "Test", "published_at": "2023-01-01", "ticker": "AAPL"}]
    with open(date_dir / "news.json", "w") as f:
        json.dump(data, f)
        
    return str(tmp_path)

def test_load_news_articles(mock_news_dir):
    df = load_news_articles(mock_news_dir)
    assert len(df) == 1
    assert df.iloc[0]['ticker'] == 'AAPL'

def test_create_labeled_dataset():
    news_df = pd.DataFrame({
        'text': ['News 1', 'News 2'],
        'published_at': ['2023-01-01', '2023-01-02'],
        'ticker': ['AAPL', 'AAPL']
    })
    
    price_df = pd.DataFrame({
        'Date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
        'Ticker': ['AAPL', 'AAPL', 'AAPL', 'AAPL', 'AAPL'],
        'Close': [100, 101, 105, 100, 90] 
    })
    # 2023-01-01: T+3 is 2023-01-04 (100). Ret = 0. Label 1 (Neutral).
    # 2023-01-02: T+3 is 2023-01-05 (90). Ret = (90-101)/101 = -0.1. Label 2 (Negative).
    
    labeled = create_labeled_dataset(news_df, price_df)
    assert len(labeled) == 2
    assert labeled.iloc[0]['label'] == 1
    assert labeled.iloc[1]['label'] == 2

def test_split_sentiment_data():
    df = pd.DataFrame({'published_at': pd.date_range('2023-01-01', periods=100)})
    train, val, test = split_sentiment_data(df)
    assert len(train) == 60
    assert len(val) == 20
    assert len(test) == 20
