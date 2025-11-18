import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime, timedelta
from transformers import PreTrainedTokenizer

def load_news_articles(
    data_dir: str, 
    tickers: Optional[List[str]] = None, 
    start_date: Optional[str] = None, 
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Load news articles from raw data directory.
    Structure: data_dir/{YYYY-MM-DD}/*.json
    """
    data_path = Path(data_dir)
    all_articles = []
    
    # Iterate over date directories
    for date_dir in data_path.iterdir():
        if not date_dir.is_dir():
            continue
            
        date_str = date_dir.name
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            continue
            
        if start_date and date_str < start_date:
            continue
        if end_date and date_str > end_date:
            continue
            
        # Load JSON files
        for file_path in date_dir.glob("*.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                # Handle list of articles or single article
                articles = data if isinstance(data, list) else [data]
                
                for article in articles:
                    ticker = article.get('ticker') or article.get('symbol')
                    if tickers and ticker not in tickers:
                        continue
                        
                    # Construct text
                    title = article.get('title', '')
                    desc = article.get('description', '')
                    content = article.get('content', '')
                    text = f"{title}. {desc}"
                    if content:
                        text += f" {content[:200]}" # Truncate content to avoid noise
                        
                    all_articles.append({
                        'text': text,
                        'published_at': article.get('published_at', date_str),
                        'ticker': ticker,
                        'source': article.get('source', {}).get('name', 'Unknown'),
                        'url': article.get('url')
                    })
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
                
    return pd.DataFrame(all_articles)

def load_social_sentiment(
    data_dir: str, 
    tickers: Optional[List[str]] = None, 
    start_date: Optional[str] = None, 
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Load social sentiment data (StockTwits).
    Structure: data_dir/{YYYY-MM-DD}/*.json
    """
    # Similar logic to load_news_articles but specific to social format
    # Assuming StockTwits format
    data_path = Path(data_dir)
    all_msgs = []
    
    for date_dir in data_path.iterdir():
        if not date_dir.is_dir():
            continue
        
        date_str = date_dir.name
        if start_date and date_str < start_date: continue
        if end_date and date_str > end_date: continue
        
        for file_path in date_dir.glob("*.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                msgs = data.get('messages', []) if isinstance(data, dict) else data
                
                for msg in msgs:
                    ticker = msg.get('symbol') # Adjust key based on actual schema
                    if tickers and ticker not in tickers: continue
                    
                    entities = msg.get('entities', {})
                    sentiment = entities.get('sentiment', {}).get('basic')
                    
                    all_msgs.append({
                        'text': msg.get('body', ''),
                        'published_at': msg.get('created_at', date_str),
                        'ticker': ticker,
                        'sentiment_label': sentiment # Bullish/Bearish/None
                    })
            except Exception:
                continue
                
    return pd.DataFrame(all_msgs)

def load_prices_for_labeling(
    data_dir: str,
    tickers: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load price data for sentiment labeling.
    Expected CSV with columns: Date, Ticker, Close
    """
    data_path = Path(data_dir)
    all_prices = []
    
    # Try loading from individual ticker files
    if tickers:
        for ticker in tickers:
            ticker_file = data_path / f"{ticker}.csv"
            if ticker_file.exists():
                df = pd.read_csv(ticker_file)
                if 'Ticker' not in df.columns:
                    df['Ticker'] = ticker
                all_prices.append(df)
    
    # Try consolidated file
    consolidated = data_path / "prices.csv"
    if consolidated.exists():
        df = pd.read_csv(consolidated)
        if tickers:
            df = df[df['Ticker'].isin(tickers)]
        all_prices.append(df)
    
    if not all_prices:
        return pd.DataFrame()
    
    result = pd.concat(all_prices, ignore_index=True)
    return result


def create_labeled_dataset(
    news_df: pd.DataFrame, 
    price_df: Optional[pd.DataFrame] = None, 
    labeling_strategy: str = 'price_change'
) -> pd.DataFrame:
    """
    Generate 3-class sentiment labels.
    0: Positive, 1: Neutral, 2: Negative
    """
    df = news_df.copy()
    
    if labeling_strategy == 'price_change':
        if price_df is None:
            raise ValueError("price_df required for price_change strategy")
            
        # Ensure datetime
        df['published_at'] = pd.to_datetime(df['published_at']).dt.normalize()
        price_df['Date'] = pd.to_datetime(price_df['Date']).dt.normalize()
        
        # Merge with price
        # We need forward returns. 
        # For each article date, get price at T+1 and T+3
        # This is computationally expensive with merge_asof or apply
        # Simplified: Calculate daily returns in price_df, then merge
        
        # Calculate forward returns in price_df
        price_df = price_df.sort_values('Date')
        price_df['fwd_ret_1d'] = price_df.groupby('Ticker')['Close'].pct_change(periods=-1) * -1 # -1 period is future? No, pct_change(1) is (t - t-1)/t-1. pct_change(-1) is (t - t+1)/t+1. We want (t+1 - t)/t
        # Actually: (Price(t+k) - Price(t)) / Price(t)
        # shift(-k) gets Price(t+k) at row t
        
        indexer = price_df.groupby('Ticker')['Close']
        price_df['price_t'] = indexer.shift(0)
        price_df['price_t1'] = indexer.shift(-1)
        price_df['price_t3'] = indexer.shift(-3)
        
        price_df['ret_1d'] = (price_df['price_t1'] - price_df['price_t']) / price_df['price_t']
        price_df['ret_3d'] = (price_df['price_t3'] - price_df['price_t']) / price_df['price_t']
        
        # Merge news with returns
        # Rename Date to published_at for merge
        price_subset = price_df[['Date', 'Ticker', 'ret_1d', 'ret_3d']].rename(
            columns={'Date': 'published_at', 'Ticker': 'ticker'}
        )
        
        df = pd.merge(df, price_subset, on=['published_at', 'ticker'], how='inner')
        
        # Labeling logic
        # > 2% -> Positive (0)
        # < -2% -> Negative (2)
        # Else -> Neutral (1)
        
        conditions = [
            (df['ret_3d'] > 0.02),
            (df['ret_3d'] < -0.02)
        ]
        choices = [0, 2] # Positive, Negative
        df['label'] = np.select(conditions, choices, default=1) # Neutral
        
    elif labeling_strategy == 'manual':
        # Assume 'sentiment_label' column exists (e.g. from StockTwits)
        # Map Bullish->0, Bearish->2, None->1
        label_map = {'Bullish': 0, 'Bearish': 2, 'None': 1, 'Neutral': 1}
        df['label'] = df['sentiment_label'].map(label_map).fillna(1).astype(int)
        
    return df[['text', 'label', 'ticker', 'published_at']]

def split_sentiment_data(
    df: pd.DataFrame, 
    train_ratio: float = 0.6, 
    val_ratio: float = 0.2,
    test_ratio: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Temporal split of data.
    """
    df = df.sort_values('published_at')
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    return train_df, val_df, test_df

class SentimentDataset(Dataset):
    """PyTorch Dataset for Sentiment Analysis."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer: PreTrainedTokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return (
            encoding['input_ids'].flatten(),
            encoding['attention_mask'].flatten(),
            torch.tensor(label, dtype=torch.long)
        )

def validate_sentiment_labels(df: pd.DataFrame):
    """Validate label distribution."""
    if 'label' not in df.columns:
        raise ValueError("Missing 'label' column")
        
    counts = df['label'].value_counts(normalize=True)
    print("Class Distribution:")
    print(counts)
    
    if any(counts < 0.1):
        print("Warning: Severe class imbalance detected (<10% for a class)")
