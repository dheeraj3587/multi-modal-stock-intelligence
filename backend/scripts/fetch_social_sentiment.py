#!/usr/bin/env python3
"""
Social sentiment data fetching script using StockTwits API.

Fetches social sentiment messages with rate limiting, pagination,
and sentiment distribution analysis.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.utils.logger import get_logger
from backend.utils.config import Config
from backend.utils.api_helpers import (
    RateLimiter,
    make_api_request,
    validate_api_response,
    handle_api_error
)
from backend.utils.file_operations import create_timestamped_directory, save_json


logger = get_logger(__name__)
config = Config()


# Rate limiter for StockTwits (200 requests per hour for authenticated users)
stocktwits_rate_limiter = RateLimiter(max_requests=200, time_window=3600)  # 1 hour


def fetch_stocktwits_messages(
    ticker: str,
    limit: int = 30,
    since_id: Optional[int] = None,
    max_id: Optional[int] = None
) -> Dict:
    """
    Fetch messages from StockTwits API for a specific ticker.
    
    Args:
        ticker: Stock ticker symbol (StockTwits format, without exchange suffix).
        limit: Number of messages to fetch (max 30 per request).
        since_id: Return messages with ID greater than this (for incremental updates).
        max_id: Return messages with ID less than or equal to this (for pagination).
        
    Returns:
        Dictionary with 'messages' list and 'cursor' for pagination.
    """
    logger.info(f"Fetching StockTwits messages for ${ticker}")
    
    try:
        # Wait for rate limiter token
        stocktwits_rate_limiter.wait_for_token()
        
        # Get access token from config
        access_token = config.stocktwits_access_token
        
        # StockTwits API endpoint
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
        
        # Build parameters
        params = {
            'access_token': access_token,
            'limit': min(limit, 30)  # API max is 30
        }
        
        if since_id:
            params['since'] = since_id
        
        if max_id:
            params['max'] = max_id
        
        # Make API request
        response = make_api_request(
            url,
            params=params,
            timeout=30,
            rate_limiter=None  # Already handled rate limiting above
        )
        
        # Validate and parse response
        data = validate_api_response(response, expected_keys=['messages'], response_format='json')
        
        messages = data.get('messages', [])
        cursor = data.get('cursor', {})
        
        logger.info(f"Fetched {len(messages)} messages for ${ticker}")
        
        return {
            'messages': messages,
            'cursor': cursor
        }
        
    except Exception as e:
        logger.error(f"Failed to fetch StockTwits messages for ${ticker}: {str(e)}")
        raise


def fetch_with_pagination(
    ticker: str,
    total_limit: int = 30,
    since_id: Optional[int] = None
) -> List[Dict]:
    """
    Fetch messages with pagination support to get more than 30 messages.
    
    Args:
        ticker: Stock ticker symbol.
        total_limit: Total number of messages to fetch (will make multiple API calls if > 30).
        since_id: Return messages with ID greater than this.
        
    Returns:
        List of message dictionaries.
    """
    all_messages = []
    max_id = None
    
    while len(all_messages) < total_limit:
        # Calculate how many messages to fetch in this request
        remaining = total_limit - len(all_messages)
        limit = min(remaining, 30)
        
        # Fetch messages
        result = fetch_stocktwits_messages(ticker, limit=limit, since_id=since_id, max_id=max_id)
        messages = result['messages']
        cursor = result['cursor']
        
        if not messages:
            logger.info("No more messages available")
            break
        
        all_messages.extend(messages)
        
        # Get max_id for next page from cursor
        max_id = cursor.get('max')
        
        if not max_id or len(messages) < limit:
            # No more pages available
            break
    
    logger.info(f"Fetched total of {len(all_messages)} messages with pagination")
    return all_messages


def normalize_messages(messages: List[Dict], ticker: str) -> List[Dict]:
    """
    Normalize and extract relevant fields from StockTwits messages.
    
    Args:
        messages: Raw messages from StockTwits API.
        ticker: Stock ticker symbol.
        
    Returns:
        List of normalized message dictionaries.
    """
    normalized = []
    
    for msg in messages:
        normalized_msg = {
            'id': msg.get('id'),
            'body': msg.get('body'),
            'created_at': msg.get('created_at'),
            'ticker': ticker,
            
            # User info
            'user_id': msg.get('user', {}).get('id'),
            'user_username': msg.get('user', {}).get('username'),
            'user_name': msg.get('user', {}).get('name'),
            'user_followers': msg.get('user', {}).get('followers'),
            'user_following': msg.get('user', {}).get('following'),
            'user_official': msg.get('user', {}).get('official'),
            
            # Sentiment (user-labeled)
            'sentiment': msg.get('entities', {}).get('sentiment', {}).get('basic') if msg.get('entities') else None,
            
            # Symbols mentioned
            'symbols': [s.get('symbol') for s in msg.get('symbols', [])],
            
            # Links
            'links': [link.get('url') for link in msg.get('links', [])],
            
            # Metadata
            'source': msg.get('source', {}).get('title'),
            'likes': msg.get('likes', {}).get('total', 0),
            'reshares': msg.get('reshares', {}).get('reshared_count', 0),
            'conversation_replies': msg.get('conversation', {}).get('replies', 0) if msg.get('conversation') else 0,
            
            # Fetch metadata
            'fetch_date': datetime.now().isoformat(),
        }
        
        normalized.append(normalized_msg)
    
    return normalized


def calculate_sentiment_distribution(messages: List[Dict]) -> Dict:
    """
    Calculate sentiment distribution statistics.
    
    Args:
        messages: List of normalized message dictionaries.
        
    Returns:
        Dictionary with sentiment counts and percentages.
    """
    sentiment_counts = {
        'bullish': 0,
        'bearish': 0,
        'neutral': 0,
        'unknown': 0
    }
    
    for msg in messages:
        sentiment = msg.get('sentiment')
        
        if sentiment == 'Bullish':
            sentiment_counts['bullish'] += 1
        elif sentiment == 'Bearish':
            sentiment_counts['bearish'] += 1
        elif sentiment is None:
            sentiment_counts['unknown'] += 1
        else:
            sentiment_counts['neutral'] += 1
    
    total = len(messages)
    
    distribution = {
        'total_messages': total,
        'bullish': sentiment_counts['bullish'],
        'bearish': sentiment_counts['bearish'],
        'neutral': sentiment_counts['neutral'],
        'unknown': sentiment_counts['unknown'],
        'bullish_pct': round(sentiment_counts['bullish'] / total * 100, 1) if total > 0 else 0,
        'bearish_pct': round(sentiment_counts['bearish'] / total * 100, 1) if total > 0 else 0,
        'neutral_pct': round(sentiment_counts['neutral'] / total * 100, 1) if total > 0 else 0,
        'unknown_pct': round(sentiment_counts['unknown'] / total * 100, 1) if total > 0 else 0,
    }
    
    return distribution


def save_social_sentiment(messages: List[Dict], ticker: str, output_dir: Path) -> Path:
    """
    Save social sentiment messages to date-partitioned directory.
    
    Args:
        messages: List of message dictionaries.
        ticker: Stock ticker symbol.
        output_dir: Base output directory.
        
    Returns:
        Path to saved file.
    """
    # Create timestamped directory
    date_dir = create_timestamped_directory(output_dir)
    
    # Create filename with ticker
    filename = f"{ticker}_stocktwits.json"
    filepath = date_dir / filename
    
    # Save to JSON
    save_json(messages, filepath)
    logger.info(f"Saved {len(messages)} messages to {filepath}")
    
    return filepath


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Fetch social sentiment data from StockTwits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch latest 30 messages
  python fetch_social_sentiment.py --ticker RELIANCE
  
  # Fetch more messages with pagination
  python fetch_social_sentiment.py --ticker TCS --limit 100
  
  # Incremental update (fetch only new messages)
  python fetch_social_sentiment.py --ticker INFY --since-id 12345678
        """
    )
    
    parser.add_argument('--ticker', type=str, required=True,
                       help='Stock ticker symbol (StockTwits format, e.g., RELIANCE)')
    parser.add_argument('--limit', type=int, default=30,
                       help='Number of messages to fetch (default: 30, max depends on pagination)')
    parser.add_argument('--since-id', type=int,
                       help='Fetch only messages with ID greater than this (for incremental updates)')
    parser.add_argument('--output-dir', type=Path, default='data/raw/social',
                       help='Output directory (default: data/raw/social)')
    
    args = parser.parse_args()
    
    logger.info("Starting social sentiment data fetch")
    logger.info(f"Ticker: ${args.ticker}")
    logger.info(f"Output directory: {args.output_dir}")
    
    try:
        # Fetch messages with pagination if needed
        if args.limit > 30:
            messages = fetch_with_pagination(args.ticker, total_limit=args.limit, since_id=args.since_id)
        else:
            result = fetch_stocktwits_messages(args.ticker, limit=args.limit, since_id=args.since_id)
            messages = result['messages']
        
        if not messages:
            logger.warning("No messages found")
            return
        
        # Normalize messages
        normalized_messages = normalize_messages(messages, args.ticker)
        
        # Calculate sentiment distribution
        sentiment_dist = calculate_sentiment_distribution(normalized_messages)
        
        # Log sentiment distribution
        logger.info(f"Sentiment distribution for ${args.ticker}:")
        logger.info(f"  Total messages: {sentiment_dist['total_messages']}")
        logger.info(f"  Bullish: {sentiment_dist['bullish']} ({sentiment_dist['bullish_pct']}%)")
        logger.info(f"  Bearish: {sentiment_dist['bearish']} ({sentiment_dist['bearish_pct']}%)")
        logger.info(f"  Neutral: {sentiment_dist['neutral']} ({sentiment_dist['neutral_pct']}%)")
        logger.info(f"  Unknown: {sentiment_dist['unknown']} ({sentiment_dist['unknown_pct']}%)")
        
        # Add sentiment distribution to data
        data_to_save = {
            'ticker': args.ticker,
            'fetch_date': datetime.now().isoformat(),
            'sentiment_distribution': sentiment_dist,
            'messages': normalized_messages
        }
        
        # Save messages
        save_social_sentiment([data_to_save], args.ticker, args.output_dir)
        
        logger.info("Social sentiment fetch completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to fetch social sentiment: {str(e)}")
        handle_api_error(e, context=f"ticker=${args.ticker}")
        sys.exit(1)


if __name__ == '__main__':
    main()
