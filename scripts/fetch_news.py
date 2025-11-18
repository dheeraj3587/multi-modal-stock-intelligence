#!/usr/bin/env python3
"""
News data fetching script using NewsAPI.

Fetches financial news articles with rate limiting, deduplication,
and date-partitioned storage.
"""

import argparse
import sys
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Set
from newsapi import NewsApiClient
from newsapi.newsapi_exception import NewsAPIException

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.utils.logger import get_logger
from backend.utils.config import Config
from backend.utils.api_helpers import RateLimiter, handle_api_error
from backend.utils.file_operations import create_timestamped_directory, save_json, load_existing_data


logger = get_logger(__name__)
config = Config()


# Rate limiter for NewsAPI (100 requests per day for free tier)
news_rate_limiter = RateLimiter(max_requests=100, time_window=86400)  # 24 hours


def fetch_news_articles(
    query: str,
    days: int = 7,
    language: str = 'en',
    country: str = None,
    sources: str = None,
    sort_by: str = 'publishedAt'
) -> List[Dict]:
    """
    Fetch news articles using NewsAPI.
    
    Args:
        query: Search query (e.g., company name, ticker, keywords).
        days: Number of days to look back.
        language: Article language (default 'en').
        country: Country code (e.g., 'in' for India). Cannot be used with sources.
        sources: Comma-separated list of news sources.
        sort_by: Sort order ('relevancy', 'popularity', 'publishedAt').
        
    Returns:
        List of article dictionaries.
    """
    logger.info(f"Fetching news articles for query: '{query}' (last {days} days)")
    
    try:
        # Wait for rate limiter token
        news_rate_limiter.wait_for_token()
        
        # Get API key from config
        api_key = config.newsapi_key
        
        # Initialize NewsAPI client
        newsapi = NewsApiClient(api_key=api_key)
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Format dates for API
        from_param = start_date.strftime("%Y-%m-%d")
        to_param = end_date.strftime("%Y-%m-%d")
        
        # Fetch articles using get_everything endpoint
        # Note: 'country' parameter is only for top-headlines, not get_everything
        response = newsapi.get_everything(
            q=query,
            from_param=from_param,
            to=to_param,
            language=language,
            sources=sources,
            sort_by=sort_by,
            page_size=100  # Max per request
        )
        
        articles = response.get('articles', [])
        total_results = response.get('totalResults', 0)
        
        logger.info(f"Found {len(articles)} articles (total available: {total_results})")
        
        # Extract and normalize article data
        normalized_articles = []
        for article in articles:
            normalized_articles.append({
                'title': article.get('title'),
                'description': article.get('description'),
                'content': article.get('content'),
                'url': article.get('url'),
                'published_at': article.get('publishedAt'),
                'source_name': article.get('source', {}).get('name'),
                'source_id': article.get('source', {}).get('id'),
                'author': article.get('author'),
                'url_to_image': article.get('urlToImage'),
                'fetch_date': datetime.now().isoformat(),
                'query': query,
            })
        
        return normalized_articles
        
    except NewsAPIException as e:
        error_msg = f"NewsAPI error: {str(e)}"
        logger.error(error_msg)
        raise
        
    except Exception as e:
        logger.error(f"Failed to fetch news articles: {str(e)}")
        raise


def deduplicate_articles(articles: List[Dict], existing_hashes: Set[str] = None) -> List[Dict]:
    """
    Remove duplicate articles based on URL and title hash.
    
    Args:
        articles: List of article dictionaries.
        existing_hashes: Set of existing article hashes to check against.
        
    Returns:
        List of deduplicated articles.
    """
    if existing_hashes is None:
        existing_hashes = set()
    
    unique_articles = []
    seen_hashes = existing_hashes.copy()
    
    for article in articles:
        # Create hash from URL or title
        url = article.get('url', '')
        title = article.get('title', '')
        
        if url:
            article_hash = hashlib.md5(url.encode()).hexdigest()
        else:
            article_hash = hashlib.md5(title.encode()).hexdigest()
        
        if article_hash not in seen_hashes:
            unique_articles.append(article)
            seen_hashes.add(article_hash)
        else:
            logger.debug(f"Skipping duplicate article: {title}")
    
    removed_count = len(articles) - len(unique_articles)
    if removed_count > 0:
        logger.info(f"Removed {removed_count} duplicate articles")
    
    return unique_articles


def load_existing_article_hashes(output_dir: Path, filename_pattern: str) -> Set[str]:
    """
    Load hashes of existing articles to avoid duplicates.
    
    Args:
        output_dir: Directory containing existing article files.
        filename_pattern: Pattern to match article files.
        
    Returns:
        Set of article URL hashes.
    """
    existing_hashes = set()
    
    try:
        # Find all existing article files
        if output_dir.exists():
            for file_path in output_dir.rglob(filename_pattern):
                try:
                    with open(file_path, 'r') as f:
                        import json
                        articles = json.load(f)
                        
                        if isinstance(articles, list):
                            for article in articles:
                                url = article.get('url', '')
                                title = article.get('title', '')
                                
                                if url:
                                    article_hash = hashlib.md5(url.encode()).hexdigest()
                                else:
                                    article_hash = hashlib.md5(title.encode()).hexdigest()
                                
                                existing_hashes.add(article_hash)
                except Exception as e:
                    logger.warning(f"Could not load hashes from {file_path}: {str(e)}")
        
        logger.info(f"Loaded {len(existing_hashes)} existing article hashes")
        
    except Exception as e:
        logger.warning(f"Error loading existing article hashes: {str(e)}")
    
    return existing_hashes


def save_news_articles(articles: List[Dict], filename: str, output_dir: Path) -> Path:
    """
    Save news articles to date-partitioned directory.
    
    Args:
        articles: List of article dictionaries.
        filename: Output filename.
        output_dir: Base output directory.
        
    Returns:
        Path to saved file.
    """
    # Create timestamped directory
    date_dir = create_timestamped_directory(output_dir)
    
    # Create filepath
    filepath = date_dir / filename
    
    # Save to JSON
    save_json(articles, filepath)
    logger.info(f"Saved {len(articles)} articles to {filepath}")
    
    return filepath


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Fetch financial news articles using NewsAPI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch news for a company
  python fetch_news.py --ticker RELIANCE --days 30
  
  # Custom search query
  python fetch_news.py --query "Reliance Industries earnings" --days 7
  
  # Filter by specific news sources
  python fetch_news.py --ticker TCS --sources "economic-times,business-standard" --days 14
  
  # Fetch with country filter (for top headlines)
  python fetch_news.py --query "stock market" --country in --days 7
        """
    )
    
    parser.add_argument('--ticker', type=str, help='Stock ticker or company name')
    parser.add_argument('--query', type=str, help='Custom search query')
    parser.add_argument('--days', type=int, default=7,
                       help='Number of days to look back (default: 7)')
    parser.add_argument('--language', type=str, default='en',
                       help='Article language (default: en)')
    parser.add_argument('--country', type=str,
                       help='Country code (e.g., "in" for India). Cannot be used with --sources.')
    parser.add_argument('--sources', type=str,
                       help='Comma-separated list of news sources')
    parser.add_argument('--sort-by', type=str, default='publishedAt',
                       choices=['relevancy', 'popularity', 'publishedAt'],
                       help='Sort order (default: publishedAt)')
    parser.add_argument('--output-dir', type=Path, default='data/raw/news',
                       help='Output directory (default: data/raw/news)')
    parser.add_argument('--no-dedupe', action='store_true',
                       help='Skip deduplication against existing articles')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.ticker and not args.query:
        parser.error("Either --ticker or --query must be specified")
    
    if args.country and args.sources:
        parser.error("Cannot use --country and --sources together")
    
    # Determine query
    query = args.query if args.query else args.ticker
    
    logger.info("Starting news data fetch")
    logger.info(f"Query: {query}")
    logger.info(f"Output directory: {args.output_dir}")
    
    try:
        # Fetch articles
        articles = fetch_news_articles(
            query=query,
            days=args.days,
            language=args.language,
            country=args.country,
            sources=args.sources,
            sort_by=args.sort_by
        )
        
        if not articles:
            logger.warning("No articles found")
            return
        
        # Deduplicate against existing articles
        if not args.no_dedupe:
            # Create filename pattern for existing articles
            ticker_safe = (args.ticker or query).replace(' ', '_').replace('.', '_')
            filename_pattern = f"*{ticker_safe}*_news.json"
            
            existing_hashes = load_existing_article_hashes(args.output_dir, filename_pattern)
            articles = deduplicate_articles(articles, existing_hashes)
        
        if not articles:
            logger.info("All articles were duplicates. No new articles to save.")
            return
        
        # Save articles
        ticker_safe = (args.ticker or query).replace(' ', '_').replace('.', '_')
        filename = f"{ticker_safe}_news.json"
        save_news_articles(articles, filename, args.output_dir)
        
        # Log summary
        logger.info(f"News fetch completed successfully. Total new articles: {len(articles)}")
        
        # Log source distribution
        source_counts = {}
        for article in articles:
            source = article.get('source_name', 'Unknown')
            source_counts[source] = source_counts.get(source, 0) + 1
        
        logger.info("Articles by source:")
        for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {source}: {count}")
        
    except Exception as e:
        logger.error(f"Failed to fetch news: {str(e)}")
        handle_api_error(e, context=f"query={query}")
        sys.exit(1)


if __name__ == '__main__':
    main()
