"""
RSS-based news fetcher for Indian stocks.
Fetches news from multiple free sources without rate limits.
"""

import feedparser
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from urllib.parse import quote
import hashlib
import logging
import time

logger = logging.getLogger(__name__)

# Robustness configuration
RSS_TIMEOUT = 15  # seconds for RSS/web requests
RSS_MAX_RETRIES = 3
RSS_BACKOFF_FACTOR = 0.5
RSS_STATUS_FORCELIST = [429, 500, 502, 503, 504]


class RSSNewsFetcher:
    """Fetches news from RSS feeds for Indian stocks."""
    
    def __init__(self, cache_hours: int = 2):
        self.cache = {}
        self.cache_expiry = timedelta(hours=cache_hours)
        self.user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        self._session = None  # Lazy initialization
        self._session = None  # Lazy initialization
    
    def _get_session(self) -> requests.Session:
        """Get or create HTTP session with retry logic."""
        if self._session is None:
            self._session = requests.Session()
            retry_strategy = Retry(
                total=RSS_MAX_RETRIES,
                backoff_factor=RSS_BACKOFF_FACTOR,
                status_forcelist=RSS_STATUS_FORCELIST,
                allowed_methods=["GET", "POST"]
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            self._session.mount("http://", adapter)
            self._session.mount("https://", adapter)
            logger.debug(f"Created HTTP session with {RSS_MAX_RETRIES} retries")
        return self._session
        
    def _get_cache_key(self, symbol: str) -> str:
        """Generate cache key for symbol."""
        return f"news_{symbol}"
    
    def _is_cache_valid(self, symbol: str) -> bool:
        """Check if cached data is still valid."""
        cache_key = self._get_cache_key(symbol)
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            return datetime.now() - timestamp < self.cache_expiry
        return False
    
    def _get_from_cache(self, symbol: str) -> Optional[List[Dict]]:
        """Get data from cache if valid."""
        if self._is_cache_valid(symbol):
            cache_key = self._get_cache_key(symbol)
            return self.cache[cache_key][0]
        return None
    
    def _save_to_cache(self, symbol: str, data: List[Dict]):
        """Save data to cache."""
        cache_key = self._get_cache_key(symbol)
        self.cache[cache_key] = (data, datetime.now())
    
    def fetch_google_news_rss(self, query: str, max_articles: int = 10) -> List[Dict]:
        """
        Fetch news from Google News RSS feed.
        
        Args:
            query: Search query (e.g., "Reliance Industries stock")
            max_articles: Maximum number of articles to fetch
            
        Returns:
            List of article dictionaries
        """
        articles = []
        try:
            encoded_query = quote(query)
            rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"
            
            logger.debug(f"Fetching Google News RSS: {rss_url}")
            
            # feedparser handles its own requests, but we can add timeout via custom handler
            feed = feedparser.parse(rss_url, request_headers={'User-Agent': self.user_agent})
            
            for entry in feed.entries[:max_articles]:
                try:
                    # Parse published date
                    published_date = datetime(*entry.published_parsed[:6]) if hasattr(entry, 'published_parsed') else datetime.now()
                    
                    article = {
                        'title': entry.title if hasattr(entry, 'title') else 'No title',
                        'description': entry.summary if hasattr(entry, 'summary') else '',
                        'url': entry.link if hasattr(entry, 'link') else '',
                        'published_at': published_date.isoformat(),
                        'source': entry.source.title if hasattr(entry, 'source') and hasattr(entry.source, 'title') else 'Google News',
                        'content': entry.summary if hasattr(entry, 'summary') else ''
                    }
                    articles.append(article)
                except Exception as e:
                    logger.warning(f"Error parsing Google News entry: {e}")
                    continue
            
            logger.info(f"Fetched {len(articles)} articles from Google News for '{query}'")
                    
        except requests.exceptions.Timeout:
            logger.error(f"Timeout fetching Google News RSS for '{query}' (>{RSS_TIMEOUT}s)")
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error fetching Google News RSS: {e}")
        except Exception as e:
            logger.error(f"Error fetching Google News RSS: {e}")
        
        return articles
    
    def fetch_moneycontrol_news(self, company_name: str, max_articles: int = 5) -> List[Dict]:
        """
        Fetch news from MoneyControl (web scraping as they don't have public RSS).
        
        Args:
            company_name: Company name to search
            max_articles: Maximum articles to fetch
            
        Returns:
            List of article dictionaries
        """
        articles = []
        try:
            headers = {'User-Agent': self.user_agent}
            search_url = f"https://www.moneycontrol.com/news/tags/{company_name.lower().replace(' ', '-')}.html"
            
            logger.debug(f"Fetching MoneyControl news: {search_url}")
            
            session = self._get_session()
            response = session.get(search_url, headers=headers, timeout=RSS_TIMEOUT)
            
            if response.status_code != 200:
                logger.warning(f"MoneyControl returned status {response.status_code}")
                return articles
            
            soup = BeautifulSoup(response.content, 'html.parser')
            news_items = soup.find_all('li', class_='clearfix', limit=max_articles)
            
            for item in news_items:
                try:
                    title_tag = item.find('h2')
                    link_tag = title_tag.find('a') if title_tag else None
                    desc_tag = item.find('p')
                    time_tag = item.find('span', class_='article_schedule')
                    
                    if title_tag and link_tag:
                        article = {
                            'title': link_tag.get_text(strip=True),
                            'description': desc_tag.get_text(strip=True) if desc_tag else '',
                            'url': link_tag.get('href', ''),
                            'published_at': time_tag.get_text(strip=True) if time_tag else str(datetime.now()),
                            'source': 'MoneyControl',
                            'content': desc_tag.get_text(strip=True) if desc_tag else ''
                        }
                        articles.append(article)
                except Exception as e:
                    logger.warning(f"Error parsing MoneyControl article: {e}")
                    continue
            
            logger.info(f"Fetched {len(articles)} articles from MoneyControl for '{company_name}'")
                    
        except requests.exceptions.Timeout:
            logger.error(f"Timeout fetching MoneyControl news for '{company_name}' (>{RSS_TIMEOUT}s)")
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error fetching MoneyControl: {e.response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error fetching MoneyControl: {e}")
        except Exception as e:
            logger.error(f"Error fetching MoneyControl news: {e}")
        
        return articles
    
    def fetch_economic_times_rss(self, query: str, max_articles: int = 10) -> List[Dict]:
        """
        Fetch news from Economic Times RSS feeds.
        
        Args:
            query: Search query
            max_articles: Maximum articles to fetch
            
        Returns:
            List of article dictionaries
        """
        articles = []
        try:
            # Economic Times has multiple RSS feeds
            rss_urls = [
                "https://economictimes.indiatimes.com/rssfeedstopstories.cms",
                "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
                "https://economictimes.indiatimes.com/news/economy/rssfeeds/1045132894.cms"
            ]
            
            query_lower = query.lower()
            
            for idx, rss_url in enumerate(rss_urls):
                try:
                    # Small delay between feeds to avoid rate limiting
                    if idx > 0:
                        time.sleep(0.3)
                    
                    logger.debug(f"Fetching ET RSS feed: {rss_url}")
                    feed = feedparser.parse(rss_url, request_headers={'User-Agent': self.user_agent})
                    
                    for entry in feed.entries:
                        # Filter by query relevance
                        title_lower = entry.title.lower() if hasattr(entry, 'title') else ''
                        desc_lower = entry.summary.lower() if hasattr(entry, 'summary') else ''
                        
                        if query_lower in title_lower or query_lower in desc_lower:
                            published_date = datetime(*entry.published_parsed[:6]) if hasattr(entry, 'published_parsed') else datetime.now()
                            
                            article = {
                                'title': entry.title if hasattr(entry, 'title') else '',
                                'description': entry.summary if hasattr(entry, 'summary') else '',
                                'url': entry.link if hasattr(entry, 'link') else '',
                                'published_at': published_date.isoformat(),
                                'source': 'Economic Times',
                                'content': entry.summary if hasattr(entry, 'summary') else ''
                            }
                            articles.append(article)
                            
                            if len(articles) >= max_articles:
                                return articles
                                
                except Exception as e:
                    logger.warning(f"Error parsing ET RSS feed {rss_url}: {e}")
                    continue
            
            logger.info(f"Fetched {len(articles)} articles from Economic Times for '{query}'")
                    
        except requests.exceptions.Timeout:
            logger.error(f"Timeout fetching Economic Times RSS for '{query}' (>{RSS_TIMEOUT}s)")
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error fetching Economic Times RSS: {e}")
        except Exception as e:
            logger.error(f"Error fetching Economic Times RSS: {e}")
        
        return articles
    
    def fetch_company_news(self, symbol: str, company_name: str, max_articles: int = 20) -> List[Dict]:
        """
        Fetch news for a company from multiple sources.
        
        Args:
            symbol: Stock symbol (e.g., "RELIANCE")
            company_name: Full company name (e.g., "Reliance Industries")
            max_articles: Maximum total articles to fetch
            
        Returns:
            Deduplicated list of articles from all sources
        """
        # Check cache first
        cached_data = self._get_from_cache(symbol)
        if cached_data:
            return cached_data
        
        all_articles = []
        seen_urls = set()
        seen_titles = set()
        
        # Search queries
        queries = [
            f"{company_name} stock India",
            f"{symbol} NSE BSE",
            company_name
        ]
        
        # Fetch from multiple sources
        for idx, query in enumerate(queries):
            # Throttle requests to avoid overwhelming providers
            if idx > 0:
                time.sleep(0.5)  # 500ms between query batches
            
            try:
                # Google News (most reliable)
                google_articles = self.fetch_google_news_rss(query, max_articles=10)
                all_articles.extend(google_articles)
                
                # Small delay between different sources
                time.sleep(0.3)
                
                # Economic Times
                et_articles = self.fetch_economic_times_rss(query, max_articles=5)
                all_articles.extend(et_articles)
            except Exception as e:
                logger.warning(f"Error fetching news for query '{query}': {e}")
                continue
        
        # Fetch from MoneyControl
        try:
            time.sleep(0.3)  # Delay before MoneyControl
            mc_articles = self.fetch_moneycontrol_news(company_name, max_articles=5)
            all_articles.extend(mc_articles)
        except Exception as e:
            logger.warning(f"Error fetching MoneyControl news: {e}")
        
        # Deduplicate by URL and title similarity
        deduplicated_articles = []
        for article in all_articles:
            url = article.get('url', '')
            title = article.get('title', '').lower().strip()
            
            # Create title hash for fuzzy deduplication
            title_hash = hashlib.md5(title.encode()).hexdigest()[:8]
            
            if url not in seen_urls and title_hash not in seen_titles:
                seen_urls.add(url)
                seen_titles.add(title_hash)
                deduplicated_articles.append(article)
                
                if len(deduplicated_articles) >= max_articles:
                    break
        
        # Sort by published date (most recent first)
        try:
            deduplicated_articles.sort(
                key=lambda x: datetime.fromisoformat(x['published_at']) if isinstance(x['published_at'], str) else x['published_at'],
                reverse=True
            )
        except Exception as e:
            logger.warning(f"Error sorting articles by date: {e}")
        
        # Cache the results
        self._save_to_cache(symbol, deduplicated_articles)
        
        logger.info(f"Fetched {len(deduplicated_articles)} articles for {symbol}")
        return deduplicated_articles


# Singleton instance
rss_fetcher = RSSNewsFetcher()
