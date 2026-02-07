"""
Redis Cache Service for Sentiment Data
Provides persistent storage with TTL and last_updated tracking.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Optional, List
import redis

logger = logging.getLogger(__name__)


class RedisCacheService:
    """Redis-based cache for sentiment data with last_updated tracking."""
    
    def __init__(self):
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.default_ttl = int(os.getenv("SENTIMENT_CACHE_TTL", "3600"))  # 1 hour default
        self._client: Optional[redis.Redis] = None
        self._connected = False
        
    def _get_client(self) -> Optional[redis.Redis]:
        """Lazy connection to Redis with retry logic."""
        if self._client is not None:
            return self._client
            
        try:
            self._client = redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            # Test connection
            self._client.ping()
            self._connected = True
            logger.info(f"Connected to Redis at {self.redis_url}")
            return self._client
        except redis.ConnectionError as e:
            logger.warning(f"Redis connection failed: {e}. Using in-memory fallback.")
            self._connected = False
            return None
        except Exception as e:
            logger.error(f"Unexpected Redis error: {e}")
            self._connected = False
            return None
    
    def is_connected(self) -> bool:
        """Check if Redis is available."""
        client = self._get_client()
        return client is not None and self._connected
    
    def set_sentiment(
        self, 
        symbol: str, 
        data: Dict, 
        ttl: Optional[int] = None
    ) -> bool:
        """
        Store sentiment data with last_updated timestamp.
        
        Args:
            symbol: Stock symbol (e.g., 'RELIANCE')
            data: Sentiment data dict
            ttl: Time-to-live in seconds (None = use default)
            
        Returns:
            True if stored successfully
        """
        client = self._get_client()
        if not client:
            return False
            
        try:
            key = f"sentiment:{symbol}"
            
            # Add last_updated timestamp
            data_with_timestamp = {
                **data,
                "last_updated": datetime.now().isoformat(),
                "symbol": symbol
            }
            
            # Serialize to JSON
            value = json.dumps(data_with_timestamp)
            
            # Store with TTL
            client.setex(key, ttl or self.default_ttl, value)
            
            logger.debug(f"Cached sentiment for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache sentiment for {symbol}: {e}")
            return False
    
    def get_sentiment(self, symbol: str) -> Optional[Dict]:
        """
        Get cached sentiment data for a symbol.
        
        Returns:
            Sentiment dict with last_updated, or None if not found/expired
        """
        client = self._get_client()
        if not client:
            return None
            
        try:
            key = f"sentiment:{symbol}"
            value = client.get(key)
            
            if value:
                data = json.loads(value)
                return data
            return None
            
        except Exception as e:
            logger.error(f"Failed to get sentiment for {symbol}: {e}")
            return None
    
    def get_all_sentiments(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Bulk get sentiment data for multiple symbols.
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dict mapping symbol to sentiment data
        """
        client = self._get_client()
        if not client:
            return {}
            
        try:
            # Use pipeline for efficiency
            pipe = client.pipeline()
            keys = [f"sentiment:{s}" for s in symbols]
            
            for key in keys:
                pipe.get(key)
                
            results = pipe.execute()
            
            sentiments = {}
            for symbol, value in zip(symbols, results):
                if value:
                    try:
                        sentiments[symbol] = json.loads(value)
                    except json.JSONDecodeError:
                        pass
                        
            return sentiments
            
        except Exception as e:
            logger.error(f"Failed to bulk get sentiments: {e}")
            return {}
    
    def set_batch_sentiments(
        self, 
        data_dict: Dict[str, Dict],
        ttl: Optional[int] = None
    ) -> int:
        """
        Bulk store sentiment data for multiple symbols.
        
        Args:
            data_dict: Dict mapping symbol to sentiment data
            ttl: Time-to-live in seconds
            
        Returns:
            Number of successfully stored items
        """
        client = self._get_client()
        if not client:
            return 0
            
        try:
            pipe = client.pipeline()
            count = 0
            now = datetime.now().isoformat()
            
            for symbol, data in data_dict.items():
                key = f"sentiment:{symbol}"
                
                # Add timestamp
                data_with_timestamp = {
                    **data,
                    "last_updated": now,
                    "symbol": symbol
                }
                
                value = json.dumps(data_with_timestamp)
                pipe.setex(key, ttl or self.default_ttl, value)
                count += 1
                
            pipe.execute()
            logger.info(f"Bulk cached {count} sentiments")
            return count
            
        except Exception as e:
            logger.error(f"Failed to bulk cache sentiments: {e}")
            return 0
    
    def delete_sentiment(self, symbol: str) -> bool:
        """Delete cached sentiment for a symbol."""
        client = self._get_client()
        if not client:
            return False
            
        try:
            key = f"sentiment:{symbol}"
            client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Failed to delete sentiment for {symbol}: {e}")
            return False
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        client = self._get_client()
        if not client:
            return {"connected": False}
            
        try:
            keys = client.keys("sentiment:*")
            return {
                "connected": True,
                "cached_symbols": len(keys),
                "symbols": [k.replace("sentiment:", "") for k in keys]
            }
        except Exception as e:
            return {"connected": False, "error": str(e)}


# Singleton instance
redis_cache = RedisCacheService()
