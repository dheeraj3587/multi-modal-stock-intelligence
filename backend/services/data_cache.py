"""
Centralised Redis data cache for market quotes, indices, fundamentals, etc.

Every piece of data that the UI needs is pre-fetched by the scheduler and
written here.  API endpoints read from here - never from external APIs
directly (except as a fallback during the first ~30 s after cold start).

Key schema (all values are JSON strings):
    cache:quotes          - {instrument_key: quote_dict, ...}
    cache:indices         - [{symbol, name, price, ...}, ...]
    cache:stocks_sentiment - [{symbol, name, price, sentiment, ...}, ...]
    cache:leaderboard     - [{rank, symbol, ...}, ...]
    cache:historical:<symbol>:<interval> - [candle, ...]
    cache:fundamentals:<symbol> - {ratios, pros, cons, ...}
    cache:meta            - {"last_refresh": ISO timestamp}
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import redis

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
DEFAULT_TTL = 600  # 10 min - scheduler refreshes every 5 min so data is always fresh


class DataCache:
    """Lazy-connecting Redis wrapper for structured market data."""

    def __init__(self):
        self._client: Optional[redis.Redis] = None

    def _r(self) -> Optional[redis.Redis]:
        if self._client is not None:
            return self._client
        try:
            self._client = redis.from_url(
                REDIS_URL, decode_responses=True, socket_connect_timeout=2, socket_timeout=2
            )
            self._client.ping()
            return self._client
        except Exception as e:
            logger.debug(f"Redis connect failed: {e}")
            self._client = None
            return None

    # --- generic get / set -----------------------------------------

    def set(self, key: str, value: Any, ttl: int = DEFAULT_TTL) -> bool:
        r = self._r()
        if not r:
            return False
        try:
            r.setex(key, ttl, json.dumps(value, default=str))
            return True
        except Exception as e:
            logger.debug(f"cache set({key}) failed: {e}")
            return False

    def get(self, key: str) -> Any:
        r = self._r()
        if not r:
            return None
        try:
            raw = r.get(key)
            return json.loads(raw) if raw else None
        except Exception as e:
            logger.debug(f"cache get({key}) failed: {e}")
            return None

    # --- typed helpers ---------------------------------------------

    # Quotes dict  {instrument_key: {last_price, ohlc, volume, ...}}
    def set_quotes(self, data: Dict, ttl: int = DEFAULT_TTL):
        self.set("cache:quotes", data, ttl)

    def get_quotes(self) -> Dict:
        return self.get("cache:quotes") or {}

    # Indices list
    def set_indices(self, data: List[Dict], ttl: int = DEFAULT_TTL):
        self.set("cache:indices", data, ttl)

    def get_indices(self) -> List[Dict]:
        return self.get("cache:indices") or []

    # Stocks with sentiment
    def set_stocks_sentiment(self, data: List[Dict], ttl: int = DEFAULT_TTL):
        self.set("cache:stocks_sentiment", data, ttl)

    def get_stocks_sentiment(self) -> List[Dict]:
        return self.get("cache:stocks_sentiment") or []

    # Leaderboard
    def set_leaderboard(self, data: List[Dict], ttl: int = DEFAULT_TTL):
        self.set("cache:leaderboard", data, ttl)

    def get_leaderboard(self) -> List[Dict]:
        return self.get("cache:leaderboard") or []

    # Historical candles per symbol/interval
    def set_historical(self, symbol: str, interval: str, data: List[Dict], ttl: int = 900):
        self.set(f"cache:historical:{symbol}:{interval}", data, ttl)

    def get_historical(self, symbol: str, interval: str) -> List[Dict]:
        return self.get(f"cache:historical:{symbol}:{interval}") or []

    # Fundamentals per symbol
    def set_fundamentals(self, symbol: str, data: Dict, ttl: int = 3600):
        self.set(f"cache:fundamentals:{symbol}", data, ttl)

    def get_fundamentals(self, symbol: str) -> Optional[Dict]:
        return self.get(f"cache:fundamentals:{symbol}")

    # Scorecard per symbol
    def set_scorecard(self, symbol: str, data: Dict, ttl: int = 3600):
        self.set(f"cache:scorecard:{symbol}", data, ttl)

    def get_scorecard(self, symbol: str) -> Optional[Dict]:
        return self.get(f"cache:scorecard:{symbol}")

    # Scorecard list (all companies summary)
    def set_scorecard_list(self, data: List[Dict], ttl: int = 3600):
        self.set("cache:scorecard_list", data, ttl)

    def get_scorecard_list(self) -> List[Dict]:
        return self.get("cache:scorecard_list") or []

    # News articles per symbol
    def set_news(self, symbol: str, data: List[Dict], ttl: int = 1800):
        self.set(f"cache:news:{symbol}", data, ttl)

    def get_news(self, symbol: str) -> List[Dict]:
        return self.get(f"cache:news:{symbol}") or []

    # Stock analysis per symbol
    def set_analysis(self, symbol: str, data: Dict, ttl: int = 600):
        self.set(f"cache:analysis:{symbol}", data, ttl)

    def get_analysis(self, symbol: str) -> Optional[Dict]:
        return self.get(f"cache:analysis:{symbol}")

    # Meta - when was the last full refresh?
    def set_last_refresh(self):
        self.set("cache:meta", {"last_refresh": datetime.utcnow().isoformat()}, ttl=86400)

    def get_last_refresh(self) -> Optional[str]:
        meta = self.get("cache:meta")
        return meta.get("last_refresh") if meta else None


# Singleton
data_cache = DataCache()
