"""
Upstox API Client - centralised wrapper for all Upstox REST v2 calls.

Features:
* LTP snapshots for up to 1000 instruments per call.
* Full-quote snapshots for up to 500 instruments per call.
* OHLC snapshots (with interval param) for up to 1000 instruments per call.
* Historical candle data (1minute, 30minute, day, week, month).
* Auto-fallback to the global Redis-stored token when no per-request token
  is available.
* All results are normalised to {instrument_key: quote_dict}.
"""

import os
import logging
from typing import Dict, List, Optional
from collections.abc import Mapping
from datetime import datetime, timedelta

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://api.upstox.com/v2"

# Batch limits per Upstox docs
_BATCH_FULL_QUOTES = 500
_BATCH_LTP = 1000
_BATCH_OHLC = 1000

# Valid historical-candle intervals
_VALID_HIST_INTERVALS = {"1minute", "30minute", "day", "week", "month"}


class UpstoxClient:
    """Thin async wrapper around Upstox v2 REST API."""

    def __init__(self):
        self.api_key = os.getenv("UPSTOX_API_KEY", "")
        self.api_secret = os.getenv("UPSTOX_API_SECRET", "")

    # --- Token helpers ------------------------------------------------

    @staticmethod
    def get_global_token() -> Optional[str]:
        """Read the global Upstox token stored in Redis by the OAuth callback."""
        try:
            import redis as _redis
            url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
            r = _redis.from_url(url, decode_responses=True, socket_connect_timeout=2)
            return r.get("upstox_global_token")
        except Exception:
            return None

    def _resolve_token(self, token: Optional[str] = None) -> Optional[str]:
        """Return whichever token is available: explicit > global Redis."""
        return token or self.get_global_token()

    # --- Market quotes ------------------------------------------------

    async def get_ltp(self, instrument_keys: List[str], token: Optional[str] = None) -> Dict:
        """Fetch Last Traded Price for up to 1000 instruments per batch."""
        resolved = self._resolve_token(token)
        if not resolved:
            return {}
        merged: Dict = {}
        for i in range(0, len(instrument_keys), _BATCH_LTP):
            batch = instrument_keys[i : i + _BATCH_LTP]
            data = await self._fetch_quote_batch("/market-quote/ltp", batch, resolved)
            if data:
                merged.update(data)
        return merged

    async def get_full_quotes(self, instrument_keys: List[str], token: Optional[str] = None) -> Dict:
        """Fetch full market quotes (OHLC, depth, volume, etc.) - 500 per batch."""
        resolved = self._resolve_token(token)
        if not resolved:
            return {}
        merged: Dict = {}
        for i in range(0, len(instrument_keys), _BATCH_FULL_QUOTES):
            batch = instrument_keys[i : i + _BATCH_FULL_QUOTES]
            data = await self._fetch_quote_batch("/market-quote/quotes", batch, resolved)
            if data:
                merged.update(data)
        return merged

    async def get_ohlc(self, instrument_keys: List[str], interval: str = "1d", token: Optional[str] = None) -> Dict:
        """Fetch OHLC data with interval param - 1000 instruments per batch."""
        resolved = self._resolve_token(token)
        if not resolved:
            return {}
        merged: Dict = {}
        for i in range(0, len(instrument_keys), _BATCH_OHLC):
            batch = instrument_keys[i : i + _BATCH_OHLC]
            data = await self._fetch_ohlc_batch(batch, interval, resolved)
            if data:
                merged.update(data)
        return merged

    # --- Historical candles -------------------------------------------

    async def get_historical_candles(
        self,
        instrument_key: str,
        interval: str = "day",
        days: int = 30,
        token: Optional[str] = None,
    ) -> List[Dict]:
        """Return historical candle data as list of dicts.

        Valid intervals: 1minute, 30minute, day, week, month.
        """
        resolved = self._resolve_token(token)
        if not resolved:
            return []

        # Normalise common aliases
        if interval == "1day":
            api_interval = "day"
        elif interval in _VALID_HIST_INTERVALS:
            api_interval = interval
        else:
            logger.error("Invalid historical interval: %s (valid: %s)", interval, _VALID_HIST_INTERVALS)
            return []

        to_date = datetime.now().strftime("%Y-%m-%d")
        from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        url = f"{BASE_URL}/historical-candle/{instrument_key}/{api_interval}/{to_date}/{from_date}"
        headers = {"Accept": "application/json", "Authorization": f"Bearer {resolved}"}

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, headers=headers, timeout=10.0)
                if resp.status_code == 200:
                    candles = resp.json().get("data", {}).get("candles", [])
                    return [
                        {"timestamp": c[0], "open": c[1], "high": c[2], "low": c[3], "close": c[4], "volume": c[5]}
                        for c in candles
                    ]
                else:
                    logger.warning("Upstox historical API %d: %s", resp.status_code, resp.text[:200])
        except Exception as e:
            logger.error("Upstox historical candle error: %s", e)
        return []

    # --- Internal helpers ---------------------------------------------

    async def _fetch_quote_batch(self, path: str, keys: List[str], access_token: str) -> Dict:
        """Fetch a batch of quotes from a /market-quote/* endpoint.

        Uses ``symbol`` as the query parameter name per Upstox v2 docs.
        """
        headers = {"Accept": "application/json", "Authorization": f"Bearer {access_token}"}
        instruments = ",".join(keys)
        url = f"{BASE_URL}/{path.lstrip('/')}"
        params = {"symbol": instruments}

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, headers=headers, params=params, timeout=8.0)
                if resp.status_code == 200:
                    return self._normalise_response(resp.json())
                elif resp.status_code == 401:
                    logger.warning("Upstox token expired (401)")
                else:
                    logger.error("Upstox API %d: %s", resp.status_code, resp.text[:200])
        except httpx.TimeoutException:
            logger.error("Upstox API timeout for %s", path)
        except Exception as e:
            logger.error("Upstox quote error: %s", e)
        return {}

    async def _fetch_ohlc_batch(self, keys: List[str], interval: str, access_token: str) -> Dict:
        """Fetch OHLC quotes with a separate ``interval`` query parameter."""
        headers = {"Accept": "application/json", "Authorization": f"Bearer {access_token}"}
        instruments = ",".join(keys)
        url = f"{BASE_URL}/market-quote/ohlc"
        params = {"symbol": instruments, "interval": interval}

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, headers=headers, params=params, timeout=8.0)
                if resp.status_code == 200:
                    return self._normalise_response(resp.json())
                elif resp.status_code == 401:
                    logger.warning("Upstox token expired (401) for OHLC")
                else:
                    logger.error("Upstox OHLC API %d: %s", resp.status_code, resp.text[:200])
        except httpx.TimeoutException:
            logger.error("Upstox OHLC API timeout")
        except Exception as e:
            logger.error("Upstox OHLC error: %s", e)
        return {}

    @staticmethod
    def _normalise_response(payload: object) -> Dict:
        """Extract the {instrument_key: quote_dict} mapping.

        Standard Upstox v2 shape: {"status": "success", "data": {"NSE_EQ|INE...": {...}}}
        """
        if not isinstance(payload, Mapping):
            return {}

        data = payload.get("data")
        if isinstance(data, Mapping):
            # Most common shape - keys contain '|' (e.g. NSE_EQ|INE...)
            if any("|" in str(k) for k in data.keys()):
                return dict(data)

        return {}


# Singleton
upstox_client = UpstoxClient()
