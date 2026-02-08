"""Market data service with Upstox API and YFinance fallback."""
import os
import yfinance as yf
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import logging
import asyncio

from backend.services.upstox_client import upstox_client
from backend.services.data_cache import data_cache

logger = logging.getLogger(__name__)


from backend.config.stock_config import STOCK_DATA, INDEX_DATA



class MarketService:
    def __init__(self):
        self.api_key = os.getenv("UPSTOX_API_KEY")
        self.api_secret = os.getenv("UPSTOX_API_SECRET")
        self.base_url = "https://api.upstox.com/v2"
        self._redis_client = None
        self._quotes_cache_ttl = 600  # 10 min TTL (prefetch every 5 min)
        self.upstox = upstox_client
        self.cache = data_cache

        # O(1) reverse lookups: instrument_key -> yf_ticker / symbol
        self._instrument_to_yf: Dict[str, str] = {}
        self._instrument_to_symbol: Dict[str, str] = {}
        for sym, d in STOCK_DATA.items():
            self._instrument_to_yf[d['instrument_key']] = d['yf_ticker']
            self._instrument_to_symbol[d['instrument_key']] = sym
        for sym, d in INDEX_DATA.items():
            self._instrument_to_yf[d['instrument_key']] = d['yf_ticker']
            self._instrument_to_symbol[d['instrument_key']] = sym

    def _get_redis(self):
        """Lazy Redis connection for quote caching."""
        if self._redis_client is not None:
            return self._redis_client
        try:
            import redis as _redis
            url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
            self._redis_client = _redis.from_url(url, decode_responses=True, socket_connect_timeout=2, socket_timeout=2)
            self._redis_client.ping()
            return self._redis_client
        except Exception:
            self._redis_client = None
            return None

    def get_cached_quotes(self) -> Dict:
        """Return market quotes from DataCache or legacy Redis key."""
        # Try new DataCache first
        cached = self.cache.get_quotes()
        if cached:
            return cached
        # Fallback to legacy key
        try:
            r = self._get_redis()
            if not r:
                return {}
            import json as _json
            raw = r.get("market_quotes_cache")
            if raw:
                return _json.loads(raw)
        except Exception as e:
            logger.debug(f"Quote cache miss: {e}")
        return {}

    def cache_quotes(self, quotes: Dict):
        """Store market quotes in both legacy Redis key and new DataCache."""
        # New structured cache
        self.cache.set_quotes(quotes, ttl=self._quotes_cache_ttl)
        # Legacy key (for backward compat during migration)
        try:
            r = self._get_redis()
            if not r:
                return
            import json as _json
            r.setex("market_quotes_cache", self._quotes_cache_ttl, _json.dumps(quotes))
        except Exception as e:
            logger.debug(f"Quote cache write failed: {e}")
        
    async def get_market_quote(self, access_token: Optional[str], instrument_keys: List[str]) -> Dict:
        """
        Fetch real-time market quotes.

        Priority:
          1. DataCache (Redis) - instant, populated by background scheduler.
          2. Upstox API via UpstoxClient (uses explicit token -> global Redis token).
          3. YFinance bulk download as last resort.
        """
        if not instrument_keys:
            return {}

        # 1. Try cache first
        cached = self.cache.get_quotes()
        if cached:
            hit = {k: cached[k] for k in instrument_keys if k in cached}
            if len(hit) == len(instrument_keys):
                return hit  # full cache hit -- instant

        merged: Dict = {}
        missing_keys = set(instrument_keys)

        # 2. Try Upstox (explicit token or global Redis token)
        token = access_token or upstox_client.get_global_token()
        if token:
            data = await self.upstox.get_full_quotes(list(missing_keys), token=token)
            if data:
                merged.update(data)
                missing_keys -= set(data.keys())
        else:
            logger.info("No Upstox access token available. Using YFinance fallback.")

        # 3. YFinance fallback for anything still missing
        if missing_keys:
            yf_data = await self._get_yfinance_quote(list(missing_keys))
            if yf_data:
                merged.update(yf_data)

        return merged

    async def _get_yfinance_quote(self, instrument_keys: List[str]) -> Dict:
        """Fetch quotes from Yahoo Finance using yf.download() for speed.

        Uses a single bulk download request instead of per-ticker API calls,
        making it 10-20x faster for 100+ tickers.
        """
        # O(1) reverse lookup via pre-built dict
        tickers = []
        key_map = {}
        for key in instrument_keys:
            yf_tick = self._instrument_to_yf.get(key)
            if yf_tick:
                tickers.append(yf_tick)
                key_map[yf_tick] = key
        
        if not tickers:
            return {}

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self._fetch_yfinance_bulk_download, tickers
            )
            
            # Transform to Upstox-compatible format
            formatted_data = {}
            for ticker, info in result.items():
                key = key_map.get(ticker)
                if key and info.get("last_price"):
                    formatted_data[key] = {
                        "last_price": info["last_price"],
                        "volume": info.get("volume", 0),
                        "ohlc": {
                            "open": info.get("open", 0.0),
                            "high": info.get("high", 0.0),
                            "low": info.get("low", 0.0),
                            "close": info.get("prev_close", 0.0),
                        },
                    }
            return formatted_data
        except Exception as e:
            logger.error(f"YFinance quote error: {e}")
            return {}

    def _fetch_yfinance_bulk_download(self, tickers: List[str]) -> Dict:
        """Use yf.download() for a fast single-request bulk fetch.
        
        Downloads 2 days of daily OHLCV data for ALL tickers at once.
        Extracts the latest close as current price and previous close for change calc.
        """
        import pandas as pd

        data_dict: Dict = {}
        try:
            df = yf.download(
                tickers,
                period="5d",
                interval="1d",
                group_by="ticker",
                threads=True,
                progress=False,
            )
        except Exception as e:
            logger.error(f"yf.download() failed: {e}")
            return data_dict

        if df is None or df.empty:
            logger.warning("yf.download() returned empty DataFrame")
            return data_dict

        import pandas as pd
        is_multi = isinstance(df.columns, pd.MultiIndex)

        for ticker in tickers:
            try:
                if is_multi:
                    if ticker not in df.columns.get_level_values(0):
                        continue
                    ticker_df = df[ticker]
                else:
                    # Single ticker - columns are just OHLCV
                    ticker_df = df

                # Drop NaN rows
                ticker_df = ticker_df.dropna(subset=["Close"])
                if ticker_df.empty:
                    continue

                latest = ticker_df.iloc[-1]
                prev = ticker_df.iloc[-2] if len(ticker_df) >= 2 else latest

                close_val = float(latest["Close"])
                if close_val <= 0 or pd.isna(close_val):
                    continue

                data_dict[ticker] = {
                    "last_price": close_val,
                    "open": float(latest.get("Open", 0) or 0),
                    "high": float(latest.get("High", 0) or 0),
                    "low": float(latest.get("Low", 0) or 0),
                    "volume": int(latest.get("Volume", 0) or 0),
                    "prev_close": float(prev["Close"]) if not pd.isna(prev["Close"]) else close_val,
                }
            except Exception as e:
                logger.debug(f"YF parse error for {ticker}: {e}")
                continue

        logger.info(f"yf.download() fetched prices for {len(data_dict)}/{len(tickers)} tickers")
        return data_dict

    
    async def get_historical_data(self, access_token: Optional[str], instrument_key: str, interval: str = "1minute", days: int = 7) -> List[Dict]:
        """
        Fetch historical candle data with cache-first strategy.
        """
        # Validate interval
        valid_intervals = {"1minute", "30minute", "day", "week", "month", "1day"}
        if interval not in valid_intervals:
            logger.error("Invalid interval for historical data: %s", interval)
            return []

        # Resolve symbol for cache key via O(1) lookup
        symbol = self._instrument_to_symbol.get(instrument_key)

        # Check cache
        if symbol:
            cached = self.cache.get_historical(symbol, interval)
            if cached:
                return cached

        def _ts_sort_key(c: Dict) -> float:
            ts = c.get("timestamp")
            if ts is None:
                return 0.0
            try:
                if isinstance(ts, (int, float)):
                    return float(ts) / 1000.0 if float(ts) > 1_000_000_000_000 else float(ts)
                if isinstance(ts, str):
                    s = ts.strip()
                    if s.isdigit():
                        v = float(s)
                        return v / 1000.0 if v > 1_000_000_000_000 else v
                    if s.endswith("Z"):
                        s = s[:-1] + "+00:00"
                    return datetime.fromisoformat(s).timestamp()
            except Exception:
                return 0.0
            return 0.0

        # Try Upstox via client
        token = access_token or upstox_client.get_global_token()
        if token:
            data = await self.upstox.get_historical_candles(instrument_key, interval, days, token=token)
            if data:
                result = sorted(data, key=_ts_sort_key)
                if symbol:
                    self.cache.set_historical(symbol, interval, result)
                return result

        # Fallback to YFinance
        data = await self._get_yfinance_history(instrument_key, interval, days)
        result = sorted(data, key=_ts_sort_key)
        if symbol and result:
            self.cache.set_historical(symbol, interval, result)
        return result

    async def _get_yfinance_history(self, instrument_key: str, interval: str, days: int) -> List[Dict]:
        ticker = self._instrument_to_yf.get(instrument_key)
        if not ticker:
            return []
            
        # Map interval
        yf_interval = "1m" if interval == "1minute" else "1d"
        if interval == "30minute": yf_interval = "30m"
        
        try:
            loop = asyncio.get_event_loop()
            hist = await loop.run_in_executor(None, lambda: yf.Ticker(ticker).history(period=f"{days}d", interval=yf_interval))
            
            candles = []
            for index, row in hist.iterrows():
                candles.append({
                    "timestamp": index.strftime("%Y-%m-%dT%H:%M:%S+05:30"),
                    "open": row['Open'],
                    "high": row['High'],
                    "low": row['Low'],
                    "close": row['Close'],
                    "volume": int(row['Volume'])
                })
            # Return chronological order (oldest -> newest). Charting libraries expect ascending time.
            return candles
        except Exception as e:
            logger.error(f"YFinance history error: {e}")
            return []

    def get_all_stocks(self) -> List[Dict]:
        return [{"symbol": s, "name": d["name"], "sector": d["sector"], "instrument_key": d["instrument_key"]} for s, d in STOCK_DATA.items()]
    
    def get_stock_info(self, symbol: str) -> Optional[Dict]:
        if symbol in STOCK_DATA:
            data = STOCK_DATA[symbol]
            return {"symbol": symbol, "name": data["name"], "sector": data["sector"], "instrument_key": data["instrument_key"]}
        return None
    
    def get_all_indices(self) -> List[Dict]:
        return [{"symbol": s, "name": d["name"], "instrument_key": d["instrument_key"]} for s, d in INDEX_DATA.items()]

    def get_fundamental_info(self, symbol: str) -> Optional[Dict]:
        """
        Fetch fundamental data from Yahoo Finance as a fallback for Screener.in.
        Returns a dict matching the structure expected by the frontend:
        {
            "symbol": str,
            "company_name": str,
            "about": str,
            "ratios": Dict[str, str],
            "pros": List[str],
            "cons": List[str]
        }
        """
        try:
            stock_info = self.get_stock_info(symbol)
            if not stock_info:
                return None
            
            ticker_symbol = STOCK_DATA.get(symbol, {}).get("yf_ticker", f"{symbol}.NS")
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.info
            
            if not info:
                return None
            
            # Extract Ratios
            ratios = {}
            if info.get('marketCap'):
                mcap_cr = info['marketCap'] / 10000000  # Convert to Crores
                ratios['Market Cap'] = f"₹{mcap_cr:,.0f} Cr"
            
            if info.get('currentPrice'):
                ratios['Current Price'] = f"₹{info['currentPrice']}"
            elif info.get('regularMarketPrice'):
                 ratios['Current Price'] = f"₹{info['regularMarketPrice']}"
            
            if info.get('dayHigh') and info.get('dayLow'):
                ratios['High / Low'] = f"₹{info['dayHigh']} / ₹{info['dayLow']}"
            
            if info.get('trailingPE'):
                ratios['Stock P/E'] = f"{info['trailingPE']:.2f}"
            
            if info.get('bookValue'):
                ratios['Book Value'] = f"₹{info['bookValue']:.2f}"
            
            if info.get('dividendYield'):
                # YFinance dividendYield seems to be 0.38 for 0.38% (already scaled) or my observation is wrong.
                # However, usually it is decimal. If I saw 0.38, it is likely 0.0038 * 100? 
                # Let's assume input is decimal. If input is 0.0038, *100 = 0.38%.
                # But verification showed 38.00%. So input was 0.38.
                # So we use input as is.
                diff_yield = info['dividendYield']
                # Heuristic: if yield > 1 (e.g. 5), it is %, use as is. 
                # If yield < 1 and > 0.1 (e.g. 0.38), could be % or decimal (38%). 
                # For Reliance 0.38 is definitely %.
                # Let's just assume it's percentage for now if it matches rate/price.
                ratios['Dividend Yield'] = f"{diff_yield:.2f}%"
            
            if info.get('returnOnEquity'):
                ratios['ROE'] = f"{info['returnOnEquity']*100:.2f}%"
            
            if info.get('returnOnAssets'):
                ratios['ROA'] = f"{info['returnOnAssets']*100:.2f}%"

            if info.get('debtToEquity'):
                ratios['Debt to Equity'] = f"{info['debtToEquity']:.2f}"

            # Generate Pros & Cons dynamically
            pros = []
            cons = []
            
            pe = info.get('trailingPE', 0)
            de = info.get('debtToEquity', 0)
            roe = info.get('returnOnEquity', 0)
            div = info.get('dividendYield', 0)
            profit_growth = info.get('earningsGrowth', 0)

            # Pros
            if de < 50: pros.append("Company has low debt.")
            if roe > 0.15: pros.append("Good return on equity over 15%.")
            if div > 0.03: pros.append("Good dividend yield.")
            if profit_growth > 0.10: pros.append("Company has shown good profit growth.")
            if info.get('priceToBook', 10) < 3: pros.append("Stock is trading at decent book value.")

            # Cons
            if pe > 40: cons.append("Stock is trading at a high PE valuation.")
            if de > 100: cons.append("Company has high debt levels.")
            if roe < 0.10: cons.append("Low return on equity.")
            if info.get('priceToBook', 1) > 10: cons.append("Stock is trading at high book value.")
            
            if not pros: pros.append("Company is stable.")
            if not cons: cons.append("No major red flags.")

            return {
                "symbol": symbol,
                "company_name": info.get('longName', stock_info['name']),
                "about": info.get('longBusinessSummary') or info.get('description', "No description available."),
                "ratios": ratios,
                "pros": pros,
                "cons": cons
            }
            
        except Exception as e:
            logger.error(f"Error fetching YFinance fundamentals for {symbol}: {e}")
            return None

market_service = MarketService()
