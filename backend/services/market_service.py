"""
Market Data Service - Fetches real market data from Upstox API with YFinance fallback
"""
import os
import httpx
import yfinance as yf
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import logging
import asyncio
from collections.abc import Mapping

logger = logging.getLogger(__name__)

# Indian Stock Mapping
STOCK_DATA = {
    'RELIANCE': {'name': 'Reliance Industries', 'isin': 'INE002A01018', 'sector': 'Energy', 'instrument_key': 'NSE_EQ|INE002A01018', 'yf_ticker': 'RELIANCE.NS'},
    'TCS': {'name': 'Tata Consultancy Services', 'isin': 'INE467B01029', 'sector': 'IT', 'instrument_key': 'NSE_EQ|INE467B01029', 'yf_ticker': 'TCS.NS'},
    'HDFCBANK': {'name': 'HDFC Bank Ltd', 'isin': 'INE040A01034', 'sector': 'Banking', 'instrument_key': 'NSE_EQ|INE040A01034', 'yf_ticker': 'HDFCBANK.NS'},
    'INFY': {'name': 'Infosys Limited', 'isin': 'INE009A01021', 'sector': 'IT', 'instrument_key': 'NSE_EQ|INE009A01021', 'yf_ticker': 'INFY.NS'},
    'ICICIBANK': {'name': 'ICICI Bank Ltd', 'isin': 'INE090A01021', 'sector': 'Banking', 'instrument_key': 'NSE_EQ|INE090A01021', 'yf_ticker': 'ICICIBANK.NS'},
    'HINDUNILVR': {'name': 'Hindustan Unilever', 'isin': 'INE030A01027', 'sector': 'FMCG', 'instrument_key': 'NSE_EQ|INE030A01027', 'yf_ticker': 'HINDUNILVR.NS'},
    'SBIN': {'name': 'State Bank of India', 'isin': 'INE062A01020', 'sector': 'Banking', 'instrument_key': 'NSE_EQ|INE062A01020', 'yf_ticker': 'SBIN.NS'},
    'BHARTIARTL': {'name': 'Bharti Airtel', 'isin': 'INE397D01024', 'sector': 'Telecom', 'instrument_key': 'NSE_EQ|INE397D01024', 'yf_ticker': 'BHARTIARTL.NS'},
    'ITC': {'name': 'ITC Limited', 'isin': 'INE154A01025', 'sector': 'FMCG', 'instrument_key': 'NSE_EQ|INE154A01025', 'yf_ticker': 'ITC.NS'},
    'KOTAKBANK': {'name': 'Kotak Mahindra Bank', 'isin': 'INE237A01028', 'sector': 'Banking', 'instrument_key': 'NSE_EQ|INE237A01028', 'yf_ticker': 'KOTAKBANK.NS'},
    'LT': {'name': 'Larsen & Toubro', 'isin': 'INE018A01030', 'sector': 'Engineering', 'instrument_key': 'NSE_EQ|INE018A01030', 'yf_ticker': 'LT.NS'},
    'AXISBANK': {'name': 'Axis Bank Ltd', 'isin': 'INE238A01034', 'sector': 'Banking', 'instrument_key': 'NSE_EQ|INE238A01034', 'yf_ticker': 'AXISBANK.NS'},
    'WIPRO': {'name': 'Wipro Limited', 'isin': 'INE075A01022', 'sector': 'IT', 'instrument_key': 'NSE_EQ|INE075A01022', 'yf_ticker': 'WIPRO.NS'},
    'SUNPHARMA': {'name': 'Sun Pharmaceutical', 'isin': 'INE044A01036', 'sector': 'Pharma', 'instrument_key': 'NSE_EQ|INE044A01036', 'yf_ticker': 'SUNPHARMA.NS'},
    'TATAMOTORS': {'name': 'Tata Motors Ltd', 'isin': 'INE155A01022', 'sector': 'Auto', 'instrument_key': 'NSE_EQ|INE155A01022', 'yf_ticker': 'TATAMOTORS.NS'},
    'MARUTI': {'name': 'Maruti Suzuki', 'isin': 'INE585B01010', 'sector': 'Auto', 'instrument_key': 'NSE_EQ|INE585B01010', 'yf_ticker': 'MARUTI.NS'},
    'HCLTECH': {'name': 'HCL Technologies', 'isin': 'INE860A01027', 'sector': 'IT', 'instrument_key': 'NSE_EQ|INE860A01027', 'yf_ticker': 'HCLTECH.NS'},
    'ASIANPAINT': {'name': 'Asian Paints', 'isin': 'INE021A01026', 'sector': 'Consumer', 'instrument_key': 'NSE_EQ|INE021A01026', 'yf_ticker': 'ASIANPAINT.NS'},
    'BAJFINANCE': {'name': 'Bajaj Finance', 'isin': 'INE296A01024', 'sector': 'Finance', 'instrument_key': 'NSE_EQ|INE296A01024', 'yf_ticker': 'BAJFINANCE.NS'},
    'TITAN': {'name': 'Titan Company', 'isin': 'INE280A01028', 'sector': 'Consumer', 'instrument_key': 'NSE_EQ|INE280A01028', 'yf_ticker': 'TITAN.NS'},
}

# Index mapping
INDEX_DATA = {
    'NIFTY50': {'name': 'NIFTY 50', 'instrument_key': 'NSE_INDEX|Nifty 50', 'yf_ticker': '^NSEI'},
    'SENSEX': {'name': 'SENSEX', 'instrument_key': 'BSE_INDEX|SENSEX', 'yf_ticker': '^BSESN'},
    'BANKNIFTY': {'name': 'BANK NIFTY', 'instrument_key': 'NSE_INDEX|Nifty Bank', 'yf_ticker': '^NSEBANK'},
}


class MarketService:
    def __init__(self):
        self.api_key = os.getenv("UPSTOX_API_KEY")
        self.api_secret = os.getenv("UPSTOX_API_SECRET")
        self.base_url = "https://api.upstox.com/v2"
        
    async def get_market_quote(self, access_token: Optional[str], instrument_keys: List[str]) -> Dict:
        """
        Fetch real-time market quotes. Tries Upstox first if token exists, else falls back to YFinance.
        """
        if not instrument_keys:
            return {}

        merged: Dict = {}
        missing_keys = set(instrument_keys)

        if access_token:
            data = await self._get_upstox_quote(access_token, instrument_keys)
            if data:
                merged.update(data)
                missing_keys -= set(data.keys())
        else:
            logger.info("No Upstox access token provided. Using YFinance fallback.")

        # Fill in any missing instruments (or all if Upstox not used/failed)
        if missing_keys:
            yf_data = await self._get_yfinance_quote(list(missing_keys))
            if yf_data:
                merged.update(yf_data)

        return merged

    async def _get_upstox_quote(self, access_token: str, instrument_keys: List[str]) -> Dict:
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {access_token}"
        }
        instruments = ",".join(instrument_keys)
        url = f"{self.base_url}/market-quote/quotes"
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    headers=headers,
                    params={"instrument_key": instruments},
                    timeout=5.0,
                )
                if response.status_code == 200:
                    payload = response.json()
                    return self._extract_upstox_quotes(payload)
                elif response.status_code == 401:
                    logger.warning("Upstox access token expired or invalid (401). Falling back to YFinance.")
                else:
                    logger.error(f"Upstox API returned error {response.status_code}: {response.text}")
        except httpx.TimeoutException:
            logger.error("Upstox API request timed out. Falling back to YFinance.")
        except Exception as e:
            logger.error(f"Upstox quote error: {e}")
        return {}

    @staticmethod
    def _extract_upstox_quotes(payload: object) -> Dict:
        """Best-effort extraction of the {instrument_key: quote} mapping.

        Upstox responses sometimes wrap the quotes under different keys depending on the endpoint/version.
        The rest of the code expects a mapping keyed by instrument_key.
        """
        if not isinstance(payload, Mapping):
            return {}

        data = payload.get("data")
        if isinstance(data, Mapping):
            # Common shape: {"data": {"NSE_EQ|...": {...}, ...}}
            if any("|" in str(k) for k in data.keys()):
                return dict(data)

            # Alternate wrappers: {"data": {"quotes": {...}}} or {"data": {"data": {...}}}
            for key in ("quotes", "data", "result", "results"):
                nested = data.get(key)
                if isinstance(nested, Mapping) and any("|" in str(k) for k in nested.keys()):
                    return dict(nested)

        # Some implementations return quotes at top-level
        for key in ("quotes", "result", "results"):
            nested = payload.get(key)
            if isinstance(nested, Mapping) and any("|" in str(k) for k in nested.keys()):
                return dict(nested)

        return {}

    async def _get_yfinance_quote(self, instrument_keys: List[str]) -> Dict:
        """Fetch quotes from Yahoo Finance and format like Upstox response"""
        # Map instrument keys back to symbols/tickers
        tickers = []
        key_map = {}
        
        # Build reverse map and ticker list
        for key in instrument_keys:
            # Check stocks
            for symbol, data in STOCK_DATA.items():
                if data['instrument_key'] == key:
                    tickers.append(data['yf_ticker'])
                    key_map[data['yf_ticker']] = key
                    break
            # Check indices
            for symbol, data in INDEX_DATA.items():
                if data['instrument_key'] == key:
                    tickers.append(data['yf_ticker'])
                    key_map[data['yf_ticker']] = key
                    break
        
        if not tickers:
            return {}

        try:
            # Run yfinance in a separate thread as it's blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self._fetch_yfinance_sync, tickers)
            
            # Transform to Upstox format
            formatted_data = {}
            for ticker, info in result.items():
                key = key_map.get(ticker)
                if key:
                    price = info.get('regularMarketPrice') or info.get('currentPrice') or 0.0
                    prev_close = info.get('regularMarketPreviousClose') or info.get('previousClose') or 0.0
                    
                    formatted_data[key] = {
                        "last_price": price,
                        "volume": info.get('regularMarketVolume') or info.get('volume') or 0,
                        "ohlc": {
                            "open": info.get('regularMarketOpen') or info.get('open') or 0.0,
                            "high": info.get('regularMarketDayHigh') or info.get('dayHigh') or 0.0,
                            "low": info.get('regularMarketDayLow') or info.get('dayLow') or 0.0,
                            "close": prev_close
                        }
                    }
            return formatted_data
        except Exception as e:
            logger.error(f"YFinance quote error: {e}")
            return {}

    def _fetch_yfinance_sync(self, tickers: List[str]) -> Dict:
        """Synchronous YFinance fetch"""
        data_dict = {}

        def _get_fast_attr(fast_obj: object, attr: str):
            if fast_obj is None:
                return None
            # FastInfo supports attributes; sometimes behaves like a mapping
            if hasattr(fast_obj, attr):
                try:
                    return getattr(fast_obj, attr)
                except Exception:
                    return None
            if isinstance(fast_obj, Mapping):
                return fast_obj.get(attr)
            return None

        def _coerce_float(value):
            try:
                if value is None:
                    return None
                return float(value)
            except Exception:
                return None

        def _coerce_int(value):
            try:
                if value is None:
                    return None
                return int(value)
            except Exception:
                return None

        # Fetch in bulk
        if len(tickers) == 1:
            t = yf.Ticker(tickers[0])
            info = t.info
            # YFinance sometimes returns empty info, try fast_info for basic price
            if not info or 'regularMarketPrice' not in info:
                 try:
                    fast = t.fast_info
                    info = {
                        'regularMarketPrice': _coerce_float(_get_fast_attr(fast, 'last_price')),
                        'regularMarketPreviousClose': _coerce_float(_get_fast_attr(fast, 'previous_close')),
                        'regularMarketOpen': _coerce_float(_get_fast_attr(fast, 'open')),
                        'regularMarketDayHigh': _coerce_float(_get_fast_attr(fast, 'day_high')),
                        'regularMarketDayLow': _coerce_float(_get_fast_attr(fast, 'day_low')),
                        'regularMarketVolume': _coerce_int(_get_fast_attr(fast, 'last_volume'))
                    }

                    # If fast_info didn't yield a usable price, fall back to slower .info
                    if not info.get('regularMarketPrice'):
                        try:
                            slow = t.info
                            if isinstance(slow, Mapping):
                                info = slow
                        except Exception:
                            pass
                 except:
                     pass
            data_dict[tickers[0]] = info
        else:
            # Group fetch
            tickers_str = " ".join(tickers)
            tickers_data = yf.Tickers(tickers_str)
            for ticker in tickers:
                try:
                    t = tickers_data.tickers[ticker]
                    # Try fast_info first for speed/reliability on live price
                    fast = t.fast_info
                    info = {
                        'regularMarketPrice': _coerce_float(_get_fast_attr(fast, 'last_price')),
                        'regularMarketPreviousClose': _coerce_float(_get_fast_attr(fast, 'previous_close')),
                        'regularMarketOpen': _coerce_float(_get_fast_attr(fast, 'open')),
                        'regularMarketDayHigh': _coerce_float(_get_fast_attr(fast, 'day_high')),
                        'regularMarketDayLow': _coerce_float(_get_fast_attr(fast, 'day_low')),
                        'regularMarketVolume': _coerce_int(_get_fast_attr(fast, 'last_volume'))
                    }

                    # If fast_info didn't yield a usable price, fall back to slower .info
                    if not info.get('regularMarketPrice'):
                        try:
                            slow = t.info
                            if isinstance(slow, Mapping):
                                info = slow
                        except Exception:
                            pass
                    data_dict[ticker] = info
                except Exception as e:
                    # Fallback to .info if fast_info fails
                    try:
                        tt = tickers_data.tickers.get(ticker)
                        data_dict[ticker] = tt.info if tt is not None else {}
                    except:
                        data_dict[ticker] = {}
                        
        return data_dict

    
    async def get_historical_data(self, access_token: Optional[str], instrument_key: str, interval: str = "1minute", days: int = 7) -> List[Dict]:
        """
        Fetch historical candle data. Fallback to YFinance if no token.
        """
        if access_token:
            data = await self._get_upstox_history(access_token, instrument_key, interval, days)
            if data:
                return data
                
        # Fallback to YFinance
        return await self._get_yfinance_history(instrument_key, interval, days)

    async def _get_upstox_history(self, access_token: str, instrument_key: str, interval: str, days: int) -> List[Dict]:
        headers = {"Accept": "application/json", "Authorization": f"Bearer {access_token}"}
        to_date = datetime.now().strftime("%Y-%m-%d")
        from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        url = f"{self.base_url}/historical-candle/{instrument_key}/{interval}/{to_date}/{from_date}"
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=headers, timeout=10.0)
                if response.status_code == 200:
                    data = response.json()
                    candles = data.get("data", {}).get("candles", [])
                    return [{"timestamp": c[0], "open": c[1], "high": c[2], "low": c[3], "close": c[4], "volume": c[5]} for c in candles]
                elif response.status_code == 401:
                    logger.warning("Upstox access token expired or invalid (401) for history. Falling back to YFinance.")
                else:
                    logger.error(f"Upstox history API returned error {response.status_code}: {response.text}")
        except httpx.TimeoutException:
            logger.error("Upstox history API request timed out. Falling back to YFinance.")
        except Exception as e:
            logger.error(f"Upstox history error: {e}")
        return []

    async def _get_yfinance_history(self, instrument_key: str, interval: str, days: int) -> List[Dict]:
        ticker = None
        for data in STOCK_DATA.values():
            if data['instrument_key'] == instrument_key:
                ticker = data['yf_ticker']
                break
        
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
            return candles[::-1] # Reverse to match typical API response if needed (latest first)
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

market_service = MarketService()
