"""
Background Data Refresher - pre-fetches ALL market data into Redis every 5 minutes.

This module is the single place responsible for keeping the DataCache warm.
API endpoints never call external APIs directly - they read from cache.

Jobs:
  1. refresh_all_quotes  - every 5 min  (Upstox LTP -> YFinance fallback)
  2. refresh_indices     - every 5 min
  3. refresh_sentiments  - every 30 min  (existing news_service logic)
  4. refresh_leaderboard - every 5 min (derived from quotes + sentiment)
"""

import asyncio
import logging
import time
from typing import List, Dict, Optional

from backend.services.market_service import market_service, STOCK_DATA, INDEX_DATA
from backend.services.data_cache import data_cache
from backend.services.upstox_client import upstox_client

logger = logging.getLogger(__name__)


# --- Quote refresh ----------------------------------------------------

async def refresh_all_quotes():
    """Fetch quotes for every tracked stock and write them into DataCache."""
    all_stocks = market_service.get_all_stocks()
    instrument_keys = [s["instrument_key"] for s in all_stocks]

    logger.info(f"[Scheduler] Refreshing quotes for {len(instrument_keys)} stocks ...")
    t0 = time.monotonic()

    quotes = await market_service.get_market_quote(None, instrument_keys)
    elapsed = time.monotonic() - t0

    if quotes:
        market_service.cache_quotes(quotes)
        data_cache.set_quotes(quotes)
        logger.info(f"[Scheduler] Quotes refreshed: {len(quotes)} in {elapsed:.1f}s")
    else:
        logger.warning(f"[Scheduler] Quote refresh returned empty after {elapsed:.1f}s")

    data_cache.set_last_refresh()


# --- Index refresh ----------------------------------------------------

async def refresh_indices():
    """Fetch market indices and cache them."""
    all_indices = market_service.get_all_indices()
    instrument_keys = [idx["instrument_key"] for idx in all_indices]

    quotes = await market_service.get_market_quote(None, instrument_keys)

    result = []
    for idx in all_indices:
        qd = quotes.get(idx["instrument_key"], {})
        ohlc = qd.get("ohlc", {})
        ltp = qd.get("last_price", 0) or 0
        prev = ohlc.get("close", ltp) or ltp
        change = ltp - prev if prev else 0
        pct = (change / prev * 100) if prev else 0
        result.append({
            "symbol": idx["symbol"],
            "name": idx["name"],
            "price": round(ltp, 2),
            "change": round(change, 2),
            "changePercent": round(pct, 2),
        })

    data_cache.set_indices(result)
    logger.info(f"[Scheduler] Indices refreshed: {len(result)}")


# --- Stocks with sentiment (builds the big list used by dashboard) ----

async def refresh_stocks_with_sentiment():
    """Merge cached quotes + cached sentiment into a single list and cache it."""
    from backend.services.news_service import news_service

    all_stocks = market_service.get_all_stocks()
    quotes = data_cache.get_quotes() or {}

    result = []
    for stock in all_stocks:
        symbol = stock["symbol"]
        qd = quotes.get(stock["instrument_key"], {})
        ohlc = qd.get("ohlc", {})
        ltp = qd.get("last_price", 0) or 0
        prev = ohlc.get("close", ltp) or ltp
        change = ltp - prev if prev else 0
        pct = (change / prev * 100) if prev else 0

        sd = news_service.get_sentiment_cached_only(symbol)
        score = sd.get("sentiment_score", 0.0)
        confidence = sd.get("confidence", 0.0)
        risk = sd.get("risk_level", "medium").lower()
        is_stale = sd.get("is_stale", False)
        last_updated = sd.get("last_updated", None)

        if score > 0.2:
            sentiment = "bullish"
        elif score < -0.2:
            sentiment = "bearish"
        else:
            sentiment = "neutral"

        result.append({
            "symbol": symbol,
            "name": stock["name"],
            "sector": stock["sector"],
            "price": round(ltp, 2),
            "change": round(change, 2),
            "changePercent": round(pct, 2),
            "forecastConfidence": round(min(99, confidence * 100) if not is_stale else 0.0, 1),
            "sentiment": sentiment,
            "riskLevel": risk,
            "lastUpdated": last_updated,
        })

    data_cache.set_stocks_sentiment(result)
    logger.info(f"[Scheduler] Stocks-with-sentiment refreshed: {len(result)}")


# --- Leaderboard (derived from stocks-with-sentiment) ----------------

async def refresh_leaderboard():
    """Compute and cache leaderboard from the stocks-with-sentiment cache."""
    stocks = data_cache.get_stocks_sentiment()
    if not stocks:
        return

    scored = []
    for s in stocks:
        bonus = 10 if s["sentiment"] == "bullish" else (-10 if s["sentiment"] == "bearish" else 0)
        gs = s.get("forecastConfidence", 0) + bonus + (s.get("changePercent", 0) * 2)
        scored.append({**s, "growthScore": round(min(100, max(0, gs)), 1)})

    scored.sort(key=lambda x: x["growthScore"], reverse=True)

    lb = []
    for i, item in enumerate(scored[:10]):
        lb.append({
            "rank": i + 1,
            "symbol": item["symbol"],
            "name": item["name"],
            "sector": item["sector"],
            "growthScore": item["growthScore"],
            "sentiment": item["sentiment"],
            "forecastPercent": item.get("changePercent", 0),
        })

    data_cache.set_leaderboard(lb)
    logger.info(f"[Scheduler] Leaderboard refreshed (top {len(lb)})")


# --- Master refresh (called by scheduler every 5 min) ----------------

async def refresh_all():
    """Run all refresh jobs in sequence (quotes first -> derived caches next)."""
    await refresh_all_quotes()
    await refresh_indices()
    await refresh_stocks_with_sentiment()
    await refresh_leaderboard()
    logger.info("[Scheduler] Full data refresh cycle complete OK")


def run_refresh_sync():
    """Wrapper for APScheduler (runs in its own event loop)."""
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(refresh_all())
    finally:
        loop.close()
