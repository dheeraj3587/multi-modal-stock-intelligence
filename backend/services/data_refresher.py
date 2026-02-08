"""
Background Data Refresher - pre-fetches ALL market data into Redis every 5 minutes.

This module is the single place responsible for keeping the DataCache warm.
API endpoints never call external APIs directly - they read from cache.

Jobs:
  1. refresh_all_quotes        - every 5 min  (Upstox LTP -> YFinance fallback)
  2. refresh_indices           - every 5 min
  3. refresh_sentiments        - every 30 min  (existing news_service logic)
  4. refresh_leaderboard       - every 5 min (derived from quotes + sentiment)
  5. refresh_fundamentals      - every 60 min (YFinance fundamentals)
  6. refresh_news              - every 30 min (RSS)
  7. refresh_scorecards        - every 60 min (from fundamentals + sentiment)
  8. refresh_analysis          - every 5 min (from quotes + sentiment + ML)
"""

import asyncio
import logging
import time
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor

from backend.services.market_service import market_service, STOCK_DATA, INDEX_DATA
from backend.services.data_cache import data_cache
from backend.services.upstox_client import upstox_client
from backend.services.redis_cache import redis_cache

logger = logging.getLogger(__name__)

_refresh_executor = ThreadPoolExecutor(max_workers=6)


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
    await refresh_news_cache()
    await refresh_analysis_cache()
    logger.info("[Scheduler] Full data refresh cycle complete OK")


async def refresh_heavy():
    """
    Heavy refresh cycle — runs less frequently (e.g. every 60 min).
    Fundamentals + scorecards are expensive, so separate from the 5-min cycle.
    """
    await refresh_fundamentals_cache()
    await refresh_scorecards_cache()
    logger.info("[Scheduler] Heavy data refresh (fundamentals + scorecards) complete OK")


def run_refresh_sync():
    """Wrapper for APScheduler (runs in its own event loop)."""
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(refresh_all())
    finally:
        loop.close()


def run_heavy_refresh_sync():
    """Wrapper for APScheduler — heavy refresh (fundamentals + scorecards)."""
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(refresh_heavy())
    finally:
        loop.close()


# --- Fundamentals refresh (YFinance) -----------------------------------

def _refresh_single_fundamental(symbol: str):
    """Fetch and cache fundamentals for one stock."""
    try:
        data = market_service.get_fundamental_info(symbol)
        if data:
            data_cache.set_fundamentals(symbol, data, ttl=3600)
    except Exception as e:
        logger.debug(f"[Scheduler] Fundamentals failed for {symbol}: {e}")


async def refresh_fundamentals_cache():
    """Refresh fundamentals for all stocks."""
    symbols = list(STOCK_DATA.keys())
    logger.info(f"[Scheduler] Refreshing fundamentals for {len(symbols)} stocks …")
    t0 = time.monotonic()

    loop = asyncio.get_event_loop()
    batch_size = 10
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        tasks = [
            loop.run_in_executor(_refresh_executor, _refresh_single_fundamental, sym)
            for sym in batch
        ]
        await asyncio.gather(*tasks, return_exceptions=True)
        if i + batch_size < len(symbols):
            await asyncio.sleep(0.3)

    elapsed = time.monotonic() - t0
    logger.info(f"[Scheduler] Fundamentals refresh done in {elapsed:.1f}s")


# --- News refresh (RSS) ------------------------------------------------

def _refresh_single_news(symbol: str, company_name: str):
    """Fetch and cache news for one stock."""
    try:
        from backend.services.rss_news_fetcher import rss_fetcher
        articles = rss_fetcher.fetch_company_news(symbol, company_name, max_articles=15)
        if articles:
            data_cache.set_news(symbol, articles, ttl=1800)
    except Exception as e:
        logger.debug(f"[Scheduler] News failed for {symbol}: {e}")


async def refresh_news_cache():
    """Refresh news for all stocks."""
    all_stocks = market_service.get_all_stocks()
    logger.info(f"[Scheduler] Refreshing news for {len(all_stocks)} stocks …")
    t0 = time.monotonic()

    loop = asyncio.get_event_loop()
    batch_size = 10
    for i in range(0, len(all_stocks), batch_size):
        batch = all_stocks[i:i + batch_size]
        tasks = [
            loop.run_in_executor(
                _refresh_executor, _refresh_single_news,
                stock["symbol"], stock["name"],
            )
            for stock in batch
        ]
        await asyncio.gather(*tasks, return_exceptions=True)
        if i + batch_size < len(all_stocks):
            await asyncio.sleep(0.2)

    elapsed = time.monotonic() - t0
    logger.info(f"[Scheduler] News refresh done in {elapsed:.1f}s")


# --- Scorecards refresh ------------------------------------------------

def _refresh_single_scorecard(symbol: str):
    """Generate and cache scorecard for one stock."""
    try:
        from backend.services.fundamentals_db import fundamentals_db
        from backend.services.scorecard_generator import generate_scorecard

        fundamentals = fundamentals_db.get_latest(symbol)
        if not fundamentals:
            return

        sentiment_data = redis_cache.get_sentiment(symbol)
        scorecard = generate_scorecard(fundamentals=fundamentals, sentiment_data=sentiment_data)
        data_cache.set_scorecard(symbol, scorecard, ttl=3600)
    except Exception as e:
        logger.debug(f"[Scheduler] Scorecard failed for {symbol}: {e}")


async def refresh_scorecards_cache():
    """Refresh scorecards for all stocks with fundamentals data."""
    from backend.services.fundamentals_db import fundamentals_db
    from backend.services.scorecard_generator import generate_scorecard

    all_fundamentals = fundamentals_db.get_all_latest()
    if not all_fundamentals:
        return

    symbols = [f["symbol"] for f in all_fundamentals]
    logger.info(f"[Scheduler] Refreshing scorecards for {len(symbols)} stocks …")
    t0 = time.monotonic()

    loop = asyncio.get_event_loop()
    scorecard_list = []
    batch_size = 20
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        tasks = [
            loop.run_in_executor(_refresh_executor, _refresh_single_scorecard, sym)
            for sym in batch
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    # Rebuild the summary list from freshly-cached scorecards
    for sym in symbols:
        sc = data_cache.get_scorecard(sym)
        if sc:
            scorecard_list.append({
                "symbol": sc.get("symbol", sym),
                "company_name": sc.get("company_name", sym),
                "sector": sc.get("sector"),
                "overall_score": sc.get("overall_score", 0),
                "overall_verdict": sc.get("overall_verdict", "N/A"),
                "overall_badge": sc.get("overall_badge", "average"),
                "pe_ratio": sc.get("key_stats", {}).get("pe_ratio"),
                "roce": sc.get("key_stats", {}).get("roce"),
                "market_cap_cr": sc.get("key_stats", {}).get("market_cap_cr"),
            })

    if scorecard_list:
        data_cache.set_scorecard_list(scorecard_list, ttl=3600)

    elapsed = time.monotonic() - t0
    logger.info(f"[Scheduler] Scorecards refresh done in {elapsed:.1f}s")


# --- Analysis refresh --------------------------------------------------

def _refresh_single_analysis(symbol: str, quotes: Dict):
    """Build and cache analysis for one stock."""
    try:
        from backend.services.model_inference_service import model_inference_service

        stock_info = market_service.get_stock_info(symbol)
        if not stock_info:
            return

        qd = quotes.get(stock_info["instrument_key"], {})
        ohlc = qd.get("ohlc", {})
        ltp = qd.get("last_price", 0) or 0
        prev = ohlc.get("close", ltp) or ltp
        change = ltp - prev if prev else 0
        pct = (change / prev * 100) if prev else 0

        sd = redis_cache.get_sentiment(symbol) or {}
        score = sd.get("sentiment_score", 0.0)
        confidence = sd.get("confidence", 0.0)
        reasoning = sd.get("reasoning", "Sentiment analysis pending.")

        if score > 0.2:
            sentiment = "bullish"
        elif score < -0.2:
            sentiment = "bearish"
        else:
            sentiment = "neutral"

        news_articles = data_cache.get_news(symbol)
        recent_news = []
        for a in (news_articles or [])[:5]:
            recent_news.append({
                "title": a.get("title", ""),
                "source": a.get("source", ""),
                "published": a.get("published_at", ""),
                "sentiment": "neutral",
            })

        risk_factors = []
        if pct < -5:
            risk_factors.append("Recent price decline")
        if score < -0.3:
            risk_factors.append("Negative sentiment in news")
        if not news_articles:
            risk_factors.append("Limited news coverage")

        risk_level = "high" if len(risk_factors) >= 3 else ("medium" if len(risk_factors) >= 1 else "low")
        growth_potential = "High" if sentiment == "bullish" and len(risk_factors) < 2 else (
            "Medium" if sentiment != "bearish" else "Low"
        )

        try:
            prediction_result = model_inference_service.predict_price(
                symbol=symbol, current_price=ltp, price_history=[], sentiment_score=score,
            )
            predicted_price = prediction_result["predicted_price"]
            forecast_confidence = prediction_result["forecast_confidence"]
            short_term = prediction_result["short_term_outlook"]
            long_term = prediction_result["long_term_outlook"]
            recommendation = prediction_result["recommendation"]
        except Exception:
            if sentiment == "bullish":
                predicted_price, short_term, long_term, recommendation = ltp * 1.08, "bullish", "bullish", "buy"
            elif sentiment == "bearish":
                predicted_price, short_term, long_term, recommendation = ltp * 0.92, "bearish", "bearish", "sell"
            else:
                predicted_price, short_term, long_term, recommendation = ltp, "neutral", "neutral", "hold"
            forecast_confidence = confidence * 100

        analysis = {
            "symbol": symbol,
            "name": stock_info.get("name", symbol),
            "sector": stock_info.get("sector", "Unknown"),
            "currentPrice": round(ltp, 2),
            "change": round(change, 2),
            "changePercent": round(pct, 2),
            "sentiment": sentiment,
            "sentimentScore": round(score, 3),
            "sentimentConfidence": round(confidence, 3),
            "sentimentReasoning": reasoning,
            "riskLevel": risk_level,
            "riskFactors": risk_factors,
            "growthPotential": growth_potential,
            "debtLevel": "Unknown",
            "predictedPrice": round(predicted_price, 2) if predicted_price else 0,
            "forecastConfidence": round(min(99, forecast_confidence), 1),
            "shortTermOutlook": short_term,
            "longTermOutlook": long_term,
            "recommendation": recommendation,
            "recentNews": recent_news,
            "newsCount": len(recent_news),
        }
        data_cache.set_analysis(symbol, analysis, ttl=600)
    except Exception as e:
        logger.debug(f"[Scheduler] Analysis failed for {symbol}: {e}")


async def refresh_analysis_cache():
    """Refresh analysis for all stocks."""
    quotes = data_cache.get_quotes() or {}
    symbols = list(STOCK_DATA.keys())
    logger.info(f"[Scheduler] Refreshing analysis for {len(symbols)} stocks …")
    t0 = time.monotonic()

    loop = asyncio.get_event_loop()
    batch_size = 20
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        tasks = [
            loop.run_in_executor(_refresh_executor, _refresh_single_analysis, sym, quotes)
            for sym in batch
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    elapsed = time.monotonic() - t0
    logger.info(f"[Scheduler] Analysis refresh done in {elapsed:.1f}s")
