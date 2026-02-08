"""
Cache Warmup Service - Pre-populates ALL Redis caches on cold start.

Ensures the UI never sees empty data.  Called once during app startup BEFORE
the server starts accepting HTTP requests.

Data populated:
  cache:quotes               - live stock prices (YFinance, always available)
  cache:indices              - NIFTY / SENSEX / BANKNIFTY
  cache:stocks_sentiment     - every stock merged with sentiment (neutral defaults)
  cache:leaderboard          - top-10 growth ranked stocks
  sentiment:<SYMBOL>         - per-stock sentiment entry in Redis (neutral stub)
  cache:fundamentals:<SYM>   - fundamental analysis per stock (YFinance)
  cache:scorecard:<SYM>      - AI scorecard per stock
  cache:scorecard_list       - list of all scorecards (summary)
  cache:news:<SYM>           - recent news articles per stock
  cache:analysis:<SYM>       - stock analysis per stock
  cache:historical:<SYM>:1day - daily historical candles
  cache:meta                 - last_refresh timestamp
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor

from backend.services.data_cache import data_cache
from backend.services.market_service import market_service, STOCK_DATA, INDEX_DATA
from backend.services.redis_cache import redis_cache

logger = logging.getLogger(__name__)

# Thread pool for blocking I/O during warmup
_warmup_executor = ThreadPoolExecutor(max_workers=8)


# ---------------------------------------------------------------------------
# 1. Quotes
# ---------------------------------------------------------------------------

async def _warmup_quotes() -> Dict:
    """Fetch quotes for every tracked stock via YFinance and cache them."""
    all_stocks = market_service.get_all_stocks()
    instrument_keys = [s["instrument_key"] for s in all_stocks]

    logger.info(f"[Warmup] Fetching quotes for {len(instrument_keys)} stocks via YFinance …")
    t0 = time.monotonic()

    quotes = await market_service.get_market_quote(None, instrument_keys)
    elapsed = time.monotonic() - t0

    if quotes:
        market_service.cache_quotes(quotes)
        data_cache.set_quotes(quotes)
        logger.info(f"[Warmup] Quotes cached: {len(quotes)} stocks in {elapsed:.1f}s")
    else:
        logger.warning("[Warmup] Quote fetch returned empty — UI will show zeros until next refresh")

    return quotes or {}


# ---------------------------------------------------------------------------
# 2. Indices
# ---------------------------------------------------------------------------

async def _warmup_indices():
    """Build index rows and cache them."""
    all_indices = market_service.get_all_indices()
    idx_keys = [idx["instrument_key"] for idx in all_indices]
    idx_quotes = await market_service.get_market_quote(None, idx_keys)

    result = []
    for idx in all_indices:
        qd = idx_quotes.get(idx["instrument_key"], {})
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
    logger.info(f"[Warmup] Indices cached: {len(result)}")


# ---------------------------------------------------------------------------
# 3. Sentiment stubs  (neutral defaults so UI is never blank)
# ---------------------------------------------------------------------------

def _warmup_sentiments():
    """Seed a neutral sentiment entry into Redis for every tracked stock."""
    now = datetime.now().isoformat()
    batch: Dict[str, Dict] = {}

    for symbol in STOCK_DATA:
        existing = redis_cache.get_sentiment(symbol)
        if existing:
            continue

        batch[symbol] = {
            "sentiment_score": 0.0,
            "confidence": 0.0,
            "risk_level": "medium",
            "reasoning": "Sentiment analysis pending — will update shortly.",
            "is_stale": True,
            "last_updated": now,
        }

    if batch:
        redis_cache.set_batch_sentiments(batch, ttl=7200)
        logger.info(f"[Warmup] Seeded {len(batch)} neutral sentiment stubs into Redis")
    else:
        logger.info("[Warmup] All sentiments already cached — nothing to seed")


# ---------------------------------------------------------------------------
# 4. Stocks-with-sentiment list  (the big dashboard payload)
# ---------------------------------------------------------------------------

def _warmup_stocks_with_sentiment(quotes: Dict):
    """Build cache:stocks_sentiment from quotes + sentiment stubs."""
    all_stocks = market_service.get_all_stocks()

    result = []
    for stock in all_stocks:
        symbol = stock["symbol"]
        qd = quotes.get(stock["instrument_key"], {})
        ohlc = qd.get("ohlc", {})
        ltp = qd.get("last_price", 0) or 0
        prev = ohlc.get("close", ltp) or ltp
        change = ltp - prev if prev else 0
        pct = (change / prev * 100) if prev else 0

        sd = redis_cache.get_sentiment(symbol) or {}
        score = sd.get("sentiment_score", 0.0)
        confidence = sd.get("confidence", 0.0)
        risk = sd.get("risk_level", "medium").lower()
        is_stale = sd.get("is_stale", True)
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
    logger.info(f"[Warmup] stocks_sentiment cached: {len(result)} rows")


# ---------------------------------------------------------------------------
# 5. Leaderboard  (derived from stocks_sentiment)
# ---------------------------------------------------------------------------

def _warmup_leaderboard():
    """Compute leaderboard from the just-cached stocks_sentiment list."""
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
    logger.info(f"[Warmup] Leaderboard cached (top {len(lb)})")


# ---------------------------------------------------------------------------
# 6. Fundamentals per stock  (YFinance — no Screener rate-limits)
# ---------------------------------------------------------------------------

def _fetch_single_fundamental(symbol: str) -> Optional[Dict]:
    """Fetch fundamental data for a single stock, return (symbol, data)."""
    # Skip if already cached
    cached = data_cache.get_fundamentals(symbol)
    if cached:
        return cached
    try:
        data = market_service.get_fundamental_info(symbol)
        if data:
            data_cache.set_fundamentals(symbol, data, ttl=3600)
        return data
    except Exception as e:
        logger.debug(f"[Warmup] Fundamentals fetch failed for {symbol}: {e}")
        return None


async def _warmup_fundamentals():
    """Pre-fetch fundamentals for all stocks using thread pool."""
    symbols = list(STOCK_DATA.keys())
    logger.info(f"[Warmup] Pre-fetching fundamentals for {len(symbols)} stocks …")
    t0 = time.monotonic()

    batch_size = 10
    fetched = 0
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        tasks = [
            asyncio.to_thread(_fetch_single_fundamental, sym)
            for sym in batch
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if r and not isinstance(r, Exception):
                fetched += 1
        # Small sleep between batches
        if i + batch_size < len(symbols):
            await asyncio.sleep(0.3)

    elapsed = time.monotonic() - t0
    logger.info(f"[Warmup] Fundamentals cached: {fetched}/{len(symbols)} in {elapsed:.1f}s")


# ---------------------------------------------------------------------------
# 7. News articles per stock  (RSS — free, no rate limits)
# ---------------------------------------------------------------------------

def _fetch_single_news(symbol: str, company_name: str) -> List[Dict]:
    """Fetch news for a single stock."""
    cached = data_cache.get_news(symbol)
    if cached:
        return cached
    try:
        from backend.services.rss_news_fetcher import rss_fetcher
        articles = rss_fetcher.fetch_company_news(symbol, company_name, max_articles=15)
        if articles:
            data_cache.set_news(symbol, articles, ttl=1800)
        return articles
    except Exception as e:
        logger.debug(f"[Warmup] News fetch failed for {symbol}: {e}")
        return []


async def _warmup_news():
    """Pre-fetch news for all stocks."""
    all_stocks = market_service.get_all_stocks()
    logger.info(f"[Warmup] Pre-fetching news for {len(all_stocks)} stocks …")
    t0 = time.monotonic()

    fetched = 0

    batch_size = 10
    for i in range(0, len(all_stocks), batch_size):
        batch = all_stocks[i:i + batch_size]
        tasks = [
            asyncio.to_thread(
                _fetch_single_news,
                stock["symbol"],
                stock["name"],
            )
            for stock in batch
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if r and not isinstance(r, Exception) and len(r) > 0:
                fetched += 1
        if i + batch_size < len(all_stocks):
            await asyncio.sleep(0.2)

    elapsed = time.monotonic() - t0
    logger.info(f"[Warmup] News cached: {fetched}/{len(all_stocks)} stocks in {elapsed:.1f}s")


# ---------------------------------------------------------------------------
# 8. Scorecards  (computed from fundamentals + sentiment — no API calls)
# ---------------------------------------------------------------------------

def _generate_single_scorecard(symbol: str) -> Optional[Dict]:
    """Generate a scorecard for a single stock from cached data."""
    cached = data_cache.get_scorecard(symbol)
    if cached:
        return cached
    try:
        from backend.services.fundamentals_db import fundamentals_db
        from backend.services.scorecard_generator import generate_scorecard

        # Try DB first
        fundamentals = fundamentals_db.get_latest(symbol)
        if not fundamentals:
            return None

        # Use cached sentiment
        sentiment_data = redis_cache.get_sentiment(symbol)

        scorecard = generate_scorecard(
            fundamentals=fundamentals,
            sentiment_data=sentiment_data,
        )

        data_cache.set_scorecard(symbol, scorecard, ttl=3600)
        return scorecard
    except Exception as e:
        logger.debug(f"[Warmup] Scorecard generation failed for {symbol}: {e}")
        return None


async def _warmup_scorecards():
    """Generate scorecards for all stocks that have fundamentals in the DB."""
    from backend.services.fundamentals_db import fundamentals_db

    all_fundamentals = fundamentals_db.get_all_latest()
    symbols_with_data = [f["symbol"] for f in all_fundamentals]

    if not symbols_with_data:
        logger.info("[Warmup] No fundamentals in DB — skipping scorecard warmup")
        return

    logger.info(f"[Warmup] Generating scorecards for {len(symbols_with_data)} stocks …")
    t0 = time.monotonic()

    generated = 0
    scorecard_list = []

    batch_size = 20
    for i in range(0, len(symbols_with_data), batch_size):
        batch = symbols_with_data[i:i + batch_size]
        tasks = [
            asyncio.to_thread(_generate_single_scorecard, sym)
            for sym in batch
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for sym, r in zip(batch, results):
            if r and not isinstance(r, Exception):
                generated += 1
                scorecard_list.append({
                    "symbol": r.get("symbol", sym),
                    "company_name": r.get("company_name", sym),
                    "sector": r.get("sector"),
                    "overall_score": r.get("overall_score", 0),
                    "overall_verdict": r.get("overall_verdict", "N/A"),
                    "overall_badge": r.get("overall_badge", "average"),
                    "pe_ratio": r.get("key_stats", {}).get("pe_ratio"),
                    "roce": r.get("key_stats", {}).get("roce"),
                    "market_cap_cr": r.get("key_stats", {}).get("market_cap_cr"),
                })

    # Cache the summary list
    if scorecard_list:
        data_cache.set_scorecard_list(scorecard_list, ttl=3600)

    elapsed = time.monotonic() - t0
    logger.info(f"[Warmup] Scorecards cached: {generated}/{len(symbols_with_data)} in {elapsed:.1f}s")


# ---------------------------------------------------------------------------
# 9. Stock analysis stubs  (basic analysis from cached data)
# ---------------------------------------------------------------------------

def _build_single_analysis(symbol: str, quotes: Dict) -> Optional[Dict]:
    """Build analysis payload for a single stock from cached data."""
    cached = data_cache.get_analysis(symbol)
    if cached:
        return cached

    try:
        from backend.services.model_inference_service import model_inference_service

        stock_info = market_service.get_stock_info(symbol)
        if not stock_info:
            return None

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
        risk_level_raw = sd.get("risk_level", "medium").lower()

        if score > 0.2:
            sentiment = "bullish"
        elif score < -0.2:
            sentiment = "bearish"
        else:
            sentiment = "neutral"

        # News from cache
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

        if len(risk_factors) >= 3:
            risk_level = "high"
        elif len(risk_factors) >= 1:
            risk_level = "medium"
        else:
            risk_level = "low"

        growth_potential = "High" if sentiment == "bullish" and len(risk_factors) < 2 else (
            "Medium" if sentiment != "bearish" else "Low"
        )

        # ML prediction (fallback to sentiment-based)

        # ML prediction (fallback to neutral if no model)
        prediction_result = model_inference_service.predict_price(
            symbol=symbol,
            current_price=ltp,
            price_history=[],
            sentiment_score=score,
        )
        
        if prediction_result:
            predicted_price = prediction_result["predicted_price"]
            forecast_confidence = prediction_result["forecast_confidence"]
            short_term = prediction_result["short_term_outlook"]
            long_term = prediction_result["long_term_outlook"]
            recommendation = prediction_result["recommendation"]
        else:
            predicted_price = ltp
            forecast_confidence = 0.0
            short_term = "neutral"
            long_term = "neutral"
            recommendation = "hold"

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
        return analysis
    except Exception as e:
        logger.debug(f"[Warmup] Analysis build failed for {symbol}: {e}")
        return None


async def _warmup_analysis(quotes: Dict):
    """Pre-build analysis payloads for all stocks."""
    symbols = list(STOCK_DATA.keys())
    logger.info(f"[Warmup] Building analysis payloads for {len(symbols)} stocks …")
    t0 = time.monotonic()

    built = 0

    batch_size = 20
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        tasks = [
            asyncio.to_thread(_build_single_analysis, sym, quotes)
            for sym in batch
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if r and not isinstance(r, Exception):
                built += 1

    elapsed = time.monotonic() - t0
    logger.info(f"[Warmup] Analysis cached: {built}/{len(symbols)} in {elapsed:.1f}s")


# ---------------------------------------------------------------------------
# 10. AI Sentiment Warmup  (real Gemini analysis, runs in background)
# ---------------------------------------------------------------------------

async def _warmup_ai_sentiments():
    """
    Fetch REAL AI sentiments (via Gemini) for all stocks and store in Redis.
    After completion, rebuild derived caches (stocks_with_sentiment, leaderboard,
    analysis) so the dashboard shows actual AI scores instead of neutral stubs.

    This runs as a background task — it does NOT block server startup.
    """
    from backend.services.news_service import news_service

    all_stocks = market_service.get_all_stocks()

    # Only analyse stocks that still have stale stubs
    to_analyse = []
    for stock in all_stocks:
        cached = redis_cache.get_sentiment(stock["symbol"])
        if not cached or cached.get("is_stale", True):
            to_analyse.append(stock)

    if not to_analyse:
        logger.info("[Warmup-AI] All stocks already have real sentiment — skipping")
        return

    logger.info(f"[Warmup-AI] Analysing {len(to_analyse)} stocks with Gemini …")
    t0 = time.monotonic()

    # Initialise RAG components
    news_service._initialize_rag()
    if not news_service._rag_initialized:
        logger.warning("[Warmup-AI] RAG not initialised (missing GEMINI_API_KEY?) — aborting")
        return

    sem = asyncio.Semaphore(3)   # limit concurrency to avoid Gemini rate-limits
    success = 0

    async def _analyse_one(stock):
        nonlocal success
        symbol = stock["symbol"]
        name = stock.get("name", symbol)
        async with sem:
            try:
                result = await asyncio.to_thread(
                    news_service.get_sentiment, symbol, name
                )
                news_service.update_sentiment_cache(symbol, result)
                success += 1
            except Exception as e:
                logger.debug(f"[Warmup-AI] Failed for {symbol}: {e}")

    await asyncio.gather(*[_analyse_one(s) for s in to_analyse], return_exceptions=True)

    elapsed = time.monotonic() - t0
    logger.info(f"[Warmup-AI] Done — {success}/{len(to_analyse)} sentiments cached in {elapsed:.1f}s")

    # Rebuild all derived caches so the dashboard reflects real AI scores
    if success > 0:
        try:
            from backend.services.data_refresher import (
                refresh_stocks_with_sentiment,
                refresh_leaderboard,
                refresh_analysis_cache,
            )
            await refresh_stocks_with_sentiment()
            await refresh_leaderboard()
            await refresh_analysis_cache()
            logger.info("[Warmup-AI] Derived caches rebuilt with real AI sentiments")
        except Exception as e:
            logger.error(f"[Warmup-AI] Failed to rebuild derived caches: {e}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def warmup_all_caches():
    """
    Run the full cache-warmup pipeline.  Call this with ``await`` during
    startup so the server is fully primed before accepting traffic.
    """
    logger.info("[Warmup] ===== Starting FULL cache warmup =====")
    t0 = time.monotonic()

    try:
        # Phase 1: Prices (needed by everything else)
        quotes = await _warmup_quotes()
        await _warmup_indices()

        # Phase 2: Sentiment stubs (needed by dashboard + analysis)
        _warmup_sentiments()

        # Phase 3: Derived dashboard caches
        _warmup_stocks_with_sentiment(quotes)
        _warmup_leaderboard()

        # Phase 4: Heavy I/O — fundamentals + news (parallelise)
        await asyncio.gather(
            _warmup_fundamentals(),
            _warmup_news(),
        )

        # Phase 5: Computed caches (depends on fundamentals + sentiment)
        await _warmup_scorecards()
        await _warmup_analysis(quotes)

        # Phase 6: Mark refresh timestamp
        data_cache.set_last_refresh()

        elapsed = time.monotonic() - t0
        logger.info(f"[Warmup] ===== FULL cache warmup complete in {elapsed:.1f}s =====")

        # Phase 7: Kick off real AI sentiment analysis in the background
        #   (non-blocking — server is already serving traffic with stubs)
        asyncio.create_task(_warmup_ai_sentiments())
        logger.info("[Warmup] AI sentiment warmup launched in background")

    except Exception as exc:
        logger.error(f"[Warmup] Cache warmup failed: {exc}", exc_info=True)
