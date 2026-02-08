"""
Scorecard API Endpoints — Screener.in → Parser → DB → RAG → AI Scorecard

Full pipeline:
  1. Fetch/parse fundamentals from Screener.in (with yfinance fallback)
  2. Store in SQLite (company_fundamentals)
  3. RAG: combine news sentiment + fundamentals
  4. Generate Tickertape-style AI scorecard
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List, Dict
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/scorecard", tags=["Scorecard"])

# Thread pool for blocking I/O
_executor = ThreadPoolExecutor(max_workers=4)

# Rate limiting: max concurrent scorecard generations
MAX_CONCURRENT_SCORECARDS = 5
_semaphore = asyncio.Semaphore(MAX_CONCURRENT_SCORECARDS)

# Cache for recent scorecard generations (prevent duplicate parallel requests)
_generation_cache: Dict[str, float] = {}  # symbol -> timestamp
CACHE_DEBOUNCE_SECONDS = 2  # Prevent duplicate requests within 2s


# ─── Response Models ─────────────────────────────────────────────────────────

class CategoryScore(BaseModel):
    name: str
    icon: str
    description: str
    weight: float
    score: float
    max_score: int
    verdict: str
    components: dict
    details: dict


class KeyStats(BaseModel):
    market_cap_cr: Optional[float] = None
    current_price: Optional[float] = None
    pe_ratio: Optional[float] = None
    roce: Optional[float] = None
    roe: Optional[float] = None
    dividend_yield: Optional[float] = None
    eps: Optional[float] = None
    debt_cr: Optional[float] = None
    promoter_holding: Optional[float] = None
    high_low: Optional[str] = None


class StrengthWeakness(BaseModel):
    category: str
    score: float
    verdict: str


class NewsSummary(BaseModel):
    sentiment: Optional[str] = None
    key_themes: Optional[str] = None
    article_count: int = 0


class ScorecardResponse(BaseModel):
    symbol: str
    company_name: str
    sector: Optional[str] = None
    industry: Optional[str] = None
    source: Optional[str] = None
    generated_at: str
    overall_score: float
    overall_max: int
    overall_verdict: str
    overall_badge: str
    categories: Dict[str, CategoryScore]
    key_stats: KeyStats
    pros: List[str]
    cons: List[str]
    strengths: List[StrengthWeakness]
    weaknesses: List[StrengthWeakness]
    peers: List[dict]
    news_summary: NewsSummary
    ai_summary: Optional[str] = None


class ScorecardListItem(BaseModel):
    symbol: str
    company_name: str
    sector: Optional[str] = None
    overall_score: float
    overall_verdict: str
    overall_badge: str
    pe_ratio: Optional[float] = None
    roce: Optional[float] = None
    market_cap_cr: Optional[float] = None


# ─── Pipeline helpers (run in thread pool) ───────────────────────────────────

def _fetch_and_store(symbol: str) -> dict:
    """Fetch from screener/yfinance, store in DB, return fundamentals."""
    from backend.services.screener_parser import fetch_and_parse
    from backend.services.fundamentals_db import fundamentals_db
    
    try:
        # Check if we have fresh data
        if not fundamentals_db.is_stale(symbol, max_age_hours=12):
            cached = fundamentals_db.get_latest(symbol)
            if cached:
                logger.info(f"Using cached fundamentals for {symbol}")
                return cached
        
        # Fetch fresh with timeout
        logger.info(f"Fetching fresh fundamentals for {symbol}")
        parsed = fetch_and_parse(symbol)
        if not parsed:
            # Try DB anyway (even if stale)
            logger.warning(f"Fresh fetch failed for {symbol}, checking cache")
            cached = fundamentals_db.get_latest(symbol)
            if cached:
                logger.info(f"Using stale cache for {symbol}")
                return cached
            raise ValueError(f"Could not fetch fundamentals for {symbol}")
        
        # Store in DB
        fundamentals_db.upsert(parsed)
        
        # Return from DB (to get the full row with defaults)
        return fundamentals_db.get_latest(symbol) or parsed
    
    except Exception as e:
        logger.error(f"Error fetching fundamentals for {symbol}: {e}")
        # Last resort: check cache even if very stale
        cached = fundamentals_db.get_latest(symbol)
        if cached:
            logger.warning(f"Using very stale cache for {symbol} due to error: {e}")
            return cached
        raise


def _get_sentiment(symbol: str, company_name: str) -> dict:
    """Get RAG sentiment analysis for the stock with error handling."""
    try:
        from backend.services.news_service import news_service
        from backend.services.rag_sentiment_analyzer import create_sentiment_analyzer
        
        # Fetch recent news with timeout
        logger.info(f"Fetching news for {symbol}")
        articles = news_service.get_company_news(symbol, company_name)
        # articles = news.get("articles", []) if isinstance(news, dict) else [] 
        # Note: get_company_news returns a List[Dict], not a dict with "articles" key
        # verify return type of get_company_news matches usage in _get_sentiment
        
        if not articles:
            logger.warning(f"No news articles found for {symbol}")
            return {
                "sentiment_score": 0.0,
                "confidence": 0.0,
                "reasoning": "No recent news available.",
                "key_themes": "no data",
                "risk_level": "MEDIUM",
                "article_count": 0,
            }
        
        # Run RAG sentiment analysis with retries
        logger.info(f"Running sentiment analysis for {symbol} with {len(articles)} articles")
        analyzer = create_sentiment_analyzer()
        result = analyzer.analyze_sentiment(
            company_name=company_name,
            symbol=symbol,
            articles=articles,
        )
        return result
    except Exception as e:
        logger.error(f"Sentiment analysis failed for {symbol}: {e}")
        return {
            "sentiment_score": 0.0,
            "confidence": 0.0,
            "reasoning": f"Sentiment analysis unavailable: {str(e)}",
            "key_themes": "error",
            "risk_level": "MEDIUM",
            "article_count": 0,
            "error": str(e),
        }


# ─── Endpoints ───────────────────────────────────────────────────────────────

@router.get("/{symbol}", response_model=ScorecardResponse)
async def get_scorecard(
    symbol: str,
    include_ai_summary: bool = Query(False, description="Generate AI-written investment summary (slower)"),
    refresh: bool = Query(False, description="Force refresh data from source"),
):
    """
    Generate a Tickertape-style AI scorecard for a stock.
    
    Pipeline: Cache-first → Screener.in (HTML) → Parser → JSON → DB → RAG (news + fundamentals) → AI Scorecard
    """
    symbol = symbol.upper()

    # 0. Try pre-built cache (unless forced refresh)
    if not refresh:
        from backend.services.data_cache import data_cache
        cached = data_cache.get_scorecard(symbol)
        if cached:
            # Optionally add AI summary if requested and missing
            if include_ai_summary and not cached.get("ai_summary"):
                pass  # Fall through to generate it
            else:
                return cached
    
    # Check for recent duplicate request (debounce)
    now = time.time()
    if symbol in _generation_cache:
        last_gen = _generation_cache[symbol]
        if now - last_gen < CACHE_DEBOUNCE_SECONDS:
            logger.warning(f"Debounced duplicate request for {symbol} (last request {now - last_gen:.1f}s ago)")
            await asyncio.sleep(0.5)  # Small delay to let first request complete
    
    _generation_cache[symbol] = now
    
    # Use semaphore to limit concurrent scorecard generations
    async with _semaphore:
        try:
            loop = asyncio.get_event_loop()
            
            # Force refresh if requested
            if refresh:
                from backend.services.fundamentals_db import fundamentals_db
                # Clear cache by marking as stale
                logger.info(f"Force refresh requested for {symbol}")
            
            # Step 1-2: Fetch/parse fundamentals + store in DB (with timeout)
            try:
                fundamentals = await asyncio.wait_for(
                    loop.run_in_executor(_executor, _fetch_and_store, symbol),
                    timeout=30.0  # 30s timeout for fundamentals fetch
                )
            except asyncio.TimeoutError:
                logger.error(f"Fundamentals fetch timed out for {symbol}")
                raise HTTPException(status_code=504, detail=f"Fundamentals fetch timed out for {symbol}")
            
            company_name = fundamentals.get("company_name", symbol)
            
            # Step 3: RAG sentiment analysis (parallel with timeout)
            try:
                sentiment_data = await asyncio.wait_for(
                    loop.run_in_executor(_executor, _get_sentiment, symbol, company_name),
                    timeout=45.0  # 45s timeout for sentiment (LLM can be slow)
                )
            except asyncio.TimeoutError:
                logger.error(f"Sentiment analysis timed out for {symbol}")
                sentiment_data = {
                    "sentiment_score": 0.0,
                    "confidence": 0.0,
                    "reasoning": "Sentiment analysis timed out",
                    "key_themes": "timeout",
                    "risk_level": "MEDIUM",
                    "article_count": 0,
                }
            
            # Step 4: Generate scorecard
            from backend.services.scorecard_generator import generate_scorecard, generate_ai_summary
            
            scorecard = generate_scorecard(
                fundamentals=fundamentals,
                sentiment_data=sentiment_data,
            )
            
            # Optional: AI summary (with timeout)
            if include_ai_summary:
                try:
                    ai_summary = await asyncio.wait_for(
                        loop.run_in_executor(_executor, generate_ai_summary, scorecard),
                        timeout=30.0
                    )
                    scorecard["ai_summary"] = ai_summary
                except asyncio.TimeoutError:
                    logger.error(f"AI summary generation timed out for {symbol}")
                    scorecard["ai_summary"] = "AI summary generation timed out."
                except Exception as e:
                    logger.error(f"AI summary generation failed for {symbol}: {e}")
                    scorecard["ai_summary"] = f"AI summary unavailable: {str(e)}"
            
            # Cache the scorecard for future requests
            from backend.services.data_cache import data_cache
            data_cache.set_scorecard(symbol, scorecard, ttl=3600)

            return scorecard
            
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Scorecard generation failed for {symbol}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to generate scorecard: {str(e)}")


@router.get("/", response_model=List[ScorecardListItem])
async def list_scorecards():
    """List all stored companies with their latest scores — cache-first."""
    from backend.services.data_cache import data_cache

    # Try pre-built cache
    cached = data_cache.get_scorecard_list()
    if cached:
        return [ScorecardListItem(**item) for item in cached]

    # Fallback: compute inline
    from backend.services.fundamentals_db import fundamentals_db
    from backend.services.scorecard_generator import generate_scorecard
    
    all_fundamentals = fundamentals_db.get_all_latest()
    
    results = []
    for f in all_fundamentals:
        try:
            sc = generate_scorecard(fundamentals=f)
            results.append(ScorecardListItem(
                symbol=f["symbol"],
                company_name=f.get("company_name", f["symbol"]),
                sector=f.get("sector"),
                overall_score=sc["overall_score"],
                overall_verdict=sc["overall_verdict"],
                overall_badge=sc["overall_badge"],
                pe_ratio=f.get("ratios", {}).get("pe_ratio"),
                roce=f.get("ratios", {}).get("roce"),
                market_cap_cr=f.get("ratios", {}).get("market_cap_cr"),
            ))
        except Exception as e:
            logger.warning(f"Failed to generate list scorecard for {f['symbol']}: {e}")
    
    return results


@router.post("/refresh/{symbol}")
async def refresh_scorecard(symbol: str):
    """Force refresh fundamentals data from Screener.in for a symbol."""
    symbol = symbol.upper()
    
    try:
        loop = asyncio.get_event_loop()
        
        from backend.services.screener_parser import fetch_and_parse
        from backend.services.fundamentals_db import fundamentals_db
        
        parsed = await loop.run_in_executor(_executor, fetch_and_parse, symbol)
        if not parsed:
            raise HTTPException(status_code=404, detail=f"Could not fetch data for {symbol}")
        
        fundamentals_db.upsert(parsed)
        
        return {
            "status": "success",
            "symbol": symbol,
            "source": parsed.get("source"),
            "company_name": parsed.get("company_name"),
            "message": f"Refreshed fundamentals for {symbol}",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch-refresh")
async def batch_refresh(symbols: List[str] = Query(
    default=["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "SBIN"],
    description="List of symbols to refresh",
)):
    """Batch refresh fundamentals for multiple symbols with throttling."""
    loop = asyncio.get_event_loop()
    results = {"success": [], "failed": []}
    
    # Process in batches of 5 to avoid overwhelming APIs
    batch_size = MAX_CONCURRENT_SCORECARDS
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        logger.info(f"Processing batch {i // batch_size + 1}: {batch}")
        
        # Process batch concurrently
        tasks = []
        for sym in batch:
            async def fetch_single(symbol):
                async with _semaphore:
                    try:
                        fundamentals = await asyncio.wait_for(
                            loop.run_in_executor(_executor, _fetch_and_store, symbol.upper()),
                            timeout=30.0
                        )
                        return {"status": "success", "symbol": symbol.upper()}
                    except asyncio.TimeoutError:
                        return {"status": "failed", "symbol": symbol.upper(), "error": "Timeout after 30s"}
                    except Exception as e:
                        return {"status": "failed", "symbol": symbol.upper(), "error": str(e)}
            
            tasks.append(fetch_single(sym))
        
        # Wait for batch to complete
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for res in batch_results:
            if isinstance(res, Exception):
                results["failed"].append({"symbol": "unknown", "error": str(res)})
            elif res["status"] == "success":
                results["success"].append(res["symbol"])
            else:
                results["failed"].append({"symbol": res["symbol"], "error": res["error"]})
        
        # Small delay between batches
        if i + batch_size < len(symbols):
            await asyncio.sleep(0.5)
    
    return {
        "total": len(symbols),
        "success_count": len(results["success"]),
        "failed_count": len(results["failed"]),
        **results,
    }


@router.get("/search/{query}")
async def search_companies(query: str):
    """Search companies in the fundamentals database."""
    from backend.services.fundamentals_db import fundamentals_db
    
    results = fundamentals_db.search(query)
    return [
        {
            "symbol": r["symbol"],
            "company_name": r.get("company_name"),
            "sector": r.get("sector"),
            "industry": r.get("industry"),
            "market_cap_cr": r.get("ratios", {}).get("market_cap_cr"),
        }
        for r in results
    ]
