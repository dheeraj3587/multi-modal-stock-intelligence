"""Market Data API Endpoints - Real market data from Upstox (with YFinance fallback).

All list endpoints read from pre-populated DataCache (Redis) for instant responses.
The background scheduler (data_refresher) keeps the cache warm every 5 minutes.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Response
from typing import List, Optional
from pydantic import BaseModel
from backend.services.market_service import market_service, STOCK_DATA
from backend.services.sentiment_analyzer import analyzer
from backend.services.news_service import news_service
from backend.services.model_inference_service import model_inference_service
from backend.services.screener_service import ScreenerService
from backend.services.data_cache import data_cache
from backend.dependencies import get_access_token, parse_symbols, CandleInterval, get_screener_service
import logging
import asyncio

router = APIRouter()
logger = logging.getLogger(__name__)


# Response Models
class StockQuote(BaseModel):
    symbol: str
    name: str
    sector: str
    price: float
    change: float
    changePercent: float
    high: float
    low: float
    open: float
    volume: int
    instrument_key: str


class IndexQuote(BaseModel):
    symbol: str
    name: str
    price: float
    change: float
    changePercent: float


class CandleData(BaseModel):
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: int


class StockWithSentiment(BaseModel):
    symbol: str
    name: str
    sector: str
    price: float
    change: float
    changePercent: float
    forecastConfidence: float
    sentiment: str  # 'bullish', 'neutral', 'bearish'
    riskLevel: str  # 'low', 'medium', 'high'
    lastUpdated: Optional[str] = None  # ISO timestamp of last sentiment update


class LeaderboardStock(BaseModel):
    rank: int
    symbol: str
    name: str
    sector: str
    growthScore: float
    sentiment: str
    forecastPercent: float


class RefreshResponse(BaseModel):
    status: str
    message: str
    refreshed_at: str


@router.post("/refresh-cache", response_model=RefreshResponse)
async def refresh_all_caches():
    """
    Manually trigger a full cache refresh.
    Refreshes: quotes, indices, sentiments, news, analysis, and leaderboard.
    """
    from datetime import datetime
    from backend.services.data_refresher import refresh_all
    from backend.services.cache_warmup import warmup_all_caches
    
    logger.info("Manual cache refresh triggered via API")
    
    try:
        # Run full warmup (includes all data)
        await warmup_all_caches()
        
        return RefreshResponse(
            status="success",
            message="All caches refreshed successfully",
            refreshed_at=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Cache refresh failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cache refresh failed: {str(e)}")


@router.get("/stocks", response_model=List[dict])
async def get_all_stocks():
    """Get list of all supported stocks"""
    return market_service.get_all_stocks()


@router.get("/stocks/quotes", response_model=List[StockQuote])
async def get_stock_quotes(
    symbol_list: List[str] = Depends(parse_symbols),
    access_token: Optional[str] = Depends(get_access_token),
):
    """
    Fetch real-time quotes for multiple stocks.
    Uses Upstox if token provided, else falls back to Yahoo Finance.
    """
    instrument_keys = []
    stock_info_map = {}
    
    for symbol in symbol_list:
        info = market_service.get_stock_info(symbol)
        if info:
            instrument_keys.append(info["instrument_key"])
            stock_info_map[info["instrument_key"]] = info
    
    # Fetch real quotes (MarketService handles fallback)
    quotes = await market_service.get_market_quote(access_token, instrument_keys)
    
    result = []
    for key, info in stock_info_map.items():
        quote_data = quotes.get(key, {})
        ohlc = quote_data.get("ohlc", {})
        
        ltp = quote_data.get("last_price", 0) or 0
        prev_close = ohlc.get("close", ltp) or ltp
        change = ltp - prev_close if prev_close else 0
        change_pct = (change / prev_close * 100) if prev_close else 0
        
        result.append(StockQuote(
            symbol=info["symbol"],
            name=info["name"],
            sector=info["sector"],
            price=ltp,
            change=change,
            changePercent=change_pct,
            high=ohlc.get("high", 0) or 0,
            low=ohlc.get("low", 0) or 0,
            open=ohlc.get("open", 0) or 0,
            volume=quote_data.get("volume", 0) or 0,
            instrument_key=info["instrument_key"]
        ))
    
    return result


@router.get("/stocks/{symbol}/historical", response_model=List[CandleData])
async def get_historical_data(
    symbol: str,
    interval: CandleInterval = CandleInterval.ONE_MINUTE,
    days: int = Query(default=7, ge=1, le=365),
    access_token: Optional[str] = Depends(get_access_token),
):
    """Fetch historical candle data for a stock"""
    stock_info = market_service.get_stock_info(symbol.upper())
    if not stock_info:
        raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")
    
    candles = await market_service.get_historical_data(
        access_token, 
        stock_info["instrument_key"],
        interval.value,
        days
    )
    
    return [CandleData(**c) for c in candles]


async def _get_stocks_with_sentiment(access_token: Optional[str]) -> List[StockWithSentiment]:
    """
    Return all stocks with sentiment.
    Reads entirely from DataCache for instant responses.
    Falls back to inline fetch only during cold-start.
    """
    # 1. Try pre-built cache (populated by data_refresher every 5 min)
    cached_list = data_cache.get_stocks_sentiment()
    if cached_list:
        return [StockWithSentiment(**s) for s in cached_list]

    # 2. Cold-start fallback: build from raw caches
    all_stocks = market_service.get_all_stocks()
    quotes = data_cache.get_quotes() or market_service.get_cached_quotes()

    if not quotes:
        # Cache still warming - do a quick inline fetch
        instrument_keys = [s["instrument_key"] for s in all_stocks]
        try:
            quotes = await asyncio.wait_for(
                market_service.get_market_quote(access_token, instrument_keys),
                timeout=30.0,
            )
            if quotes:
                market_service.cache_quotes(quotes)
        except asyncio.TimeoutError:
            logger.warning("Inline quote fetch timeout -- cache still warming")
            quotes = {}

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

        result.append(StockWithSentiment(
            symbol=symbol,
            name=stock["name"],
            sector=stock["sector"],
            price=ltp,
            change=change,
            changePercent=pct,
            forecastConfidence=min(99, confidence * 100) if not is_stale else 0.0,
            sentiment=sentiment,
            riskLevel=risk,
            lastUpdated=last_updated,
        ))

    return result


@router.get("/stocks/with-sentiment", response_model=List[StockWithSentiment])
async def get_stocks_with_sentiment(
    response: Response,
    access_token: Optional[str] = Depends(get_access_token),
):
    # Helps UI and any intermediate caches avoid hammering the backend.
    response.headers["Cache-Control"] = "public, max-age=30, stale-while-revalidate=60"
    return await _get_stocks_with_sentiment(access_token)


class RealTimeSentiment(BaseModel):
    """Response model for real-time sentiment lookup"""
    symbol: str
    name: str
    sentiment_score: float
    sentiment_label: str  # 'bullish', 'neutral', 'bearish'
    confidence: float
    risk_level: str
    reasoning: str
    last_updated: Optional[str] = None
    cached: bool = False  # True if from cache, False if freshly analyzed


@router.get("/stocks/{symbol}/sentiment", response_model=RealTimeSentiment)
async def get_stock_sentiment(symbol: str):
    """
    Get real-time AI sentiment for ANY stock symbol.
    This is the endpoint for users to search and analyze any stock they want.
    
    - If cached in Redis: returns instantly
    - If not cached: calls Gemini API for real-time analysis (takes ~5 seconds)
    """
    symbol = symbol.upper()
    
    # Get stock info (creates dynamic entry if not in STOCK_DATA)
    stock_info = market_service.get_stock_info(symbol)
    if stock_info:
        company_name = stock_info['name']
    else:
        # For stocks not in our list, use symbol as name
        company_name = symbol
    
    # Check cache first
    cached_sentiment = news_service.get_sentiment_cached_only(symbol)
    is_stale = cached_sentiment.get("is_stale", True)
    
    if not is_stale:
        # Return cached result
        score = cached_sentiment.get("sentiment_score", 0.0)
        return RealTimeSentiment(
            symbol=symbol,
            name=company_name,
            sentiment_score=score,
            sentiment_label="bullish" if score > 0.2 else ("bearish" if score < -0.2 else "neutral"),
            confidence=cached_sentiment.get("confidence", 0.0),
            risk_level=cached_sentiment.get("risk_level", "medium"),
            reasoning=cached_sentiment.get("reasoning", ""),
            last_updated=cached_sentiment.get("last_updated"),
            cached=True
        )
    
    # Not cached - fetch real-time from Gemini
    logger.info(f"Fetching real-time sentiment for {symbol} ({company_name})")
    
    try:
        result = await asyncio.to_thread(
            news_service.get_sentiment,
            symbol,
            company_name
        )
        
        # Cache the result
        news_service.update_sentiment_cache(symbol, result)
        
        score = result.get("sentiment_score", 0.0)
        return RealTimeSentiment(
            symbol=symbol,
            name=company_name,
            sentiment_score=score,
            sentiment_label="bullish" if score > 0.2 else ("bearish" if score < -0.2 else "neutral"),
            confidence=result.get("confidence", 0.0),
            risk_level=result.get("risk_level", "medium"),
            reasoning=result.get("reasoning", ""),
            last_updated=result.get("last_updated"),
            cached=False
        )
    except Exception as e:
        logger.error(f"Failed to get sentiment for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze {symbol}: {str(e)}")


@router.get("/indices", response_model=List[IndexQuote])
async def get_indices(
    access_token: Optional[str] = Depends(get_access_token),
):
    """Fetch market indices (NIFTY, SENSEX, etc.) - cache-first."""
    # Try pre-built cache
    cached = data_cache.get_indices()
    if cached:
        return [IndexQuote(**i) for i in cached]

    # Fallback to inline fetch
    all_indices = market_service.get_all_indices()
    instrument_keys = [idx["instrument_key"] for idx in all_indices]
    quotes = await market_service.get_market_quote(access_token, instrument_keys)

    result = []
    for idx in all_indices:
        qd = quotes.get(idx["instrument_key"], {})
        ohlc = qd.get("ohlc", {})
        ltp = qd.get("last_price", 0) or 0
        prev = ohlc.get("close", ltp) or ltp
        change = ltp - prev if prev else 0
        pct = (change / prev * 100) if prev else 0
        result.append(IndexQuote(
            symbol=idx["symbol"], name=idx["name"],
            price=ltp, change=change, changePercent=pct,
        ))
    return result


@router.get("/leaderboard", response_model=List[LeaderboardStock])
async def get_leaderboard(
    access_token: Optional[str] = Depends(get_access_token),
):
    """Get stocks ranked by growth potential - cache-first."""
    cached = data_cache.get_leaderboard()
    if cached:
        return [LeaderboardStock(**lb) for lb in cached]

    # Fall back to computing in-line
    stocks_with_sentiment = await _get_stocks_with_sentiment(access_token)
    
    # Sort by forecast confidence and sentiment
    scored = []
    for stock in stocks_with_sentiment:
        # Calculate growth score
        sentiment_bonus = 10 if stock.sentiment == "bullish" else (-10 if stock.sentiment == "bearish" else 0)
        growth_score = stock.forecastConfidence + sentiment_bonus + (stock.changePercent * 2)
        
        scored.append({
            "stock": stock,
            "growth_score": min(100, max(0, growth_score))
        })
    
    # Sort by growth score
    scored.sort(key=lambda x: x["growth_score"], reverse=True)
    
    result = []
    for i, item in enumerate(scored[:10]):
        stock = item["stock"]
        result.append(LeaderboardStock(
            rank=i + 1,
            symbol=stock.symbol,
            name=stock.name,
            sector=stock.sector,
            growthScore=item["growth_score"],
            sentiment=stock.sentiment,
            forecastPercent=stock.changePercent
        ))
    
    return result

class NewsItem(BaseModel):
    title: str
    source: str
    published: str
    sentiment: str


class StockAnalysisResponse(BaseModel):
    symbol: str
    name: str
    sector: str
    currentPrice: float
    change: float
    changePercent: float
    sentiment: str
    sentimentScore: float
    sentimentConfidence: float
    sentimentReasoning: str
    riskLevel: str
    riskFactors: List[str]
    growthPotential: str
    debtLevel: str
    predictedPrice: float
    forecastConfidence: float
    shortTermOutlook: str
    longTermOutlook: str
    recommendation: str
    recentNews: List[NewsItem]
    newsCount: int


@router.get("/stocks/{symbol}/analysis", response_model=StockAnalysisResponse)
async def get_stock_analysis(
    symbol: str,
    access_token: Optional[str] = Depends(get_access_token),
):
    """Get comprehensive AI analysis for a specific stock — cache-first."""
    symbol = symbol.upper()

    # 1. Try pre-built cache
    cached = data_cache.get_analysis(symbol)
    if cached:
        return cached

    # 2. Fallback to live computation
    try:
        # Get basic stock info
        stock_info = market_service.get_stock_info(symbol)
        if not stock_info:
            raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")
        
        quotes = await market_service.get_market_quote(access_token, [stock_info["instrument_key"]])
        quote_data = quotes.get(stock_info["instrument_key"], {})
        ohlc = quote_data.get("ohlc", {})
        
        ltp = quote_data.get("last_price", 0) or 0
        prev_close = ohlc.get("close", ltp) or ltp
        change = ltp - prev_close if prev_close else 0
        change_pct = (change / prev_close * 100) if prev_close else 0
        
        # Get RAG-powered sentiment analysis
        company_name = stock_info.get("name", symbol)
        try:
            # Start RAG task in background
            rag_task = asyncio.create_task(
                asyncio.to_thread(news_service.get_sentiment, symbol, company_name)
            )
            
            # Wait with timeout (fast response for UI)
            rag_result = await asyncio.wait_for(rag_task, timeout=5.0)
            
            sentiment_score = rag_result.get("sentiment_score", 0.0)
            sentiment_confidence = rag_result.get("confidence", 0.5)
            sentiment_reasoning = rag_result.get("reasoning", "Analysis based on recent news and market sentiment")
        except asyncio.TimeoutError:
            # Timeout hit - let background task finish!
            logger.warning(f"Sentiment analysis timeout for {symbol} (background analysis continuing)")
            sentiment_score = 0.0
            sentiment_confidence = 0.3
            sentiment_reasoning = "Deep analysis in progress. Check back shortly."
        except Exception as e:
            logger.warning(f"Sentiment analysis timeout/failed for {symbol}: {e}")
            sentiment_score = 0.0
            sentiment_confidence = 0.3
            sentiment_reasoning = "Unable to fetch sentiment analysis. Using neutral stance."
        
        # Determine sentiment category
        if sentiment_score > 0.2:
            sentiment = "bullish"
        elif sentiment_score < -0.2:
            sentiment = "bearish"
        else:
            sentiment = "neutral"
        
        # Get recent news
        try:
            articles = await asyncio.wait_for(
                asyncio.to_thread(news_service.get_company_news, symbol, company_name),
                timeout=5.0
            )
            recent_news = [
                {
                    "title": a.get("title", ""),
                    "source": a.get("source", ""),
                    "published": a.get("published_at", ""),
                    "sentiment": "bullish" if analyzer.get_sentiment(a.get("title", "")) > 0.1 else ("bearish" if analyzer.get_sentiment(a.get("title", "")) < -0.1 else "neutral")
                }
                for a in articles[:5]
            ]
        except Exception as e:
            logger.warning(f"News fetch failed for {symbol}: {e}")
            recent_news = []
        
        # Calculate risk factors
        risk_factors = []
        if change_pct < -5:
            risk_factors.append("Recent price decline")
        if sentiment_score < -0.3:
            risk_factors.append("Negative sentiment in news")
        if not recent_news:
            risk_factors.append("Limited news coverage")
        
        # Determine risk level
        if len(risk_factors) >= 3:
            risk_level = "high"
        elif len(risk_factors) >= 1:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        # Growth potential
        growth_potential = "High" if sentiment == "bullish" and len(risk_factors) < 2 else ("Medium" if sentiment != "bearish" else "Low")
        debt_level = "Unknown"
        
        # Get ML-based price prediction

        # Get ML-based price prediction
        predicted_price = ltp
        forecast_confidence = 0.0
        short_term = "neutral"
        long_term = "neutral"
        recommendation = "hold"

        try:
            # Fetch historical data for better prediction
            historical_candles = await market_service.get_historical_data(
                access_token,
                stock_info["instrument_key"],
                interval="1day",
                days=60
            )
            price_history = [c["close"] for c in historical_candles] if historical_candles else []
            
            # Get ML prediction
            prediction_result = model_inference_service.predict_price(
                symbol=symbol,
                current_price=ltp,
                price_history=price_history,
                sentiment_score=sentiment_score
            )
            
            if prediction_result:
                predicted_price = prediction_result["predicted_price"]
                forecast_confidence = prediction_result["forecast_confidence"]
                short_term = prediction_result["short_term_outlook"]
                long_term = prediction_result["long_term_outlook"]
                recommendation = prediction_result["recommendation"]
                logger.info(f"ML prediction for {symbol}: {predicted_price} (confidence: {forecast_confidence}%)")
            else:
                logger.debug(f"ML prediction unavailable for {symbol} (insufficient data or no model)")
            
        except Exception as e:
            logger.error(f"ML prediction failed for {symbol}: {e}")
            # Defaults (neutral) will be used
        
        result = {
            "symbol": symbol,
            "name": company_name,
            "sector": stock_info.get("sector", "Unknown"),
            "currentPrice": ltp,
            "change": round(change, 2),
            "changePercent": round(change_pct, 2),
            "sentiment": sentiment,
            "sentimentScore": round(sentiment_score, 3),
            "sentimentConfidence": round(sentiment_confidence, 3),
            "sentimentReasoning": sentiment_reasoning,
            "riskLevel": risk_level,
            "riskFactors": risk_factors,
            "growthPotential": growth_potential,
            "debtLevel": debt_level,
            "predictedPrice": round(predicted_price, 2),
            "forecastConfidence": round(min(99, forecast_confidence), 1),
            "shortTermOutlook": short_term,
            "longTermOutlook": long_term,
            "recommendation": recommendation,
            "recentNews": recent_news,
            "newsCount": len(recent_news)
        }

        # Cache the result for future requests
        data_cache.set_analysis(symbol, result, ttl=600)
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing stock {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing stock: {str(e)}")


@router.get("/stocks/{symbol}/fundamental-analysis", response_model=dict)
async def get_fundamental_analysis(
    symbol: str,
    service: ScreenerService = Depends(get_screener_service),
):
    """Get fundamental analysis — cache-first, then Screener.in / Yahoo Finance."""
    symbol = symbol.upper()

    # 0. Try pre-built cache
    cached = data_cache.get_fundamentals(symbol)
    if cached:
        return cached

    # 1. Try Screener.in first
    try:
        # Get company name
        stock_info = market_service.get_stock_info(symbol)
        company_name = stock_info.get("name") if stock_info else None
        
        data = await asyncio.to_thread(service.get_company_details, symbol, company_name)
        
        if data:
            data_cache.set_fundamentals(symbol, data, ttl=3600)
            return data
            
    except Exception as e:
        logger.warning(f"Screener.in scraping failed for {symbol}: {e}. Trying fallback.")

    # 2. Fallback to Yahoo Finance
    try:
        logger.info(f"Attempting Yahoo Finance fallback for {symbol} fundamentals")
        fallback_data = await asyncio.to_thread(market_service.get_fundamental_info, symbol)
        
        if fallback_data:
            data_cache.set_fundamentals(symbol, fallback_data, ttl=3600)
            return fallback_data
            
        raise HTTPException(status_code=404, detail="Fundamental data not found on Screener.in or Yahoo Finance")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching fundamental analysis for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching fundamental data: {str(e)}")