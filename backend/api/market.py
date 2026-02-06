"""
Market Data API Endpoints - Real market data from Upstox (with YFinance fallback)
"""
from fastapi import APIRouter, HTTPException, Header
from typing import List, Optional
from pydantic import BaseModel
from backend.services.market_service import market_service, STOCK_DATA
from backend.services.sentiment_analyzer import analyzer
from backend.services.news_service import news_service
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


class LeaderboardStock(BaseModel):
    rank: int
    symbol: str
    name: str
    sector: str
    growthScore: float
    sentiment: str
    forecastPercent: float


@router.get("/stocks", response_model=List[dict])
async def get_all_stocks():
    """Get list of all supported stocks"""
    return market_service.get_all_stocks()


@router.get("/stocks/quotes", response_model=List[StockQuote])
async def get_stock_quotes(
    symbols: str = "RELIANCE,TCS,HDFCBANK,INFY,ICICIBANK",
    authorization: Optional[str] = Header(None)
):
    """
    Fetch real-time quotes for multiple stocks.
    Uses Upstox if token provided, else falls back to Yahoo Finance.
    """
    access_token = None
    if authorization and authorization.startswith("Bearer "):
        access_token = authorization.split(" ")[1]
    
    symbol_list = [s.strip() for s in symbols.split(",")]
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
    interval: str = "1minute",
    days: int = 7,
    authorization: Optional[str] = Header(None)
):
    """Fetch historical candle data for a stock"""
    access_token = None
    if authorization and authorization.startswith("Bearer "):
        access_token = authorization.split(" ")[1]
    
    stock_info = market_service.get_stock_info(symbol.upper())
    if not stock_info:
        raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")
    
    candles = await market_service.get_historical_data(
        access_token, 
        stock_info["instrument_key"],
        interval,
        days
    )
    
    return [CandleData(**c) for c in candles]


@router.get("/stocks/with-sentiment", response_model=List[StockWithSentiment])
async def get_stocks_with_sentiment(
    authorization: Optional[str] = Header(None)
):
    """Get all stocks with AI sentiment analysis"""
    access_token = None
    if authorization and authorization.startswith("Bearer "):
        access_token = authorization.split(" ")[1]
    
    all_stocks = market_service.get_all_stocks()
    instrument_keys = [s["instrument_key"] for s in all_stocks]
    
    # Fetch quotes (handles fallback)
    quotes = await market_service.get_market_quote(access_token, instrument_keys)
    
    # Process sentiment in parallel with timeout
    async def get_stock_with_sentiment(stock):
        """Fetch sentiment for a single stock with timeout"""
        symbol = stock["symbol"]
        company_name = stock.get("name", symbol)
        quote_data = quotes.get(stock["instrument_key"], {})
        ohlc = quote_data.get("ohlc", {})
        
        ltp = quote_data.get("last_price", 0) or 0
        prev_close = ohlc.get("close", ltp) or ltp
        change = ltp - prev_close if prev_close else 0
        change_pct = (change / prev_close * 100) if prev_close else 0
        
        # Default sentiment (neutral) with timeout
        avg_sentiment = 0.0
        rag_confidence = 0.5
        rag_risk = "medium"
        
        try:
            # Try to get RAG-powered sentiment with 5 second timeout
            rag_result = await asyncio.wait_for(
                asyncio.to_thread(news_service.get_sentiment, symbol, company_name),
                timeout=5.0
            )
            avg_sentiment = rag_result.get("sentiment_score", 0.0)
            rag_confidence = rag_result.get("confidence", 0.5)
            rag_risk = rag_result.get("risk_level", "MEDIUM").lower()
        except asyncio.TimeoutError:
            logger.warning(f"Sentiment timeout for {symbol}, using default neutral")
        except Exception as e:
            logger.warning(f"RAG sentiment failed for {symbol}: {e}")
        
        # Determine sentiment category
        if avg_sentiment > 0.2:
            sentiment = "bullish"
        elif avg_sentiment < -0.2:
            sentiment = "bearish"
        else:
            sentiment = "neutral"
        
        # Use RAG risk level or derive from volatility
        risk_level = rag_risk if rag_risk in ("low", "medium", "high") else "medium"
        
        # Forecast confidence from RAG or derived
        confidence = min(95, max(50, rag_confidence * 100))
        
        return StockWithSentiment(
            symbol=symbol,
            name=stock["name"],
            sector=stock["sector"],
            price=ltp,
            change=change,
            changePercent=change_pct,
            forecastConfidence=confidence,
            sentiment=sentiment,
            riskLevel=risk_level
        )
    
    # Fetch sentiment for all stocks in parallel (with limits to avoid overwhelming resources)
    result = []
    batch_size = 10  # Process 10 stocks at a time
    for i in range(0, len(all_stocks), batch_size):
        batch = all_stocks[i:i+batch_size]
        try:
            batch_results = await asyncio.gather(
                *[get_stock_with_sentiment(stock) for stock in batch],
                return_exceptions=True
            )
            result.extend([r for r in batch_results if not isinstance(r, Exception)])
        except Exception as e:
            logger.error(f"Error processing sentiment batch: {e}")
    
    return result


@router.get("/indices", response_model=List[IndexQuote])
async def get_indices(
    authorization: Optional[str] = Header(None)
):
    """Fetch market indices (NIFTY, SENSEX, etc.)"""
    access_token = None
    if authorization and authorization.startswith("Bearer "):
        access_token = authorization.split(" ")[1]
    
    all_indices = market_service.get_all_indices()
    
    instrument_keys = [idx["instrument_key"] for idx in all_indices]
    
    # Fetch quotes (handles fallback)
    quotes = await market_service.get_market_quote(access_token, instrument_keys)
    
    result = []
    for idx in all_indices:
        quote_data = quotes.get(idx["instrument_key"], {})
        ohlc = quote_data.get("ohlc", {})
        
        ltp = quote_data.get("last_price", 0) or 0
        prev_close = ohlc.get("close", ltp) or ltp
        change = ltp - prev_close if prev_close else 0
        change_pct = (change / prev_close * 100) if prev_close else 0
        
        result.append(IndexQuote(
            symbol=idx["symbol"],
            name=idx["name"],
            price=ltp,
            change=change,
            changePercent=change_pct
        ))
    
    return result


@router.get("/leaderboard", response_model=List[LeaderboardStock])
async def get_leaderboard(
    authorization: Optional[str] = Header(None)
):
    """Get stocks ranked by growth potential with AI analysis"""
    stocks_with_sentiment = await get_stocks_with_sentiment(authorization)
    
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

@router.get("/stocks/{symbol}/analysis", response_model=dict)
async def get_stock_analysis(
    symbol: str,
    authorization: Optional[str] = Header(None)
):
    """Get comprehensive AI analysis for a specific stock"""
    try:
        # Get basic stock info
        stock_info = market_service.get_stock_info(symbol)
        if not stock_info:
            raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")
        
        # Get current price
        access_token = None
        if authorization and authorization.startswith("Bearer "):
            access_token = authorization.split(" ")[1]
        
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
            rag_result = await asyncio.wait_for(
                asyncio.to_thread(news_service.get_sentiment, symbol, company_name),
                timeout=10.0
            )
            sentiment_score = rag_result.get("sentiment_score", 0.0)
            sentiment_confidence = rag_result.get("confidence", 0.5)
            sentiment_reasoning = rag_result.get("reasoning", "Analysis based on recent news and market sentiment")
        except (asyncio.TimeoutError, Exception) as e:
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
        except:
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
        
        # Predicted price and recommendation
        if sentiment == "bullish":
            predicted_price = ltp * 1.08
            short_term = "bullish"
            long_term = "bullish"
            recommendation = "buy"
        elif sentiment == "bearish":
            predicted_price = ltp * 0.92
            short_term = "bearish"
            long_term = "bearish"
            recommendation = "sell"
        else:
            predicted_price = ltp
            short_term = "neutral"
            long_term = "neutral"
            recommendation = "hold"
        
        return {
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
            "forecastConfidence": round(min(99, sentiment_confidence * 100), 1),
            "shortTermOutlook": short_term,
            "longTermOutlook": long_term,
            "recommendation": recommendation,
            "recentNews": recent_news,
            "newsCount": len(recent_news)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing stock {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing stock: {str(e)}")