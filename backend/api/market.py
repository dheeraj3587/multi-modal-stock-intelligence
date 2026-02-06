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
    
    result = []
    for stock in all_stocks:
        symbol = stock["symbol"]
        company_name = stock.get("name", symbol)
        quote_data = quotes.get(stock["instrument_key"], {})
        ohlc = quote_data.get("ohlc", {})
        
        ltp = quote_data.get("last_price", 0) or 0
        prev_close = ohlc.get("close", ltp) or ltp
        change = ltp - prev_close if prev_close else 0
        change_pct = (change / prev_close * 100) if prev_close else 0
        
        # Get RAG-powered sentiment from news (RSS feeds + Gemini AI)
        try:
            rag_result = news_service.get_sentiment(symbol, company_name)
            avg_sentiment = rag_result.get("sentiment_score", 0.0)
            rag_confidence = rag_result.get("confidence", 0.5)
            rag_risk = rag_result.get("risk_level", "MEDIUM").lower()
        except Exception as e:
            logger.warning(f"RAG sentiment failed for {symbol}: {e}, falling back to VADER")
            articles = news_service.get_company_news(symbol)
            total_sentiment = 0
            for article in articles[:5]:
                title = article.get('title', '')
                total_sentiment += analyzer.get_sentiment(title)
            avg_sentiment = total_sentiment / len(articles) if articles else 0
            rag_confidence = 0.5
            rag_risk = "medium"
        
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
        
        result.append(StockWithSentiment(
            symbol=symbol,
            name=stock["name"],
            sector=stock["sector"],
            price=ltp,
            change=change,
            changePercent=change_pct,
            forecastConfidence=confidence,
            sentiment=sentiment,
            riskLevel=risk_level
        ))
    
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
