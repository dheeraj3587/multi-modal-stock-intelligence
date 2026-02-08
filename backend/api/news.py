from fastapi import APIRouter, HTTPException
from typing import List, Dict, Optional
from pydantic import BaseModel
from backend.services.news_service import news_service
from backend.services.sentiment_analyzer import analyzer
from backend.services.data_cache import data_cache
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

class NewsItem(BaseModel):
    title: str
    source: str
    url: str
    published_at: str
    sentiment_score: Optional[float] = 0.0
    description: Optional[str] = ""

class SentimentResponse(BaseModel):
    symbol: str
    overall_sentiment: float
    confidence: Optional[float] = 0.5
    reasoning: Optional[str] = ""
    key_themes: Optional[str] = ""
    risk_level: Optional[str] = "MEDIUM"
    article_count: int
    news: List[NewsItem]
    model: Optional[str] = ""

@router.get("/news/{symbol}", response_model=SentimentResponse)
async def get_stock_sentiment(symbol: str, company_name: Optional[str] = None):
    """
    Fetches news for a stock and calculates AI-powered sentiment using RAG.
    Cache-first: returns from Redis if available, otherwise fetches live.
    """
    # Clean symbol (remove .NS if present for news search)
    clean_symbol = symbol.split('.')[0].upper()
    
    # 1. Try cached news + cached sentiment for instant response
    cached_news = data_cache.get_news(clean_symbol)
    cached_sentiment = news_service.get_sentiment_cached_only(clean_symbol)
    is_stale = cached_sentiment.get("is_stale", True)

    if cached_news and not is_stale:
        processed_news = []
        for article in cached_news[:10]:
            title = article.get('title', '')
            article_score = analyzer.get_sentiment(title)
            processed_news.append(NewsItem(
                title=title,
                source=article.get('source', 'Unknown'),
                url=article.get('url', '#'),
                published_at=article.get('published_at', ''),
                sentiment_score=article_score,
                description=article.get('description', '')[:200]
            ))

        return SentimentResponse(
            symbol=symbol,
            overall_sentiment=cached_sentiment.get('sentiment_score', 0.0),
            confidence=cached_sentiment.get('confidence', 0.5),
            reasoning=cached_sentiment.get('reasoning', ''),
            key_themes=cached_sentiment.get('key_themes', ''),
            risk_level=cached_sentiment.get('risk_level', 'MEDIUM'),
            article_count=cached_sentiment.get('article_count', len(cached_news)),
            news=processed_news,
            model=cached_sentiment.get('model', 'cached')
        )

    # 2. Fallback to live fetch
    articles = news_service.get_company_news(clean_symbol, company_name or clean_symbol)
    
    # Cache the articles
    if articles:
        data_cache.set_news(clean_symbol, articles, ttl=1800)

    # Get RAG-powered sentiment analysis
    sentiment_result = news_service.get_sentiment(clean_symbol, company_name or clean_symbol)
    
    processed_news = []
    
    for article in articles[:10]:
        title = article.get('title', '')
        article_score = analyzer.get_sentiment(title) if not sentiment_result.get('error') else 0.0
        
        processed_news.append(NewsItem(
            title=title,
            source=article.get('source', 'Unknown'),
            url=article.get('url', '#'),
            published_at=article.get('published_at', ''),
            sentiment_score=article_score,
            description=article.get('description', '')[:200]
        ))
    
    return SentimentResponse(
        symbol=symbol,
        overall_sentiment=sentiment_result.get('sentiment_score', 0.0),
        confidence=sentiment_result.get('confidence', 0.5),
        reasoning=sentiment_result.get('reasoning', ''),
        key_themes=sentiment_result.get('key_themes', ''),
        risk_level=sentiment_result.get('risk_level', 'MEDIUM'),
        article_count=sentiment_result.get('article_count', len(articles)),
        news=processed_news,
        model=sentiment_result.get('model', 'vader')
    )

