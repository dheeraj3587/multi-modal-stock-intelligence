from fastapi import APIRouter, HTTPException
from typing import List, Dict, Optional
from pydantic import BaseModel
from backend.services.news_service import news_service
from backend.services.sentiment_analyzer import analyzer
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
    Symbol format expected: 'RELIANCE' or 'RELIANCE.NS'
    """
    # Clean symbol (remove .NS if present for news search)
    clean_symbol = symbol.split('.')[0]
    
    # Fetch news articles
    articles = news_service.get_company_news(clean_symbol, company_name or clean_symbol)
    
    # Get RAG-powered sentiment analysis
    sentiment_result = news_service.get_sentiment(clean_symbol, company_name or clean_symbol)
    
    processed_news = []
    
    for article in articles[:10]:  # Limit to top 10 for response
        title = article.get('title', '')
        # Use simple VADER for individual article scores
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

