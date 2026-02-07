#!/usr/bin/env python3
"""
Bootstrap Script: Pre-populate Redis with sentiment data for all stocks.
Run this ONCE after starting the containers to initialize the cache.

Usage:
    docker exec stock-intelligence-backend python -m backend.scripts.bootstrap_sentiments
    OR
    python -m backend.scripts.bootstrap_sentiments  (if running locally)
"""

import os
import sys
import asyncio
import logging
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

from backend.services.market_service import market_service
from backend.services.news_service import news_service
from backend.services.redis_cache import redis_cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def bootstrap_sentiments():
    """
    Fetch sentiment for all stocks and store in Redis.
    This should be run once on initial deployment.
    """
    logger.info("=" * 60)
    logger.info("🚀 SENTIMENT BOOTSTRAP STARTING")
    logger.info("=" * 60)
    
    # Check Redis connection
    if not redis_cache.is_connected():
        logger.error("❌ Redis is not connected! Please ensure Redis is running.")
        logger.error("   Try: docker-compose up -d redis")
        return False
    
    logger.info("✅ Redis connection verified")
    
    # Get all stocks
    all_stocks = market_service.get_all_stocks()
    logger.info(f"📊 Found {len(all_stocks)} stocks to analyze")
    
    # Check which stocks already have cached sentiment
    existing = redis_cache.get_all_sentiments([s['symbol'] for s in all_stocks])
    already_cached = [s for s in all_stocks if s['symbol'] in existing]
    to_analyze = [s for s in all_stocks if s['symbol'] not in existing]
    
    logger.info(f"   - Already cached: {len(already_cached)}")
    logger.info(f"   - Need analysis: {len(to_analyze)}")
    
    if not to_analyze:
        logger.info("✅ All stocks already have cached sentiment!")
        return True
    
    # Initialize RAG components
    logger.info("🔧 Initializing RAG components...")
    news_service._initialize_rag()
    
    if not news_service._rag_initialized:
        logger.error("❌ Failed to initialize RAG. Check GEMINI_API_KEY.")
        return False
    
    logger.info("✅ RAG initialized successfully")
    
    # Analyze each stock
    success_count = 0
    failed = []
    
    # Limit concurrent requests to avoid rate limiting
    sem = asyncio.Semaphore(3)
    
    async def analyze_stock(stock):
        nonlocal success_count
        symbol = stock['symbol']
        company_name = stock.get('name', symbol)
        
        async with sem:
            try:
                logger.info(f"📈 Analyzing {symbol} ({company_name})...")
                
                # Call Gemini API for sentiment
                result = await asyncio.to_thread(
                    news_service.get_sentiment,
                    symbol,
                    company_name
                )
                
                # Store in Redis
                news_service.update_sentiment_cache(symbol, result)
                
                sentiment = result.get('sentiment_score', 0)
                confidence = result.get('confidence', 0)
                
                sentiment_label = "bullish" if sentiment > 0.2 else ("bearish" if sentiment < -0.2 else "neutral")
                
                logger.info(f"   ✓ {symbol}: {sentiment_label} ({confidence*100:.0f}% confidence)")
                success_count += 1
                
            except Exception as e:
                logger.error(f"   ✗ {symbol}: Failed - {e}")
                failed.append(symbol)
    
    # Process all stocks
    start_time = datetime.now()
    
    tasks = [analyze_stock(s) for s in to_analyze]
    await asyncio.gather(*tasks, return_exceptions=True)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    # Summary
    logger.info("=" * 60)
    logger.info("📊 BOOTSTRAP COMPLETE")
    logger.info("=" * 60)
    logger.info(f"   Total stocks: {len(all_stocks)}")
    logger.info(f"   Successfully cached: {success_count}")
    logger.info(f"   Failed: {len(failed)}")
    logger.info(f"   Time elapsed: {elapsed:.1f}s")
    
    if failed:
        logger.warning(f"   Failed symbols: {', '.join(failed)}")
    
    # Verify Redis
    stats = redis_cache.get_cache_stats()
    logger.info(f"   Redis cache now has: {stats.get('cached_symbols', 0)} sentiments")
    
    return len(failed) == 0


def main():
    """Main entry point."""
    print("\n" + "=" * 60)
    print("STOCK INTELLIGENCE - SENTIMENT BOOTSTRAP")
    print("=" * 60 + "\n")
    
    success = asyncio.run(bootstrap_sentiments())
    
    if success:
        print("\n✅ Bootstrap completed successfully!")
        print("   Your dashboard will now load instantly with real sentiment data.")
    else:
        print("\n⚠️  Bootstrap completed with some errors.")
        print("   Check the logs above for details.")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
