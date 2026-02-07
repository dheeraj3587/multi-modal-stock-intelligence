
import asyncio
import os
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock env vars if needed (assuming .env is loaded by the app environment)
# os.environ["GEMINI_API_KEY"] = "..." 

async def test_rag():
    print("🚀 Starting RAG Test...")
    
    try:
        from backend.services.news_service import news_service
        
        symbol = "RELIANCE"
        company_name = "Reliance Industries"
        
        start_time = time.time()
        print(f"📊 Fetching sentiment for {symbol}...")
        
        # Call directly without the 2s timeout wrapper from market.py
        result = await asyncio.to_thread(news_service.get_sentiment, symbol, company_name)
        
        duration = time.time() - start_time
        
        print(f"\n✅ RAG Completed in {duration:.2f} seconds")
        print(f"Sentiment Score: {result.get('sentiment_score')}")
        print(f"Confidence: {result.get('confidence')}")
        print(f"Reasoning: {result.get('reasoning')}")
        print(f"Article Count: {result.get('article_count')}")
        
        if duration > 2.0:
            print("\n⚠️  WARNING: Execution took longer than the 2.0s backend timeout!")
            
    except Exception as e:
        print(f"\n❌ RAG Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Ensure we run in an async context
    asyncio.run(test_rag())
