import os
from newsapi import NewsApiClient
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging

# Import new RAG-based services
from backend.services.rss_news_fetcher import rss_fetcher
from backend.services.vector_store import create_vector_store
from backend.services.rag_sentiment_analyzer import create_sentiment_analyzer
from backend.services.gemini_file_search import GeminiFileSearchRAG
from backend.services.model_validator import ModelValidator

# Configure logger
logger = logging.getLogger(__name__)

class NewsService:
    def __init__(self, use_rag: bool = True):
        """
        Initialize news service.
        
        Args:
            use_rag: If True, use RSS + RAG pipeline. If False, use legacy NewsAPI.
        """
        self.use_rag = use_rag
        
        # Legacy NewsAPI setup (fallback)
        self.api_key = os.getenv("NEWSAPI_KEY")
        if not self.api_key:
            logger.warning("NEWSAPI_KEY not found - using RSS-based news fetching")
            self.client = None
            self.use_rag = True  # Force RAG mode if no API key
        else:
            self.client = NewsApiClient(api_key=self.api_key)
        
        self.cache = {}
        self.cache_expiry = timedelta(hours=2)
        
        # RAG components (lazy initialization)
        self._vector_store = None
        self._sentiment_analyzer = None
        self._rag_initialized = False
        self._file_search_rag = None
        self._use_managed_rag = os.getenv("USE_GEMINI_FILE_SEARCH", "false").lower() == "true"

    def _initialize_rag(self):
        """Lazy initialization of RAG components."""
        if self._rag_initialized:
            return
        
        try:
            # Validate model availability BEFORE initializing RAG
            gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if gemini_key and self._use_managed_rag:
                validator = ModelValidator(gemini_api_key=gemini_key)
                available_models = validator.validate_gemini_models()
                
                gemini_model = os.getenv("GEMINI_MODEL", "gemini-3-flash")
                if not validator.validate_model_name(gemini_model, available_models):
                    logger.warning(
                        f"Requested model '{gemini_model}' not available. "
                        "Attempting to use it anyway (may fail at runtime)."
                    )
            
            # Determine embedding model type from environment
            embedding_model = os.getenv("EMBEDDING_MODEL_TYPE", "local").lower()
            
            # Fallback logic if no keys available
            if embedding_model == "openai" and not os.getenv("OPENAI_API_KEY"):
                embedding_model = "local"
            if embedding_model == "huggingface" and not (os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")):
                embedding_model = "local"
            if embedding_model == "gemini" and not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
                embedding_model = "local"
            
            gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            gemini_model = os.getenv("GEMINI_MODEL", "gemini-3-flash")

            if self._use_managed_rag and gemini_key:
                cache_hours = int(os.getenv("NEWS_CACHE_HOURS", "2"))
                self._file_search_rag = GeminiFileSearchRAG(
                    api_key=gemini_key,
                    model_name=gemini_model,
                    cache_hours=cache_hours
                )
                logger.info("Gemini File Search enabled")

            # Create vector store with persistence
            data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'vector_store')
            self._vector_store = create_vector_store(
                model_type=embedding_model,
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                huggingface_token=os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN"),
                huggingface_model=os.getenv("HUGGINGFACE_MODEL", "google/embeddinggemma-300m"),
                gemini_api_key=gemini_key,
                persist_dir=data_dir
            )

            # Create sentiment analyzer
            self._sentiment_analyzer = create_sentiment_analyzer()

            self._rag_initialized = True
            logger.info(f"RAG components initialized successfully with {embedding_model} embeddings")
        except Exception as e:
            logger.error(f"Failed to initialize RAG components: {e}")
            self.use_rag = False

    def get_company_news(self, symbol: str, company_name: str = "") -> List[Dict]:
        """
        Fetches news for a specific company or symbol.
        Uses RSS + RAG if enabled, otherwise falls back to NewsAPI.
        """
        if self.use_rag:
            return self._get_company_news_rag(symbol, company_name)
        else:
            return self._get_company_news_legacy(symbol, company_name)
    
    def _get_company_news_rag(self, symbol: str, company_name: str = "") -> List[Dict]:
        """Fetch news using RSS feeds (no rate limits)."""
        # Check cache
        cache_key = f"rag_{symbol}"
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if datetime.now() - timestamp < self.cache_expiry:
                return cached_data
        
        try:
            # Fetch from RSS feeds
            articles = rss_fetcher.fetch_company_news(symbol, company_name, max_articles=20)
            
            # Update cache
            self.cache[cache_key] = (articles, datetime.now())
            
            return articles
        except Exception as e:
            logger.error(f"Error fetching RSS news for {symbol}: {e}")
            return []
    
    def _get_company_news_legacy(self, symbol: str, company_name: str = "") -> List[Dict]:
        """Legacy NewsAPI implementation."""
        if not self.client:
            return []

        # Check cache
        if symbol in self.cache:
            cached_data, timestamp = self.cache[symbol]
            if datetime.now() - timestamp < self.cache_expiry:
                return cached_data

        query = f"{symbol} stock"
        if company_name:
            query = f"{company_name} OR {symbol}"

        try:
            from_date = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')
            
            response = self.client.get_everything(
                q=query,
                from_param=from_date,
                language='en',
                sort_by='publishedAt',
                page_size=10
            )

            if response['status'] == 'ok':
                articles = response['articles']
                valid_articles = [
                    a for a in articles 
                    if a.get('title') and a.get('source', {}).get('name') != "[Removed]"
                ]
                
                self.cache[symbol] = (valid_articles, datetime.now())
                return valid_articles
            return []
            
        except Exception as e:
            error_msg = str(e)
            if "rateLimited" in error_msg:
                logger.warning(f"NewsAPI rate limit reached for {symbol}. Returning cached/empty.")
                if symbol in self.cache:
                    return self.cache[symbol][0]
            else:
                logger.error(f"Error fetching news for {symbol}: {e}")
            return []
    
    def get_sentiment(self, symbol: str, company_name: str = "") -> Dict:
        """
        Get AI-powered sentiment analysis for a company using RAG.
        
        Args:
            symbol: Stock symbol
            company_name: Company name
            
        Returns:
            Sentiment analysis dict with score, confidence, reasoning, etc.
        """
        if not self.use_rag:
            # Fallback to simple sentiment
            from backend.services.sentiment_analyzer import analyzer
            articles = self.get_company_news(symbol, company_name)
            if not articles:
                return {"sentiment_score": 0.0, "confidence": 0.0}
            
            combined_text = " ".join([
                f"{a.get('title', '')} {a.get('description', '')}" 
                for a in articles[:5]
            ])
            score = analyzer.get_sentiment(combined_text)
            return {"sentiment_score": score, "confidence": 0.5}
        
        # Initialize RAG if needed
        self._initialize_rag()
        if not self._rag_initialized:
            return {"sentiment_score": 0.0, "confidence": 0.0, "error": "RAG not available"}
        
        try:
            # Get recent articles
            articles = self.get_company_news(symbol, company_name)
            
            if not articles:
                return {
                    "sentiment_score": 0.0,
                    "confidence": 0.0,
                    "reasoning": "No news articles available",
                    "article_count": 0
                }
            
            if self._use_managed_rag and self._file_search_rag:
                sentiment_result = self._file_search_rag.analyze_sentiment(
                    symbol=symbol,
                    company_name=company_name,
                    articles=articles
                )
                if "error" not in sentiment_result:
                    return sentiment_result
                if not self._vector_store:
                    return sentiment_result

            # Add articles to vector store for context
            if not self._vector_store:
                return self._default_sentiment_fallback("Vector store unavailable")

            self._vector_store.add_documents(articles, text_key='content')

            # Search for similar historical articles
            query = f"{company_name} {symbol} stock market news"
            context_articles = self._vector_store.similarity_search(query, k=5, score_threshold=0.6)

            # Analyze sentiment with RAG
            sentiment_result = self._sentiment_analyzer.analyze_sentiment(
                company_name=company_name,
                symbol=symbol,
                articles=articles,
                context_articles=context_articles
            )

            # Save vector store periodically
            if len(self._vector_store) % 50 == 0:
                self._vector_store.save()

            return sentiment_result
            
        except Exception as e:
            logger.error(f"Error in RAG sentiment analysis for {symbol}: {e}")
            return {
                "sentiment_score": 0.0,
                "confidence": 0.0,
                "error": str(e),
                "article_count": 0
            }

    def _default_sentiment_fallback(self, reason: str) -> Dict:
        return {
            "sentiment_score": 0.0,
            "confidence": 0.0,
            "reasoning": reason,
            "article_count": 0
        }
    
    def get_batch_sentiment(self, stocks: List[Dict]) -> Dict[str, Dict]:
        """
        Get sentiment for multiple stocks efficiently.
        
        Args:
            stocks: List of dicts with 'symbol' and 'company_name' keys
            
        Returns:
            Dict mapping symbol to sentiment analysis
        """
        if not self.use_rag or not self._rag_initialized:
            self._initialize_rag()
        
        results = {}
        for stock in stocks:
            symbol = stock.get('symbol', '')
            company_name = stock.get('company_name', stock.get('name', ''))
            results[symbol] = self.get_sentiment(symbol, company_name)
        
        return results

# Singleton instance
news_service = NewsService(use_rag=True)

