"""
RAG-based sentiment analyzer using AI models with vector search context.
Supports both OpenAI and local models (via Ollama).
"""

import os
import json
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class RAGSentimentAnalyzer:
    """Sentiment analyzer using RAG (Retrieval Augmented Generation)."""
    
    def __init__(
        self,
        model_type: str = "local",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """
        Initialize RAG sentiment analyzer.
        
        Args:
            model_type: "openai", "anthropic", "gemini", or "local" (Ollama)
            model_name: Specific model name (e.g., "gpt-4", "llama3", "claude-3-5-sonnet")
            api_key: API key for cloud models
        """
        self.model_type = model_type
        self.model_name = model_name
        self.api_key = api_key or os.getenv(f"{model_type.upper()}_API_KEY")
        
        # Initialize appropriate client
        if model_type == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
            self.model_name = model_name or "gpt-4o-mini"
            logger.info(f"Initialized OpenAI with model {self.model_name}")
            
        elif model_type == "anthropic":
            from anthropic import Anthropic
            self.client = Anthropic(api_key=self.api_key)
            self.model_name = model_name or "claude-3-5-sonnet-20241022"
            logger.info(f"Initialized Anthropic with model {self.model_name}")
            
        elif model_type == "gemini":
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
            self.model_name = model_name or "gemini-3-flash-preview"
            logger.info(f"Initialized Gemini with model {self.model_name}")
            
        elif model_type == "local":
            # Use Ollama for local inference
            try:
                import ollama
                self.client = ollama
                self.model_name = model_name or "llama3.2"
                logger.info(f"Initialized Ollama with model {self.model_name}")
            except ImportError:
                logger.warning("Ollama not available. Install with: pip install ollama")
                self.client = None
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
    
    def _create_sentiment_prompt(
        self,
        company_name: str,
        symbol: str,
        articles: List[Dict],
        context_articles: List[Tuple[Dict, float]]
    ) -> str:
        """
        Create a detailed prompt for sentiment analysis.
        
        Args:
            company_name: Company name
            symbol: Stock symbol
            articles: Recent articles
            context_articles: Similar articles from vector search with scores
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""You are a financial sentiment analyst. Analyze the sentiment for {company_name} ({symbol}) based on recent news articles.

## Recent News Articles:
"""
        
        for i, article in enumerate(articles[:5], 1):
            prompt += f"{i}. **{article.get('title', 'No title')}**\n"
            prompt += f"   Source: {article.get('source', 'Unknown')}\n"
            prompt += f"   Date: {article.get('published_at', 'Unknown')}\n"
            if article.get('description'):
                prompt += f"   Summary: {article['description'][:200]}...\n"
            prompt += "\n"
        
        if context_articles:
            prompt += "\n## Historical Context (Similar Past Articles):\n"
            for i, (article, score) in enumerate(context_articles[:3], 1):
                prompt += f"{i}. {article.get('title', 'No title')} (relevance: {score:.2f})\n"
                if article.get('description'):
                    prompt += f"   {article['description'][:150]}...\n\n"
        
        prompt += """
## Task:
Analyze the overall sentiment and provide:
1. **Sentiment Score**: A number between -1.0 (very negative) and +1.0 (very positive)
2. **Confidence**: How confident you are (0.0 to 1.0)
3. **Reasoning**: Brief explanation (2-3 sentences)
4. **Key Themes**: Main topics/concerns (comma-separated)
5. **Risk Level**: LOW, MEDIUM, or HIGH based on negative news

Respond ONLY with valid JSON in this exact format:
{
    "sentiment_score": 0.0,
    "confidence": 0.0,
    "reasoning": "string",
    "key_themes": "string",
    "risk_level": "MEDIUM"
}
"""
        return prompt
    
    def _parse_llm_response(self, response: str) -> Dict:
        """Parse LLM response and extract JSON."""
        try:
            # Try to find JSON in the response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            else:
                # Couldn't find JSON, return default
                return self._default_sentiment_response()
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM response as JSON")
            return self._default_sentiment_response()
    
    def _default_sentiment_response(self) -> Dict:
        """Return default neutral sentiment response."""
        return {
            "sentiment_score": 0.0,
            "confidence": 0.3,
            "reasoning": "Unable to determine sentiment from available data.",
            "key_themes": "insufficient data",
            "risk_level": "MEDIUM"
        }
    
    def analyze_sentiment(
        self,
        company_name: str,
        symbol: str,
        articles: List[Dict],
        context_articles: Optional[List[Tuple[Dict, float]]] = None
    ) -> Dict:
        """
        Analyze sentiment using RAG approach.
        
        Args:
            company_name: Company name
            symbol: Stock symbol
            articles: Recent news articles
            context_articles: Historical context from vector search
            
        Returns:
            Sentiment analysis dict with score, confidence, reasoning, etc.
        """
        if not articles:
            return {
                "sentiment_score": 0.0,
                "confidence": 0.0,
                "reasoning": "No news articles available for analysis.",
                "key_themes": "no data",
                "risk_level": "MEDIUM",
                "article_count": 0
            }
        
        context_articles = context_articles or []
        prompt = self._create_sentiment_prompt(company_name, symbol, articles, context_articles)
        
        try:
            if self.model_type == "openai":
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a financial sentiment analysis expert. Always respond with valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=500
                )
                result_text = response.choices[0].message.content
                
            elif self.model_type == "anthropic":
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=500,
                    temperature=0.3,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                result_text = response.content[0].text
                
            elif self.model_type == "gemini":
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt
                )
                result_text = getattr(response, "text", "")
                
            elif self.model_type == "local" and self.client:
                response = self.client.chat(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a financial sentiment analyst. Respond only with valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    options={"temperature": 0.3}
                )
                result_text = response['message']['content']
            else:
                return self._default_sentiment_response()
            
            # Parse the response
            result = self._parse_llm_response(result_text)
            
            # Add metadata
            result['article_count'] = len(articles)
            result['analyzed_at'] = datetime.now().isoformat()
            result['model'] = f"{self.model_type}:{self.model_name}"
            
            logger.info(f"Analyzed sentiment for {symbol}: {result['sentiment_score']:.2f} (confidence: {result['confidence']:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return {
                **self._default_sentiment_response(),
                "article_count": len(articles),
                "error": str(e)
            }
    
    def analyze_batch(
        self,
        stocks: List[Dict[str, str]],
        articles_dict: Dict[str, List[Dict]],
        context_dict: Optional[Dict[str, List[Tuple[Dict, float]]]] = None
    ) -> Dict[str, Dict]:
        """
        Analyze sentiment for multiple stocks.
        
        Args:
            stocks: List of dicts with 'symbol' and 'company_name' keys
            articles_dict: Dict mapping symbol to articles list
            context_dict: Dict mapping symbol to context articles
            
        Returns:
            Dict mapping symbol to sentiment analysis
        """
        results = {}
        context_dict = context_dict or {}
        
        for stock in stocks:
            symbol = stock['symbol']
            company_name = stock['company_name']
            articles = articles_dict.get(symbol, [])
            context = context_dict.get(symbol, [])
            
            results[symbol] = self.analyze_sentiment(
                company_name=company_name,
                symbol=symbol,
                articles=articles,
                context_articles=context
            )
        
        return results


# Factory function
def create_sentiment_analyzer(
    model_type: str = None,
    model_name: str = None,
    api_key: str = None
) -> RAGSentimentAnalyzer:
    """
    Create a sentiment analyzer with the specified configuration.
    
    Args:
        model_type: "openai", "anthropic", "gemini", or "local"
        model_name: Specific model name
        api_key: API key for cloud models
        
    Returns:
        RAGSentimentAnalyzer instance
    """
    # Read from env first, then auto-detect
    if not model_type:
        model_type = os.getenv("AI_MODEL_TYPE", "").lower()
    
    if not model_type:
        # Auto-detect from available API keys
        if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
            model_type = "gemini"
        elif os.getenv("OPENAI_API_KEY"):
            model_type = "openai"
        elif os.getenv("ANTHROPIC_API_KEY"):
            model_type = "anthropic"
        else:
            model_type = "local"
            logger.info("No API keys found, using local model (Ollama)")
    
    # Read model name from env if not provided
    if not model_name:
        model_name = os.getenv(f"{model_type.upper()}_MODEL")
    
    # Read API key from env if not provided
    if not api_key:
        api_key = os.getenv(f"{model_type.upper()}_API_KEY")
        if model_type == "gemini" and not api_key:
            api_key = os.getenv("GOOGLE_API_KEY")
    
    logger.info(f"Creating sentiment analyzer: type={model_type}, model={model_name}")
    return RAGSentimentAnalyzer(model_type=model_type, model_name=model_name, api_key=api_key)
