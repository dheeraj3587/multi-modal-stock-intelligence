"""RAG-based sentiment analyzer using AI models with vector search."""

import os
import json
import time
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Timeout configurations (increased for LLM calls)
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "30"))  # 30s default
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))  # 2 retries
LLM_RETRY_DELAY = float(os.getenv("LLM_RETRY_DELAY", "2.0"))  # 2s initial delay


class RAGSentimentAnalyzer:
    """Sentiment analyzer using RAG (Retrieval Augmented Generation)."""
    
    def __init__(
        self,
        model_type: str = "local",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """Initialize with model_type: 'openai', 'anthropic', 'gemini', or 'local'."""
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
        """Create prompt for sentiment analysis."""
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
                result = json.loads(json_str)
                
                # Normalize confidence: if > 1.0, LLM returned as percentage (e.g., 72.5 instead of 0.725)
                confidence = float(result.get("confidence", 0.5))
                if confidence > 1.0:
                    confidence = confidence / 100.0
                result["confidence"] = max(0.0, min(1.0, confidence))  # Clamp to 0-1
                
                # Normalize sentiment_score to -1.0 to 1.0 range
                score = float(result.get("sentiment_score", 0.0))
                result["sentiment_score"] = max(-1.0, min(1.0, score))
                
                return result
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
        """Analyze sentiment using RAG approach with retry logic."""
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
        
        # Attempt analysis with retries and exponential backoff
        last_error = None
        for attempt in range(LLM_MAX_RETRIES + 1):
            try:
                result_text = self._call_llm(prompt, attempt)
                
                # Parse the response
                result = self._parse_llm_response(result_text)
                
                # Add metadata
                result['article_count'] = len(articles)
                result['analyzed_at'] = datetime.now().isoformat()
                result['model'] = f"{self.model_type}:{self.model_name}"
                
                logger.info(f"Analyzed sentiment for {symbol}: {result['sentiment_score']:.2f} (confidence: {result['confidence']:.2f})")
                return result
                
            except Exception as e:
                last_error = e
                if attempt < LLM_MAX_RETRIES:
                    delay = LLM_RETRY_DELAY * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Sentiment analysis attempt {attempt + 1} failed for {symbol}: {e}. Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"Sentiment analysis failed for {symbol} after {LLM_MAX_RETRIES + 1} attempts: {e}")
        
        # Return default response if all retries failed
        return {
            **self._default_sentiment_response(),
            "article_count": len(articles),
            "error": str(last_error)
        }
    
    def _call_llm(self, prompt: str, attempt: int = 0) -> str:
        """Call LLM with appropriate timeout and error handling."""
        timeout = LLM_TIMEOUT + (attempt * 10)  # Increase timeout on retries
        
        if self.model_type == "openai":
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a financial sentiment analysis expert. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500,
                timeout=timeout,
            )
            return response.choices[0].message.content
            
        elif self.model_type == "anthropic":
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=500,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
                timeout=timeout,
            )
            return response.content[0].text
            
        elif self.model_type == "gemini":
            # Gemini SDK may not support timeout directly, wrap in threading
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    self.client.models.generate_content,
                    model=self.model_name,
                    contents=prompt
                )
                try:
                    response = future.result(timeout=timeout)
                    return getattr(response, "text", "")
                except concurrent.futures.TimeoutError:
                    raise TimeoutError(f"Gemini API call timed out after {timeout}s")
            
        elif self.model_type == "local" and self.client:
            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a financial sentiment analyst. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                options={"temperature": 0.3},
            )
            return response['message']['content']
        else:
            raise ValueError(f"No valid LLM client configured for {self.model_type}")
    
    def analyze_batch(
        self,
        stocks: List[Dict[str, str]],
        articles_dict: Dict[str, List[Dict]],
        context_dict: Optional[Dict[str, List[Tuple[Dict, float]]]] = None
    ) -> Dict[str, Dict]:
        """Analyze sentiment for multiple stocks in a single LLM call."""
        if not stocks:
            return {}

        context_dict = context_dict or {}
        
        # Prepare the batch prompt
        prompt = """You are a financial sentiment analyst.
For each stock listed below, analyze the provided news headlines and return a sentiment score (-1.0 to 1.0) and confidence (0.0 to 1.0).

STOCKS DATA:
"""
        
        valid_stocks = []
        
        for i, stock in enumerate(stocks, 1):
            symbol = stock.get('symbol')
            company = stock.get('company_name', symbol)
            articles = articles_dict.get(symbol, [])
            
            # Skip if no articles
            if not articles:
                continue
                
            valid_stocks.append(symbol)
            
            prompt += f"\n=== {i}. {company} ({symbol}) ===\n"
            prompt += "News Headlines:\n"
            for art in articles[:4]:  # Top 4 articles only to save tokens
                title = art.get('title', 'No title').replace('\n', ' ')
                source = art.get('source', 'Unknown')
                prompt += f"- {title} [{source}]\n"
                
            # Add context if available
            context = context_dict.get(symbol, [])
            if context:
                prompt += "Historical Context:\n"
                for art, score in context[:2]:
                    prompt += f"- {art.get('title', '')} (relevance: {score:.2f})\n"

        if not valid_stocks:
            return {}

        prompt += """
\nINSTRUCTIONS:
For each stock above, determine:
1. Sentiment Score (-1.0 to 1.0)
2. Confidence (0.0 to 1.0)
3. Risk Level (LOW, MEDIUM, HIGH)

Return ONLY valid JSON mapping stock symbol to result.
Example Format:
{
  "RELIANCE": {"sentiment_score": 0.5, "confidence": 0.8, "risk_level": "LOW"},
  "TCS": {"sentiment_score": -0.2, "confidence": 0.6, "risk_level": "MEDIUM"}
}
"""

        # Call LLM
        results = {}
        try:
            logger.info(f"Batch analyzing sentiment for {len(valid_stocks)} stocks")
            response_text = self._call_llm(prompt)
            parsed = self._parse_llm_response(response_text)
            
            # Validate and format results
            for symbol in valid_stocks:
                if symbol in parsed:
                    data = parsed[symbol]
                    # Normalize confidence: if > 1.0, LLM returned as percentage
                    conf = float(data.get("confidence", 0.5))
                    if conf > 1.0:
                        conf = conf / 100.0
                    conf = max(0.0, min(1.0, conf))
                    
                    score = float(data.get("sentiment_score", 0.0))
                    score = max(-1.0, min(1.0, score))
                    
                    results[symbol] = {
                        "sentiment_score": score,
                        "confidence": conf,
                        "risk_level": str(data.get("risk_level", "MEDIUM")),
                        "reasoning": "Batch analysis based on recent headlines",
                        "analyzed_at": datetime.now().isoformat()
                    }
                else:
                    results[symbol] = self._default_sentiment_response()
                    
        except Exception as e:
            logger.error(f"Batch sentiment analysis failed: {e}")
            for symbol in valid_stocks:
                results[symbol] = self._default_sentiment_response()
        
        # Fill in missing stocks with empty response
        for stock in stocks:
            symbol = stock.get('symbol')
            if symbol not in results:
                results[symbol] = {
                    "sentiment_score": 0.0,
                    "confidence": 0.0,
                    "reasoning": "No news data available",
                    "article_count": 0
                }
                
        return results


# Factory function
def create_sentiment_analyzer(
    model_type: str = None,
    model_name: str = None,
    api_key: str = None
) -> RAGSentimentAnalyzer:
    """Factory function to create sentiment analyzer from environment config."""
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
