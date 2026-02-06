"""
Managed RAG with Gemini File Search stores.
"""

import json
import logging
import os
import tempfile
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class GeminiFileSearchRAG:
    """RAG sentiment analysis backed by Gemini File Search."""

    def __init__(
        self,
        api_key: str,
        model_name: str,
        cache_hours: int = 2,
        store_prefix: str = "stock-news"
    ):
        from google import genai

        self.genai = genai
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.cache_expiry = timedelta(hours=cache_hours)
        self.store_prefix = store_prefix
        self._store_cache: Dict[str, Dict[str, str]] = {}

    def analyze_sentiment(self, symbol: str, company_name: str, articles: List[Dict]) -> Dict:
        if not articles:
            return self._default_response("No news articles available for analysis.")

        store_name = self._get_or_create_store(symbol, company_name, articles)
        if not store_name:
            return self._default_response("File search store unavailable.")

        prompt = self._build_prompt(symbol, company_name, len(articles))

        try:
            tool = self._build_file_search_tool(store_name)
            config = self.genai.types.GenerateContentConfig(
                tools=[tool],
                response_mime_type="application/json"
            )
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config
            )
            result = self._parse_response(getattr(response, "text", ""))
            result["article_count"] = len(articles)
            result["analyzed_at"] = datetime.now().isoformat()
            result["model"] = f"gemini:{self.model_name}"
            return result
        except Exception as exc:
            logger.error(f"File Search sentiment error: {exc}")
            return {
                **self._default_response("Unable to analyze sentiment."),
                "article_count": len(articles),
                "error": str(exc)
            }

    def _build_prompt(self, symbol: str, company_name: str, article_count: int) -> str:
        return (
            f"TASK: Analyze sentiment for {company_name} ({symbol}).\n"
            f"CONTEXT: You have access to {article_count} financial news articles via file search.\n"
            "OUTPUT: Return ONLY valid JSON with keys: "
            "sentiment_score (-1.0..1.0), confidence (0.0..1.0), reasoning, "
            "key_themes (comma-separated), risk_level (LOW|MEDIUM|HIGH)."
        )

    def _build_file_search_tool(self, store_name: str):
        try:
            return self.genai.types.Tool(
                file_search=self.genai.types.FileSearch(
                    file_search_store_names=[store_name]
                )
            )
        except Exception:
            return {"file_search": {"file_search_store_names": [store_name]}}

    def _get_or_create_store(self, symbol: str, company_name: str, articles: List[Dict]) -> Optional[str]:
        cache_key = f"{symbol.lower()}"
        cached = self._store_cache.get(cache_key)
        if cached:
            cached_at = cached.get("timestamp")
            if cached_at and datetime.now() - cached_at < self.cache_expiry:
                return cached.get("store_name")

        store_name = self._create_store(symbol)
        if not store_name:
            return None

        file_path = self._write_articles_file(symbol, company_name, articles)
        if not file_path:
            return None

        if not self._upload_file(store_name, file_path):
            return None

        self._store_cache[cache_key] = {
            "store_name": store_name,
            "timestamp": datetime.now()
        }
        return store_name

    def _create_store(self, symbol: str) -> Optional[str]:
        try:
            display_name = f"{self.store_prefix}-{symbol.lower()}-{datetime.now():%Y%m%d%H%M%S}"
            store = self.client.file_search_stores.create(
                config={"display_name": display_name}
            )
            return store.name
        except Exception as exc:
            logger.error(f"Failed to create file search store: {exc}")
            return None

    def _upload_file(self, store_name: str, file_path: str) -> bool:
        try:
            self.client.file_search_stores.upload_to_file_search_store(
                file_search_store_name=store_name,
                file=file_path
            )
            return True
        except Exception as exc:
            logger.error(f"Failed to upload file to file search store: {exc}")
            return False

    def _write_articles_file(self, symbol: str, company_name: str, articles: List[Dict]) -> Optional[str]:
        try:
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as tmp:
                tmp.write(f"Company: {company_name} ({symbol})\n\n")
                for idx, article in enumerate(articles, 1):
                    tmp.write(f"Article {idx}\n")
                    tmp.write(f"Title: {article.get('title', '')}\n")
                    tmp.write(f"Source: {article.get('source', '')}\n")
                    tmp.write(f"Date: {article.get('published_at', '')}\n")
                    tmp.write(f"Summary: {article.get('description', '')}\n")
                    tmp.write("\n---\n\n")
                return tmp.name
        except Exception as exc:
            logger.error(f"Failed to write articles file: {exc}")
            return None

    def _parse_response(self, response_text: str) -> Dict:
        try:
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                return json.loads(response_text[start_idx:end_idx])
        except json.JSONDecodeError:
            logger.warning("Failed to parse file search response as JSON")
        return self._default_response("Unable to parse response.")

    def _default_response(self, reason: str) -> Dict:
        return {
            "sentiment_score": 0.0,
            "confidence": 0.3,
            "reasoning": reason,
            "key_themes": "insufficient data",
            "risk_level": "MEDIUM"
        }
