"""
AI Chat Service for Stock Intelligence Platform.

Provides a conversational AI interface backed by Gemini (or fallback LLMs)
with Redis caching for fast repeated queries.

Supports two modes:
  - **Quick** (default): live quote + sentiment context → fast response.
  - **Deep**: fundamentals, news articles, RAG vector search, scorecard
              data all injected into the prompt for thorough analysis.
"""

import os
import json
import hashlib
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

from backend.services.redis_cache import redis_cache

logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────
CHAT_CACHE_TTL = int(os.getenv("CHAT_CACHE_TTL", "1800"))  # 30 min
DEEP_CACHE_TTL = int(os.getenv("DEEP_CACHE_TTL", "3600"))   # 60 min (deep is expensive)
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "30"))
DEEP_LLM_TIMEOUT = int(os.getenv("DEEP_LLM_TIMEOUT", "60"))


def _get_redis():
    """Get raw Redis client (or None)."""
    try:
        client = redis_cache._get_client()
        return client if redis_cache._connected else None
    except Exception:
        return None


def _cache_key(user_msg: str, symbol: Optional[str], deep: bool = False) -> str:
    """Deterministic cache key from user message + optional symbol + mode."""
    prefix = "deepchat" if deep else "chat"
    raw = f"{symbol or ''}:{user_msg.strip().lower()}"
    return f"{prefix}:{hashlib.sha256(raw.encode()).hexdigest()[:24]}"


class AIChatService:
    """Conversational AI chat about stocks, powered by Gemini / OpenAI / Anthropic."""

    def __init__(self):
        self._client = None
        self._model_type: Optional[str] = None
        self._model_name: Optional[str] = None
        self._ready = False
        self._init()

    # ── Lazy LLM init ───────────────────────────────────────
    def _init(self):
        try:
            gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            openai_key = os.getenv("OPENAI_API_KEY")
            anthropic_key = os.getenv("ANTHROPIC_API_KEY")

            if gemini_key:
                from google import genai
                self._client = genai.Client(api_key=gemini_key)
                self._model_type = "gemini"
                self._model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
                self._ready = True
                logger.info(f"AI Chat: Gemini ({self._model_name})")

            elif openai_key:
                from openai import OpenAI
                self._client = OpenAI(api_key=openai_key)
                self._model_type = "openai"
                self._model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
                self._ready = True
                logger.info(f"AI Chat: OpenAI ({self._model_name})")

            elif anthropic_key:
                from anthropic import Anthropic
                self._client = Anthropic(api_key=anthropic_key)
                self._model_type = "anthropic"
                self._model_name = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
                self._ready = True
                logger.info(f"AI Chat: Anthropic ({self._model_name})")

            else:
                logger.warning("AI Chat: No API key found – chat will be unavailable")

        except Exception as exc:
            logger.error(f"AI Chat init error: {exc}")

    # ── Public API ──────────────────────────────────────────
    def chat(
        self,
        message: str,
        symbol: Optional[str] = None,
        history: Optional[List[Dict]] = None,
        stock_context: Optional[Dict] = None,
        deep: bool = False,
    ) -> Dict:
        """
        Send a message to the AI and get a response.

        Args:
            message:       User's chat message.
            symbol:        Optional stock symbol for context.
            history:       Previous conversation turns [{role, content}, ...].
            stock_context: Live stock data dict injected as system context.
            deep:          If True, gather extensive data (fundamentals, news, RAG).

        Returns:
            {"reply": str, "cached": bool, "model": str, "timestamp": str, "deep": bool}
        """
        if not self._ready:
            return {
                "reply": "AI chat is currently unavailable. Please configure a GEMINI_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY.",
                "cached": False,
                "model": "none",
                "deep": deep,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        # ── 1. Check cache(only standalone questions – no history)
        ttl = DEEP_CACHE_TTL if deep else CHAT_CACHE_TTL
        ck = _cache_key(message, symbol, deep)
        if not history:
            r = _get_redis()
            if r:
                try:
                    cache_hit = r.get(ck)
                    if cache_hit:
                        cached = json.loads(cache_hit) if isinstance(cache_hit, str) else cache_hit
                        cached["cached"] = True
                        return cached
                except Exception:
                    pass

        # ── 2. Build messages ──────────────────────────────
        system_prompt = self._build_system_prompt(symbol, stock_context, deep)
        messages = self._build_messages(system_prompt, message, history)

        # ── 3. Call LLM ───────────────────────────────────
        timeout = DEEP_LLM_TIMEOUT if deep else LLM_TIMEOUT
        max_tokens = 2048 if deep else 1024
        try:
            reply_text = self._call_llm(messages, timeout=timeout, max_tokens=max_tokens)
        except Exception as exc:
            logger.error(f"AI Chat LLM error: {exc}")
            return {
                "reply": f"Sorry, I couldn't process your request right now. Error: {str(exc)[:120]}",
                "cached": False,
                "model": f"{self._model_type}:{self._model_name}",
                "deep": deep,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        result = {
            "reply": reply_text,
            "cached": False,
            "model": f"{self._model_type}:{self._model_name}",
            "deep": deep,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # ── 4. Cache standalone answers───────────────────
        if not history:
            r = _get_redis()
            if r:
                try:
                    r.setex(ck, ttl, json.dumps(result))
                except Exception as exc:
                    logger.debug(f"Chat cache write failed: {exc}")

        return result

    # ── System prompt builders ──────────────────────────────
    def _build_system_prompt(self, symbol: Optional[str], ctx: Optional[Dict], deep: bool) -> str:
        if deep and symbol:
            return self._build_deep_system_prompt(symbol, ctx)
        return self._build_quick_system_prompt(symbol, ctx)

    def _build_quick_system_prompt(self, symbol: Optional[str], ctx: Optional[Dict]) -> str:
        base = (
            "You are StockSense AI, an expert Indian stock market analyst assistant. "
            "You help users understand stocks, sectors, fundamentals, technicals, "
            "news sentiment, and market trends for NSE/BSE-listed companies. "
            "Be concise, data-driven, and actionable. Use bullet points when helpful. "
            "Always mention that your analysis is not financial advice."
        )
        if symbol:
            base += f"\n\nThe user is asking about stock symbol: {symbol}."
        if ctx:
            base += "\n\n### Live Stock Data\n```json\n" + json.dumps(ctx, default=str) + "\n```"
        return base

    def _build_deep_system_prompt(self, symbol: str, ctx: Optional[Dict]) -> str:
        """
        Build an extensive system prompt with fundamentals, news, RAG context,
        and scorecard data for deep analysis.
        """
        sections: List[str] = [
            "You are StockSense AI performing a DEEP ANALYSIS. "
            "Provide a comprehensive, multi-dimensional analysis of the stock. "
            "Cover: fundamentals, valuation, growth, risks, sentiment, news catalysts, "
            "and a clear verdict. Use data points from the context below. "
            "Structure your answer with clear headings: "
            "## Overview, ## Fundamentals, ## Sentiment & News, ## Risk Assessment, ## Verdict. "
            "This is not financial advice."
        ]

        if symbol:
            sections.append(f"\n**Stock Symbol:** {symbol}")

        # Live quote
        if ctx:
            sections.append("\n### Live Market Data\n```json\n" + json.dumps(ctx, default=str) + "\n```")

        # Fundamentals from DB
        try:
            from backend.services.fundamentals_db import fundamentals_db
            fund = fundamentals_db.get_latest(symbol)
            if fund:
                ratios = fund.get("ratios", {})
                fund_data = {
                    "company_name": fund.get("company_name"),
                    "sector": fund.get("sector"),
                    "industry": fund.get("industry"),
                    "market_cap_cr": ratios.get("market_cap_cr"),
                    "pe_ratio": ratios.get("pe_ratio"),
                    "industry_pe": ratios.get("industry_pe"),
                    "roe": ratios.get("roe"),
                    "roce": ratios.get("roce"),
                    "eps": ratios.get("eps"),
                    "debt_cr": ratios.get("debt_cr"),
                    "dividend_yield": ratios.get("dividend_yield"),
                    "book_value": ratios.get("book_value"),
                    "promoter_holding": ratios.get("promoter_holding"),
                    "revenue_growth": fund.get("revenue_growth"),
                    "earnings_growth": fund.get("earnings_growth"),
                    "debt_to_equity": fund.get("debt_to_equity"),
                    "profit_margins": fund.get("profit_margins"),
                    "operating_margins": fund.get("operating_margins"),
                    "current_ratio": fund.get("current_ratio"),
                    "free_cashflow": fund.get("free_cashflow"),
                    "business_summary": (fund.get("business_summary") or "")[:500],
                }
                # Clean None values
                fund_data = {k: v for k, v in fund_data.items() if v is not None}
                if fund_data:
                    sections.append("\n### Fundamentals\n```json\n" + json.dumps(fund_data, default=str) + "\n```")

                # Pros / Cons
                pros = fund.get("pros", [])
                cons = fund.get("cons", [])
                if pros:
                    sections.append("\n### Strengths\n" + "\n".join(f"- {p}" for p in pros[:8]))
                if cons:
                    sections.append("\n### Weaknesses\n" + "\n".join(f"- {c}" for c in cons[:8]))

                # Quarterly results (last 4)
                qr = fund.get("quarterly_results", [])
                if qr and isinstance(qr, list):
                    sections.append("\n### Recent Quarterly Results\n```json\n" + json.dumps(qr[:4], default=str) + "\n```")

        except Exception as exc:
            logger.debug(f"Deep mode: fundamentals fetch failed: {exc}")

        # News articles from RSS
        try:
            from backend.services.news_service import news_service
            from backend.services.market_service import STOCK_DATA
            company_name = STOCK_DATA.get(symbol.upper(), {}).get("name", symbol)
            articles = news_service.get_company_news(symbol, company_name)
            if articles:
                news_lines = ["\n### Recent News Headlines"]
                for a in articles[:8]:
                    title = a.get("title", "")
                    source = a.get("source", "")
                    date = a.get("published_at", "")
                    desc = (a.get("description") or "")[:150]
                    news_lines.append(f"- **{title}** ({source}, {date})\n  {desc}")
                sections.append("\n".join(news_lines))
        except Exception as exc:
            logger.debug(f"Deep mode: news fetch failed: {exc}")

        # RAG vector search for historical context
        try:
            from backend.services.news_service import news_service
            news_service._initialize_rag()
            if news_service._vector_store and len(news_service._vector_store) > 0:
                query = f"{symbol} stock market analysis outlook"
                rag_results = news_service._vector_store.similarity_search(query, k=5, score_threshold=0.4)
                if rag_results:
                    rag_lines = ["\n### Historical Context (RAG Vector Search)"]
                    for doc, score in rag_results:
                        title = doc.get("title", "")
                        desc = (doc.get("description") or "")[:120]
                        rag_lines.append(f"- {title} (relevance: {score:.2f}) – {desc}")
                    sections.append("\n".join(rag_lines))
        except Exception as exc:
            logger.debug(f"Deep mode: RAG search failed: {exc}")

        # Scorecard data (cached)
        try:
            from backend.services.scorecard_generator import ScorecardGenerator
            gen = ScorecardGenerator()
            # Try to generate or get cached scorecard
            from backend.services.fundamentals_db import fundamentals_db
            fund = fundamentals_db.get_latest(symbol)
            if fund:
                sc = gen.generate(fund)
                if sc:
                    score_summary = {
                        "overall_score": sc.get("overall_score"),
                        "overall_verdict": sc.get("overall_verdict"),
                        "overall_badge": sc.get("overall_badge"),
                    }
                    cats = sc.get("categories", {})
                    for cat_key, cat_val in cats.items():
                        score_summary[f"{cat_key}_score"] = f"{cat_val.get('score', 0)}/{cat_val.get('max_score', 10)}"
                        score_summary[f"{cat_key}_verdict"] = cat_val.get("verdict", "")
                    sections.append("\n### AI Scorecard Summary\n```json\n" + json.dumps(score_summary, default=str) + "\n```")
        except Exception as exc:
            logger.debug(f"Deep mode: scorecard failed: {exc}")

        return "\n".join(sections)

    def _build_messages(
        self, system: str, user_msg: str, history: Optional[List[Dict]]
    ) -> List[Dict]:
        msgs: List[Dict] = [{"role": "system", "content": system}]
        if history:
            for h in history[-10:]:  # keep last 10 turns
                msgs.append({"role": h.get("role", "user"), "content": h.get("content", "")})
        msgs.append({"role": "user", "content": user_msg})
        return msgs

    def _call_llm(self, messages: List[Dict], timeout: int = LLM_TIMEOUT, max_tokens: int = 1024) -> str:
        import concurrent.futures

        if self._model_type == "gemini":
            parts = []
            for m in messages:
                role = m["role"]
                content = m["content"]
                if role == "system":
                    parts.append(f"[System Instructions]\n{content}\n")
                elif role == "assistant":
                    parts.append(f"Assistant: {content}\n")
                else:
                    parts.append(f"User: {content}\n")
            prompt = "\n".join(parts)

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    self._client.models.generate_content,
                    model=self._model_name,
                    contents=prompt,
                )
                response = future.result(timeout=timeout)
                return getattr(response, "text", "") or "I couldn't generate a response."

        elif self._model_type == "openai":
            response = self._client.chat.completions.create(
                model=self._model_name,
                messages=messages,
                temperature=0.4,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            return response.choices[0].message.content

        elif self._model_type == "anthropic":
            system_text = ""
            filtered = []
            for m in messages:
                if m["role"] == "system":
                    system_text += m["content"] + "\n"
                else:
                    filtered.append(m)
            response = self._client.messages.create(
                model=self._model_name,
                system=system_text,
                max_tokens=max_tokens,
                temperature=0.4,
                messages=filtered,
                timeout=timeout,
            )
            return response.content[0].text

        raise ValueError(f"Unknown model type: {self._model_type}")


# ── Singleton ───────────────────────────────────────────────
ai_chat_service = AIChatService()
