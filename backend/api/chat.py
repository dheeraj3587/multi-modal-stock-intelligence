"""
AI Chat API Endpoint – conversational stock assistant.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import asyncio
import logging

from backend.services.ai_chat_service import ai_chat_service
from backend.services.market_service import market_service
from backend.services.news_service import news_service

router = APIRouter(prefix="/chat", tags=["AI Chat"])
logger = logging.getLogger(__name__)


# ── Request / Response schemas ──────────────────────────────
class ChatMessage(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    symbol: Optional[str] = Field(None, description="Optional stock symbol for context")
    history: Optional[List[ChatMessage]] = Field(default_factory=list, description="Previous turns")
    deep: bool = Field(False, description="Enable deep analysis mode (fundamentals + news + RAG)")


class ChatResponse(BaseModel):
    reply: str
    cached: bool = False
    model: str = ""
    timestamp: str = ""
    deep: bool = False


# ── Helpers ─────────────────────────────────────────────────
def _gather_stock_context(symbol: str) -> Optional[Dict]:
    """Pull live data for a symbol so the LLM has real numbers."""
    try:
        all_stocks = market_service.get_all_stocks()
        stock_info = next((s for s in all_stocks if s["symbol"].upper() == symbol.upper()), None)

        ctx: Dict = {}
        if stock_info:
            ctx["name"] = stock_info.get("name", symbol)
            ctx["sector"] = stock_info.get("sector", "")

        # Quote
        try:
            quotes = market_service.get_quotes([symbol])
            if quotes:
                q = quotes[0] if isinstance(quotes, list) else quotes
                ctx.update({
                    "price": q.get("price") or q.get("last_price"),
                    "change": q.get("change"),
                    "changePercent": q.get("changePercent") or q.get("change_percent"),
                    "high": q.get("high"),
                    "low": q.get("low"),
                    "volume": q.get("volume"),
                })
        except Exception:
            pass

        # Sentiment
        try:
            sent = news_service.get_sentiment_cached_only(symbol)
            if sent:
                ctx["sentiment_score"] = sent.get("sentiment_score")
                ctx["sentiment_confidence"] = sent.get("confidence")
                ctx["risk_level"] = sent.get("risk_level")
                ctx["sentiment_reasoning"] = sent.get("reasoning")
        except Exception:
            pass

        return ctx if ctx else None
    except Exception as exc:
        logger.debug(f"Context gather failed for {symbol}: {exc}")
        return None


# ── Endpoint ────────────────────────────────────────────────
@router.post("/", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Chat with the AI stock assistant.

    Optionally pass a `symbol` for stock-specific context, and a `history`
    list for multi-turn conversations.
    """
    # Gather live context in a thread so we don't block the event loop
    stock_ctx = None
    if req.symbol:
        stock_ctx = await asyncio.to_thread(_gather_stock_context, req.symbol)

    history_dicts = [h.dict() for h in (req.history or [])]

    result = await asyncio.to_thread(
        ai_chat_service.chat,
        message=req.message,
        symbol=req.symbol,
        history=history_dicts,
        stock_context=stock_ctx,
        deep=req.deep,
    )

    return ChatResponse(**result)


@router.get("/health")
async def chat_health():
    """Check if AI chat service is available."""
    return {
        "available": ai_chat_service._ready,
        "model_type": ai_chat_service._model_type,
        "model_name": ai_chat_service._model_name,
    }
