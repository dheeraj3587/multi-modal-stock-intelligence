"""
Multi-Modal Stock Intelligence Platform - FastAPI Main Application
"""
# Load environment variables FIRST before any other imports
from dotenv import load_dotenv
load_dotenv()

import os
import logging
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Configure root logger so warmup / scheduler messages appear in docker logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# Initialize FastAPI application
from backend.api import auth, news, market, scorecard, chat

# Database
from backend.db.session import init_db

app = FastAPI(
    title="Multi-Modal Stock Intelligence Platform",
    description="AI-driven stock intelligence system with time-series forecasting, sentiment analysis, and live market data",
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.include_router(auth.router)
app.include_router(news.router)
app.include_router(market.router)
app.include_router(scorecard.router)
app.include_router(chat.router)

# CORS Configuration
CORS_ORIGINS = os.getenv("BACKEND_CORS_ORIGINS", '["http://localhost:3000"]')
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, use specific origins from env
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """
    Health check endpoint for container orchestration and monitoring.
    """
    from backend.services.data_cache import data_cache
    last_refresh = data_cache.get_last_refresh()
    return {
        "status": "healthy",
        "service": "stock-intelligence-backend",
        "version": "0.2.0",
        "timestamp": datetime.utcnow().isoformat(),
        "environment": os.getenv("APP_ENV", "development"),
        "last_data_refresh": last_refresh,
    }


@app.get("/")
async def root():
    """Root endpoint - redirects to API documentation"""
    return {
        "message": "Multi-Modal Stock Intelligence Platform API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health"
    }


from backend.services.scheduler_service import scheduler_service
from backend.services.news_service import news_service
from backend.services.market_service import market_service
from backend.services.data_refresher import refresh_all, run_refresh_sync, run_heavy_refresh_sync
from backend.services.cache_warmup import warmup_all_caches
import asyncio
import logging

_logger = logging.getLogger(__name__)


# Application startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    print("🚀 Stock Intelligence Platform Backend starting...")

    # ── Initialise PostgreSQL tables ──────────────────────────────
    try:
        init_db()
        print("✅ PostgreSQL tables initialised")
    except Exception as e:
        _logger.error(f"DB init failed (will retry on first request): {e}")

    # ── Start APScheduler ─────────────────────────────────────────
    scheduler_service.start()

    # ── Warm up ALL Redis caches (blocks until done) ──────────────
    #    This ensures every UI endpoint returns real data right away.
    await warmup_all_caches()

    all_stocks = market_service.get_all_stocks()

    # Sentiment refresh every 30 min
    scheduler_service.add_job(
        news_service.refresh_all_sentiments,
        "interval",
        minutes=30,
        id="sentiment_refresh_job",
        replace_existing=True,
        args=[all_stocks],
    )

    # Immediate one-shot sentiment bootstrap
    scheduler_service.add_job(
        news_service.refresh_all_sentiments,
        "date",
        run_date=datetime.now(),
        id="sentiment_initial_boot",
        replace_existing=True,
        args=[all_stocks],
    )

    # ── Full data refresh every 5 minutes (quotes + indices + derived caches) ──
    scheduler_service.add_job(
        run_refresh_sync,
        "interval",
        minutes=5,
        id="full_data_refresh",
        replace_existing=True,
    )

    # ── Heavy refresh every 60 minutes (fundamentals + scorecards) ──
    scheduler_service.add_job(
        run_heavy_refresh_sync,
        "interval",
        minutes=60,
        id="heavy_data_refresh",
        replace_existing=True,
    )

    # (warmup_all_caches already pre-populated quotes, indices, leaderboard)
    # The scheduler will keep them fresh from here on.

    print("🚀 Sentiment scheduler initialised (30 min interval)")
    print("📊 Full data refresh scheduled (5 min interval)")
    print("🏗️  Heavy data refresh scheduled (60 min interval)")
    print(f"📝 API Documentation: http://localhost:8000/docs")
    print(f"💚 Health Check: http://localhost:8000/health")


# Application shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    scheduler_service.shutdown()
    print("👋 Stock Intelligence Platform Backend shutting down...")

