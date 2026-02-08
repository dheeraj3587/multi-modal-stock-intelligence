
from contextlib import asynccontextmanager
from fastapi import FastAPI
from datetime import datetime
import logging
import asyncio

# Services
from backend.db.session import init_db
from backend.services.scheduler_service import scheduler_service
from backend.services.news_service import news_service
from backend.services.market_service import market_service
from backend.services.data_refresher import run_refresh_sync, run_heavy_refresh_sync
from backend.services.cache_warmup import warmup_all_caches

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup configuration, service initialization, and shutdown cleanup.
    """
    # ── STARTUP PHASE ─────────────────────────────────────────────
    print("🚀 Stock Intelligence Platform Backend starting...")

    # 1. Initialise PostgreSQL tables
    try:
        init_db()
        print("✅ PostgreSQL tables initialised")
    except Exception as e:
        logger.error(f"DB init failed (will retry on first request): {e}")

    # 2. Start APScheduler
    scheduler_service.start()

    # 3. Warm up ALL Redis caches (blocks until done)
    #    This ensures every UI endpoint returns real data right away.
    await warmup_all_caches()

    # 4. Schedule Background Jobs
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

    # Full data refresh every 5 minutes (quotes + indices + derived caches)
    scheduler_service.add_job(
        run_refresh_sync,
        "interval",
        minutes=5,
        id="full_data_refresh",
        replace_existing=True,
        args=[],
    )

    # Heavy refresh every 60 minutes (fundamentals + scorecards)
    scheduler_service.add_job(
        run_heavy_refresh_sync,
        "interval",
        minutes=60,
        id="heavy_data_refresh",
        replace_existing=True,
        args=[],
    )

    print("🚀 Sentiment scheduler initialised (30 min interval)")
    print("📊 Full data refresh scheduled (5 min interval)")
    print("🏗️  Heavy data refresh scheduled (60 min interval)")
    print(f"📝 API Documentation: http://localhost:8000/docs")
    print(f"💚 Health Check: http://localhost:8000/health")

    yield

    # ── SHUTDOWN PHASE ────────────────────────────────────────────
    scheduler_service.shutdown()
    print("👋 Stock Intelligence Platform Backend shutting down...")
