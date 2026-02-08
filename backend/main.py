"""
Multi-Modal Stock Intelligence Platform - FastAPI Main Application
"""
# Load environment variables FIRST before any other imports
from dotenv import load_dotenv
load_dotenv()

import os
import logging
from datetime import datetime, timezone
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Application Lifecycle
from backend.lifecycle import lifespan

# API Routers
from backend.api import auth, news, market, scorecard, chat

# Configure root logger so warmup / scheduler messages appear in docker logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(
    title="Multi-Modal Stock Intelligence Platform",
    description="AI-driven stock intelligence system with time-series forecasting, sentiment analysis, and live market data",
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Register Routers
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
        "timestamp": datetime.now(timezone.utc).isoformat(),
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

