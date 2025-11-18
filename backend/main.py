"""
Multi-Modal Stock Intelligence Platform - FastAPI Main Application
"""
import os
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI application
app = FastAPI(
    title="Multi-Modal Stock Intelligence Platform",
    description="AI-driven stock intelligence system with time-series forecasting, sentiment analysis, and live market data",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

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
    Returns 200 OK with service status and metadata.
    
    This endpoint does not depend on databases or external services,
    so it can be used for container health checks even when dependencies
    are still starting up.
    """
    return {
        "status": "healthy",
        "service": "stock-intelligence-backend",
        "version": "0.1.0",
        "timestamp": datetime.utcnow().isoformat(),
        "environment": os.getenv("APP_ENV", "development")
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


# Application startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    print("🚀 Stock Intelligence Platform Backend starting...")
    print(f"📝 API Documentation: http://localhost:8000/docs")
    print(f"💚 Health Check: http://localhost:8000/health")


# Application shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("👋 Stock Intelligence Platform Backend shutting down...")

