"""FastAPI application main entry point."""

import logging
import time
from collections.abc import Callable
from contextlib import asynccontextmanager

import redis
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import generate, status, styles

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown."""
    # Startup
    logger.info("Starting up Drum Pattern Generator API...")

    # Verify Redis connection
    try:
        r = redis.Redis(host="localhost", port=6379, decode_responses=True)
        r.ping()
        logger.info("✓ Redis connection successful")
    except redis.ConnectionError as e:
        logger.error(f"✗ Redis connection failed: {e}")
        logger.warning("API will start but task queue will not function")

    yield

    # Shutdown
    logger.info("Shutting down Drum Pattern Generator API...")


app = FastAPI(
    title="Drum Pattern Generator API",
    description="AI-powered MIDI drum pattern generation with producer style emulation",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next: Callable) -> Response:
    """Log all HTTP requests with timing information."""
    start_time = time.time()

    # Log request
    logger.info(f"→ {request.method} {request.url.path}")

    # Process request
    response = await call_next(request)

    # Calculate duration
    duration = time.time() - start_time

    # Log response
    logger.info(
        f"← {request.method} {request.url.path} " f"[{response.status_code}] ({duration:.3f}s)"
    )

    return response


# Global error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle uncaught exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "path": str(request.url.path),
        },
    )


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Drum Pattern Generator API",
        "version": "0.1.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    """Health check endpoint with dependency status."""
    health_status = {"status": "healthy", "redis": "unknown"}

    # Check Redis connection
    try:
        r = redis.Redis(host="localhost", port=6379, decode_responses=True)
        r.ping()
        health_status["redis"] = "connected"
    except redis.ConnectionError:
        health_status["redis"] = "disconnected"
        health_status["status"] = "degraded"

    return health_status


# Register routers
app.include_router(generate.router, prefix="/api/v1", tags=["generation"])
app.include_router(status.router, prefix="/api/v1", tags=["status"])
app.include_router(styles.router, prefix="/api/v1", tags=["styles"])

logger.info("✓ API routes registered")
