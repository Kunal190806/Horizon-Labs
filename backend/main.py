"""
backend/main.py
FastAPI application entry point with full middleware, WebSocket support, and lifespan.
"""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from backend.core.config import settings
from backend.core.database import init_db, close_db
from backend.core.cache import close_cache
from backend.core.logging_config import configure_logging

# Configure logging before any imports
configure_logging(debug=settings.debug)
logger = logging.getLogger(__name__)

# Import all routers
from backend.api.routes.datasets import router as datasets_router
from backend.api.routes.preprocessing import router as preprocessing_router
from backend.api.routes.detection import router as detection_router
from backend.api.routes.validation import router as validation_router
from backend.api.routes.characterization import router as characterization_router
from backend.api.routes.visualization import router as visualization_router
from backend.api.routes.reports import router as reports_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application lifespan: initialize and teardown resources."""
    logger.info("🚀 Starting %s v%s [%s]", settings.app_name, settings.app_version, settings.app_env)

    # Initialize database
    await init_db()

    # Create required directories
    import os
    for d in [settings.datasets_dir, settings.model_dir, settings.reports_dir, settings.mast_cache_dir]:
        os.makedirs(d, exist_ok=True)

    logger.info("✅ All services initialized. API ready.")
    yield

    logger.info("🛑 Shutting down...")
    await close_db()
    await close_cache()


# ── FastAPI App ────────────────────────────────────────────────────────────────
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "🔭 **Horizon Exoplanet Platform** — Automated TESS exoplanet detection and characterization.\n\n"
        "Full pipeline: data acquisition → preprocessing → transit detection (TLS/BLS) → "
        "ML candidate validation → planet characterization → reports."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── Middleware ─────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# ── Routers ────────────────────────────────────────────────────────────────────
app.include_router(datasets_router)
app.include_router(preprocessing_router)
app.include_router(detection_router)
app.include_router(validation_router)
app.include_router(characterization_router)
app.include_router(visualization_router)
app.include_router(reports_router)


# ── Health Check ───────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
async def health_check():
    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.app_version,
        "env": settings.app_env,
    }


@app.get("/", tags=["System"])
async def root():
    return {
        "message": f"Welcome to {settings.app_name}",
        "docs": "/docs",
        "version": settings.app_version,
    }


# ── WebSocket: Real-time Job Status ────────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active: dict[str, WebSocket] = {}

    async def connect(self, job_id: str, ws: WebSocket):
        await ws.accept()
        self.active[job_id] = ws

    def disconnect(self, job_id: str):
        self.active.pop(job_id, None)

    async def send_status(self, job_id: str, data: dict):
        ws = self.active.get(job_id)
        if ws:
            try:
                await ws.send_text(json.dumps(data))
            except Exception:
                self.disconnect(job_id)


manager = ConnectionManager()


@app.websocket("/ws/jobs/{job_id}")
async def job_status_ws(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for real-time job progress updates."""
    await manager.connect(job_id, websocket)
    try:
        while True:
            # Keep-alive ping
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        manager.disconnect(job_id)



# ── Serve Frontend Static Files & SPA Routing ──────────────────────────────────
import os
from fastapi.responses import FileResponse

frontend_dist = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend", "dist"))

@app.get("/{catchall:path}", tags=["System"])
async def serve_frontend(catchall: str):
    # Check if the requested file exists in the frontend dist folder (assets, logo, favicon, etc)
    file_path = os.path.join(frontend_dist, catchall)
    if os.path.exists(file_path) and os.path.isfile(file_path):
        return FileResponse(file_path)
        
    # Otherwise fallback to index.html for React SPA client-side routing
    index_file = os.path.join(frontend_dist, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {
        "message": "Welcome to Horizon API.",
        "status": "online",
        "frontend": "Not built yet. Run 'npm run build' in 'frontend/' directory to serve UI."
    }


if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info",
    )
