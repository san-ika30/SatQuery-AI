"""
SatQuery AI — FastAPI Main Application
"""
import os
import structlog
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from dotenv import load_dotenv

from routers import analyze, health, sessions, orchestrator

load_dotenv()

log = structlog.get_logger()

# Local file storage directory (when R2 is not configured)
LOCAL_STORAGE_DIR = Path(os.getenv("LOCAL_STORAGE_DIR", "./local_storage"))
LOCAL_STORAGE_DIR.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("SatQuery AI backend starting up")
    log.info(f"Local storage directory: {LOCAL_STORAGE_DIR.resolve()}")
    yield
    log.info("SatQuery AI backend shutting down")


app = FastAPI(
    title="SatQuery AI",
    description=(
        "Agentic Vision-Language Assistant for Multimodal Remote Sensing "
        "Image Analysis. Supports single optical/SAR images, cross-modal "
        "optical+SAR pairs, and bi-temporal change analysis."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000,*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(health.router, prefix="/api", tags=["Health"])
app.include_router(orchestrator.router, prefix="/api", tags=["Orchestrator"])
app.include_router(analyze.router, prefix="/api", tags=["Analysis"])
app.include_router(sessions.router, prefix="/api", tags=["Sessions"])


@app.get("/health", tags=["Health"])
async def root_health_check():
    """Root health check for microservice status."""
    return await orchestrator.satquery_health_check()


@app.get("/", tags=["Root"])
async def root():
    return {
        "service": "SatQuery AI",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs",
    }


@app.get("/api/local-files/{file_path:path}", tags=["Local Files"])
async def serve_local_file(file_path: str):
    """
    Serve locally stored files (images, PDFs) when R2 is not configured.
    The file_path is relative to LOCAL_STORAGE_DIR.
    """
    target = LOCAL_STORAGE_DIR / file_path
    if not target.exists() or not target.is_file():
        raise HTTPException(404, f"Local file not found: {file_path}")

    # Determine content type
    suffix = target.suffix.lower()
    content_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".pdf": "application/pdf",
        ".json": "application/json",
    }
    content_type = content_types.get(suffix, "application/octet-stream")
    return FileResponse(str(target), media_type=content_type)
