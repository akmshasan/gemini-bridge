"""FastAPI Gemini Bridge application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from gemini_bridge.api.v1.router import api_router
from gemini_bridge.core.config import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description=settings.description,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.get("/")
def read_root() -> dict[str, str]:
    """Return welcome metadata and link to interactive API docs."""
    return {
        "message": f"Welcome to {settings.app_name}!",
        "version": settings.version,
        "docs": "/docs",
        "health": "/health",
        "ready": "/ready",
    }


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return application liveness status."""
    return {"status": "ok"}


@app.get("/ready")
def ready_check() -> dict[str, str]:
    """Return application readiness status."""
    has_key = bool(settings.gemini_api_key)
    return {
        "status": "ready" if has_key else "not_configured",
        "gemini_api_key_configured": str(has_key),
    }
