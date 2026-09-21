"""FastAPI Gemini Bridge application entry point."""

from typing import Annotated

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from gemini_bridge.api.v1.router import api_router
from gemini_bridge.core.config import Settings, get_settings

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
def read_root(
    current_settings: Annotated[Settings, Depends(get_settings)],
) -> dict[str, str]:
    """Return welcome metadata and link to interactive API docs."""
    return {
        "message": f"Welcome to {current_settings.app_name}!",
        "version": current_settings.version,
        "docs": "/docs",
        "health": "/health",
        "ready": "/ready",
    }


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return application liveness status."""
    return {"status": "ok"}


@app.get("/ready")
def ready_check(
    current_settings: Annotated[Settings, Depends(get_settings)],
) -> dict[str, str]:
    """Return application readiness status."""
    has_key = bool(current_settings.gemini_api_key)
    return {
        "status": "ready" if has_key else "not_configured",
        "gemini_api_key_configured": str(has_key),
    }
