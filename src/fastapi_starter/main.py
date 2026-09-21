from fastapi import FastAPI

app = FastAPI(
    title="FastAPI Starter",
    version="0.1.0",
    description="A minimal FastAPI starter application",
)


@app.get("/")
def read_root() -> dict[str, str]:
    """Return welcome message."""
    return {"message": "Hello from FastAPI Starter!"}


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return application health status."""
    return {"status": "ok"}


@app.get("/ready")
def ready_check() -> dict[str, str]:
    """Return application readiness status."""
    return {"status": "ready"}
