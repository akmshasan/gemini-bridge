import uvicorn

from fastapi_starter.main import app


def main() -> None:
    """Run the FastAPI application via uvicorn."""
    uvicorn.run("fastapi_starter.main:app", host="0.0.0.0", port=8000, reload=True)


__all__ = ["app", "main"]
