import uvicorn

from gemini_bridge.main import app


def main() -> None:
    """Run the FastAPI application via uvicorn."""
    uvicorn.run("gemini_bridge.main:app", host="0.0.0.0", port=8000, reload=True)


__all__ = ["app", "main"]
