"""Application configuration module."""

import os
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    app_name: str = "Agent Gateway"
    version: str = "0.1.0"
    description: str = "AI / LLM Agent Gateway & RAG Service powered by Google Gemini and FastAPI"

    # Gemini API configuration
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.6-flash"
    gemini_embedding_model: str = "gemini-embedding-2"

    # Vector store configuration
    chroma_persist_dir: str = "./data/chroma"
    default_chunk_size: int = 800
    default_chunk_overlap: int = 100

    # CORS settings
    cors_origins: list[str] = ["*"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def model_post_init(self, context: object, /) -> None:
        """Resolve API key if alternative environment variable names are used."""
        super().model_post_init(context)
        if not self.gemini_api_key:
            self.gemini_api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings singleton."""
    return Settings()
