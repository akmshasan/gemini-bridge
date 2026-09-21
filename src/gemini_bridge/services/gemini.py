"""Google Gemini client integration service."""

import asyncio
from collections.abc import AsyncGenerator

from google import genai
from google.genai import types

from gemini_bridge.core.config import Settings, get_settings


class GeminiService:
    """Service wrapping Google GenAI SDK for generation and embeddings."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the Gemini client using application settings."""
        self.settings = settings or get_settings()
        if not self.settings.gemini_api_key:
            msg = "GEMINI_API_KEY is not configured in environment or settings"
            raise ValueError(msg)
        self.client = genai.Client(api_key=self.settings.gemini_api_key)

    async def generate_text(
        self,
        prompt: str,
        system_instruction: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
    ) -> str:
        """Generate a complete text response for a given prompt."""
        target_model = model or self.settings.gemini_model
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
        )
        response = await self.client.aio.models.generate_content(
            model=target_model,
            contents=prompt,
            config=config,
        )
        return response.text or ""

    async def generate_stream(
        self,
        prompt: str,
        system_instruction: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
    ) -> AsyncGenerator[str]:
        """Stream generated text chunks asynchronously as they arrive."""
        target_model = model or self.settings.gemini_model
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
        )
        stream = await self.client.aio.models.generate_content_stream(
            model=target_model,
            contents=prompt,
            config=config,
        )
        async for chunk in stream:
            if chunk.text:
                yield chunk.text

    async def get_embedding(
        self,
        text: str,
        model: str | None = None,
    ) -> list[float]:
        """Generate embedding vector for a single text."""
        target_model = model or self.settings.gemini_embedding_model
        res = await self.client.aio.models.embed_content(
            model=target_model,
            contents=text,
        )
        if not res.embeddings:
            return []
        return list(res.embeddings[0].values)

    async def get_embeddings(
        self,
        texts: list[str],
        model: str | None = None,
    ) -> list[list[float]]:
        """Generate embedding vectors for multiple texts concurrently."""
        tasks = [self.get_embedding(text, model=model) for text in texts]
        return await asyncio.gather(*tasks)
