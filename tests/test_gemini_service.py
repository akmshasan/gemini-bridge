"""Unit tests for GeminiService covering 100% of methods and branches."""

import os
from unittest import mock
from unittest.mock import AsyncMock, MagicMock

import pytest

from gemini_bridge.core.config import Settings
from gemini_bridge.services.gemini import GeminiService


def test_gemini_service_init_missing_key() -> None:
    """Test ValueError when initializing without API key."""
    with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "", "GOOGLE_API_KEY": ""}, clear=False):
        settings = Settings(gemini_api_key="")
        with pytest.raises(ValueError, match="GEMINI_API_KEY is not configured"):
            GeminiService(settings=settings)


@pytest.mark.asyncio
async def test_generate_text() -> None:
    """Test GeminiService.generate_text with response and empty response."""
    settings = Settings(gemini_api_key="mock_key")
    service = GeminiService(settings=settings)

    mock_response = MagicMock()
    mock_response.text = "Generated Gemini text"
    service.client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    result = await service.generate_text(
        prompt="Test prompt",
        system_instruction="System instructions",
        model="custom-model",
        temperature=0.7,
    )
    assert result == "Generated Gemini text"
    service.client.aio.models.generate_content.assert_called_once()

    # Test empty response fallback
    mock_response.text = None
    empty_result = await service.generate_text(prompt="Test empty")
    assert empty_result == ""


@pytest.mark.asyncio
async def test_generate_stream() -> None:
    """Test GeminiService.generate_stream yielding non-empty chunks."""
    settings = Settings(gemini_api_key="mock_key")
    service = GeminiService(settings=settings)

    chunk1 = MagicMock()
    chunk1.text = "Hello "
    chunk2 = MagicMock()
    chunk2.text = ""  # Should be skipped
    chunk3 = MagicMock()
    chunk3.text = "stream!"

    async def mock_stream_coro(*args: object, **kwargs: object):
        async def mock_stream_gen():
            for c in [chunk1, chunk2, chunk3]:
                yield c
        return mock_stream_gen()

    service.client.aio.models.generate_content_stream = mock_stream_coro

    chunks = []
    async for chunk in service.generate_stream(prompt="Stream prompt"):
        chunks.append(chunk)

    assert chunks == ["Hello ", "stream!"]


@pytest.mark.asyncio
async def test_get_embedding() -> None:
    """Test GeminiService.get_embedding with results and empty result."""
    settings = Settings(gemini_api_key="mock_key")
    service = GeminiService(settings=settings)

    mock_emb = MagicMock()
    mock_emb.values = [0.1, 0.2, 0.3]
    mock_res = MagicMock()
    mock_res.embeddings = [mock_emb]

    service.client.aio.models.embed_content = AsyncMock(return_value=mock_res)

    vec = await service.get_embedding("hello", model="custom-embed")
    assert vec == [0.1, 0.2, 0.3]

    # Test empty embeddings result
    mock_res.embeddings = []
    empty_vec = await service.get_embedding("empty")
    assert empty_vec == []


@pytest.mark.asyncio
async def test_get_embeddings() -> None:
    """Test GeminiService.get_embeddings with multiple texts."""
    settings = Settings(gemini_api_key="mock_key")
    service = GeminiService(settings=settings)

    mock_emb = MagicMock()
    mock_emb.values = [0.5, 0.6]
    mock_res = MagicMock()
    mock_res.embeddings = [mock_emb]

    service.client.aio.models.embed_content = AsyncMock(return_value=mock_res)

    vecs = await service.get_embeddings(["text 1", "text 2"])
    assert vecs == [[0.5, 0.6], [0.5, 0.6]]
