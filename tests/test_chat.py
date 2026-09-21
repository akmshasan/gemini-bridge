from collections.abc import AsyncGenerator
import os
from unittest import mock

from fastapi.testclient import TestClient
import pytest
from pytest_mock import MockerFixture

from gemini_bridge.core.config import Settings, get_settings
from gemini_bridge.main import app
from gemini_bridge.services.gemini import GeminiService

client = TestClient(app)


@pytest.mark.asyncio
async def test_chat_generate_success(mocker: MockerFixture) -> None:
    """Test generating a chat response."""
    mocker.patch.object(
        GeminiService,
        "generate_text",
        return_value="This is a test answer from Gemini.",
    )

    response = client.post(
        "/api/v1/chat/generate",
        json={"prompt": "Hello Gemini!", "system_instruction": "Be concise."},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["response"] == "This is a test answer from Gemini."
    assert "model" in data


@pytest.mark.asyncio
async def test_chat_generate_empty_prompt() -> None:
    """Test validation error on empty prompt."""
    response = client.post(
        "/api/v1/chat/generate",
        json={"prompt": ""},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_chat_generate_service_error(mocker: MockerFixture) -> None:
    """Test generation failure handling."""
    mocker.patch.object(
        GeminiService,
        "generate_text",
        side_effect=RuntimeError("API Quota Exceeded"),
    )

    response = client.post(
        "/api/v1/chat/generate",
        json={"prompt": "Hello error test"},
    )
    assert response.status_code == 500
    assert "Generation failed" in response.json()["detail"]


@pytest.mark.asyncio
async def test_chat_missing_api_key() -> None:
    """Test calling chat endpoint when API key is missing."""
    with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "", "GOOGLE_API_KEY": ""}, clear=False):
        test_settings = Settings(gemini_api_key="")
        app.dependency_overrides[get_settings] = lambda: test_settings
        try:
            response = client.post(
                "/api/v1/chat/generate",
                json={"prompt": "Should fail"},
            )
            assert response.status_code == 500
            assert "GEMINI_API_KEY is not configured" in response.json()["detail"]
        finally:
            app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_chat_stream_success(mocker: MockerFixture) -> None:
    """Test streaming chat tokens as SSE."""

    async def mock_stream(*args: object, **kwargs: object) -> AsyncGenerator[str, None]:
        yield "Hello "
        yield "world!"

    mocker.patch.object(GeminiService, "generate_stream", side_effect=mock_stream)

    response = client.post(
        "/api/v1/chat/stream",
        json={"prompt": "Stream test"},
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    content = response.text
    assert "Hello " in content
    assert "world!" in content
    assert '"done": true' in content


@pytest.mark.asyncio
async def test_chat_stream_error(mocker: MockerFixture) -> None:
    """Test streaming error propagation via SSE."""

    async def mock_stream_error(*args: object, **kwargs: object) -> AsyncGenerator[str, None]:
        raise RuntimeError("Stream disconnected")
        yield "Never reached"  # noqa: unreachable

    mocker.patch.object(GeminiService, "generate_stream", side_effect=mock_stream_error)

    response = client.post(
        "/api/v1/chat/stream",
        json={"prompt": "Stream error test"},
    )
    assert response.status_code == 200
    content = response.text
    assert "Stream disconnected" in content
    assert '"done": true' in content
