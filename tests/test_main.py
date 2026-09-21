import os
from unittest import mock

from fastapi.testclient import TestClient
from pytest_mock import MockerFixture

from gemini_bridge import main
from gemini_bridge.core.config import Settings, get_settings
from gemini_bridge.main import app

client = TestClient(app)


def test_read_root() -> None:
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "Gemini Bridge" in data["message"]
    assert data["version"] == "0.1.0"
    assert data["docs"] == "/docs"


def test_health_check() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_check() -> None:
    """Test the readiness endpoint with configured key."""
    test_settings = Settings(gemini_api_key="mock_key")
    app.dependency_overrides[get_settings] = lambda: test_settings
    try:
        response = client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert data["gemini_api_key_configured"] == "True"
    finally:
        app.dependency_overrides.clear()


def test_ready_check_unconfigured() -> None:
    """Test readiness endpoint when API key is not configured."""
    with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "", "GOOGLE_API_KEY": ""}, clear=False):
        test_settings = Settings(gemini_api_key="")
        app.dependency_overrides[get_settings] = lambda: test_settings
        try:
            response = client.get("/ready")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "not_configured"
            assert data["gemini_api_key_configured"] == "False"
        finally:
            app.dependency_overrides.clear()


def test_config_env_fallback() -> None:
    """Test environment variable fallback for GEMINI_API_KEY."""
    with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "", "GOOGLE_API_KEY": "fallback_google_key"}, clear=False):
        settings = Settings(gemini_api_key="")
        assert settings.gemini_api_key == "fallback_google_key"


def test_main(mocker: MockerFixture) -> None:
    """Test the CLI entrypoint."""
    mock_run = mocker.patch("uvicorn.run")
    main()
    mock_run.assert_called_once_with(
        "gemini_bridge.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
