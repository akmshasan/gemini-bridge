from fastapi.testclient import TestClient
from pytest_mock import MockerFixture

from agent_gateway import main
from agent_gateway.main import app

client = TestClient(app)


def test_read_root() -> None:
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "Agent Gateway" in data["message"]
    assert data["version"] == "0.1.0"
    assert data["docs"] == "/docs"


def test_health_check() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_check() -> None:
    """Test the readiness endpoint."""
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "gemini_api_key_configured" in data


def test_main(mocker: MockerFixture) -> None:
    """Test the CLI entrypoint."""
    mock_run = mocker.patch("uvicorn.run")
    main()
    mock_run.assert_called_once_with(
        "agent_gateway.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
