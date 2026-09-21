from fastapi.testclient import TestClient
from pytest_mock import MockerFixture

from fastapi_starter import main
from fastapi_starter.main import app

client = TestClient(app)


def test_read_root() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello from FastAPI Starter!"}


def test_health_check() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_check() -> None:
    """Test the readiness endpoint."""
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_main(mocker: MockerFixture) -> None:
    """Test the CLI entrypoint."""
    mock_run = mocker.patch("uvicorn.run")
    main()
    mock_run.assert_called_once_with(
        "fastapi_starter.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )

