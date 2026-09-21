from pathlib import Path

from fastapi.testclient import TestClient
import pytest
from pytest_mock import MockerFixture

from gemini_bridge.core.config import Settings, get_settings
from gemini_bridge.main import app
from gemini_bridge.services.gemini import GeminiService
from gemini_bridge.services.vector_store import VectorStoreService

client = TestClient(app)


def test_chunk_text_logic() -> None:
    """Test chunking logic with overlap."""
    settings = Settings(default_chunk_size=50, default_chunk_overlap=10)
    service = VectorStoreService(settings=settings)

    text = "A" * 120
    chunks = service.chunk_text(text)
    assert len(chunks) >= 2
    assert service.chunk_text("") == []


@pytest.mark.asyncio
async def test_rag_ingest_and_query_flow(mocker: MockerFixture, tmp_path: Path) -> None:
    """Test full ingestion, document listing, query, and clearing lifecycle."""
    test_settings = Settings(
        gemini_api_key="mock_key",
        chroma_persist_dir=str(tmp_path),
    )
    app.dependency_overrides[get_settings] = lambda: test_settings

    dummy_embedding = [0.1] * 128

    async def mock_get_embeddings(self: object, texts: list[str], *args: object, **kwargs: object) -> list[list[float]]:
        return [dummy_embedding for _ in texts]

    mocker.patch.object(GeminiService, "get_embeddings", mock_get_embeddings)
    mocker.patch.object(
        GeminiService,
        "get_embedding",
        return_value=dummy_embedding,
    )
    mocker.patch.object(
        GeminiService,
        "generate_text",
        return_value="FastAPI is a modern, high-performance web framework for Python.",
    )

    try:
        # 1. Ingest document
        ingest_res = client.post(
            "/api/v1/rag/ingest",
            json={
                "content": "FastAPI is built on Starlette and Pydantic. It provides automatic OpenAPI docs.",
                "source": "fastapi_overview.md",
                "metadata": {"category": "documentation"},
            },
        )
        assert ingest_res.status_code == 201
        data = ingest_res.json()
        assert data["source"] == "fastapi_overview.md"
        assert data["chunks_count"] >= 1
        assert len(data["chunk_ids"]) >= 1

        # 2. List documents
        list_res = client.get("/api/v1/rag/documents")
        assert list_res.status_code == 200
        list_data = list_res.json()
        assert list_data["total_chunks"] >= 1
        assert list_data["documents"][0]["source"] == "fastapi_overview.md"

        # 3. Query RAG
        query_res = client.post(
            "/api/v1/rag/query",
            json={"question": "What is FastAPI?", "top_k": 2},
        )
        assert query_res.status_code == 200
        query_data = query_res.json()
        assert query_data["question"] == "What is FastAPI?"
        assert "FastAPI is a modern" in query_data["answer"]
        assert len(query_data["context_chunks"]) >= 1

        # 4. Clear documents
        del_res = client.delete("/api/v1/rag/documents")
        assert del_res.status_code == 200

        # 5. Verify list is now empty
        list_after = client.get("/api/v1/rag/documents")
        assert list_after.json()["total_chunks"] == 0
    finally:
        app.dependency_overrides.clear()
