"""Retrieval-Augmented Generation (RAG) endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from agent_gateway.core.config import Settings, get_settings
from agent_gateway.schemas.rag import (
    DocumentListResponse,
    IngestResponse,
    IngestTextRequest,
    RAGQueryRequest,
    RAGQueryResponse,
)
from agent_gateway.services.gemini import GeminiService
from agent_gateway.services.rag import RAGService
from agent_gateway.services.vector_store import VectorStoreService

router = APIRouter(prefix="/rag", tags=["RAG"])


def get_rag_service(settings: Annotated[Settings, Depends(get_settings)]) -> RAGService:
    """Dependency provider for RAGService."""
    try:
        gemini = GeminiService(settings=settings)
        vector_store = VectorStoreService(settings=settings)
    except ValueError as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(err),
        ) from err
    else:
        return RAGService(gemini_service=gemini, vector_store=vector_store, settings=settings)


@router.post("/ingest", status_code=status.HTTP_201_CREATED)
async def ingest_document(
    request: IngestTextRequest,
    rag_service: Annotated[RAGService, Depends(get_rag_service)],
) -> IngestResponse:
    """Ingest raw text or markdown into the local vector store."""
    try:
        res = await rag_service.ingest_text(
            content=request.content,
            source=request.source,
            metadata=request.metadata,
        )
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {err}",
        ) from err
    else:
        return res


@router.post("/query")
async def query_knowledge_base(
    request: RAGQueryRequest,
    rag_service: Annotated[RAGService, Depends(get_rag_service)],
) -> RAGQueryResponse:
    """Query knowledge base and synthesize an answer grounded in indexed context."""
    try:
        res = await rag_service.query(
            question=request.question,
            top_k=request.top_k,
            system_instruction=request.system_instruction,
        )
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {err}",
        ) from err
    else:
        return res


@router.get("/documents")
def list_indexed_documents(
    rag_service: Annotated[RAGService, Depends(get_rag_service)],
) -> DocumentListResponse:
    """List summary of all document chunks indexed in the vector store."""
    try:
        res = rag_service.list_documents()
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list documents: {err}",
        ) from err
    else:
        return res


@router.delete("/documents")
def clear_documents(
    rag_service: Annotated[RAGService, Depends(get_rag_service)],
) -> dict[str, str]:
    """Clear all indexed documents from the vector store."""
    try:
        rag_service.clear()
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear knowledge base: {err}",
        ) from err
    else:
        return {"status": "success", "message": "Knowledge base cleared."}
