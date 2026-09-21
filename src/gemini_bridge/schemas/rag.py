"""RAG ingestion and query schemas."""

from pydantic import BaseModel, Field


class IngestTextRequest(BaseModel):
    """Request payload to ingest text content into the local vector store."""

    content: str = Field(..., min_length=1, description="Raw text, markdown, or notes to ingest.")
    source: str = Field(default="direct_input", description="Source identifier (e.g. filename, url, title).")
    metadata: dict[str, str] = Field(default_factory=dict, description="Custom metadata tags.")


class IngestResponse(BaseModel):
    """Response payload after ingesting text."""

    source: str = Field(..., description="Source identifier of the ingested document.")
    chunks_count: int = Field(..., description="Number of chunks created and indexed.")
    chunk_ids: list[str] = Field(..., description="Identifiers of the stored chunks.")
    message: str = Field(..., description="Status summary message.")


class DocumentChunkResult(BaseModel):
    """Retrieved document chunk with similarity metadata."""

    chunk_id: str = Field(..., description="Unique chunk identifier.")
    content: str = Field(..., description="Chunk text content.")
    source: str = Field(..., description="Original document source.")
    distance: float | None = Field(default=None, description="Vector cosine distance (lower is closer).")
    metadata: dict[str, str] = Field(default_factory=dict, description="Associated chunk metadata.")


class RAGQueryRequest(BaseModel):
    """Request payload to query the RAG pipeline."""

    question: str = Field(..., min_length=1, description="User question to answer using grounded context.")
    top_k: int = Field(default=4, ge=1, le=20, description="Number of relevant chunks to retrieve.")
    system_instruction: str | None = Field(default=None, description="Optional custom system instruction.")


class RAGQueryResponse(BaseModel):
    """Response payload containing generated answer and source context chunks."""

    question: str = Field(..., description="Original user question.")
    answer: str = Field(..., description="Synthesized answer grounded in retrieved context.")
    context_chunks: list[DocumentChunkResult] = Field(..., description="Retrieved context chunks used as grounding.")
    model: str = Field(..., description="Model identifier used for synthesis.")


class DocumentListItem(BaseModel):
    """Summary of an indexed chunk in the vector store."""

    chunk_id: str = Field(..., description="Unique chunk ID.")
    source: str = Field(..., description="Document source name.")
    char_length: int = Field(..., description="Character length of chunk.")
    preview: str = Field(..., description="Preview snippet of the chunk.")


class DocumentListResponse(BaseModel):
    """Response payload listing indexed chunks."""

    total_chunks: int = Field(..., description="Total number of chunks indexed.")
    documents: list[DocumentListItem] = Field(..., description="List of document chunk summaries.")
