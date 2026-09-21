"""Retrieval-Augmented Generation (RAG) orchestrator service."""

from gemini_bridge.core.config import Settings, get_settings
from gemini_bridge.schemas.rag import (
    DocumentListResponse,
    IngestResponse,
    RAGQueryResponse,
)
from gemini_bridge.services.gemini import GeminiService
from gemini_bridge.services.vector_store import VectorStoreService


class RAGService:
    """Coordinates document ingestion, semantic search, and grounded generation."""

    def __init__(
        self,
        gemini_service: GeminiService | None = None,
        vector_store: VectorStoreService | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Initialize RAG orchestrator with Gemini and VectorStore services."""
        self.settings = settings or get_settings()
        self.gemini = gemini_service or GeminiService(self.settings)
        self.vector_store = vector_store or VectorStoreService(self.settings)

    async def ingest_text(
        self,
        content: str,
        source: str = "direct_input",
        metadata: dict[str, str] | None = None,
    ) -> IngestResponse:
        """Chunk, embed, and store document in the local vector store."""
        chunks = self.vector_store.chunk_text(content)
        if not chunks:
            return IngestResponse(
                source=source,
                chunks_count=0,
                chunk_ids=[],
                message="No content to ingest.",
            )

        embeddings = await self.gemini.get_embeddings(chunks)
        chunk_ids = self.vector_store.add_document(
            chunks=chunks,
            source=source,
            embeddings=embeddings,
            custom_metadata=metadata,
        )

        return IngestResponse(
            source=source,
            chunks_count=len(chunks),
            chunk_ids=chunk_ids,
            message=f"Successfully ingested {len(chunks)} chunks from source '{source}'.",
        )

    async def query(
        self,
        question: str,
        top_k: int = 4,
        system_instruction: str | None = None,
    ) -> RAGQueryResponse:
        """Retrieve relevant context chunks and synthesize an answer."""
        query_embedding = await self.gemini.get_embedding(question)
        chunks = self.vector_store.search(query_embedding=query_embedding, top_k=top_k)

        if not chunks:
            fallback_prompt = f"Question: {question}\n\nNote: No indexed documents were found in the knowledge base."
            answer = await self.gemini.generate_text(
                prompt=fallback_prompt,
                system_instruction=system_instruction,
            )
            return RAGQueryResponse(
                question=question,
                answer=answer,
                context_chunks=[],
                model=self.settings.gemini_model,
            )

        context_blocks = []
        for idx, chunk in enumerate(chunks, 1):
            context_blocks.append(f"[{idx}] Source: {chunk.source}\n{chunk.content}")
        joined_context = "\n\n".join(context_blocks)

        augmented_prompt = (
            "You are a helpful assistant answering questions using the retrieved reference context below.\n\n"
            "=== RETRIEVED CONTEXT ===\n"
            f"{joined_context}\n"
            "=========================\n\n"
            f"Question: {question}\n\n"
            "Instructions: Provide a clear, accurate answer based on the context above. "
            "Cite sources [1], [2], etc., when referencing specific facts."
        )

        answer = await self.gemini.generate_text(
            prompt=augmented_prompt,
            system_instruction=system_instruction,
        )

        return RAGQueryResponse(
            question=question,
            answer=answer,
            context_chunks=chunks,
            model=self.settings.gemini_model,
        )

    def list_documents(self) -> DocumentListResponse:
        """List summary of all indexed chunks."""
        docs = self.vector_store.list_documents()
        return DocumentListResponse(total_chunks=len(docs), documents=docs)

    def clear(self) -> None:
        """Reset the vector database."""
        self.vector_store.clear()
