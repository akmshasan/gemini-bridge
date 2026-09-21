"""Persistent embedded vector store service using ChromaDB."""

import uuid
from pathlib import Path
from typing import Any

import chromadb

from gemini_bridge.core.config import Settings, get_settings
from gemini_bridge.schemas.rag import DocumentChunkResult, DocumentListItem

_PREVIEW_LENGTH = 120


class VectorStoreService:
    """Local embedded vector store for document indexing and retrieval."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize ChromaDB persistent client and document collection."""
        self.settings = settings or get_settings()
        persist_dir = Path(self.settings.chroma_persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection = self.client.get_or_create_collection(
            name="rag_knowledge_base",
            metadata={"hnsw:space": "cosine"},
        )

    def chunk_text(
        self,
        text: str,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> list[str]:
        """Split text into overlapping chunks for indexing."""
        size = chunk_size or self.settings.default_chunk_size
        overlap = chunk_overlap or self.settings.default_chunk_overlap

        text = text.strip()
        if not text:
            return []
        if len(text) <= size:
            return [text]

        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = start + size
            chunk = text[start:end]
            chunks.append(chunk.strip())
            start += size - overlap

        return [c for c in chunks if c]

    def add_document(
        self,
        chunks: list[str],
        source: str,
        embeddings: list[list[float]],
        custom_metadata: dict[str, str] | None = None,
    ) -> list[str]:
        """Add text chunks with their precomputed embeddings into ChromaDB."""
        if not chunks:
            return []

        doc_prefix = str(uuid.uuid4())[:8]
        chunk_ids: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for idx, _ in enumerate(chunks):
            chunk_id = f"{doc_prefix}_{idx}"
            chunk_ids.append(chunk_id)
            meta: dict[str, Any] = {
                "source": source,
                "chunk_index": idx,
                **(custom_metadata or {}),
            }
            metadatas.append(meta)

        self.collection.add(
            ids=chunk_ids,
            documents=chunks,
            embeddings=embeddings,  # type: ignore[arg-type]
            metadatas=metadatas,
        )
        return chunk_ids

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 4,
    ) -> list[DocumentChunkResult]:
        """Retrieve the top-k most similar document chunks."""
        count = self.collection.count()
        if count == 0:
            return []

        n_results = min(top_k, count)
        results = self.collection.query(
            query_embeddings=[query_embedding],  # type: ignore[arg-type]
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        output: list[DocumentChunkResult] = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0] if "distances" in results else []

        for idx, chunk_id in enumerate(ids):
            content = docs[idx] if idx < len(docs) else ""
            raw_meta = metas[idx] if idx < len(metas) else {}
            source = str(raw_meta.get("source", "unknown"))
            distance = float(dists[idx]) if idx < len(dists) else None
            metadata = {str(k): str(v) for k, v in raw_meta.items()}

            output.append(
                DocumentChunkResult(
                    chunk_id=chunk_id,
                    content=content,
                    source=source,
                    distance=distance,
                    metadata=metadata,
                )
            )

        return output

    def list_documents(self) -> list[DocumentListItem]:
        """List all indexed document chunks in the vector store."""
        data = self.collection.get(include=["metadatas", "documents"])
        ids = data.get("ids", [])
        metas = data.get("metadatas", [])
        docs = data.get("documents", [])

        items: list[DocumentListItem] = []
        for idx, chunk_id in enumerate(ids):
            meta = metas[idx] if idx < len(metas) and metas[idx] else {}
            doc = docs[idx] if idx < len(docs) else ""
            source = str(meta.get("source", "unknown"))
            preview = doc[:_PREVIEW_LENGTH].replace("\n", " ") + "..." if len(doc) > _PREVIEW_LENGTH else doc
            items.append(
                DocumentListItem(
                    chunk_id=chunk_id,
                    source=source,
                    char_length=len(doc),
                    preview=preview,
                )
            )
        return items

    def clear(self) -> None:
        """Clear all indexed documents from the collection."""
        self.client.delete_collection("rag_knowledge_base")
        self.collection = self.client.get_or_create_collection(
            name="rag_knowledge_base",
            metadata={"hnsw:space": "cosine"},
        )
