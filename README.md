# Agent Gateway

> **Embedded AI / LLM Agent Gateway & RAG Service powered by Google Gemini and FastAPI.**  
> 100% local, zero Docker dependencies, sub-second responses, and in-process vector search.

---

## Features

- ⚡ **Gemini 2.0 / Flash Support**: Fast text completions and Server-Sent Events (SSE) token streaming via Google's `google-genai` SDK.
- 🔍 **Embedded RAG & Vector Search**: In-process ChromaDB vector store (`./data/chroma`) with automatic text chunking and cosine similarity retrieval.
- 🚫 **Zero Docker Needed**: Runs purely in Python using embedded persistent file storage.
- 🛡️ **Type-Safe & Tested**: Built on FastAPI, Pydantic v2, and `pydantic-settings` with 87%+ automated test coverage.
- 📖 **Interactive OpenAPI Docs**: Complete Swagger UI available out of the box at `/docs`.

---

## Quickstart

### 1. Prerequisites
- Python 3.10+ (or 3.14)
- [uv](https://github.com/astral-sh/uv)
- Google Gemini API key exported as `GEMINI_API_KEY`:
  ```bash
  export GEMINI_API_KEY="your_api_key_here"
  ```

### 2. Install & Run
```bash
# Sync dependencies
make install

# Start development server with auto-reload (port 8000)
make run
```

Access:
- **Interactive Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Alternative ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

---

## Key Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/v1/chat/generate` | `POST` | Full text generation with optional system prompt |
| `/api/v1/chat/stream` | `POST` | Real-time Server-Sent Events (SSE) token stream |
| `/api/v1/rag/ingest` | `POST` | Chunk, embed, and store documents in local vector store |
| `/api/v1/rag/query` | `POST` | Query knowledge base and generate grounded answer with citations |
| `/api/v1/rag/documents` | `GET` | List all indexed document chunks |
| `/api/v1/rag/documents` | `DELETE` | Clear the knowledge base |

---

## Developer Commands

```bash
make test        # Run pytest test suite
make test-cov    # Run tests with coverage report
make check       # Run ruff linter & format verification
make format      # Auto-format codebase with ruff
make lint-fix    # Auto-fix linting issues
```

---

## Documentation

For an in-depth guide on architecture, day-to-day productivity workflows, and cURL recipes, see:
👉 **[Architecture & Practical Guide](docs/architecture_and_guide.md)**
