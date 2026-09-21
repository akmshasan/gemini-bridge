# Gemini Bridge

[![CI](https://github.com/akmshasan/gemini-bridge/actions/workflows/ci.yml/badge.svg)](https://github.com/akmshasan/gemini-bridge/actions/workflows/ci.yml)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

> **Embedded AI / LLM Gemini Bridge & RAG Service powered by Google Gemini and FastAPI.**  
> 100% local, zero Docker dependencies required, sub-second responses, and in-process vector search.

---

## Features

- ⚡ **Gemini 2.0 / Flash Support**: Fast text completions and Server-Sent Events (SSE) token streaming via Google's official `google-genai` SDK.
- 🔍 **Embedded RAG & Vector Search**: In-process ChromaDB vector store (`./data/chroma`) with automatic text chunking and cosine similarity retrieval.
- 🚫 **Zero Docker Needed**: Runs purely in Python using embedded persistent file storage (optional Dockerfile and docker-compose also included for deployment).
- 🛡️ **Type-Safe & Tested**: Built on FastAPI, Pydantic v2, and `pydantic-settings` with 92%+ automated test coverage.
- 📂 **Bulk Folder Ingestion CLI**: Recursively ingest directories of Markdown, text, and source code files directly into your knowledge base with a single command.
- 📖 **Interactive OpenAPI Docs**: Complete Swagger UI available out of the box at `/docs`.

---

## Quickstart

### 1. Prerequisites
- Python 3.10+ (compatible up to Python 3.14)
- [uv](https://github.com/astral-sh/uv) (fast Python package manager)
- Google Gemini API key (get a free key at [Google AI Studio](https://aistudio.google.com/app/apikey))

### 2. Configuration
Copy the template configuration and add your Gemini API key:
```bash
cp .env.example .env
```

Edit `.env`:
```env
GEMINI_API_KEY="your-actual-api-key"
```

Or export it directly in your environment:
```bash
export GEMINI_API_KEY="your_api_key_here"
```

### 3. Install & Run
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
- **Readiness Check**: [http://localhost:8000/ready](http://localhost:8000/ready)

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

## Bulk Document Ingestion CLI

To index an entire folder of notes, documentation, or code into the knowledge base:

```bash
# Ingest all markdown, text, and code files in a directory
uv run python scripts/ingest_folder.py ./docs

# Specify target file extensions
uv run python scripts/ingest_folder.py ./my-notes --extensions=.md,.txt
```

---

## Optional Container Deployment (Docker)

If you prefer running in a container or deploying to Google Cloud Run:

```bash
# Using Docker Compose
GEMINI_API_KEY="your-api-key" docker compose up -d

# Or build the Docker image directly
docker build -t gemini-bridge .
docker run -p 8000:8000 -e GEMINI_API_KEY="your-api-key" gemini-bridge
```

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

For an in-depth guide on architecture, day-to-day productivity workflows, terminal aliases, and cURL recipes, see:
👉 **[Architecture & Practical Guide](docs/architecture_and_guide.md)**

---

## License

This project is licensed under the [MIT License](LICENSE).
