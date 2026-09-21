# Gemini Bridge: Architecture & Day-to-Day Practical Guide

An embedded, high-performance AI Gateway & Retrieval-Augmented Generation (RAG) backend powered by **FastAPI** and **Google Gemini 2.0 / Flash**. 

Designed to run **100% locally** on your machine with **zero Docker requirements**.

---

## 1. What Is Gemini Bridge?

Gemini Bridge is your local intelligence bridge. Instead of scattering bespoke LLM calls, API keys, and prompt logic across multiple scripts, Gemini Bridge centralizes all your AI tasks into a single, high-speed, local API service on `localhost:8000`.

### Key Capabilities
- **Direct LLM Generation & SSE Streaming**: Sub-second text completions and real-time token streaming using Gemini Flash (`gemini-3.6-flash`).
- **Local RAG & Embedded Vector Search**: In-process document ingestion and vector storage powered by ChromaDB (`./data/chroma`) and Gemini embeddings (`gemini-embedding-2`).
- **Zero-Docker Footprint**: Runs completely in your local Python virtual environment using SQLite/Chroma file storage.
- **Type-Safe & Production-Ready**: Strict Pydantic models, full OpenAPI/Swagger documentation (`/docs`), and comprehensive test coverage.

---

## 2. Architecture & Data Flow

```
                               ┌─────────────────────────────────────────┐
                               │       Client Consumers & Tools          │
                               │  - Terminal / Shell scripts             │
                               │  - Raycast / Alfred / Shortcuts         │
                               │  - Python Automations & Web Frontends   │
                               └────────────────────┬────────────────────┘
                                                    │ HTTP / SSE
                                                    ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       Gemini Bridge (Port 8000)                                        │
│                                                                                                        │
│  ┌───────────────────────┐   ┌──────────────────────────┐   ┌────────────────────────────────────────┐  │
│  │   /health & /ready    │   │  /api/v1/chat/generate   │   │  /api/v1/rag/ingest                    │  │
│  │   Liveness & Status   │   │  /api/v1/chat/stream     │   │  /api/v1/rag/query                     │  │
│  └───────────────────────┘   └────────────┬─────────────┘   │  /api/v1/rag/documents                 │  │
│                                           │                 └────────────────────┬───────────────────┘  │
│                                           │                                      │                      │
│                                           ▼                                      ▼                      │
│                           ┌───────────────────────────────┐     ┌────────────────────────────────────┐  │
│                           │         GeminiService         │     │         VectorStoreService         │  │
│                           │ - Text generation & streaming │     │ - Text chunker (size + overlap)    │  │
│                           │ - Model: gemini-3.6-flash     │     │ - In-process ChromaDB persistence  │  │
│                           │ - Embed: gemini-embedding-2   │     │ - Location: ./data/chroma          │  │
│                           └───────────────┬───────────────┘     └─────────────────┬──────────────────┘  │
│                                           │                                       │                      │
└───────────────────────────────────────────┼───────────────────────────────────────┼──────────────────────┘
                                            │                                       │
                                            ▼                                       ▼
                               ┌─────────────────────────┐             ┌─────────────────────────┐
                               │    Google Gemini API    │             │   Local File Storage    │
                               │  (Cloud Model Endpoint) │             │    (./data/chroma/)     │
                               └─────────────────────────┘             └─────────────────────────┘
```

---

## 3. How It Helps Your Day-to-Day Workflow

### Use Case 1: Your Personal "Second Brain" Search Engine
- **The Problem**: You have project notes, system designs, README files, or meeting notes spread across directories, but searching via keyword search (`grep` or Spotlight) misses synonyms and conceptual relationships.
- **How Gemini Bridge Solves It**: Ingest your notes into `/api/v1/rag/ingest`. Ask complex questions via `/api/v1/rag/query`. The service retrieves the relevant sections and synthesizes an answer with exact source citations.

### Use Case 2: Instant Terminal AI Copilot
- **The Problem**: You frequently need quick code explanations, shell command help, or regex generation without opening a browser tab and losing terminal focus.
- **How Gemini Bridge Solves It**: Create a bash/zsh alias calling `/api/v1/chat/stream` or `/api/v1/chat/generate` directly from your command line:
  ```bash
  # In your ~/.zshrc:
  ask() {
    curl -s -X POST http://localhost:8000/api/v1/chat/generate \
      -H "Content-Type: application/json" \
      -d "{\"prompt\": \"$*\", \"system_instruction\": \"Answer concisely in 1-3 lines for a terminal user.\"}" \
      | python3 -c "import sys, json; print(json.load(sys.stdin)['response'])"
  }
  ```
  Now in any terminal:
  ```bash
  ask "How do I untar a .tar.gz file to a specific directory?"
  ```

### Use Case 3: Summarize & Extract from Files / URLs
- **The Problem**: Long log outputs, JSON dumps, or technical articles take too long to scan manually.
- **How Gemini Bridge Solves It**: Pipe text directly into the gateway for structured extraction:
  ```bash
  cat error.log | curl -s -X POST http://localhost:8000/api/v1/chat/generate \
    -H "Content-Type: application/json" \
    -d "{\"prompt\": \"Identify the root cause and error stack traces:\n$(cat error.log)\"}"
  ```

### Use Case 4: Zero-Friction Local AI Gateway for Custom Tools
- If you build small apps (Raycast extensions, local Streamlit apps, VS Code extensions), point them to `http://localhost:8000/api/v1/chat/stream` or `http://localhost:8000/api/v1/rag/query`. You never have to bundle API keys or embedding engines into those clients.

---

## 4. API Reference & cURL Examples

Ensure the service is running (`make run` or `uv run gemini-bridge`).

### 1. Health & Readiness
```bash
curl -s http://localhost:8000/health
# {"status":"ok"}

curl -s http://localhost:8000/ready
# {"status":"ready","gemini_api_key_configured":"True"}
```

### 2. Standard Chat Completion
**Endpoint**: `POST /api/v1/chat/generate`

```bash
curl -X POST http://localhost:8000/api/v1/chat/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain the difference between SQLite and PostgreSQL in two bullet points.",
    "system_instruction": "You are an expert database engineer."
  }'
```

**Response**:
```json
{
  "response": "* **Architecture:** SQLite is an embedded, serverless library...\n* **Scalability & Concurrency:** PostgreSQL is a full client-server database...",
  "model": "gemini-3.6-flash"
}
```

### 3. Real-Time Token Streaming (Server-Sent Events)
**Endpoint**: `POST /api/v1/chat/stream`

Use `curl -N` (no-buffer) to watch tokens stream in real-time:
```bash
curl -N -X POST http://localhost:8000/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a haiku about Python programming."
  }'
```

**Stream output**:
```text
data: {"text": "Clean code", "done": false}
data: {"text": " reads like prose,", "done": false}
data: {"text": "\nIndents shape the flowing thought,\nPython comes alive.", "done": false}
data: {"text": "", "done": true}
```

### 4. Ingest Documents into RAG Vector Store
**Endpoint**: `POST /api/v1/rag/ingest`

```bash
curl -X POST http://localhost:8000/api/v1/rag/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Project Phoenix Architecture: The frontend is built in SvelteKit. The backend is an asynchronous FastAPI service communicating with BigQuery and Redis for caching. Deployment runs on Google Cloud Run.",
    "source": "phoenix_architecture.md",
    "metadata": {"project": "phoenix", "author": "shafiul"}
  }'
```

**Response**:
```json
{
  "source": "phoenix_architecture.md",
  "chunks_count": 1,
  "chunk_ids": ["a7c39f10_0"],
  "message": "Successfully ingested 1 chunks from source 'phoenix_architecture.md'."
}
```

### 5. Query the RAG Knowledge Base
**Endpoint**: `POST /api/v1/rag/query`

```bash
curl -X POST http://localhost:8000/api/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What caching layer and cloud platform does Project Phoenix use?",
    "top_k": 2
  }'
```

**Response**:
```json
{
  "question": "What caching layer and cloud platform does Project Phoenix use?",
  "answer": "Project Phoenix uses Redis for caching and runs on Google Cloud Run [1].",
  "context_chunks": [
    {
      "chunk_id": "a7c39f10_0",
      "content": "Project Phoenix Architecture: The frontend is built in SvelteKit...",
      "source": "phoenix_architecture.md",
      "distance": 0.124
    }
  ],
  "model": "gemini-3.6-flash"
}
```

### 6. List and Clear Indexed Documents
```bash
# List all indexed chunks
curl -s http://localhost:8000/api/v1/rag/documents

# Reset / clear knowledge base
curl -s -X DELETE http://localhost:8000/api/v1/rag/documents
```

---

## 5. Configuration & Environment Variables

All settings can be customized in `.env` or passed via system environment variables:

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `GEMINI_API_KEY` | *(read from env)* | Your Google Gemini API Key |
| `GEMINI_MODEL` | `gemini-3.6-flash` | Default text generation model |
| `GEMINI_EMBEDDING_MODEL` | `gemini-embedding-2` | Default embedding model for RAG |
| `CHROMA_PERSIST_DIR` | `./data/chroma` | Directory where vector embeddings are persisted |
| `DEFAULT_CHUNK_SIZE` | `800` | Target characters per document chunk |
| `DEFAULT_CHUNK_OVERLAP` | `100` | Overlap characters between chunks |

---

## 6. Development Workflow & Commands

```bash
# Run tests
make test

# Run tests with coverage breakdown
make test-cov

# Check linting and formatting
make check

# Auto-fix lint and formatting
make lint-fix
make format

# Start local server with hot reload
make run
```

---

## 7. Next Steps & Recommended Additions

When you are ready to expand:
1. **Directory Ingest CLI Script**: A small Python CLI script `scripts/ingest_folder.py` to ingest an entire folder of markdown or PDF files in one command.
2. **Autonomous Tool Calling**: Allow Gemini to call Python tools (e.g., query local databases, check git status, fetch weather).
3. **Web Dashboard**: A minimal single-page web UI (vanilla HTML/JS or React) with a chat box and drag-and-drop document uploader.
