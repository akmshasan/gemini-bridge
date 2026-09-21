FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy dependency definition files
COPY pyproject.toml uv.lock ./

# Install project dependencies without project itself
RUN uv sync --frozen --no-install-project --no-dev

# Copy application source code and documentation
COPY src/ ./src/
COPY README.md ./

# Install project
RUN uv sync --frozen --no-dev

# Create data directory for ChromaDB persistence
RUN mkdir -p /app/data/chroma

ENV PATH="/app/.venv/bin:$PATH"
ENV CHROMA_PERSIST_DIR="/app/data/chroma"

EXPOSE 8000

CMD ["uvicorn", "gemini_bridge.main:app", "--host", "0.0.0.0", "--port", "8000"]
