# Build stage
FROM rust:1.98.1 as builder

WORKDIR /app

COPY Cargo.toml Cargo.lock* ./
RUN mkdir src && echo "fn main() {}" > src/main.rs

# Cache dependencies
RUN cargo build --release 2>&1 | grep -v "warning\|error\[" || true
RUN rm -rf src

# Copy source code
COPY src ./src

# Build application
RUN cargo build --release

# Runtime stage
FROM debian:bookworm-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy binary from builder
COPY --from=builder /app/target/release/gemini_bridge /usr/local/bin/

# Create data directory
RUN mkdir -p /app/data

# Expose port
EXPOSE 8000

# Set environment
ENV RUST_LOG=info
ENV SERVER_HOST=0.0.0.0

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["gemini_bridge"]
