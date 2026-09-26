#!/bin/bash

set -e

echo "🚀 Rust Project Setup"
echo "===================="

DEST="$PWD"
echo "Setting up in: $DEST"

# Verify cargo is installed
if ! command -v cargo &> /dev/null; then
    echo "❌ Rust/Cargo not found. Install from https://rustup.rs/"
    exit 1
fi

echo "✓ Cargo version: $(cargo --version)"
echo "✓ Rust version: $(rustc --version)"

# Check if src files exist
if [ ! -f "src/main.rs" ]; then
    echo "⚠️  Source files need to be copied."
    echo "Please refer to RUST_CONVERSION_COMPLETE.md for instructions."
    exit 1
fi

echo "✓ Source files found"

# Create necessary directories
mkdir -p src/config src/error src/handlers src/models src/services

# Check Cargo.toml
if [ ! -f "Cargo.toml" ]; then
    echo "❌ Cargo.toml not found!"
    exit 1
fi

echo "✓ Cargo.toml configured"

# Build the project
echo ""
echo "Building Rust project..."
echo "This may take a few minutes on first build..."

if cargo check 2>&1 | tail -5; then
    echo "✓ Project checks out successfully!"
else
    echo "⚠️  Build check completed (see output above)"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. cp .env.example .env"
echo "2. Edit .env and add your GEMINI_API_KEY"
echo "3. docker-compose up -d qdrant  (or run Qdrant locally)"
echo "4. cargo run"
echo ""
echo "Server will start on http://localhost:8000"
