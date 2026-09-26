.PHONY: help build test test-verbose test-coverage test-coverage-html clean fmt lint doc release docker-build docker-push run stop logs

SHELL := /bin/bash
CARGO_FLAGS ?= 
RUST_LOG ?= debug

help:
	@echo "Available commands:"
	@echo "  make build              - Build the project"
	@echo "  make test               - Run tests"
	@echo "  make test-verbose       - Run tests with verbose output"
	@echo "  make test-coverage      - Run tests with coverage report"
	@echo "  make test-coverage-html - Generate HTML coverage report"
	@echo "  make clean              - Clean build artifacts"
	@echo "  make fmt                - Format code"
	@echo "  make lint               - Run clippy linter"
	@echo "  make doc                - Generate documentation"
	@echo "  make release            - Build release binary"
	@echo "  make docker-build       - Build Docker image"
	@echo "  make docker-push        - Push Docker image"
	@echo "  make run                - Run locally with docker-compose"
	@echo "  make stop               - Stop docker-compose"
	@echo "  make logs               - Show docker-compose logs"

build:
	@echo "Building project..."
	cargo build $(CARGO_FLAGS)

test:
	@echo "Running tests..."
	cargo test --lib --tests $(CARGO_FLAGS)

test-verbose:
	@echo "Running tests (verbose)..."
	cargo test --lib --tests -- --nocapture $(CARGO_FLAGS)

# main.rs is excluded: it's just server bootstrap (bind + run), which needs a
# live bound socket to exercise and belongs to deployment/e2e checks, not
# unit coverage. Every other module (handlers, services, config, error,
# models) is measured and expected to sit at ~100%.
test-coverage:
	@echo "Running tests with coverage..."
	@command -v cargo-tarpaulin >/dev/null 2>&1 || cargo install cargo-tarpaulin
	cargo tarpaulin --out Stdout --exclude-files 'tests/*' --exclude-files 'src/main.rs' $(CARGO_FLAGS)

test-coverage-html:
	@echo "Generating HTML coverage report..."
	@command -v cargo-tarpaulin >/dev/null 2>&1 || cargo install cargo-tarpaulin
	@mkdir -p coverage
	cargo tarpaulin --out Html --output-dir coverage --exclude-files 'tests/*' --exclude-files 'src/main.rs' --timeout 300 -v $(CARGO_FLAGS)
	@echo "✅ Coverage report generated at coverage/index.html"

clean:
	@echo "Cleaning build artifacts..."
	cargo clean
	rm -rf coverage/
	rm -rf target/

fmt:
	@echo "Formatting code..."
	cargo fmt

lint:
	@echo "Running clippy..."
	cargo clippy --all-targets --all-features -- -D warnings

doc:
	@echo "Generating documentation..."
	cargo doc --no-deps --open

release:
	@echo "Building release binary..."
	cargo build --release $(CARGO_FLAGS)
	@echo "✅ Release binary at target/release/gemini_bridge"

docker-build:
	@echo "Building Docker image..."
	docker build -t gemini-bridge:latest .

docker-push:
	@echo "Pushing Docker image..."
	docker push gemini-bridge:latest

run:
	@echo "Starting services with docker-compose..."
	docker compose up -d
	@echo "✅ Services started. Check logs with: make logs"

stop:
	@echo "Stopping services..."
	docker compose down

logs:
	@echo "Showing docker-compose logs..."
	docker compose logs -f

dev: build test lint
	@echo "✅ Development checks complete"

pre-commit: fmt lint test
	@echo "✅ Pre-commit checks complete"

all: clean build test lint doc
	@echo "✅ All checks complete"
