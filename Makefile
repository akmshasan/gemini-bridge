.PHONY: help install sync lock test test-cov lint lint-fix format format-check check clean clean-all run

export PYTHONPATH := src

help: ## Show available Makefile targets

	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: sync ## Install dependencies and sync virtual environment

sync: ## Sync dependencies with uv
	uv sync

lock: ## Update and lock dependencies
	uv lock

run: ## Run the application
	uv run agent-gateway

test: ## Run test suite
	uv run pytest

test-cov: ## Run test suite with coverage report
	uv run pytest --cov=src --cov-report=term-missing

lint: ## Run linter (ruff)
	uv run ruff check .

lint-fix: ## Automatically fix linter issues
	uv run ruff check --fix .

format: ## Format code with ruff
	uv run ruff format .

format-check: ## Check code formatting with ruff
	uv run ruff format --check .

check: lint format-check ## Run all code quality checks (lint & format check)

clean: ## Remove caches and temporary build artifacts
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf .coverage htmlcov
	rm -rf build dist *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

clean-all: clean ## Remove caches, temporary files, and virtual environment
	rm -rf .venv
