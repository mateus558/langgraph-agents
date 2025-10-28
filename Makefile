.PHONY: help install install-dev format lint type-check test test-cov clean run-chat run-websearch

help:
	@echo "Available commands:"
	@echo "  make install       - Install production dependencies"
	@echo "  make install-dev   - Install development dependencies"
	@echo "  make format        - Format code with black"
	@echo "  make lint          - Lint code with ruff"
	@echo "  make type-check    - Type check with mypy"
	@echo "  make test          - Run tests"
	@echo "  make test-cov      - Run tests with coverage"
	@echo "  make clean         - Remove build artifacts"
	@echo "  make pre-commit    - Install pre-commit hooks"
	@echo "  make run-chat      - Run chat agent"
	@echo "  make run-websearch - Run websearch agent"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

format:
	black src/
	ruff check --fix src/

lint:
	ruff check src/

type-check:
	mypy src/

test:
	pytest

test-cov:
	pytest --cov=src --cov-report=html --cov-report=term

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

pre-commit:
	pre-commit install

run-chat:
	python -m src.chatagent.agent

run-websearch:
	python -m src.websearch.agent
