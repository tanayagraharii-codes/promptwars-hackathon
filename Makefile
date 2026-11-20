# Developer shortcuts. Run `make help` to list them.
.DEFAULT_GOAL := help
PY ?= python3

.PHONY: help install run test lint format typecheck audit check

help: ## List available commands
	@grep -E '^[a-z]+:.*## ' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

install: ## Install Python and frontend dev dependencies
	$(PY) -m pip install -r requirements-dev.txt
	npm ci

run: ## Run the app locally on http://localhost:8080
	uvicorn app.main:app --reload --port 8080

test: ## Run Python (with coverage) and frontend tests
	pytest --cov
	npm test

lint: ## Lint Python and JavaScript, and check formatting
	ruff check app tests
	ruff format --check app tests
	npm run lint
	npm run format:check

format: ## Auto-format Python and frontend code
	ruff format app tests
	ruff check --fix app tests
	npm run format

typecheck: ## Strict static type checking
	mypy

audit: ## Scan dependencies for known vulnerabilities
	pip-audit -r requirements.txt --strict
	npm audit --omit=optional

check: lint typecheck test audit ## Everything CI runs
