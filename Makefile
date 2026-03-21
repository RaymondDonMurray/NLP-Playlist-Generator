# Makefile
# NLP Music Playlist Retrieval System
# ADR-006: Single entry point for all pipeline operations.
# scope.md §Docker: docker-compose up → full demo.

.PHONY: help install test test-unit test-smoke lint \
        catalog descriptors index eval eval-smoke \
        serve-retrieve serve-rerank serve-explain \
        docker-build docker-up docker-down tensorboard clean

CONFIG ?= config/config.yaml

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
install:  ## Install Python dependencies
	pip install -r requirements.txt

install-dev:  ## Install deps + dev tools
	pip install -r requirements.txt
	pip install black isort mypy

# ---------------------------------------------------------------------------
# Testing (scope.md §CI/CD)
# ---------------------------------------------------------------------------
test:  ## Run full test suite with coverage
	pytest tests/ -v --cov=src --cov-report=term-missing

test-unit:  ## Run unit tests only (no integration)
	pytest tests/ -v -m "not integration"

test-smoke:  ## Run fast smoke test subset (CI mode — ~2min)
	pytest tests/ -v -m "smoke"

test-schema:  ## Run schema validation tests only
	pytest tests/data/test_schema.py -v

test-metrics:  ## Run metrics tests only (pure functions, fast)
	pytest tests/eval/test_metrics.py -v

# ---------------------------------------------------------------------------
# Data pipeline (scope.md §hourly timeline)
# ---------------------------------------------------------------------------
catalog:  ## Step 1: Build FMA+Spotify+Genius catalog → catalog.parquet
	python scripts/build_catalog.py --config $(CONFIG)

catalog-dry-run:  ## Validate pipeline setup without API calls
	python scripts/build_catalog.py --config $(CONFIG) --dry-run

descriptors:  ## Step 2: Generate Mistral 7B descriptors → descriptors.parquet
	python scripts/generate_descriptors.py --config $(CONFIG)

descriptors-resume:  ## Resume descriptor generation from latest checkpoint
	python scripts/generate_descriptors.py --config $(CONFIG) --resume

index:  ## Step 3: Embed + bulk load → ChromaDB dual indexes
	python scripts/build_index.py --config $(CONFIG)

pipeline:  ## Run full pipeline: catalog → descriptors → index
	$(MAKE) catalog && $(MAKE) descriptors && $(MAKE) index

# ---------------------------------------------------------------------------
# Evaluation (scope.md §evaluation suite)
# ---------------------------------------------------------------------------
eval:  ## Run full ablation eval matrix → eval/results.csv + TensorBoard
	python scripts/run_eval.py --config $(CONFIG) --print-table

eval-smoke:  ## CI eval: 10 queries only (~2 min) — scope.md §CI/CD
	python scripts/run_eval.py --config $(CONFIG) --smoke-test

eval-regression:  ## Check for NDCG regression vs baseline (CI gate)
	python scripts/run_eval.py --config $(CONFIG) --smoke-test --check-regression

eval-config:  ## Run a single ablation config: make eval-config CONFIG_NAME=hybrid_rerank
	python scripts/run_eval.py --config $(CONFIG) --config-name $(CONFIG_NAME)

# ---------------------------------------------------------------------------
# Serving CLI (ADR-005: three decomposed commands)
# ---------------------------------------------------------------------------
serve-retrieve:  ## Retrieve top-50 candidates for a query
	@echo "Usage: make serve-retrieve QUERY='chill rainy afternoon jazz'"
	python -m src.serving.cli retrieve \
		--query "$(QUERY)" \
		--top-k 50 \
		--index-tier auto \
		--json-output \
		--config $(CONFIG)

serve-rerank:  ## Rerank candidates from a retrieve output file
	@echo "Usage: make serve-rerank QUERY='...' CANDIDATES=candidates.json"
	python -m src.serving.cli rerank \
		--query "$(QUERY)" \
		--candidates $(CANDIDATES) \
		--top-k 10 \
		--json-output \
		--config $(CONFIG)

serve-explain:  ## Generate per-track explanations (async/offline)
	@echo "Usage: make serve-explain QUERY='...' TRACKS=tracks.json"
	python -m src.serving.cli explain \
		--query "$(QUERY)" \
		--tracks $(TRACKS) \
		--json-output \
		--config $(CONFIG)

# ---------------------------------------------------------------------------
# Docker (scope.md §Docker container)
# ---------------------------------------------------------------------------
docker-build:  ## Build Docker image
	docker build -t music-nlp-retrieval .

docker-up:  ## Start full stack: Ollama sidecar + pipeline
	docker-compose up

docker-up-detached:  ## Start stack in background
	docker-compose up -d

docker-down:  ## Stop and remove containers
	docker-compose down

docker-logs:  ## Tail pipeline service logs
	docker-compose logs -f pipeline

# ---------------------------------------------------------------------------
# Experiment tracking (ADR-004: TensorBoard)
# ---------------------------------------------------------------------------
tensorboard:  ## Launch TensorBoard on runs/
	tensorboard --logdir runs/

# ---------------------------------------------------------------------------
# Code quality
# ---------------------------------------------------------------------------
lint:  ## Run flake8 + mypy type checks
	flake8 src/ tests/ --max-line-length 100
	mypy src/ --ignore-missing-imports

format:  ## Auto-format with black + isort
	black src/ tests/ scripts/ --line-length 100
	isort src/ tests/ scripts/

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
clean-checkpoints:  ## Remove descriptor generation checkpoints
	rm -rf checkpoints/desc_*.parquet

clean-staging:  ## Remove ChromaDB staging index
	rm -rf chroma_db/staging

clean-runs:  ## Remove TensorBoard logs (careful — irreversible)
	rm -rf runs/

clean-pyc:  ## Remove Python cache files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true

clean: clean-pyc  ## Safe cleanup (pyc only)
