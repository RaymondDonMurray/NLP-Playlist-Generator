# Dockerfile
# ADR-005: Base image python:3.11-slim (not python:3.11 — bloated, not python:latest — non-deterministic)
# scope.md §Docker: All data artifacts mounted as volumes, never baked into image.

FROM python:3.12-slim

# System dependencies
# - curl: Ollama health checks
# - gcc, g++: Some Python packages need compilation (e.g., rank_bm25, rapidfuzz)
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies before copying source (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY config/ ./config/

# Create directories for mounted volumes (scope.md §Docker: mount as volumes)
# data/, chroma_db/, runs/, checkpoints/ are gitignored — mounted at runtime
RUN mkdir -p data/raw data/processed chroma_db runs checkpoints/biencoder eval

# eval/queries.json is NOT gitignored — it's committed as part of the project
COPY eval/ ./eval/

# Make Python find the src package
ENV PYTHONPATH=/app

# Default: run the full eval pipeline
# Override with docker-compose command or docker run args
CMD ["python", "scripts/run_eval.py", "--config", "config/config.yaml"]
