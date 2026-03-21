# CLAUDE.md
## NLP Music Playlist Retrieval System — Agent Conventions

This file defines coding standards, project conventions, and key constraints for any coding agent working in this repository. Read this before writing any code.

---

## Project Overview

A music playlist retrieval system framed as a search problem. A user submits a free-text query ("chill rainy afternoon jazz with some melancholy") and receives a ranked playlist of 10 tracks with per-track explanations. The system enriches a track catalog with LLM-generated descriptors, indexes them into a vector store, and performs semantic retrieval + cross-encoder reranking.

**Stack:** Python 3.11 · PyTorch (MPS backend, M1 Max) · sentence-transformers · ChromaDB · Mistral 7B via Ollama · TensorBoard · Docker

---

## Repository Layout

```
music-nlp-retrieval/
├── src/
│   ├── data/
│   │   ├── ingest.py          # FMA, Spotify, Genius API clients
│   │   ├── preprocess.py      # fuzzy join, lyrics cleaning, validation
│   │   └── schema.py          # all Pydantic models — single source of truth
│   ├── models/
│   │   ├── descriptor.py      # Mistral 7B prompt + output parser
│   │   ├── encoder.py         # bge-base-en-v1.5 inference wrapper
│   │   ├── reranker.py        # cross-encoder ms-marco wrapper
│   │   └── finetune.py        # sentence-transformers training (Tier 2)
│   ├── serving/
│   │   ├── retriever.py       # Retriever class with index_tier logic
│   │   ├── ranker.py          # two-stage pipeline: retrieve → rerank
│   │   ├── explainer.py       # per-track explanation via Mistral 7B
│   │   └── cli.py             # three decomposed CLI commands
│   └── eval/
│       ├── metrics.py         # NDCG, MRR, Precision@k, Novelty@k
│       ├── runner.py          # loads queries.json, runs retrieval
│       └── ablations.py       # full ablation matrix orchestration
├── scripts/
│   ├── build_catalog.py       # end-to-end data pipeline entry point
│   ├── generate_descriptors.py
│   ├── build_index.py
│   └── run_eval.py
├── eval/
│   ├── queries.json           # 50 evaluation queries with relevant_ids
│   └── results.csv            # ablation table output
├── config/
│   └── config.yaml            # single config file, loaded at entry points
├── runs/                      # TensorBoard logs (gitignored)
├── checkpoints/               # model + descriptor checkpoints (gitignored)
├── data/                      # raw + processed data artifacts (gitignored)
├── docs/
│   ├── ADR.md
│   ├── scope.md
│   ├── failure_modes.md
│   └── CLAUDE.md              # this file
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

**Import discipline:** strict downward-only. `serving/` imports from `models/`. `models/` imports from `data/schema.py` only. `eval/` imports from `serving/` via the public `Retriever` interface. No circular imports. No `eval/` importing from `models/` directly.

--- 

## Imports

@docs/scope.md
@docs/architecture_decision_record.md
@docs/failure_modes.md

---

## Configuration

All tunable parameters live in `config/config.yaml`. Entry point scripts load this file via a Pydantic settings model. API keys (Spotify, Genius) live in `.env` only — never in `config.yaml`, never hardcoded.

```yaml
# config/config.yaml — canonical parameter list
pipeline:
  checkpoint_interval: 100        # tracks per checkpoint
  fuzzy_match_threshold: 85       # rapidFuzz title threshold
  artist_match_threshold: 70      # rapidFuzz artist threshold (two-pass)
  max_lyrics_tokens: 200
  min_lyrics_tokens: 20
  speechiness_max: 0.5            # filter podcasts/comedy
  duration_ms_min: 30000          # filter intros/outros

enrichment:
  ollama_model: "mistral:7b-instruct-q4_K_M"
  ollama_timeout_s: 30
  max_retries: 3
  prompt_version: "v1"            # bump when prompt template changes

retrieval:
  bi_encoder: "BAAI/bge-base-en-v1.5"
  cross_encoder: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  ann_top_k: 50
  rerank_top_k: 10
  confidence_threshold: 0.65      # auto-routing threshold (tunable)
  rerank_variance_threshold: 0.05 # fallback if cross-encoder is flat

training:
  output_dir: "checkpoints/biencoder"
  num_epochs: 10
  batch_size: 32
  hard_negative_weight: 0.3       # combined with in-batch negatives

eval:
  queries_path: "eval/queries.json"
  results_path: "eval/results.csv"
  ndcg_k: 10
  early_stop_patience: 3
  early_stop_min_delta: 0.005

tensorboard:
  log_dir: "runs/"
  log_every_n_steps: 50
```

**Rule:** If a number appears in source code that affects pipeline behavior or model quality, it belongs in `config.yaml` with a comment, not as a magic constant inline.

---

## Coding Standards

### General

- Python 3.12. Type-annotate all function signatures — parameters and return types.
- Line length: 100 characters.
- Docstrings on all public functions and classes. One-line for simple helpers, full Google-style for anything with non-obvious behavior.
- No `print()` in library code. Use `logging.getLogger(__name__)` everywhere. Entry point scripts configure the root logger; library modules only call `logger.warning(...)`, `logger.info(...)`, etc.
- Prefer explicit over implicit. `enrichment_tier: Literal["full", "metadata_only"]` is better than `is_full_enrichment: bool`.

### Pydantic

All data contracts are defined in `src/data/schema.py`. This is the single source of truth for every data structure in the pipeline.

- Use Pydantic v2 (`model_validate`, `field_validator`, `model_validator`).
- Schema violations at pipeline ingestion are **hard failures** — raise, don't log-and-continue.
- Schema violations at enrichment (descriptor generation, reranking) are **soft failures** — log a warning, return `None`, continue pipeline.
- Never pass `None` into an LLM prompt. Use the `prompt_features` property pattern to convert nullable fields to `"unknown X"` strings.

```python
# CORRECT
@property
def prompt_features(self) -> dict:
    return {
        "tempo": f"{self.tempo:.0f} BPM" if self.tempo else "unknown tempo",
        "energy": f"{self.energy:.2f}" if self.energy else "unknown energy",
    }

# WRONG — passes None into prompt template
{"tempo": self.tempo, "energy": self.energy}
```

### Error Handling

Two categories of failure — treat them differently:

| Category | Examples | Behavior |
|---|---|---|
| Data contract violation | Missing required field, bad fuzzy join, corrupt parquet schema | Raise immediately. Do not continue. |
| Quality degradation | Low confidence descriptor, OOV query token, flat rerank scores | Log warning. Set `warning` field in response. Continue. |

Never catch `Exception` broadly. Catch specific exception types (`httpx.TimeoutException`, `ValidationError`, `FileNotFoundError`) and handle each explicitly.

Retry logic: use exponential backoff for transient failures (network timeout, 503). Never retry `ValidationError` — the prompt or schema is wrong and retrying produces the same output.

```python
# CORRECT — specific, differentiated
except httpx.TimeoutException:
    time.sleep(2 ** attempt)   # retry with backoff
except ValidationError:
    return None                # don't retry schema failures

# WRONG — swallows everything
except Exception:
    return None
```

### Data Pipeline

- Checkpoint descriptor generation every `config.pipeline.checkpoint_interval` tracks to `checkpoints/desc_{prompt_hash}_{batch_idx:04d}.parquet`.
- Write checkpoints with an atomic manifest: write to `.tmp` then `os.replace()`.
- Validate every checkpoint against its manifest row count before reading.
- Tag every checkpoint filename with the prompt version hash (`hashlib.md5(prompt_template.encode()).hexdigest()[:8]`). Assert all checkpoints share the same hash at merge time.
- Never write to the production ChromaDB index directly. Build to staging, validate with spot checks, then `shutil.move()` to production.

### Serving

The CLI exposes three decomposed commands. Keep them separate — do not merge into a single `recommend` command:

```bash
python -m music_retrieval.serving retrieve  --query "..." --top-k 50  --index-tier auto  --json-output
python -m music_retrieval.serving rerank    --query "..." --candidates candidates.json --top-k 10 --json-output
python -m music_retrieval.serving explain   --query "..." --tracks tracks.json --json-output
```

All CLI inputs are validated through Pydantic request models before any inference code runs. All responses are Pydantic response models serialized to JSON. The `warning` field in every response is the designated channel for soft failure signals (retrieval collapse, low rerank variance, OOV tokens) — use it.

### TensorBoard Logging

Log to `runs/{YYYYMMDD}_{prompt_hash_6char}_{short_description}`. One run per experiment.

Log these and only these:

| Scope | Keys |
|---|---|
| Training | `train/loss` every N steps · `val/loss` per epoch · `val/ndcg_at_10`, `val/mrr`, `val/precision_at_10` per eval checkpoint |
| Pipeline | `pipeline/coverage_rate` · `pipeline/retry_rate` · `pipeline/prompt_version` |
| Retrieval eval | `eval/ndcg_at_10` · `eval/mrr` · `eval/precision_at_10` · `eval/novelty_at_10` |

No embedding projector. No custom dashboards beyond the scalar summaries above.

---

## Key Models & Their Roles

| Model | Role | Notes |
|---|---|---|
| `BAAI/bge-base-en-v1.5` | Bi-encoder for ANN retrieval and query embedding | MPS backend on M1 Max; ~35ms/query |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker (top-50 → top-10) | ~12ms for 50 pairs on MPS |
| `mistral:7b-instruct-q4_K_M` (Ollama) | Descriptor generation + per-track explanation | ~2s/track; runs as sidecar in docker-compose |

Never swap models without updating `config.yaml` and re-running the full eval suite. Model names in source code should always read from config, not be hardcoded strings.

---

## Dual Index Architecture

Two ChromaDB collections, never one:

| Collection | Contents | When used |
|---|---|---|
| `full_enrichment` | Tracks with lyrics + audio features | Query confidence ≥ `config.retrieval.confidence_threshold` |
| `metadata_only` | Tracks with audio features only (no lyrics) | Query confidence < threshold, or fallback |

The `Retriever` class handles routing transparently via `index_tier="auto"`. Confidence is the cosine similarity between the query embedding and the centroid of the top-5 candidates from `full_enrichment`. The threshold is a tunable hyperparameter — report its sensitivity curve in the eval notebook.

When reporting eval metrics, always specify which index tier was used. Do not mix results from both tiers in a single NDCG number.

---

## Evaluation

The eval harness in `eval/` is first-class code — not a notebook, not a script that runs once. It must be:

- Runnable from a single command: `python scripts/run_eval.py --config config/config.yaml`
- Deterministic: same config → same results
- Logged to TensorBoard under a named run

The ablation matrix (5 configurations minimum):

| Config | Description |
|---|---|
| `dense_only` | bge-base ANN, no reranking, no BM25 |
| `bm25_only` | BM25 over descriptor text, no dense |
| `hybrid_rrf` | Dense + BM25 fused via RRF |
| `hybrid_rerank` | Hybrid + cross-encoder reranking |
| `hybrid_rerank_expanded` | Hybrid + reranking + LLM query expansion |

Every row in the README ablation table must correspond to a named, reproducible TensorBoard run. If the run doesn't exist, the number doesn't go in the table.

Eval dataset: `eval/queries.json` with structure:
```json
[
  {
    "query": "chill rainy afternoon jazz with some melancholy",
    "relevant_ids": ["fma_012345", "fma_067890"],
    "source": "author_labeled",
    "notes": "optional"
  }
]
```

The `source: "author_labeled"` field is required and must be acknowledged in the README. Do not present NDCG numbers without the caveat that the eval set is self-labeled.

---

## Defensive Code Patterns

These patterns appear throughout the codebase. Use them consistently.

**Two-pass fuzzy match validation** (`data/preprocess.py`): validate both title AND artist similarity; write suspicious matches to a review CSV before proceeding.

**Audio feature range guards** (`data/schema.py`): validate all Spotify features against `FEATURE_VALID_RANGES`; set to `None` on out-of-range rather than passing sentinel values downstream.

**Retrieval collapse detection** (`serving/retriever.py`): if >80% of top-50 candidates share one genre and the query doesn't name that genre, log a warning and set `response.warning`.

**Rerank fallback** (`serving/ranker.py`): if cross-encoder score std < `config.retrieval.rerank_variance_threshold`, return ANN order and set `response.warning = "low_rerank_variance"`.

**Prompt feature safety** (`data/schema.py`): all nullable fields are converted to `"unknown X"` strings via `CatalogRecord.prompt_features` before being passed to the LLM. Never pass `None`.

**Checkpoint manifest** (`scripts/generate_descriptors.py`): write row count manifest atomically alongside every checkpoint; validate before reading.

**Staging index promotion** (`scripts/build_index.py`): build to `chroma_db/staging/`, validate with spot checks, then `shutil.move()` to `chroma_db/production/`.

---

## What Not to Build

These were explicitly cut from scope. Do not implement them:

- Streamlit or web UI
- Real-time Spotify OAuth integration
- Deployed HTTP endpoint (Railway, Fly.io, etc.)
- Collaborative filtering or matrix factorization
- Multi-language lyrics support
- Fine-tuning the LLM for descriptor generation
- Graph-based artist similarity
- Embedding projector in TensorBoard

If a new feature is proposed that isn't in `docs/scope.md`, check the cut list first. If it's there, don't build it without explicit instruction.

---

## Docker

Ollama runs as a sidecar service in `docker-compose.yml` with the model volume mounted. Data artifacts are mounted as volumes — never baked into the image. The target behavior: `docker-compose up` builds indexes, runs eval, and prints results.

```yaml
# docker-compose.yml structure
services:
  ollama:
    image: ollama/ollama
    volumes:
      - ollama_models:/root/.ollama
  pipeline:
    build: .
    volumes:
      - ./data:/app/data
      - ./chroma_db:/app/chroma_db
      - ./runs:/app/runs
    depends_on:
      - ollama
```

Base image: `python:3.11-slim`. Do not use `python:3.11` (bloated) or `python:latest` (non-deterministic).

---

*Last updated: 2026-03-21*