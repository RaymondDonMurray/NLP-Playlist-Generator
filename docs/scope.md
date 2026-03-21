# Project Scope: NLP Music Playlist Retrieval System
### *A Recommender-as-Search Portfolio Project for MLE (RecSys) Roles*

---

## North Star

> A user types *"chill rainy afternoon jazz with some melancholy"* and receives a ranked playlist of 10 tracks with per-track explanations. The system works by enriching a track catalog with LLM-generated descriptors, indexing into a vector store, and performing semantic retrieval + cross-encoder reranking — exactly the architecture behind production music recommenders.

**What makes this impressive to a hiring team:** The project demonstrates the full vertical slice of a RecSys MLE — data pipeline, feature engineering via LLM, dense retrieval, reranking, offline evaluation, and operational tooling. The story is not "I used an embedding model." The story is "I understood the failure modes of dense retrieval in recommendation contexts and systematically addressed them."

---

## MVP — Target: 20 Hours

*A shippable, evaluated, containerized system. No UI required.*

### Core Model & Retrieval Pipeline

**Enrichment layer** — the centerpiece of engineering depth.

For each track, generate a 2–3 sentence natural language descriptor by prompting a local LLM (Mistral 7B Instruct Q4_K_M via Ollama) with:
- Lyrics excerpt (first ~200 tokens, stripped of structural annotations)
- Structured metadata: tempo, key, energy, valence, acousticness, danceability (from Spotify audio features)
- Genre tags (from FMA)

Target descriptor format (enforced via structured prompting + Pydantic validation):
```
Mood: [X]. Tempo feel: [X]. Genre: [X]. Lyrical themes: [X]. [1–2 sentence narrative.]
```

Example output: *"Mood: melancholy, introspective. Tempo feel: slow. Genre: piano jazz. Lyrical themes: late-night loneliness, rain, isolation. Sparse acoustic piano with hushed brushed drums; feels like watching rain on a window at 2am."*

Fallback path for tracks with no lyrics: descriptor generated from metadata only; flag in catalog for downstream analysis.

**Vector index**

- Embed all descriptors with `BAAI/bge-base-en-v1.5` via `sentence-transformers` (runs fast on M1 Max unified memory)
- Store in ChromaDB with full metadata payload per record
- Target catalog: **3K–5K tracks** with complete enrichment (quality over quantity)
- Load strategy: generate all embeddings first → save to parquet → single bulk load into Chroma (avoid incremental writes across sessions due to known M1 macOS persistence quirks)

**Two-stage retrieval**

| Stage | Method | Output |
|---|---|---|
| Stage 1: ANN Retrieval | Dense cosine similarity, `bge-base-en-v1.5` | Top-50 candidates |
| Stage 2: Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Top-10 final playlist |

Additionally: BM25 retrieval over descriptor text (`rank_bm25`) fused with dense scores via Reciprocal Rank Fusion (RRF). Hybrid retrieval is a concrete ablation and directly reflects production architectures.

**Playlist output** — each track returned with: title, artist, descriptor snippet, retrieval score, rerank score, and a one-line LLM-generated explanation of *why* it matches the query.

---

### Basic Data Pipeline

**Sources:**

| Source | Data | Method | Notes |
|---|---|---|---|
| FMA (Free Music Archive) | Track metadata, genres, play counts | Direct download (`fma_metadata`, ~342MB) | `github.com/mdeff/fma` |
| Spotify Web API | Audio features per track | `spotipy`, ~2 min for 5K tracks at free tier | Fuzzy-join to FMA on (artist, title) |
| Genius API | Lyrics | Async crawler, ~5hr crawl for 5K tracks | ~60–70% coverage on indie tracks |

**Preprocessing steps:**

1. Fuzzy match FMA → Spotify IDs using `rapidfuzz` (threshold ≥ 85); drop unmatched rows rather than carrying bad joins forward
2. Lyrics cleaning: strip `[Verse]`/`[Chorus]` annotations, truncate to 200 tokens, drop non-English tracks
3. Normalize all Spotify audio features to [0, 1] range before including in LLM prompts
4. Deduplicate on (artist, track_name) post-matching
5. Save final `catalog.parquet` and `lyrics.parquet` before any enrichment

**Schema for one enriched track record:**
```json
{
  "track_id": "fma_012345",
  "title": "...",
  "artist": "...",
  "genre_top": "Jazz",
  "spotify_id": "...",
  "tempo": 72.4,
  "energy": 0.31,
  "valence": 0.18,
  "acousticness": 0.89,
  "lyrics_available": true,
  "descriptor": "Mood: melancholy...",
  "embedding": [0.032, -0.114, ...]
}
```

---

### Evaluation Suite

*This is what separates a portfolio project from a notebook demo. Build this before you finish the pipeline.*

**Offline evaluation dataset:**
- 50 hand-authored queries spanning: mood-based (*"euphoric summer driving"*), genre-specific (*"modal jazz with spiritual undertones"*), hybrid (*"melancholy indie folk for a long train ride"*), and abstract (*"music that sounds like the end of something"*)
- ~12 relevant tracks labeled per query from the catalog (author-labeled; documented explicitly in README)
- Stored as `eval/queries.json` with structure `{"query": "...", "relevant_ids": ["fma_...", ...]}`

**Metrics reported:**

| Metric | Why |
|---|---|
| Precision@10 | Did the top-10 contain relevant tracks? |
| NDCG@10 | Were the most relevant tracks ranked highest? |
| MRR | Was there at least one relevant track near the top? |
| Novelty@10 | Inverse popularity of retrieved tracks (anti-popularity-bias signal) |

**Ablation table (required):**

| Config | P@10 | NDCG@10 | MRR | Novelty |
|---|---|---|---|---|
| Dense only (bge-base) | | | | |
| BM25 only | | | | |
| Hybrid (RRF) | | | | |
| Hybrid + cross-encoder rerank | | | | |

This table *is* the engineering contribution. Run it. Put it in the README.

---

### Docker Container

```
music-nlp-retrieval/
├── Dockerfile
├── docker-compose.yml        # mounts ./data and ./chroma_db volumes
├── requirements.txt
├── scripts/
│   ├── build_catalog.py      # FMA + Spotify + Genius pipeline
│   ├── generate_descriptors.py
│   ├── build_index.py
│   └── run_eval.py
├── src/
│   ├── retriever.py
│   ├── ranker.py
│   └── pipeline.py           # end-to-end query → playlist
└── eval/
    ├── queries.json
    └── results.csv
```

- Base image: `python:3.11-slim`
- Ollama runs as a sidecar service in `docker-compose.yml` with the model volume mounted
- All data artifacts mounted as volumes (not baked into image)
- `docker-compose up` → indexes built, eval runs, results printed
- Tested on both macOS (M1) and Linux (AMD64)

---

### Comprehensive README

Sections:

1. **System overview** — architecture diagram (Mermaid), one-paragraph narrative
2. **Motivation** — why this framing (recommender as search) maps to production systems
3. **Quickstart** — `docker-compose up` to full working demo in under 5 commands
4. **Data pipeline** — sources, preprocessing, coverage rates, fuzzy matching strategy
5. **Enrichment layer** — prompt design, LLM choice rationale, failure modes
6. **Retrieval & reranking** — two-stage architecture, why cross-encoder reranking, why hybrid BM25+dense
7. **Evaluation** — methodology, synthetic dataset caveat (stated explicitly), ablation table
8. **Known limitations** — catalog coverage ceiling, LLM descriptor quality variance, cold-start on very niche genres
9. **What I would do with more time** — brief roadmap pointing to stretch goals

---

## Stretch Tier 1 — Hours 21–40
*Production engineering signals. Each is an independent module; prioritize in order listed.*

### Popularity Bias Analysis & Debiasing
*(~6 hrs, highest impressiveness-to-effort ratio)*

Compute the popularity distribution of retrieved tracks vs. the full catalog. Naive dense retrieval over-retrieves popular tracks because their descriptors are richer (more lyrics coverage, more complete metadata). Implement inverse-popularity score reweighting at retrieval time. Report before/after NDCG + Novelty@10 on the eval set.

This is a *classic* RecSys production problem. Any interviewer focused on recommender systems will ask about it, and having a concrete measured result signals you understand the domain, not just the tools.

### Experiment Tracking Integration
*(~4 hrs)*

Integrate MLflow (lightweight, runs locally):
- Track every eval run: hyperparameters (retrieval top-k, reranking model, hybrid fusion weight), all four metrics
- Log descriptor generation: prompt version, LLM model, coverage rate
- Store ablation results as named experiment runs with tags
- Export a `experiments/` directory with `mlflow ui`-compatible artifacts

Document: *"Every result in the README ablation table is reproducible from a named MLflow run."*

### Query Expansion via LLM
*(~4 hrs)*

Before embedding a user query, prompt a local LLM to expand it:

> *"chill rainy afternoon jazz"* → *"slow tempo, minor key, acoustic or piano-led jazz, melancholy, introspective, sparse instrumentation, late night, Bill Evans, Kind of Blue adjacent"*

Measure retrieval improvement on the eval set. Add to ablation table as a fifth row. Frame this as a concrete ablation — not just a feature addition.

### CI/CD Pipeline
*(~5 hrs)*

GitHub Actions workflow with two jobs:

**`test` job** (triggers on every push):
- Unit tests for `retriever.py` and `ranker.py` (mock ChromaDB, test scoring logic)
- Schema validation: assert enriched catalog parquet matches expected columns/dtypes
- Descriptor generation: test prompt formatting + Pydantic output validation on 5 fixture tracks

**`eval` job** (triggers on pushes to `main`):
- Runs the full eval harness on a 10-query smoke-test subset (fast, ~2 min)
- Fails if NDCG@10 drops >5% from the baseline stored in `eval/baseline_metrics.json`
- Posts metrics summary to the PR as a comment

This is the signal that you understand MLOps, not just ML. A GitHub Actions badge + test coverage report in the README is immediately visible to a reviewer.

### Monitoring Dashboard
*(~5 hrs)*

A lightweight Streamlit dashboard (local, not deployed) that reads `experiments/` artifacts and displays:
- Metric trends across eval runs over time
- Popularity distribution of retrieved vs. catalog tracks (histogram)
- Per-query NDCG breakdown (which queries are hardest)
- Descriptor coverage rates by genre

Framing: *"This is what a RecSys team would use to track retrieval quality during catalog updates or prompt iteration."* The point is not the UI — it's demonstrating you thought about observability.

---

## Stretch Tier 2 — Hours 41–80
*Senior/staff-level signals. Only attempt if Tier 1 is solid and stable.*

### Model Optimization: Embedding Quantization + Latency Benchmarking
*(~8 hrs)*

Quantize `bge-base-en-v1.5` to INT8 using ONNX Runtime:
1. Export model to ONNX
2. Apply dynamic INT8 quantization
3. Benchmark: FP32 vs INT8 on M1 Max — latency per query, throughput (queries/sec), retrieval quality delta on eval set

Document the tradeoff table explicitly. The narrative: *"For a production system serving thousands of queries per minute, a 2× latency reduction with <1% NDCG degradation is an easy engineering decision."*

Cross-encoder pruning: use `transformers` structured pruning to reduce the reranker by 30–40% while measuring Precision@10 on the eval set.

### A/B Testing Framework
*(~8 hrs)*

Implement a lightweight A/B framework for comparing retrieval configurations:
- Assign incoming queries to variant A or B (dense-only vs. hybrid)
- Log per-query results, variant assignment, and timestamps to SQLite
- Compute statistical significance of metric differences using a permutation test (directly applicable from your existing Python permutation testing work in the DA pipeline)
- Visualize A/B results in the monitoring dashboard

This is the single highest signal item for a RecSys MLE role. Production recommender teams live in A/B test results. Implementing even a toy version demonstrates you understand the full deployment lifecycle.

### Contrastive Fine-Tuning of the Embedding Model
*(~15 hrs)*

Use `sentence-transformers` training API to fine-tune `bge-base-en-v1.5` on music-specific triplets:
- Positive pairs: (query, relevant track descriptor) from your eval set + LLM-augmented variants
- Hard negatives: (query, same-genre but different-mood track) mined from catalog

Training runs on M1 Max unified memory; use MPS backend. Target: show NDCG@10 improvement of ≥3 points on the eval set vs. the off-the-shelf embedding. Document training curves in MLflow.

### Playlist Coherence Reranking via MMR
*(~10 hrs)*

Playlist quality is not just query-item relevance — a good playlist has internal coherence. After retrieving candidates, use Maximal Marginal Relevance (MMR) to select a diverse-but-coherent subset:

```
score(d) = λ · sim(query, d) − (1 − λ) · max_{d' ∈ selected} sim(d, d')
```

Tune `λ` on the eval set. Report a coherence metric (intra-playlist embedding variance) alongside the standard retrieval metrics. Show that MMR playlists score higher on coherence without significantly sacrificing relevance.

---

## Cut List
*Features that sound compelling but add no signal to an MLE RecSys hiring packet.*

| Feature | Why It's Cut |
|---|---|
| **Streamlit / web UI with audio playback** | Frontend engineering, not RecSys. The system is evaluated by the eval harness, not by how it looks. A demo notebook is sufficient. |
| **Real-time Spotify integration / OAuth flow** | API integration work, zero RecSys signal. Adds 8+ hours of auth/rate-limit debugging. |
| **Last.fm tag enrichment, Wikipedia artist bios, music theory analysis** | Marginal enrichment quality improvement at high pipeline complexity cost. Descriptor from lyrics + audio features is sufficient to demonstrate the concept. |
| **Deployed endpoint (Railway, Fly.io, etc.)** | DevOps work. Adds cost, maintenance overhead, and zero signal to a RecSys portfolio. "It runs in Docker" is sufficient. |
| **Collaborative filtering / matrix factorization layer** | Would require user interaction data you don't have. A synthetic user-item matrix is not convincing. The project's thesis is content-based retrieval — stay on thesis. |
| **CLAP audio embeddings (unless you have 40+ free hours)** | Excellent idea, but getting licensed audio for 5K tracks from FMA is painful, CLAP inference is slow, and this is Tier 3 scope. The project is strong without it. |
| **Multi-language support** | Requires separate lyrics cleaning, evaluation queries, and LLM prompting logic. Restrict to English for MVP and stretch goals. |
| **User preference learning / session modeling** | Requires user data. Out of scope for a catalog enrichment + retrieval project. |
| **Fine-tuning the LLM for descriptor generation** | You'd need ground-truth human-written descriptors. You don't have them. Prompt engineering is the right tool here. |
| **Graph-based artist similarity modeling** | Interesting research direction, but requires a graph construction pipeline and a GNN. This is a different project. |

---

## Risk Register

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Fuzzy join (FMA → Spotify) eats time | High | Medium | Hard threshold at 85; drop unmatched; 3K clean tracks > 5K dirty ones |
| Lyrics crawl bottleneck blocks enrichment | Medium | High | Start crawl overnight before enrichment day; build fallback path first |
| LLM descriptor quality inconsistent | High | High | Tune prompt on 20 tracks before batching; enforce structured output + Pydantic; retry on malformed |
| Evaluation feels circular (self-labeled) | Certain | Low | Acknowledge explicitly in README; frame correctly; the methodology is standard at this scale |
| ChromaDB persistence issues on M1 macOS | Medium | Medium | Generate embeddings → parquet → single bulk load; never write incrementally |
| Scope creep on enrichment schema | Certain | High | Freeze schema after hour 6; document schema in design doc before writing code |
| Demo feels bad even when metrics are fine | Medium | Low | Build 6 curated queries you know work for the README walkthrough; cover limitations honestly |

---

## Hourly Timeline

| Hours | Work | Deliverable |
|---|---|---|
| 1–4 | FMA metadata, Spotify API setup, fuzzy join | `catalog.parquet` (~4K tracks, audio features merged) |
| 5–6 | Genius API crawler, lyrics cleaning | `lyrics.parquet`, coverage rate documented |
| 7–9 | Descriptor prompt tuning + batch generation (Mistral 7B via Ollama) | `descriptors.parquet` |
| 10–12 | Embed all descriptors, bulk load ChromaDB, smoke test retrieval | Working `retriever.py` |
| 13–15 | Cross-encoder reranking + BM25/RRF hybrid fusion | Working `ranker.py`, end-to-end query → playlist |
| 16–18 | Write 50 eval queries, label relevant tracks, run ablation table | `eval/queries.json`, `eval/results.csv` |
| 19–20 | Dockerfile, docker-compose, README, demo notebook | **Shippable MVP** |
| 21–26 | Popularity bias analysis + debiasing | Updated ablation table with Novelty@10 |
| 27–30 | MLflow experiment tracking + query expansion | Reproducible eval runs, expansion ablation |
| 31–35 | CI/CD (GitHub Actions: test + eval jobs) | Green CI badge |
| 36–40 | Monitoring dashboard (Streamlit, local) | `dashboard/` with metric trend views |
| 41–48 | Embedding quantization + latency benchmarking | FP32 vs INT8 tradeoff table |
| 49–56 | A/B testing framework + permutation significance test | `ab_testing/` module |
| 57–72 | Contrastive fine-tuning of embedding model | Fine-tuned checkpoint, NDCG delta |
| 73–80 | MMR coherence reranking, final polish, portfolio writeup | Complete project |

---

## Pre-Code Checklist

Before writing a single line of implementation code, complete and commit the following:

- [ ] `design_doc.md`: schema for one enriched track record, evaluation dataset structure, list of three primary ablations
- [ ] `eval/queries.json`: stub file with first 10 evaluation queries written
- [ ] Prompt template v0 tested manually against 5 tracks in an Ollama session
- [ ] Fuzzy join threshold validated on a 50-track sample before full catalog run
- [ ] `scope.md` (this document) reviewed and signed off

*The design doc makes evaluation a first-class citizen from hour zero. That discipline is the difference between an MLE candidate and a data scientist who codes.*

---

*Last updated: 2026-03-21*