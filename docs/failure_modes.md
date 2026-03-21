# Failure Modes & Defensive Engineering
## NLP Music Playlist Retrieval System

**Date:** 2026-03-21

> **Design principle:** Fail loudly on data contract violations, fail gracefully on quality degradation. A missing Pydantic field should crash the pipeline immediately at ingest. A low-confidence descriptor should log a warning and continue. Conflating these two categories produces code that is either too brittle or too permissive.

---

## Table of Contents

1. [Data Quality Issues](#1-data-quality-issues)
2. [Model Failure Cases](#2-model-failure-cases)
3. [Pipeline Edge Cases](#3-pipeline-edge-cases)
4. [Infrastructure Failure Modes](#4-infrastructure-failure-modes)
5. [Failure Mode Register](#5-failure-mode-register)

---

## 1. Data Quality Issues

### 1.1 Fuzzy Join False Positives

**What goes wrong:** rapidFuzz at threshold 85 will confidently match "Blue (Da Ba Dee)" by Eiffel 65 to "Blue" by LeAnn Rimes. False matches produce LLM descriptors seeded with incorrect genre and audio features. This poisons the index silently — there is no runtime error, only degraded NDCG.

**When it surfaces:** Not until eval. A query for "electronic dance" returns a country ballad with high confidence.

**Defense:** Two-pass validation using both artist AND title similarity. Matches where artist similarity drops below 70 despite title similarity above 85 are flagged for manual review, not silently accepted.

```python
from rapidfuzz import fuzz
from enum import Enum

class MatchQuality(Enum):
    HIGH = "high"
    SUSPICIOUS = "suspicious"
    REJECT = "reject"

def validate_match(fma_row, spotify_row) -> MatchQuality:
    title_sim = fuzz.token_sort_ratio(fma_row.title, spotify_row.title)
    artist_sim = fuzz.token_sort_ratio(fma_row.artist, spotify_row.artist)

    if title_sim >= 85 and artist_sim >= 70:
        return MatchQuality.HIGH
    elif title_sim >= 85 and artist_sim >= 40:
        return MatchQuality.SUSPICIOUS   # flag, don't drop
    else:
        return MatchQuality.REJECT

# Write suspicious matches to a review CSV before proceeding
```

> **Operational note:** Spot-check 50 HIGH and 50 SUSPICIOUS matches manually before running descriptor generation. A 3-hour LLM batch seeded from bad joins cannot be recovered cheaply.

---

### 1.2 Lyrics Encoding Garbage

**What goes wrong:** Genius serves HTML with escaped Unicode, non-breaking spaces (`\xa0`), and occasionally mixed-script annotations. A descriptor generated from `"سئم\xa0تعبت\xa0weariness and longing"` produces incoherent embeddings that silently poison the index.

**Defense:** `ftfy` + ASCII filter at ingest. Treat any result under 20 tokens as missing rather than passing garbage downstream.

```python
import ftfy
import re

def clean_lyrics(raw: str) -> str | None:
    text = ftfy.fix_text(raw)
    text = re.sub(r'\[.*?\]', '', text)          # strip [Verse 1], [Chorus]
    text = re.sub(r'\xa0', ' ', text)             # non-breaking spaces
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)   # strip non-ASCII
    text = re.sub(r'\s+', ' ', text).strip()

    if len(text.split()) < 20:
        return None   # treat as missing
    return text
```

---

### 1.3 Spotify Sentinel Values

**What goes wrong:** Spotify returns `tempo: 0.0` for unanalyzed tracks and `loudness: 0.0` for silence-flagged tracks. These are sentinel values, not real measurements. A `tempo: 0.0` in a prompt produces a descriptor claiming "no discernible pulse" — accurate for the sentinel, wrong for the actual track.

**Defense:** Hard range validation at ingest; null on out-of-range rather than passing sentinel downstream.

```python
FEATURE_VALID_RANGES = {
    "tempo":        (40.0,  250.0),
    "loudness":     (-60.0,   0.0),
    "energy":       (0.0,     1.0),
    "valence":      (0.0,     1.0),
    "acousticness": (0.0,     1.0),
    "danceability": (0.0,     1.0),
}

def validate_audio_features(features: dict) -> dict:
    clean = {}
    for k, v in features.items():
        if k in FEATURE_VALID_RANGES:
            lo, hi = FEATURE_VALID_RANGES[k]
            clean[k] = v if (v is not None and lo <= v <= hi) else None
        else:
            clean[k] = v
    return clean
```

Also filter at ingest:
- Drop tracks with `speechiness > 0.5` (podcasts/comedy albums miscategorized in FMA)
- Drop tracks with `duration_ms < 30000` (intros, outros, stingers)

---

### 1.4 Duplicate Tracks Under Different IDs

**What goes wrong:** FMA has the same recording under multiple track IDs — live vs. studio, remaster vs. original. Near-identical embeddings cluster together and inflate Precision@10 artificially (a single query retrieves the same song 3 times).

**Defense:** Post-index cosine dedup. After building the index, query each track against its nearest neighbors and flag pairs with similarity > 0.98 as duplicates, keeping the highest enrichment quality version.

```python
def dedup_index(collection, similarity_threshold: float = 0.98) -> list[str]:
    """Returns list of track_ids to remove as duplicates."""
    to_remove = []
    results = collection.query(query_embeddings=all_embeddings, n_results=3)
    for i, neighbors in enumerate(results["distances"]):
        for dist, neighbor_id in zip(neighbors[1:], results["ids"][i][1:]):
            similarity = 1 - dist   # ChromaDB returns L2 distance by default
            if similarity > similarity_threshold:
                # Keep higher enrichment tier; mark other for removal
                to_remove.append(pick_duplicate_to_remove(track_ids[i], neighbor_id))
    return list(set(to_remove))
```

---

## 2. Model Failure Cases

### 2.1 Hallucinated Lyrical Themes (Instrumental Tracks)

**What goes wrong:** Mistral 7B invents lyrical themes for instrumental tracks. A piano jazz piece with no lyrics receives a descriptor claiming "themes of heartbreak and lost love" because that's statistically likely for piano jazz. This is worse than a missing descriptor — it confidently misdirects retrieval.

**Defense:** Structured output validation that enforces null `lyrical_themes` when `has_lyrics=False`. Prompt the model explicitly to set `lyrical_themes` to null for instrumentals.

```python
from pydantic import BaseModel, field_validator
from typing import Literal

class TrackDescriptor(BaseModel):
    mood: str
    tempo_feel: str
    genre: str
    lyrical_themes: str | None   # None for instrumentals
    narrative: str
    has_lyrics: bool
    confidence: Literal["high", "medium", "low"]

    @field_validator('lyrical_themes')
    def no_themes_without_lyrics(cls, v, info):
        if not info.data.get('has_lyrics') and v is not None:
            raise ValueError("Cannot have lyrical themes for instrumental track")
        return v
```

Any descriptor with `confidence: low` receives `enrichment_tier: metadata_only` and routes to the metadata-only index regardless of lyrics availability.

---

### 2.2 Retrieval Collapse (All One Genre)

**What goes wrong:** Dense retrieval on a 5K-track catalog can return 50 candidates dominated by a single genre. The cross-encoder then reranks within this bad candidate set and cannot recover — the failure is upstream of reranking.

**When it occurs:** Queries containing strong genre-adjacent signals (e.g., instrument names) that cluster tightly in the embedding space. Most common with niche genres that have few catalog representatives.

**Defense:** Post-retrieval genre concentration check. If the candidate set is >80% from one genre and the query doesn't explicitly name that genre, log as a retrieval anomaly and optionally add a genre diversity filter.

```python
from collections import Counter

def detect_retrieval_collapse(
    candidates: list[Track],
    query_genres: list[str],
    threshold: float = 0.8
) -> bool:
    """Returns True if candidate set has collapsed into a single genre."""
    genre_counts = Counter(t.genre_top for t in candidates)
    top_genre = max(genre_counts, key=genre_counts.get)
    top_genre_pct = genre_counts[top_genre] / len(candidates)

    is_collapsed = top_genre_pct > threshold
    is_expected = top_genre.lower() in [g.lower() for g in query_genres]

    if is_collapsed and not is_expected:
        logger.warning(
            "Retrieval collapse detected: %.0f%% of candidates are '%s'",
            top_genre_pct * 100, top_genre
        )
    return is_collapsed and not is_expected
```

---

### 2.3 Low Cross-Encoder Score Variance

**What goes wrong:** Abstract or spatial queries ("music that sounds like standing in a cathedral") produce near-uniform cross-encoder scores because ms-marco-MiniLM was trained on web search data, not music semantics. All 50 candidates score ~0.5; reranking is effectively random.

**Defense:** Measure score variance before committing to the reranked order. Fall back to ANN order when variance is below threshold — at least that ranking reflects embedding similarity.

```python
import numpy as np

def rerank_with_fallback(
    query: str,
    candidates: list[Track],
    cross_encoder,
    variance_threshold: float = 0.05
) -> tuple[list[Track], dict]:
    pairs = [(query, c.descriptor) for c in candidates]
    scores = cross_encoder.predict(pairs)
    score_variance = float(np.std(scores))

    if score_variance < variance_threshold:
        logger.warning(
            "Low rerank variance (%.3f) for query '%s' — returning ANN order",
            score_variance, query[:60]
        )
        return candidates, {
            "reranker_used": False,
            "reason": "low_variance",
            "score_std": score_variance
        }

    ranked = [c for c, _ in sorted(zip(candidates, scores), key=lambda x: -x[1])]
    return ranked, {"reranker_used": True, "score_std": score_variance}
```

The metadata dict surfaces in the `RetrieveResponse.warning` field so quality degradation is observable without crashing.

---

### 2.4 Out-of-Vocabulary Genre Terms

**What goes wrong:** bge-base-en-v1.5 has poor representations for niche genre terms — "hyperpop," "djent," "zeuhl," "fourth-world." Queries containing these terms land in generic "electronic" or "jazz" neighborhoods regardless of their actual meaning.

**Defense:** Vocabulary coverage check per query. Log a warning when query tokens fall outside the model's top-50K vocabulary. This doesn't fix the failure but provides a diagnostic signal when eval results look unexpectedly bad.

```python
def check_query_vocabulary(query: str, tokenizer, top_k_vocab: int = 50_000) -> list[str]:
    """Returns list of tokens likely OOV for the embedding model."""
    tokens = tokenizer.tokenize(query.lower())
    vocab_ids = tokenizer.convert_tokens_to_ids(tokens)
    oov = [t for t, i in zip(tokens, vocab_ids) if i >= top_k_vocab]
    if oov:
        logger.warning("Possible OOV tokens in query: %s", oov)
    return oov
```

---

## 3. Pipeline Edge Cases

### 3.1 Missing Values — Centralized Contract

Never scatter `if track.get('energy') is not None` across multiple files. Define the data contract once in `data/schema.py` and enforce it at the boundary.

The `prompt_features` property is the critical pattern: the LLM receives `"unknown tempo"` rather than `None` or an empty string. The latter causes the model to invent a value.

```python
from pydantic import BaseModel, Field
from typing import Literal

class CatalogRecord(BaseModel):
    track_id: str
    title: str
    artist: str
    genre_top: str = "Unknown"

    # Spotify features — all nullable, all have documented fallbacks
    tempo:        float | None = None
    energy:       float | None = None
    valence:      float | None = None
    acousticness: float | None = None

    # Enrichment
    lyrics_text:      str | None = None
    descriptor:       str | None = None
    enrichment_tier:  Literal["full", "metadata_only"] = "metadata_only"

    @property
    def prompt_features(self) -> dict:
        """Returns prompt-safe feature dict — never passes None to LLM."""
        return {
            "tempo":        f"{self.tempo:.0f} BPM" if self.tempo else "unknown tempo",
            "energy":       f"{self.energy:.2f}"    if self.energy else "unknown energy",
            "valence":      f"{self.valence:.2f}"   if self.valence else "unknown valence",
            "acousticness": f"{self.acousticness:.2f}" if self.acousticness else "unknown",
        }
```

---

### 3.2 Corrupt Checkpoint Parquet Files

**What goes wrong:** Parquet corruption happens when a write is interrupted — exactly what happens if the descriptor batch crashes mid-checkpoint. A corrupted checkpoint read silently drops rows or raises a cryptic Arrow error mid-pipeline.

**Defense:** Write a row-count manifest atomically alongside each checkpoint. Validate against the manifest before reading.

```python
import json
import os
import pandas as pd
from pathlib import Path

def write_checkpoint(df: pd.DataFrame, path: Path, manifest_path: Path):
    """Write checkpoint with atomic manifest update."""
    df.to_parquet(path, index=False)

    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())

    manifest[path.name] = {"row_count": len(df), "columns": list(df.columns)}

    # Atomic write: temp file then os.replace (safe on macOS + Linux)
    tmp = manifest_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    os.replace(tmp, manifest_path)

def load_checkpoint(path: Path, manifest_path: Path) -> pd.DataFrame | None:
    manifest = json.loads(manifest_path.read_text())
    expected = manifest.get(path.name)
    if not expected:
        logger.error("No manifest entry for checkpoint %s", path.name)
        return None

    try:
        df = pd.read_parquet(path)
        if len(df) != expected["row_count"]:
            logger.error(
                "Checkpoint %s corrupted: expected %d rows, got %d",
                path.name, expected["row_count"], len(df)
            )
            return None
        return df
    except Exception as e:
        logger.error("Failed to read checkpoint %s: %s", path.name, e)
        return None
```

---

### 3.3 Mixed Prompt Versions Across Checkpoints

**What goes wrong:** Iterating on the descriptor prompt between batch runs produces checkpoint files with different schemas — e.g., prompt v1 has no `tempo_feel` field, prompt v2 adds it. Merging checkpoint files from two versions produces a DataFrame with 40% NaN in the new column.

**Defense:** Tag every checkpoint file with the prompt template's MD5 hash. Assert version consistency at merge time.

```python
import hashlib

def prompt_version_hash(prompt_template: str) -> str:
    return hashlib.md5(prompt_template.encode()).hexdigest()[:8]

# In checkpoint filename:
# desc_{prompt_hash}_{batch_idx:04d}.parquet
# e.g. desc_a3f91c2b_0042.parquet

def merge_checkpoints(checkpoints_dir: Path, prompt_hash: str) -> pd.DataFrame:
    files = list(checkpoints_dir.glob(f"desc_{prompt_hash}_*.parquet"))

    wrong_version = [
        f for f in checkpoints_dir.glob("desc_*.parquet")
        if prompt_hash not in f.name
    ]
    if wrong_version:
        raise RuntimeError(
            f"Found {len(wrong_version)} checkpoints from a different prompt version. "
            f"Delete or archive them before merging: {wrong_version}"
        )

    dfs = [pd.read_parquet(f) for f in sorted(files)]
    merged = pd.concat(dfs, ignore_index=True)
    return merged.drop_duplicates(subset="track_id")
```

---

## 4. Infrastructure Failure Modes

### 4.1 Training Crash Mid-Run

**What goes wrong:** A crash at epoch 50 of 100 loses all progress if checkpoints aren't configured correctly. More subtly: restarting from the most recent checkpoint is wrong if epoch 50 was worse than epoch 47 — the best model checkpoint is what matters.

**Defense:** Configure `save_strategy="epoch"`, `load_best_model_at_end=True`, and `metric_for_best_model="ndcg_at_10"`. Set `resume_from_checkpoint=True` so re-running the script auto-detects the latest checkpoint. Test this deliberately on a short run before committing to a full training session.

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="checkpoints/biencoder",
    num_train_epochs=10,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="ndcg_at_10",   # custom compute_metrics function
    greater_is_better=True,
    save_total_limit=3,                   # keep only 3 checkpoints — disk space
    resume_from_checkpoint=True,          # auto-detects latest checkpoint on rerun
)
```

Log the checkpoint path that was ultimately used for evaluation in TensorBoard under `training/best_checkpoint_epoch`.

---

### 4.2 ChromaDB Index Corruption on macOS

**What goes wrong:** ChromaDB has intermittent issues with index corruption when the process is killed during a write on macOS (M1). The index appears to load but silently returns incorrect or empty results.

**Defense:** Never write to the production index directly. Build into a staging directory, validate it, then promote atomically.

```python
import shutil
import chromadb

def build_index_safely(
    records: list[CatalogRecord],
    config,
    n_spot_checks: int = 10
):
    staging_path = config.chroma_dir / "staging"
    production_path = config.chroma_dir / "production"
    backup_path = config.chroma_dir / "production_backup"

    # Build into staging
    if staging_path.exists():
        shutil.rmtree(staging_path)
    staging_client = chromadb.PersistentClient(path=str(staging_path))
    _populate_index(staging_client, records)

    # Validate: spot-check N random queries return non-empty results
    _validate_index(staging_client, records, n_spot_checks=n_spot_checks)

    # Atomic promotion
    if production_path.exists():
        if backup_path.exists():
            shutil.rmtree(backup_path)
        shutil.move(str(production_path), str(backup_path))

    shutil.move(str(staging_path), str(production_path))
    logger.info("Index promoted to production (%d records)", len(records))

def _validate_index(client, records, n_spot_checks: int):
    import random
    collection = client.get_collection("tracks")
    sample = random.sample(records, n_spot_checks)
    for record in sample:
        results = collection.query(
            query_embeddings=[record.embedding],
            n_results=5
        )
        assert len(results["ids"][0]) > 0, \
            f"Spot check failed: no results for track {record.track_id}"
    logger.info("Index validation passed (%d spot checks)", n_spot_checks)
```

---

### 4.3 Ollama Process Hang

**What goes wrong:** Ollama is a separate process. If it crashes or is OOM-killed during descriptor generation, the batch loop hangs on the next HTTP request rather than failing fast. A 30-second hang per track on 5K tracks is 40+ hours of blocked pipeline.

**Defense:** Hard 30-second timeout on every request. Exponential backoff on transient failures (timeout, 503). Do not retry schema validation failures — the prompt template is wrong and retrying produces the same bad output.

```python
import httpx
import time

def generate_descriptor_with_retry(
    track: CatalogRecord,
    client: httpx.Client,
    max_retries: int = 3
) -> TrackDescriptor | None:
    for attempt in range(max_retries):
        try:
            response = client.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "mistral:7b-instruct-q4_K_M",
                    "prompt": build_prompt(track),
                    "format": "json",
                    "stream": False,
                },
                timeout=30.0,   # hard timeout — never hang
            )
            response.raise_for_status()
            return TrackDescriptor.model_validate_json(
                extract_json(response.text)
            )

        except httpx.TimeoutException:
            logger.warning(
                "Ollama timeout on track %s (attempt %d/%d)",
                track.track_id, attempt + 1, max_retries
            )
            time.sleep(2 ** attempt)   # 1s, 2s, 4s

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 503:
                logger.warning("Ollama unavailable — waiting 10s")
                time.sleep(10)
            else:
                logger.error("Unexpected HTTP error for track %s: %s", track.track_id, e)
                return None   # don't retry HTTP errors

        except ValidationError as e:
            # Schema failure: retrying will produce the same bad output
            logger.warning(
                "Bad descriptor schema for track %s (prompt issue, not retrying): %s",
                track.track_id, e
            )
            return None

    logger.error("Exhausted retries for track %s", track.track_id)
    return None
```

---

### 4.4 Malformed Serving Requests

**What goes wrong:** `argparse` does not enforce type contracts — it converts everything to strings unless explicitly typed. A `top_k` of `-1` or a 2,000-character query string will either crash at inference time or silently degrade results.

**Defense:** Validate all inputs through Pydantic request models before any inference code runs. Surface soft failures (retrieval anomalies, rerank fallback) via a `warning` field in the response rather than hard-crashing.

```python
from pydantic import BaseModel, Field, field_validator
from typing import Literal

class RetrieveRequest(BaseModel):
    query: str
    top_k: int = Field(default=50, ge=1, le=200)
    index_tier: Literal["auto", "full", "metadata_only"] = "auto"

    @field_validator('query')
    def query_not_empty(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("Query cannot be empty")
        if len(v) > 500:
            raise ValueError(f"Query too long ({len(v)} chars, max 500)")
        return v

class RetrieveResponse(BaseModel):
    query: str
    candidates: list[CandidateTrack]
    index_tier_used: Literal["full", "metadata_only"]
    retrieval_ms: float
    reranker_used: bool
    warning: str | None = None   # soft failure signal: collapse, low variance, OOV
```

The `warning` field is the key pattern: it distinguishes between system errors (exceptions, hard crashes) and quality degradation (model failure modes). Both are observable; only one is fatal.

---

## 5. Failure Mode Register

| Failure | Category | Severity | Handler |
|---|---|---|---|
| Fuzzy join false positive | Data quality | High | Two-pass artist+title validation; suspicious log; manual review CSV |
| Lyrics encoding garbage | Data quality | Medium | `ftfy` + ASCII filter; treat as missing if <20 tokens |
| Spotify sentinel values (`0.0`) | Data quality | Medium | `FEATURE_VALID_RANGES` validator; null on out-of-range |
| `speechiness > 0.5` non-music tracks | Data quality | Medium | Hard filter at ingest |
| Sub-30s tracks (intros/outros) | Data quality | Low | Hard filter on `duration_ms` at ingest |
| Duplicate tracks (same song, different IDs) | Data quality | Medium | Post-index cosine dedup at similarity > 0.98 |
| Hallucinated lyrical themes (instrumental) | Model failure | High | Pydantic: `lyrical_themes` null-enforced when `has_lyrics=False` |
| Retrieval collapse (all one genre) | Model failure | High | `detect_retrieval_collapse()`; log anomaly; optional genre filter |
| Low cross-encoder score variance | Model failure | Medium | `rerank_with_fallback()`; return ANN order; `warning` in response |
| OOV genre terms in query | Model failure | Low | Vocabulary coverage check; warning logged per query |
| `None` fields in LLM prompt | Pipeline edge case | High | `prompt_features` property returns `"unknown X"` strings |
| Corrupt checkpoint parquet | Pipeline edge case | High | Row count manifest; atomic manifest write via `os.replace()` |
| Mixed prompt versions in checkpoints | Pipeline edge case | High | Prompt hash in filename; assert at merge time |
| Audio feature extreme outliers | Pipeline edge case | Medium | `validate_audio_features()` with hard range guards |
| Training crash mid-run | Infrastructure | High | `resume_from_checkpoint=True`; best-NDCG checkpoint retained |
| ChromaDB index corruption (macOS) | Infrastructure | High | Staging → validate → atomic promote pattern |
| Ollama process hang | Infrastructure | High | 30s hard timeout; exponential backoff; no retry on schema errors |
| Malformed CLI request | Infrastructure | Medium | Pydantic `RetrieveRequest` validators; `warning` field in `RetrieveResponse` |

---

## Appendix: Severity Definitions

| Severity | Meaning |
|---|---|
| **High** | Causes silent data corruption, pipeline halt, or meaningfully wrong results with no observable signal. Must be caught before propagating. |
| **Medium** | Degrades result quality or causes recoverable failures. Should be logged and surfaced in response metadata. |
| **Low** | Minor quality signal. Log as warning; does not require intervention. |

---

*Last updated: 2026-03-21*