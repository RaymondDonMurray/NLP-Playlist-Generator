# TASKS.md — Implementation Checklist
## NLP Music Playlist Retrieval System

**Generated:** 2026-04-04
**Last updated:** 2026-04-16
**Skeleton status:** Phase 1 (data layer) is fully implemented. T01, T02, T03, T04, T05, T06, T24 (descriptor generation) are complete. T07–T16 (Retriever class) are complete. Remaining `NotImplementedError` stubs are in Phase 3 (T17–T19) and Phase 4–6.

---

## What Is Already Implemented

The following files are **complete** and require no implementation tasks:

| File | Status |
|---|---|
| `src/data/schema.py` | Complete — all Pydantic models, AppConfig, prompt_features, chroma_metadata |
| `src/data/ingest.py` | Complete — FMAClient, SpotifyClient, GeniusClient |
| `src/data/preprocess.py` | Complete — validate_match, clean_lyrics, validate_audio_features, write_checkpoint, load_checkpoint, merge_checkpoints, dedup_catalog |
| `scripts/build_catalog.py` | Complete — full pipeline entry point |
| `tests/conftest.py` | Complete — all fixtures |
| `tests/data/test_schema.py` | Complete — all schema tests pass once schema.py is done |
| `tests/data/test_preprocess.py` | Complete — all preprocess tests pass against current code |
| `config/config.yaml` | Complete |
| `src/models/descriptor.py` | Complete — build_prompt, extract_json, generate_descriptor, generate_descriptors_batch (T02, T03, T04, T05) |
| `scripts/generate_descriptors.py` | Complete — main() with catalog+lyrics merge, validation, batch generation, descriptors.parquet output (T24) |
| `src/models/encoder.py` | Complete — _load_model (CUDA→MPS→CPU), encode_query (instruction prefix), encode_descriptors, encode_batch (T01) |
| `tests/models/test_encoder.py` | Complete — 29 tests covering device selection, lazy loading, return types, prefix, normalization, batch forwarding |
| `src/models/reranker.py` | Complete — _load_model, _score_pairs, rerank with low-variance ANN fallback (T06) |
| `src/serving/retriever.py` | Complete — _get_chroma_client, _build_bm25_indexes, _compute_confidence, _select_index_tier, _dense_retrieve, _bm25_retrieve, _rrf_fuse, _detect_retrieval_collapse, _check_query_vocabulary, retrieve (T07–T16) |

---

## Phase 2 — Model Wrappers

*Milestone: `pytest tests/models/ tests/data/` passes. Descriptors can be generated for 5 fixture tracks.*

---

### T01: Encoder — lazy model load and encode methods ✅
- **File:** `src/models/encoder.py` → `Encoder._load_model`, `Encoder.encode_query`, `Encoder.encode_descriptors`, `Encoder.encode_batch`
- **Complexity:** [S]
- **Depends on:** none
- **Context files:** `src/data/schema.py`, `src/models/encoder.py`
- **Description:** Implement four methods on the `Encoder` class.
  1. `_load_model`: Detect available device (CUDA if `torch.cuda.is_available()`, MPS if `torch.backends.mps.is_available()`, else CPU). Instantiate `SentenceTransformer(self.model_name, device=device)`. Log the device chosen. Return the model. Model name must be read from `self.model_name` (which comes from `config.retrieval.bi_encoder`), never hardcoded.
  2. `encode_query`: bge-base-en-v1.5 uses an asymmetric instruction prefix for queries. Prepend `"Represent this sentence for searching relevant passages: "` to the query string, then call `self.model.encode([prefixed_query], normalize_embeddings=True)`. Return as `list[float]`.
  3. `encode_descriptors`: Encode a list of descriptor strings in batches using `self.model.encode(texts, batch_size=batch_size, show_progress_bar=show_progress, normalize_embeddings=True)`. Return `list[list[float]]`.
  4. `encode_batch`: General-purpose encode without query prefix. Same as `encode_descriptors` but no `show_progress` arg semantics concern.
- **Signature:**
  ```python
  def _load_model(self) -> SentenceTransformer: ...
  def encode_query(self, query: str) -> list[float]: ...
  def encode_descriptors(self, texts: list[str], batch_size: int = 64, show_progress: bool = True) -> list[list[float]]: ...
  def encode_batch(self, texts: list[str], batch_size: int = 64) -> list[list[float]]: ...
  ```
- **Acceptance:** `python -c "from src.models.encoder import Encoder; from src.data.schema import AppConfig; e = Encoder(AppConfig()); v = e.encode_query('test'); assert len(v) == 768"` exits 0.
- **Watch out:** MPS only available on Apple Silicon macOS. Wrap in `try/except` or check both `torch.backends.mps.is_available()` and `torch.backends.mps.is_built()`. Return `list[float]` not `np.ndarray` — the rest of the codebase expects native Python lists.

---

### T02: build_prompt — render descriptor prompt for a single track ✅
- **File:** `src/models/descriptor.py` → `build_prompt`
- **Complexity:** [S]
- **Depends on:** none (schema.py already implemented)
- **Context files:** `src/models/descriptor.py`, `src/data/schema.py`
- **Description:** Render `DESCRIPTOR_PROMPT_V1` by substituting track metadata. Use `track.prompt_features` (already implemented in schema.py) for all audio feature values — this guarantees no `None` reaches the prompt. The lyrics section is conditional: if `track.lyrics_available` and `track.lyrics_text` is not None, include `LYRICS_SECTION_TEMPLATE.format(lyrics=...)` with the text truncated to `~200 tokens` (split by whitespace, join first 200 words); otherwise, omit the lyrics section entirely (leave the `{lyrics_section}` placeholder as an empty string). Set `has_lyrics` in the prompt to `true` or `false` (lowercase, as the LLM will parse JSON).
- **Signature:** `def build_prompt(track: CatalogRecord, prompt_template: str = DESCRIPTOR_PROMPT_V1) -> str`
- **Acceptance:** Run `pytest tests/models/test_descriptor.py::TestBuildPrompt -v`. All 6 tests must pass.
- **Watch out:** `failure_modes.md §3.1` — the string `": None"` must never appear in the rendered prompt. The `prompt_features` property already handles this, but double-check the `lyrics_section` substitution doesn't inject raw Python `None`.

---

### T03: extract_json — extract JSON object from Ollama response text ✅
- **File:** `src/models/descriptor.py` → `extract_json`
- **Complexity:** [S]
- **Depends on:** none
- **Context files:** `src/models/descriptor.py`
- **Description:** Ollama sometimes wraps its JSON response in markdown code fences (` ```json ... ``` `). This function must handle three cases: (1) bare JSON object; (2) JSON wrapped in ` ```json\n...\n``` `; (3) JSON with surrounding prose (e.g., "Here is the descriptor: {…} Done."). Strategy: strip code fences first, then find the substring from the first `{` to the last `}`.
- **Signature:** `def extract_json(response_text: str) -> str`
- **Acceptance:** Run `pytest tests/models/test_descriptor.py::TestExtractJson -v`. All 3 tests must pass.
- **Watch out:** Use `rfind('}')` not `find('}')` — nested JSON objects would otherwise truncate early.

---

### T04: generate_descriptor — single-track descriptor via Ollama with retry ✅
- **File:** `src/models/descriptor.py` → `generate_descriptor`
- **Complexity:** [M]
- **Depends on:** T02, T03
- **Context files:** `src/models/descriptor.py`, `docs/failure_modes.md` (§4.3)
- **Description:** POST to Ollama `/api/generate` with the rendered prompt and `"format": "json"`. Implement exactly the retry logic from `failure_modes.md §4.3`:
  - `httpx.TimeoutException` → `time.sleep(2 ** attempt)` (1s, 2s, 4s), retry up to `config.enrichment.max_retries`
  - `httpx.HTTPStatusError` with 503 → `time.sleep(10)`, retry
  - `httpx.HTTPStatusError` other → return `None` immediately (no retry)
  - `ValidationError` → return `None` immediately (no retry — prompt issue, not transient)
  - After exhausting retries → log error, return `None`

  On success: call `extract_json()` on `response.json()["response"]`, then `TrackDescriptor.model_validate_json(extracted)`. Use `config.enrichment.ollama_base_url` and `config.enrichment.ollama_model` (never hardcoded). Timeout is `config.enrichment.ollama_timeout_s`.
- **Signature:** `def generate_descriptor(track: CatalogRecord, client: httpx.Client, config: AppConfig, prompt_template: str = DESCRIPTOR_PROMPT_V1) -> TrackDescriptor | None`
- **Acceptance:** With Ollama running locally and `mistral:7b-instruct-q4_K_M` loaded, test manually with one fixture track. Unit-testable without Ollama by patching `client.post`.
- **Watch out:** `ValidationError` must NOT be retried. The failure_modes.md pattern is explicit: "The prompt or schema is wrong and retrying produces the same output." Catch `pydantic.ValidationError` specifically, not `Exception`.

---

### T05: generate_descriptors_batch — batch generation with checkpoint/resume ✅
- **File:** `src/models/descriptor.py` → `generate_descriptors_batch`
- **Complexity:** [M]
- **Depends on:** T02, T03, T04
- **Context files:** `src/models/descriptor.py`, `src/data/preprocess.py`, `docs/failure_modes.md` (§3.2, §3.3)
- **Description:** Generate descriptors for all records with parquet checkpointing and resume support. Algorithm:
  1. Compute `prompt_hash = prompt_version_hash(prompt_template)` (function already in `preprocess.py`).
  2. Scan `checkpoints_dir` for existing checkpoint files matching `desc_{prompt_hash}_*.parquet`. Collect all `track_id` values from these checkpoints → `processed_ids`. (Use `load_checkpoint` from `preprocess.py` for manifest validation.)
  3. Filter `records` to only unprocessed ones.
  4. Open one `httpx.Client` for the full batch (reuse connection).
  5. Iterate in batches of `config.pipeline.checkpoint_interval`. For each track, call `generate_descriptor()`. Collect successful descriptors. On None result, log warning, set `enrichment_tier = "metadata_only"` on the record, continue.
  6. After each batch of N tracks, call `write_checkpoint(df, path, manifest_path)` where path is `desc_{prompt_hash}_{batch_idx:04d}.parquet` and manifest is `checkpoints_dir / "manifest.json"`.
  7. After all records processed, call `merge_checkpoints(checkpoints_dir, prompt_hash)` to produce the final merged DataFrame.
  8. Update each `CatalogRecord.descriptor` and `enrichment_tier` from the merged data. Return updated records list.
- **Signature:** `def generate_descriptors_batch(records: list[CatalogRecord], config: AppConfig, checkpoints_dir: Path, prompt_template: str = DESCRIPTOR_PROMPT_V1) -> list[CatalogRecord]`
- **Acceptance:** Run with `--max-tracks 5` against fixture tracks. Verify `checkpoints/` contains parquet files. Kill mid-run and re-run with `--resume`; verify it skips already-processed tracks.
- **Watch out:** `failure_modes.md §3.3` — all checkpoint filenames must include `prompt_hash`. If checkpoints from a different hash exist, `merge_checkpoints` raises `RuntimeError`. Log pipeline TensorBoard metrics: `pipeline/coverage_rate` and `pipeline/retry_rate`.

---

### T06: Reranker — model load, score pairs, rerank with variance fallback ✅
- **File:** `src/models/reranker.py` → `Reranker._load_model`, `Reranker._score_pairs`, `Reranker.rerank`
- **Complexity:** [M]
- **Depends on:** none (schema.py already implemented)
- **Context files:** `src/models/reranker.py`, `src/data/schema.py`, `docs/failure_modes.md` (§2.3)
- **Description:**
  1. `_load_model`: Instantiate `CrossEncoder(self.model_name)`. Model name from `self.model_name` (set in `__init__` from config). Log model name.
  2. `_score_pairs`: Build a list of `(query, candidate.descriptor or "")` pairs for all candidates. Call `self.model.predict(pairs)`. Return `np.ndarray` of scores.
  3. `rerank`: Call `_score_pairs`. Compute `score_std = float(np.std(scores))`. If `score_std < config.retrieval.rerank_variance_threshold`, return `(candidates, {"reranker_used": False, "reason": "low_variance", "score_std": score_std})` with candidates in **original input order**. Otherwise, sort candidates by score descending, populate `candidate.rerank_score` for each, return `(sorted_candidates, {"reranker_used": True, "score_std": score_std})`.
- **Signature:**
  ```python
  def _load_model(self) -> CrossEncoder: ...
  def _score_pairs(self, query: str, candidates: list[CandidateTrack]) -> np.ndarray: ...
  def rerank(self, query: str, candidates: list[CandidateTrack]) -> tuple[list[CandidateTrack], dict]: ...
  ```
- **Acceptance:** Run `pytest tests/serving/test_ranker.py -v`. All 4 tests must pass. They mock `_score_pairs` so no model download is required for CI.
- **Watch out:** `failure_modes.md §2.3` — the fallback must preserve the *input* order (ANN order), not sort by score. The test explicitly checks `[c.track_id for c in ranked] == [c.track_id for c in fixture_candidates]`.

---

## Phase 3 — Serving Layer

*Milestone: `python -m src.serving.cli retrieve --query "chill jazz" --json-output` returns valid JSON. `pytest tests/serving/` passes.*

**Retriever (T07–T16): complete.** Remaining: T17 (Ranker), T18 (Explainer), T19 (CLI).

---

### T07: Retriever._get_chroma_client — lazy ChromaDB init ✅
- **File:** `src/serving/retriever.py` → `Retriever._get_chroma_client`
- **Complexity:** [S]
- **Depends on:** none
- **Context files:** `src/serving/retriever.py`, `src/data/schema.py`
- **Description:** Lazy-initialize `chromadb.PersistentClient` pointing to `config.paths.chroma_dir + "production"`. Cache in `self._chroma_client`. Log the path on first initialization.
- **Signature:** `def _get_chroma_client(self) -> chromadb.ClientAPI`
- **Acceptance:** `python -c "from src.serving.retriever import Retriever; from src.data.schema import AppConfig; from unittest.mock import MagicMock; r = Retriever(AppConfig(), MagicMock()); r._get_chroma_client()"` — raises `chromadb.errors.NotFoundError` or similar if `chroma_db/production/` doesn't exist, which is expected pre-index-build.
- **Watch out:** Use `chroma_db/production/` path, not staging. This path is created by `scripts/build_index.py` atomic promotion.

---

### T08: Retriever._build_bm25_indexes — build in-memory BM25 over all descriptors ✅
- **File:** `src/serving/retriever.py` → `Retriever._build_bm25_indexes`
- **Complexity:** [M]
- **Depends on:** T07
- **Context files:** `src/serving/retriever.py`
- **Description:** For each collection (`COLLECTION_FULL` and `COLLECTION_META_ONLY`), fetch all documents from ChromaDB using `collection.get(include=["documents", "metadatas"])`. Tokenize each document by whitespace-splitting. Build `BM25Okapi(tokenized_corpus)`. Store the BM25 index in `self._bm25_full` / `self._bm25_meta`.

  **Critical implementation detail not in the stub:** You must also store the parallel list of track IDs alongside the corpus, because BM25Okapi results are positional (index-based) and you need to map positions back to `track_id`. Add instance attributes `self._bm25_full_ids: list[str]` and `self._bm25_meta_ids: list[str]` to the class. Similarly store `self._bm25_full_docs: list[str]` and `self._bm25_meta_docs: list[str]` for use in `_bm25_retrieve`. Log the corpus sizes.
- **Signature:** `def _build_bm25_indexes(self) -> None`
- **Acceptance:** After index is built (`scripts/build_index.py`), calling `_build_bm25_indexes()` should log "BM25 full_enrichment: N docs, BM25 metadata_only: M docs".
- **Watch out:** `collection.get()` with no query returns all stored documents. ChromaDB `get()` returns a dict with keys `"ids"`, `"documents"`, `"metadatas"`. Documents may be `None` for records stored with no document; handle gracefully by using `descriptor or ""`.

---

### T09: Retriever._compute_confidence — routing confidence score ✅
- **File:** `src/serving/retriever.py` → `Retriever._compute_confidence`
- **Complexity:** [S]
- **Depends on:** T07
- **Context files:** `src/serving/retriever.py`, `docs/architecture_decision_record.md` (ADR-006)
- **Description:** Query the `full_enrichment` collection for the top `top_k` (default 5) candidates for `query_embedding`. Compute the centroid of the returned embeddings (mean across candidates). Compute cosine similarity between `query_embedding` and the centroid.

  ChromaDB returns distances, not embeddings, by default. To get embeddings, pass `include=["embeddings"]` to `collection.query()`. If the collection returns fewer than `top_k` results (small catalog edge case), use what's returned.
- **Signature:** `def _compute_confidence(self, query_embedding: list[float], collection: chromadb.Collection, top_k: int = 5) -> float`
- **Acceptance:** With a populated index, verify that a genre-specific query ("piano jazz") returns confidence ≥ 0.5 when matched tracks exist.
- **Watch out:** ChromaDB L2 distances must be converted to cosine similarity only if the collection was created with `hnsw:space: cosine`. `build_index.py` (T25) must create collections with cosine space. If cosine, ChromaDB returns `1 - cosine_distance`, not raw cosine similarity. Verify collection space setting matches.

---

### T10: Retriever._select_index_tier — resolve auto-routing ✅
- **File:** `src/serving/retriever.py` → `Retriever._select_index_tier`
- **Complexity:** [S]
- **Depends on:** T09
- **Context files:** `src/serving/retriever.py`
- **Description:** If `self.index_tier` is `"full"` or `"metadata_only"`, return it directly (no confidence computation). If `"auto"`, call `_compute_confidence()` against the full_enrichment collection. If confidence ≥ `config.retrieval.confidence_threshold`, return `"full"`. Otherwise return `"metadata_only"`.
- **Signature:** `def _select_index_tier(self, query_embedding: list[float]) -> Literal["full", "metadata_only"]`
- **Acceptance:** Run `pytest tests/serving/test_retriever.py::TestIndexTierSelection -v`. All 3 tests pass. They mock `_compute_confidence` so no index is required.

---

### T11: Retriever._dense_retrieve — ANN retrieval from ChromaDB ✅
- **File:** `src/serving/retriever.py` → `Retriever._dense_retrieve`
- **Complexity:** [S]
- **Depends on:** T07
- **Context files:** `src/serving/retriever.py`, `src/data/schema.py`
- **Description:** Call `collection.query(query_embeddings=[query_embedding], n_results=top_k, include=["metadatas", "distances"])`. Parse the results into a list of `CandidateTrack` objects. Populate `dense_score` from the distance (if cosine space, distance = 1 − similarity, so `dense_score = 1 − distance`). Fields to populate: `track_id`, `title`, `artist`, `genre_top`, `descriptor` (from metadata), `dense_score`.
- **Signature:** `def _dense_retrieve(self, query_embedding: list[float], collection: chromadb.Collection, top_k: int) -> list[CandidateTrack]`
- **Acceptance:** With a populated index, returns `top_k` CandidateTrack objects with non-None `dense_score`.
- **Watch out:** ChromaDB metadata values may include extra keys not in `CandidateTrack`. Only extract fields that CandidateTrack has. Missing metadata keys (e.g., no descriptor for metadata-only tracks) should produce `None` for optional fields.

---

### T12: Retriever._bm25_retrieve — BM25 retrieval over descriptor text ✅
- **File:** `src/serving/retriever.py` → `Retriever._bm25_retrieve`
- **Complexity:** [M]
- **Depends on:** T08
- **Context files:** `src/serving/retriever.py`
- **Description:** Tokenize the query by whitespace-splitting. Call `bm25.get_scores(tokenized_query)` on the appropriate BM25 index (`_bm25_full` or `_bm25_meta` based on `tier`). Get top-k indices by score. Look up track_id from `_bm25_full_ids` / `_bm25_meta_ids` (added in T08). Return list of `CandidateTrack` objects with `bm25_score` populated. Also populate `descriptor` from the stored docs list.

  If BM25 indexes haven't been built yet (None), call `_build_bm25_indexes()` lazily and log a warning that this is slow on first call.
- **Signature:** `def _bm25_retrieve(self, query: str, tier: Literal["full", "metadata_only"], top_k: int) -> list[CandidateTrack]`
- **Acceptance:** With a populated index, returns up to `top_k` CandidateTrack objects with `bm25_score` set.
- **Watch out:** `BM25Okapi.get_scores()` returns scores for the full corpus in corpus order (not sorted). Use `np.argsort(-scores)[:top_k]` to get the top-k indices. Scores of 0.0 are valid (no match) — don't filter them out before truncation.

---

### T13: Retriever._rrf_fuse — Reciprocal Rank Fusion ✅
- **File:** `src/serving/retriever.py` → `Retriever._rrf_fuse`
- **Complexity:** [S]
- **Depends on:** none (pure logic)
- **Context files:** `src/serving/retriever.py`
- **Description:** Implement standard RRF: for each track, sum `1 / (k + rank)` across both input lists, where `k = config.retrieval.rrf_k` (default 60) and rank is 1-based position. A track not present in a list contributes 0 from that list. Merge the two input lists by track_id, compute RRF scores, sort descending, return the fused list. Populate `rrf_score` on each CandidateTrack.
- **Signature:** `def _rrf_fuse(self, dense_results: list[CandidateTrack], bm25_results: list[CandidateTrack], k: int | None = None) -> list[CandidateTrack]`
- **Acceptance:** Run `pytest tests/serving/test_retriever.py::TestRRFFusion -v`. Both tests pass.
- **Watch out:** The fused result set should contain the **union** of both lists (not the intersection). A track present in only one list still appears — it just gets 0 contribution from the missing list.

---

### T14: Retriever._detect_retrieval_collapse — genre concentration check ✅
- **File:** `src/serving/retriever.py` → `Retriever._detect_retrieval_collapse`
- **Complexity:** [S]
- **Depends on:** none (pure logic)
- **Context files:** `src/serving/retriever.py`, `docs/failure_modes.md` (§2.2)
- **Description:** Count genres in `candidates` using `collections.Counter`. Find the top genre and its fraction. If top genre fraction > `config.retrieval.collapse_genre_threshold` (default 0.8) AND the top genre name does not appear (case-insensitive) in the query string, log a warning and return `True`. Otherwise return `False`.
- **Signature:** `def _detect_retrieval_collapse(self, candidates: list[CandidateTrack], query: str) -> bool`
- **Acceptance:** Run `pytest tests/serving/test_retriever.py::TestRetrievalCollapseDetection -v`. All 3 tests pass.
- **Watch out:** The genre check is a simple substring check: `top_genre.lower() in query.lower()`. "Jazz" in "smooth jazz relaxing" → True (expected, not collapsed). This is intentionally simple.

---

### T15: Retriever._check_query_vocabulary — OOV token diagnostic ✅
- **File:** `src/serving/retriever.py` → `Retriever._check_query_vocabulary`
- **Complexity:** [S]
- **Depends on:** T01
- **Context files:** `src/serving/retriever.py`, `docs/failure_modes.md` (§2.4)
- **Description:** Access `self.encoder.model.tokenizer` to get the SentenceTransformer's underlying tokenizer. Tokenize the query: `tokens = tokenizer.tokenize(query.lower())`. Convert to IDs: `ids = tokenizer.convert_tokens_to_ids(tokens)`. Any token with ID ≥ 50,000 (an approximation for OOV in a 30K–50K vocab model) is flagged. Log a warning listing the flagged tokens. Return the list of OOV tokens.
- **Signature:** `def _check_query_vocabulary(self, query: str) -> list[str]`
- **Acceptance:** With Encoder loaded, `retriever._check_query_vocabulary("djent zeuhl hyperpop")` logs a warning. `retriever._check_query_vocabulary("chill jazz")` returns an empty list.
- **Watch out:** `bge-base-en-v1.5` uses a BERT-based tokenizer with ~30K vocab. IDs ≥ 30,000 are OOV or subword boundaries. Use 50,000 as a conservative threshold (as in `failure_modes.md §2.4`). If `self.encoder.model` hasn't been loaded yet, this will trigger lazy load — that's acceptable.

---

### T16: Retriever.retrieve — full retrieval pipeline ✅
- **File:** `src/serving/retriever.py` → `Retriever.retrieve`
- **Complexity:** [M]
- **Depends on:** T07, T08, T09, T10, T11, T12, T13, T14, T15
- **Context files:** `src/serving/retriever.py`, `src/data/schema.py`
- **Description:** Implement the 7-step pipeline documented in the stub docstring:
  1. Start timer.
  2. Embed query: `query_embedding = self.encoder.encode_query(request.query)`.
  3. OOV check: `self._check_query_vocabulary(request.query)` — warning only.
  4. Select index tier: `tier = self._select_index_tier(query_embedding)`. Get the collection.
  5. Dense ANN retrieve: `dense_candidates = self._dense_retrieve(query_embedding, collection, request.top_k)`.
  6. BM25 retrieve: `bm25_candidates = self._bm25_retrieve(request.query, tier, request.top_k)`.
  7. RRF fusion: `fused = self._rrf_fuse(dense_candidates, bm25_candidates)`, truncate to `request.top_k`.
  8. Retrieval collapse check: if `_detect_retrieval_collapse(fused, request.query)`, set `warning = "retrieval_collapse"`.
  9. Stop timer. Return `RetrieveResponse(query=..., candidates=fused, index_tier_used=tier, retrieval_ms=elapsed_ms, reranker_used=False, warning=warning)`.
- **Signature:** `def retrieve(self, request: RetrieveRequest) -> RetrieveResponse`
- **Acceptance:** `python -m src.serving.cli retrieve --query "chill rainy jazz" --json-output` returns valid JSON with `candidates` list and `index_tier_used` field.
- **Watch out:** `reranker_used=False` in the RetrieveResponse is correct here — reranking happens in the Ranker layer, not the Retriever. The `warning` field must be `None` if no soft failure occurred.

---

### T17: Ranker — retrieve, rerank, and retrieve_and_rerank
- **File:** `src/serving/ranker.py` → `Ranker.retrieve`, `Ranker.rerank`, `Ranker.retrieve_and_rerank`
- **Complexity:** [S]
- **Depends on:** T06, T16
- **Context files:** `src/serving/ranker.py`, `src/data/schema.py`
- **Description:**
  1. `retrieve`: Delegate directly to `self.retriever.retrieve(request)`. Return the `RetrieveResponse`.
  2. `rerank`: Start timer. Call `self.reranker.rerank(request.query, request.candidates)` → `(ranked, meta)`. Truncate to `request.top_k`. Return `RerankResponse(query=..., tracks=ranked[:top_k], rerank_ms=elapsed_ms, reranker_used=meta["reranker_used"], warning="low_rerank_variance" if not meta["reranker_used"] else None)`.
  3. `retrieve_and_rerank`: Build a `RetrieveRequest`, call `self.retrieve()`, build a `RerankRequest` from the candidates, call `self.rerank()`. Use `top_k_retrieve` (default `config.retrieval.ann_top_k`) and `top_k_rerank` (default `config.retrieval.rerank_top_k`). Return both response objects as a tuple.
- **Signature:**
  ```python
  def retrieve(self, request: RetrieveRequest) -> RetrieveResponse: ...
  def rerank(self, request: RerankRequest) -> RerankResponse: ...
  def retrieve_and_rerank(self, query: str, top_k_retrieve: int | None = None, top_k_rerank: int | None = None, index_tier: str = "auto") -> tuple[RetrieveResponse, RerankResponse]: ...
  ```
- **Acceptance:** With mocked Retriever and Reranker, both stage responses should be returned from `retrieve_and_rerank`.

---

### T18: Explainer — build prompt, explain_track, explain
- **File:** `src/serving/explainer.py` → `Explainer._build_explanation_prompt`, `Explainer.explain_track`, `Explainer.explain`
- **Complexity:** [M]
- **Depends on:** none (uses only schema.py and httpx)
- **Context files:** `src/serving/explainer.py`, `src/data/schema.py`, `docs/failure_modes.md` (§4.3)
- **Description:**
  1. `_build_explanation_prompt`: Format `EXPLANATION_PROMPT_TEMPLATE` using query and track fields. Handle `None` descriptor with a fallback string (`"[no descriptor available]"`). Handle `None` genre_top with `"Unknown"`.
  2. `explain_track`: POST to Ollama `/api/generate` with the prompt. Unlike descriptor generation, the response is plain text (no `"format": "json"`). Extract the explanation from `response.json()["response"]`. Strip leading/trailing whitespace. Apply the same retry/timeout logic as `generate_descriptor` (T04): 30s timeout, exponential backoff on timeout, no retry on schema errors. Return `None` on failure.
  3. `explain`: Start timer. Create one `httpx.Client`. For each track, call `explain_track()`. If successful, create a copy of the track with `explanation` populated (or use `.model_copy(update={"explanation": text})`). Tracks with failed explanations keep `explanation=None`. Return `ExplainResponse(query=..., tracks=tracks_with_explanations, explain_ms=elapsed_ms)`.
- **Signature:**
  ```python
  def _build_explanation_prompt(self, query: str, track: CandidateTrack) -> str: ...
  def explain_track(self, query: str, track: CandidateTrack, client: httpx.Client) -> str | None: ...
  def explain(self, request: ExplainRequest) -> ExplainResponse: ...
  ```
- **Acceptance:** With Ollama running, `explain()` on 1 fixture candidate returns `ExplainResponse` with `tracks[0].explanation` populated.
- **Watch out:** `CandidateTrack` is a Pydantic model; use `.model_copy(update={...})` to produce a copy with `explanation` set, not in-place mutation. A missing explanation is a soft failure — it must not raise, it should set `explanation=None` and continue.

---

### T19: CLI — load_config, build_components, cmd_retrieve, cmd_rerank, cmd_explain
- **File:** `src/serving/cli.py` → `load_config`, `build_components`, `cmd_retrieve`, `cmd_rerank`, `cmd_explain`
- **Complexity:** [M]
- **Depends on:** T16, T17, T18
- **Context files:** `src/serving/cli.py`, `src/data/schema.py`
- **Description:**
  1. `load_config`: Open YAML file at `config_path`, parse with `yaml.safe_load`, validate via `AppConfig.model_validate()`. Any missing/misconfigured key raises a `ValidationError` (hard failure, not caught). Return `AppConfig`.
  2. `build_components`: Instantiate in order: `Encoder(config)`, `Retriever(config, encoder)`, `Reranker(config)`, `Ranker(config, retriever, reranker)`, `Explainer(config)`. Return `(retriever, ranker, explainer)`.
  3. `cmd_retrieve`: Validate args via `RetrieveRequest(query=args.query, top_k=args.top_k, index_tier=args.index_tier)`. Build retriever via `build_components`. Call `retriever.retrieve(request)`. Print `response.model_dump_json(indent=2)` if `--json-output`, else print a human-readable summary.
  4. `cmd_rerank`: Load candidates JSON from `args.candidates` file. Validate via `RerankRequest`. Call `ranker.rerank(request)`. Print response JSON.
  5. `cmd_explain`: Load tracks JSON from `args.tracks` file. Validate via `ExplainRequest`. Call `explainer.explain(request)`. Print response JSON.
- **Signature:** (see existing stubs)
- **Acceptance:**
  ```bash
  python -m src.serving.cli retrieve --query "chill jazz afternoon" --json-output
  python -m src.serving.cli --help
  ```
  Both exit 0 (first requires built index).
- **Watch out:** Loading `.env` (for any API keys) should happen before components are initialized. Use `from dotenv import load_dotenv; load_dotenv()` at the top of `main()`.

---

## Phase 4 — Evaluation Harness

*Milestone: `pytest tests/eval/ tests/serving/` passes. `python scripts/run_eval.py --smoke-test` exits 0.*

---

### T20: eval/metrics.py — ndcg_at_k, mrr, precision_at_k, novelty_at_k
- **File:** `src/eval/metrics.py` → `ndcg_at_k`, `mrr`, `precision_at_k`, `novelty_at_k`
- **Complexity:** [S]
- **Depends on:** none (pure math, no imports)
- **Context files:** `src/eval/metrics.py`
- **Description:** Implement the four retrieval evaluation metrics. All are pure functions (no I/O, no side effects).
  1. `ndcg_at_k`: Compute DCG for `retrieved_ids[:k]` with binary relevance (1 if in `relevant_ids`, else 0). DCG = Σ rel_i / log2(i+2) for i in [0, k). Compute ideal DCG (IDCG) by sorting all relevant results to the top (min(k, |relevant_ids|) hits at positions 0..|relevant|). Return DCG / IDCG. Return 0.0 if `relevant_ids` is empty.
  2. `mrr`: Iterate `retrieved_ids[:k]`. Return `1 / (rank + 1)` at the first hit (0-indexed rank). Return 0.0 if no hit in top-k.
  3. `precision_at_k`: Count hits in `retrieved_ids[:k]` divided by `k`.
  4. `novelty_at_k`: For each track in `retrieved_ids[:k]`, get popularity from `track_popularity` dict (default 0.5 if missing). Novelty = 1 − popularity. Return mean novelty over top-k.
- **Signature:** (see existing stubs)
- **Acceptance:** Run `pytest tests/eval/test_metrics.py -v`. All 16 tests pass.
- **Watch out:** IDCG denominator uses `log2(i+2)` not `log2(i+1)` — the standard NDCG formula with 1-based ranks starts at `log2(2)=1` for rank 1, meaning rank 1 denominator = `log2(1+1)=1.0`. Use `math.log2(rank + 2)` for 0-indexed rank.

---

### T21: runner — load_eval_queries, write_results, check_regression
- **File:** `src/eval/runner.py` → `load_eval_queries`, `write_results`, `check_regression`
- **Complexity:** [S]
- **Depends on:** none (schema.py already done)
- **Context files:** `src/eval/runner.py`, `src/data/schema.py`
- **Description:**
  1. `load_eval_queries`: Read JSON from `queries_path`. Validate each entry via `EvalQuery.model_validate()`. Raise `ValidationError` on the first invalid entry (hard failure — don't silently skip). Return `list[EvalQuery]`.
  2. `write_results`: Write `list[EvalResult]` to CSV at `output_path`. Use `csv.DictWriter` with the field names from `EvalResult.model_fields`. Create parent directories if needed.
  3. `check_regression`: Load `baseline_path` JSON (if it doesn't exist, log warning and return `True` — no baseline means no regression). Compute mean NDCG@10 across `results`. Compare to baseline value `baseline["hybrid_rerank"]["ndcg_at_k"]` (or the first key if `hybrid_rerank` not present). If current mean drops > `threshold` (e.g., 0.05 means >5% drop), log an error and return `False`. Otherwise return `True`.
- **Signature:** (see existing stubs)
- **Acceptance:** `pytest tests/eval/ -v` (no new tests specifically for these helpers, but T22 will exercise them).

---

### T22: runner — run_single_query, run_eval
- **File:** `src/eval/runner.py` → `run_single_query`, `run_eval`
- **Complexity:** [M]
- **Depends on:** T17, T20, T21
- **Context files:** `src/eval/runner.py`, `src/data/schema.py`
- **Description:**
  1. `run_single_query`: Call `ranker.retrieve_and_rerank(query.query, ...)`. Extract top-k track IDs from `rerank_response.tracks`. Compute all four metrics via `compute_all_metrics()`. Return an `EvalResult` with the config name, index tier, and warning from the response.
  2. `run_eval`: Load queries via `load_eval_queries` (or use `queries_override`). Build `track_popularity` dict from the catalog if not passed in. Loop over queries, calling `run_single_query`. Collect `EvalResult` objects. Compute aggregate metrics (mean NDCG, MRR, P@k, Novelty@k). Log to TensorBoard: `eval/ndcg_at_10`, `eval/mrr`, `eval/precision_at_10`, `eval/novelty_at_10` (ADR-004). Call `write_results()`. Return results list.
- **Signature:** (see existing stubs)
- **Acceptance:** With a smoke test override of 3 fixture queries and mocked retriever, `run_eval()` returns a list of `EvalResult` and writes `eval/results.csv`.
- **Watch out:** `scope.md` §evaluation: "When reporting eval metrics, always specify which index tier was used. Do not mix results from both tiers in a single NDCG number." The `EvalResult.index_tier` field exists precisely for this. When logging aggregate metrics to TensorBoard, tag them by `config_name`.

---

### T23: ablations — make_run_name, apply_ablation_config, run_ablation_matrix, print_ablation_table
- **File:** `src/eval/ablations.py` → `make_run_name`, `apply_ablation_config`, `run_ablation_matrix`, `print_ablation_table`
- **Complexity:** [L]
- **Depends on:** T22
- **Context files:** `src/eval/ablations.py`, `src/data/schema.py`, `src/eval/runner.py`
- **Description:**
  1. `make_run_name`: Format `f"{datetime.now().strftime('%Y%m%d')}_{hash6}_{ablation.name}"` where `hash6` is the first 6 chars of `hashlib.md5(ablation.name.encode()).hexdigest()`.
  2. `apply_ablation_config`: Returns a modified `AppConfig` reflecting the `AblationConfig` flags. **See Unresolved Questions for the architectural gap here.** Recommended approach: add `use_dense: bool = True` and `use_bm25: bool = True` fields to `RetrievalConfig` in `schema.py`, then set them in `apply_ablation_config`. The Retriever's `retrieve()` method (T16) should check these flags to skip dense or BM25 retrieval when False. For `use_reranker=False`, set a config flag like `retrieval.use_reranker: bool = True` that `Ranker.rerank()` (T17) checks; when False, `rerank()` returns candidates in ANN order.
  3. `run_ablation_matrix`: Iterate `ABLATION_CONFIGS` (or override list). For each: `apply_ablation_config`, create `SummaryWriter` for the run name, instantiate `Encoder`, `Retriever`, `Reranker`, `Ranker`, call `run_eval`. Collect results. Log `eval/ndcg_at_10` per config to TensorBoard. Return dict of config_name → results.
  4. `print_ablation_table`: Print a Markdown table with rows matching `scope.md §ablation table` structure: `| Config | P@10 | NDCG@10 | MRR | Novelty |`.
- **Signature:** (see existing stubs)
- **Acceptance:** `python scripts/run_eval.py --smoke-test --print-table` prints a markdown table to stdout.
- **Watch out:** `run_ablation_matrix` creates one `Encoder` and reuses it across all ablations (model loading is expensive). Only `Retriever`/`Reranker`/`Ranker` need re-instantiation per ablation. The `hybrid_rerank_expanded` ablation requires query expansion (see Unresolved Questions); skip it if the query expander is not implemented, logging a warning.

---

## Phase 5 — Script Entry Points

*Milestone: Full pipeline runnable end-to-end via 4 script commands. `docker-compose up` produces results.*

---

### T24: scripts/generate_descriptors.py — main() ✅
- **File:** `scripts/generate_descriptors.py` → `main`
- **Complexity:** [M]
- **Depends on:** T05
- **Context files:** `scripts/generate_descriptors.py`, `src/models/descriptor.py`, `src/data/preprocess.py`
- **Description:** Implement the 6-step process documented in the file's stub comments:
  1. Load `catalog.parquet` and `lyrics.parquet` from `config.paths.processed_dir`.
  2. Merge lyrics into catalog records by joining on `track_id`.
  3. Validate and construct `list[CatalogRecord]`.
  4. Apply `--max-tracks` limit if provided.
  5. Call `generate_descriptors_batch(records, config, checkpoints_dir, prompt_template)`.
  6. Save the updated records as `descriptors.parquet` in `config.paths.processed_dir`. Include `track_id`, `descriptor`, `enrichment_tier` columns at minimum.

  Log TensorBoard pipeline metrics: `pipeline/coverage_rate` (fraction of tracks with non-None descriptor) and `pipeline/retry_rate`. Use `config.enrichment.prompt_version` to select the prompt template.
- **Signature:** `def main() -> None`
- **Acceptance:** `python scripts/generate_descriptors.py --config config/config.yaml --max-tracks 5` exits 0, produces `data/processed/descriptors.parquet`.
- **Watch out:** The `generate_descriptors_batch` function returns `list[CatalogRecord]` with `descriptor` populated where successful. Tracks that failed descriptor generation have `descriptor=None` and `enrichment_tier="metadata_only"`. Both must be written to the output parquet.

---

### T25: scripts/build_index.py — main()
- **File:** `scripts/build_index.py` → `main`
- **Complexity:** [L]
- **Depends on:** T01, T07
- **Context files:** `scripts/build_index.py`, `src/models/encoder.py`, `docs/failure_modes.md` (§1.4, §4.2)
- **Description:** Implement the 8-step process from the stub comments:
  1. Load `descriptors.parquet`.
  2. Split into `full_enrichment` (rows where `enrichment_tier == "full"` and `descriptor` is not None) and `metadata_only` (all others).
  3. If `--skip-embedding` is not set, encode all descriptors with `Encoder.encode_descriptors()`. Save `embeddings.parquet` alongside. If `--skip-embedding`, load from existing `embeddings.parquet`.
  4. Build **staging** ChromaDB collections (path: `config.paths.chroma_dir + "staging/"`). Create each collection with `metadata={"hnsw:space": "cosine"}`. Bulk-add documents with `collection.add(ids=..., embeddings=..., documents=descriptors, metadatas=chroma_metadata_list)`.
  5. Cosine dedup within each staging collection (`failure_modes.md §1.4`): query each track against its neighbors, remove duplicates with similarity > `config.pipeline.dedup_similarity_threshold`. Log removed count.
  6. Spot-check validation: for `args.n_spot_checks` randomly sampled tracks, query the staging index and assert non-empty results.
  7. Atomic promotion: backup existing production index (if present), `shutil.move(staging, production)`.
  8. Log final collection sizes.
- **Signature:** `def main() -> None`
- **Acceptance:** `python scripts/build_index.py --config config/config.yaml` exits 0. `chroma_db/production/` exists with two collections.
- **Watch out:** ChromaDB's `add()` has a per-batch limit (~5000 items). Split into batches if catalog > 5000 tracks. The `chroma_metadata` dict for each record must not contain `None` values — use `CatalogRecord.chroma_metadata` property (already implemented in schema.py). The `embedding` column in parquet is stored as a list of floats; ChromaDB expects `list[list[float]]`.

---

### T26: scripts/run_eval.py — main()
- **File:** `scripts/run_eval.py` → `main`
- **Complexity:** [M]
- **Depends on:** T19, T23
- **Context files:** `scripts/run_eval.py`, `src/eval/ablations.py`
- **Description:** Implement the 8-step process from the stub comments:
  1. Load AppConfig.
  2. Load eval queries from `config.eval.queries_path`.
  3. Build Encoder, Retriever, Reranker, Ranker (shared across ablations).
  4. Load `catalog.parquet` and compute `track_popularity` as normalized FMA play counts (or uniform 0.5 if play counts unavailable).
  5. If `--config-name` specified, run only that ablation config. Otherwise run all via `run_ablation_matrix`.
  6. Write results to `config.eval.results_path`.
  7. If `--print-table`, call `print_ablation_table`.
  8. If `--check-regression`, call `check_regression`; exit 1 on failure.
- **Signature:** `def main() -> None`
- **Acceptance:** `python scripts/run_eval.py --config config/config.yaml --smoke-test --config-name dense_only` exits 0 and prints a 1-row table.
- **Watch out:** FMA `tracks.csv` contains a `track/listens` column that can serve as play count. Normalize by max play count in catalog for a [0,1] popularity score. If the column is missing or all zero, log a warning and use 0.5 for all tracks.

---

## Phase 6 — Tier 2: Fine-Tuning (Stretch Goals)

*These tasks implement `src/models/finetune.py`. Only start after Phase 5 is complete and the eval harness shows stable baselines.*

---

### T27: finetune.mine_hard_negatives — mine same-genre, different-mood negatives
- **File:** `src/models/finetune.py` → `mine_hard_negatives`
- **Complexity:** [M]
- **Depends on:** T01
- **Context files:** `src/models/finetune.py`, `src/data/schema.py`, `docs/architecture_decision_record.md` (ADR-003)
- **Description:** For each `EvalQuery`, encode the query. Find top-`top_k` similar descriptors from the catalog via cosine similarity (using the encoder). Among those, identify "hard negatives" — descriptors from the same genre as at least one relevant track, but not in `query.relevant_ids`. Return a list of `(query_text, positive_descriptor, hard_negative_descriptor)` triples where positive descriptors come from `relevant_ids`.
- **Signature:** `def mine_hard_negatives(queries: list[EvalQuery], catalog_descriptors: list[str], catalog_ids: list[str], encoder, top_k: int = 20) -> list[tuple[str, str, str]]`
- **Acceptance:** With 10 eval queries and a 100-track catalog sample, returns at least 5 triples.

---

### T28: finetune.build_training_dataset — 80/20 train/val split
- **File:** `src/models/finetune.py` → `build_training_dataset`
- **Complexity:** [M]
- **Depends on:** T27
- **Context files:** `src/models/finetune.py`, `docs/architecture_decision_record.md` (ADR-003)
- **Description:** Split `queries` 80/20 (train/val). For train queries, build `InputExample` objects using `sentence_transformers.InputExample`. Positive pairs `(query, relevant_descriptor)` for `MultipleNegativesRankingLoss`. Triplets `(query, positive, hard_negative)` for triplet margin loss. For val queries, build a separate list. Return `(train_examples, val_examples)`.
- **Signature:** `def build_training_dataset(queries: list[EvalQuery], catalog_records: list, config: AppConfig, encoder) -> tuple[list, list]`
- **Acceptance:** Returns two non-empty lists when given ≥10 queries with populated `relevant_ids`.

---

### T29: finetune.train — contrastive fine-tuning with NDCG early stopping
- **File:** `src/models/finetune.py` → `train`
- **Complexity:** [L]
- **Depends on:** T28
- **Context files:** `src/models/finetune.py`, `docs/architecture_decision_record.md` (ADR-003), `docs/failure_modes.md` (§4.1)
- **Description:** Fine-tune `bge-base-en-v1.5` using the `sentence-transformers` `SentenceTransformerTrainer`. Key implementation decisions from ADR-003:
  - Combined loss: `MultipleNegativesRankingLoss` (weight 0.7 in-batch) + `TripletLoss` (weight 0.3 hard negatives).
  - `TrainingArguments` with `save_strategy="epoch"`, `load_best_model_at_end=True`, `metric_for_best_model="ndcg_at_10"`, `save_total_limit=config.training.save_total_limit`.
  - Custom `compute_metrics` callback that runs retrieval eval on val queries and returns `{"ndcg_at_10": ...}`.
  - `resume_from_checkpoint=True` for crash recovery (`failure_modes.md §4.1`).
  - Log `train/loss`, `val/loss`, `val/ndcg_at_10`, `val/mrr`, `val/precision_at_10` to TensorBoard.
  - Return path to best checkpoint.
- **Signature:** `def train(config: AppConfig, train_examples: list, val_examples: list, val_queries: list[EvalQuery], tensorboard_run_name: str) -> Path`
- **Acceptance:** Run for 1 epoch with 10 train examples: checkpoint saved to `checkpoints/biencoder/`. TensorBoard run created under `runs/`.
- **Watch out:** NDCG@10 early stopping requires a full retrieval pass per epoch on the val set. This is slow — run it every `config.eval.early_stop_patience` epochs (not every epoch) to keep training tractable. The `sentence-transformers` Trainer API expects a specific dataset format; check `sentence_transformers.datasets.SentenceLabelDataset` compatibility.

---

## Unresolved Questions

These items are architecturally ambiguous in the current skeleton. Do **not** implement them arbitrarily — resolve them first.

### UQ-1: apply_ablation_config — how ablation flags flow to Retriever/Ranker

`AblationConfig` has `use_dense`, `use_bm25`, `use_reranker` flags, but `AppConfig.RetrievalConfig` has no corresponding fields. The Retriever and Ranker check `config.*` values — they cannot currently be told "skip BM25" or "skip reranking."

**Recommended resolution:** Add `use_dense: bool = True`, `use_bm25: bool = True`, `use_reranker: bool = True` to `RetrievalConfig` in `schema.py`. Then `apply_ablation_config` sets these flags on a copy of the config. The Retriever's `retrieve()` (T16) checks them before dense/BM25 retrieval. The Ranker's `rerank()` (T17) checks `use_reranker` and short-circuits if False. This requires a minor addition to `schema.py` and `config.yaml`.

### UQ-2: eval/queries.json — relevant_ids are empty

All 10 stub queries in `eval/queries.json` have `relevant_ids: []`. The eval harness will produce 0.0 for all metrics until these are populated with real FMA track IDs. This is a **data labeling task**, not a code task. It must be done manually after `build_catalog.py` and `generate_descriptors.py` complete. Budget ~2 hours to label ~12 relevant tracks per query (scope.md §evaluation: "50 hand-authored queries with ~12 relevant tracks labeled per query").

### UQ-3: hybrid_rerank_expanded — query expansion has no skeleton

The 5th ablation config (`hybrid_rerank_expanded`) requires LLM query expansion before retrieval. There is no stub for a query expander anywhere in the codebase. Scope.md §Tier 1 describes this feature (~4 hrs), but no `src/serving/query_expander.py` stub was created. Either create this stub explicitly before implementing T23, or skip `hybrid_rerank_expanded` in the ablation matrix and document the omission.

### UQ-4: ChromaDB distance space coupling

`Retriever._compute_confidence` (T09) and `Retriever._dense_retrieve` (T11) interpret ChromaDB distances as cosine distances (converting via `score = 1 − distance`). This only works correctly if `build_index.py` (T25) creates collections with `metadata={"hnsw:space": "cosine"}`. This coupling must be explicit in T25's implementation. If the default L2 space is used instead, the confidence scores will be wrong.

### UQ-5: BM25 instance attributes not in stub

`Retriever._build_bm25_indexes` (T08) must store parallel track ID lists (e.g., `_bm25_full_ids: list[str]`) alongside the `BM25Okapi` objects. These are required by `_bm25_retrieve` (T12) but are not declared in the `__init__` stub. The implementer should add them in `__init__` as `self._bm25_full_ids: list[str] = []` etc. This is an implementation detail, not an architecture decision.

### UQ-6: Explainer response is plain text, not JSON

`generate_descriptor` (T04) uses `"format": "json"` in the Ollama request and calls `extract_json()`. The `explain_track` (T18) prompt asks for a plain English sentence. Ollama's plain-text response may include trailing newlines or occasional preamble text ("Here is the explanation: ..."). A simple `.strip()` should suffice, but the implementer should test with the actual model to verify response format.

---

## Recommended Model Routing

### Safe for a local/smaller model (Haiku, 7B local):
| Task | Rationale |
|---|---|
| T01 | Straightforward API wrapping, well-documented |
| T02 | Simple string substitution |
| T03 | Simple string parsing |
| T13 | Pure math, well-defined algorithm |
| T14 | Pure logic, well-defined |
| T07 | Single line of real logic |
| T10 | Thin conditional wrapper |
| T11 | Straightforward ChromaDB → Pydantic mapping |
| T20 | Pure math functions with known formulas |
| T21 | Simple I/O with clear contracts |

### Route to Sonnet (requires multi-step logic, cross-file context):
| Task | Rationale |
|---|---|
| T04 | Retry logic with specific exception handling per failure mode |
| T06 | Variance fallback logic, numpy, must match test expectations exactly |
| T08 | ChromaDB API + BM25 index construction + ID mapping |
| T09 | ChromaDB query + centroid computation + cosine distance interpretation |
| T12 | BM25 scores + positional index mapping |
| T15 | Tokenizer API usage, vocabulary threshold selection |
| T17 | Orchestration, must preserve correct response schemas |
| T18 | Retry logic + plain-text extraction + model copy pattern |
| T19 | CLI wiring of all components |
| T22 | Retrieval eval loop + TensorBoard logging |
| T24 | Script orchestration + TensorBoard metrics |
| T26 | Script with regression check + exit codes |
| T27 | Embedding similarity mining with genre filtering |
| T28 | sentence-transformers dataset construction |

### Route to Opus or Sonnet with extended thinking (complex integration, architectural decisions):
| Task | Rationale |
|---|---|
| T05 | Resume logic + checkpoint versioning + batch loop + merge; many interacting constraints |
| T16 | Orchestrates all 7 steps; must propagate warnings correctly across multiple subsystems |
| T23 | **Blocked by UQ-1**; `apply_ablation_config` requires schema changes before implementation |
| T25 | Cosine dedup logic + staging/validation/atomic promotion; most complex script |
| T29 | Combined losses + custom NDCG early stopping callback; sentence-transformers Trainer internals |
