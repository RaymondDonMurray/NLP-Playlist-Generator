"""src/serving/retriever.py

Retriever class: dense ANN + BM25 hybrid retrieval with auto-routing.
ADR-006: Retriever(index_tier="auto") selects index by query confidence score.
ADR-002: Dual ChromaDB indexes — full_enrichment and metadata_only.
failure_modes.md §2.2: Retrieval collapse detection.
failure_modes.md §2.4: OOV genre term detection.

Import discipline (CLAUDE.md): serving/ imports from models/ and data/schema.py.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from pathlib import Path
from typing import Literal

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi

from src.data.schema import (
    AppConfig,
    CandidateTrack,
    RetrieveRequest,
    RetrieveResponse,
)
from src.models.encoder import Encoder

logger = logging.getLogger(__name__)

# Collection names — must match build_index.py
COLLECTION_FULL      = "full_enrichment"
COLLECTION_META_ONLY = "metadata_only"


class Retriever:
    """Two-index ChromaDB retriever with auto-routing and BM25 hybrid fusion.

    ADR-006: index_tier="auto" computes confidence as cosine similarity between
    query embedding and centroid of top-5 candidates from full_enrichment.
    Routes to full_enrichment if score ≥ config.retrieval.confidence_threshold,
    else metadata_only.

    Initialized once at startup; stateless between requests (ADR-006 invariant 2).
    """

    def __init__(
        self,
        config: AppConfig,
        encoder: Encoder,
        index_tier: Literal["auto", "full", "metadata_only"] = "auto",
    ) -> None:
        self.config = config
        self.encoder = encoder
        self.index_tier = index_tier
        self._chroma_client: chromadb.ClientAPI | None = None
        self._bm25_full: BM25Okapi | None = None
        self._bm25_meta: BM25Okapi | None = None
        # Parallel ID and document lists for BM25 positional index mapping (T08/T12)
        self._bm25_full_ids: list[str] = []
        self._bm25_meta_ids: list[str] = []
        self._bm25_full_docs: list[str] = []
        self._bm25_meta_docs: list[str] = []

    # ------------------------------------------------------------------
    # T07: Lazy ChromaDB client init
    # ------------------------------------------------------------------

    def _get_chroma_client(self) -> chromadb.ClientAPI:
        """Lazy-init ChromaDB persistent client (connection pooled at startup).

        failure_modes.md §4.2: Never write to production directly.
        Points to chroma_db/production/ (built by scripts/build_index.py).
        """
        if self._chroma_client is None:
            production_path = str(Path(self.config.paths.chroma_dir) / "production")
            logger.info("Initialising ChromaDB client at '%s'", production_path)
            self._chroma_client = chromadb.PersistentClient(path=production_path)
        return self._chroma_client

    # ------------------------------------------------------------------
    # T08: Build in-memory BM25 indexes
    # ------------------------------------------------------------------

    def _build_bm25_indexes(self) -> None:
        """Build BM25 indexes over descriptor text for both collections.

        scope.md §hybrid retrieval: rank_bm25 over descriptor text, fused
        with dense scores via Reciprocal Rank Fusion.

        Stores parallel track ID lists alongside BM25 indexes so positional
        results (index-based) can be mapped back to track_id in _bm25_retrieve.
        """
        client = self._get_chroma_client()

        for collection_name, id_attr, doc_attr, bm25_attr in [
            (COLLECTION_FULL,      "_bm25_full_ids",  "_bm25_full_docs",  "_bm25_full"),
            (COLLECTION_META_ONLY, "_bm25_meta_ids",  "_bm25_meta_docs",  "_bm25_meta"),
        ]:
            try:
                collection = client.get_collection(collection_name)
            except Exception:
                logger.warning(
                    "Collection '%s' not found — BM25 index will be empty", collection_name
                )
                setattr(self, bm25_attr, BM25Okapi([[""]]))
                setattr(self, id_attr, [])
                setattr(self, doc_attr, [])
                continue

            result = collection.get(include=["documents", "metadatas"])
            ids: list[str] = result["ids"]
            documents: list[str] = [
                (doc or "") for doc in (result["documents"] or [])
            ]

            tokenized_corpus = [doc.split() for doc in documents]
            # BM25Okapi requires a non-empty corpus
            if not tokenized_corpus:
                tokenized_corpus = [[""]]

            setattr(self, bm25_attr, BM25Okapi(tokenized_corpus))
            setattr(self, id_attr, ids)
            setattr(self, doc_attr, documents)

            logger.info(
                "BM25 %s: %d docs", collection_name, len(ids)
            )

    # ------------------------------------------------------------------
    # T09: Routing confidence score
    # ------------------------------------------------------------------

    def _compute_confidence(
        self,
        query_embedding: list[float],
        collection: chromadb.Collection | None = None,
        top_k: int = 5,
    ) -> float:
        """Compute routing confidence for a query against the full_enrichment index.

        ADR-006: Cosine similarity between query embedding and centroid of top-5
        candidates from full_enrichment. If score ≥ confidence_threshold, route
        to full_enrichment; else metadata_only.

        Both query and stored embeddings are L2-normalised (bge-base with
        normalize_embeddings=True), so cosine similarity = dot product.

        If collection is None, retrieves full_enrichment from the ChromaDB client.
        Passing collection explicitly is supported for callers that already hold
        a reference (e.g. direct integration tests).
        """
        if collection is None:
            client = self._get_chroma_client()
            try:
                collection = client.get_collection(COLLECTION_FULL)
            except Exception as exc:
                logger.warning("full_enrichment not found: %s — confidence 0.0", exc)
                return 0.0

        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["embeddings"],
            )
        except Exception as exc:
            logger.warning("Confidence query failed: %s — defaulting to 0.0", exc)
            return 0.0

        raw_embeddings = results.get("embeddings") or [[]]
        embeddings = raw_embeddings[0] if raw_embeddings else []
        if len(embeddings) == 0:
            return 0.0

        candidate_matrix = np.array(embeddings, dtype=np.float32)
        centroid = candidate_matrix.mean(axis=0)
        # Normalise centroid before dot product (query is already normalised)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm == 0.0:
            return 0.0
        centroid = centroid / centroid_norm

        query_vec = np.array(query_embedding, dtype=np.float32)
        confidence = float(np.dot(query_vec, centroid))
        return confidence

    # ------------------------------------------------------------------
    # T10: Index tier selection
    # ------------------------------------------------------------------

    def _select_index_tier(
        self,
        query_embedding: list[float],
    ) -> Literal["full", "metadata_only"]:
        """Resolve auto-routing to a concrete index tier.

        ADR-006: Only calls _compute_confidence when index_tier="auto".
        Otherwise returns the explicitly specified tier.

        _compute_confidence is called with no collection argument so that it
        can be patched cleanly in tests without requiring a live ChromaDB client.
        """
        if self.index_tier != "auto":
            return self.index_tier  # type: ignore[return-value]

        confidence = self._compute_confidence(query_embedding)
        threshold = self.config.retrieval.confidence_threshold

        if confidence >= threshold:
            logger.info("Routing to full_enrichment (confidence %.3f ≥ %.3f)", confidence, threshold)
            return "full"

        logger.info(
            "Routing to metadata_only (confidence %.3f < %.3f)", confidence, threshold
        )
        return "metadata_only"

    # ------------------------------------------------------------------
    # T11: Dense ANN retrieval
    # ------------------------------------------------------------------

    def _dense_retrieve(
        self,
        query_embedding: list[float],
        collection: chromadb.Collection,
        top_k: int,
    ) -> list[CandidateTrack]:
        """ANN retrieval from ChromaDB using cosine similarity.

        ChromaDB with hnsw:space=cosine returns distance = 1 − cosine_similarity,
        so dense_score = 1 − distance.
        """
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["metadatas", "distances"],
        )

        ids: list[str] = (results["ids"] or [[]])[0]
        metadatas: list[dict] = (results["metadatas"] or [[]])[0]
        distances: list[float] = (results["distances"] or [[]])[0]

        candidates: list[CandidateTrack] = []
        for track_id, meta, dist in zip(ids, metadatas, distances):
            dense_score = 1.0 - dist  # cosine space: distance = 1 - similarity
            candidates.append(
                CandidateTrack(
                    track_id=track_id,
                    title=meta.get("title", ""),
                    artist=meta.get("artist", ""),
                    genre_top=meta.get("genre_top", "Unknown"),
                    descriptor=meta.get("descriptor"),
                    dense_score=dense_score,
                )
            )
        return candidates

    # ------------------------------------------------------------------
    # T12: BM25 retrieval
    # ------------------------------------------------------------------

    def _bm25_retrieve(
        self,
        query: str,
        tier: Literal["full", "metadata_only"],
        top_k: int,
    ) -> list[CandidateTrack]:
        """BM25 retrieval over descriptor text.

        Uses BM25Okapi positional index; maps positions back to track_id via
        the parallel ID list built in _build_bm25_indexes.
        """
        if self._bm25_full is None:
            logger.warning("BM25 indexes not built — building now (slow on first call)")
            self._build_bm25_indexes()

        if tier == "full":
            bm25: BM25Okapi | None = self._bm25_full
            ids = self._bm25_full_ids
            docs = self._bm25_full_docs
        else:
            bm25 = self._bm25_meta
            ids = self._bm25_meta_ids
            docs = self._bm25_meta_docs

        if not ids or bm25 is None:
            return []

        tokenized_query = query.split()
        scores: np.ndarray = bm25.get_scores(tokenized_query)
        top_indices = np.argsort(-scores)[:top_k]

        candidates: list[CandidateTrack] = []
        for idx in top_indices:
            candidates.append(
                CandidateTrack(
                    track_id=ids[idx],
                    title="",       # not stored in BM25 index; enriched after RRF if needed
                    artist="",
                    genre_top="Unknown",
                    descriptor=docs[idx] or None,
                    bm25_score=float(scores[idx]),
                )
            )
        return candidates

    # ------------------------------------------------------------------
    # T13: Reciprocal Rank Fusion
    # ------------------------------------------------------------------

    def _rrf_fuse(
        self,
        dense_results: list[CandidateTrack],
        bm25_results: list[CandidateTrack],
        k: int | None = None,
    ) -> list[CandidateTrack]:
        """Reciprocal Rank Fusion of dense and BM25 result lists.

        scope.md: Hybrid BM25 + dense RRF fusion. Concrete ablation vs dense-only.

        RRF score = Σ 1/(k + rank) across both lists (1-based rank).
        Tracks absent from a list contribute 0 from that list.
        Result is the union of both input lists sorted by RRF score descending.
        """
        rrf_k = k if k is not None else self.config.retrieval.rrf_k

        # Build a map from track_id → CandidateTrack (prefer dense metadata)
        track_map: dict[str, CandidateTrack] = {}
        for candidate in bm25_results:
            track_map[candidate.track_id] = candidate
        for candidate in dense_results:
            track_map[candidate.track_id] = candidate  # dense wins for metadata

        rrf_scores: dict[str, float] = {}

        for rank_0, candidate in enumerate(dense_results):
            tid = candidate.track_id
            rrf_scores[tid] = rrf_scores.get(tid, 0.0) + 1.0 / (rrf_k + rank_0 + 1)

        for rank_0, candidate in enumerate(bm25_results):
            tid = candidate.track_id
            rrf_scores[tid] = rrf_scores.get(tid, 0.0) + 1.0 / (rrf_k + rank_0 + 1)

        fused: list[CandidateTrack] = []
        for tid, score in sorted(rrf_scores.items(), key=lambda x: -x[1]):
            fused.append(track_map[tid].model_copy(update={"rrf_score": score}))

        return fused

    # ------------------------------------------------------------------
    # T14: Retrieval collapse detection
    # ------------------------------------------------------------------

    def _detect_retrieval_collapse(
        self,
        candidates: list[CandidateTrack],
        query: str,
    ) -> bool:
        """Check if candidates are dominated by a single genre unexpectedly.

        failure_modes.md §2.2: If >80% of top-50 candidates share one genre and
        the query doesn't name that genre, log warning and set response.warning.
        """
        if not candidates:
            return False

        genre_counts = Counter(c.genre_top for c in candidates)
        top_genre, top_count = genre_counts.most_common(1)[0]
        top_fraction = top_count / len(candidates)
        threshold = self.config.retrieval.collapse_genre_threshold

        if top_fraction > threshold and top_genre.lower() not in query.lower():
            logger.warning(
                "Retrieval collapse detected: %.0f%% of candidates are '%s'",
                top_fraction * 100,
                top_genre,
            )
            return True
        return False

    # ------------------------------------------------------------------
    # T15: OOV vocabulary check
    # ------------------------------------------------------------------

    def _check_query_vocabulary(self, query: str) -> list[str]:
        """Check for likely OOV tokens in the query.

        failure_modes.md §2.4: bge-base has poor representations for niche
        genre terms (hyperpop, djent, zeuhl, fourth-world). Log a warning
        when query tokens are likely OOV. Does not fix the failure, provides
        diagnostic signal.

        Uses a conservative threshold of 50,000 — bge-base-en-v1.5 has ~30K vocab.
        """
        tokenizer = self.encoder.model.tokenizer
        tokens = tokenizer.tokenize(query.lower())
        ids = tokenizer.convert_tokens_to_ids(tokens)
        oov = [t for t, i in zip(tokens, ids) if i >= 50_000]
        if oov:
            logger.warning("Possible OOV tokens in query: %s", oov)
        return oov

    # ------------------------------------------------------------------
    # T16: Full retrieval pipeline
    # ------------------------------------------------------------------

    def retrieve(self, request: RetrieveRequest) -> RetrieveResponse:
        """Full retrieval pipeline for a single query.

        Steps:
        1. Embed query
        2. OOV check (warning only)
        3. Select index tier (auto-routing if index_tier="auto")
        4. Dense ANN retrieve top-k
        5. BM25 retrieve top-k
        6. RRF fusion
        7. Retrieval collapse check (warning if triggered)

        Returns RetrieveResponse with warning field populated for any soft
        failures (failure_modes.md §4.4).
        """
        start_ms = time.monotonic()
        warning: str | None = None

        # Step 1: Embed query
        query_embedding = self.encoder.encode_query(request.query)

        # Step 2: OOV diagnostic (warning-only, does not block)
        self._check_query_vocabulary(request.query)

        # Step 3: Select index tier
        tier = self._select_index_tier(query_embedding)

        # Step 4: Dense ANN retrieve (skip if use_dense=False for ablations)
        if self.config.retrieval.use_dense:
            client = self._get_chroma_client()
            collection_name = COLLECTION_FULL if tier == "full" else COLLECTION_META_ONLY
            try:
                collection = client.get_collection(collection_name)
            except Exception as exc:
                raise RuntimeError(
                    f"ChromaDB collection '{collection_name}' does not exist. "
                    "Run 'python scripts/build_index.py' to build the index before serving."
                ) from exc
            dense_candidates = self._dense_retrieve(query_embedding, collection, request.top_k)
        else:
            dense_candidates = []

        # Step 5: BM25 retrieve (skip if use_bm25=False for ablations)
        if self.config.retrieval.use_bm25:
            bm25_candidates = self._bm25_retrieve(request.query, tier, request.top_k)
        else:
            bm25_candidates = []

        # Step 6: RRF fusion — truncate to top_k after fusion
        fused = self._rrf_fuse(dense_candidates, bm25_candidates)
        fused = fused[: request.top_k]

        # Step 7: Retrieval collapse check
        if self._detect_retrieval_collapse(fused, request.query):
            warning = "retrieval_collapse"

        elapsed_ms = (time.monotonic() - start_ms) * 1000.0

        return RetrieveResponse(
            query=request.query,
            candidates=fused,
            index_tier_used=tier,
            retrieval_ms=elapsed_ms,
            reranker_used=False,
            warning=warning,
        )
