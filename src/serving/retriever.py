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
from typing import Literal

import chromadb
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

    def _get_chroma_client(self) -> chromadb.ClientAPI:
        """Lazy-init ChromaDB persistent client (connection pooled at startup)."""
        raise NotImplementedError

    def _build_bm25_indexes(self) -> None:
        """Build BM25 indexes over descriptor text for both collections.

        scope.md §hybrid retrieval: rank_bm25 over descriptor text, fused
        with dense scores via Reciprocal Rank Fusion.
        """
        raise NotImplementedError

    def _compute_confidence(
        self,
        query_embedding: list[float],
        collection: chromadb.Collection,
        top_k: int = 5,
    ) -> float:
        """Compute routing confidence for a query against the full_enrichment index.

        ADR-006: Cosine similarity between query embedding and centroid of top-5
        candidates from full_enrichment. If score ≥ confidence_threshold, route
        to full_enrichment; else metadata_only.
        """
        raise NotImplementedError

    def _select_index_tier(
        self,
        query_embedding: list[float],
    ) -> Literal["full", "metadata_only"]:
        """Resolve auto-routing to a concrete index tier.

        ADR-006: Only called when index_tier="auto". Otherwise returns
        the explicitly specified tier.
        """
        raise NotImplementedError

    def _dense_retrieve(
        self,
        query_embedding: list[float],
        collection: chromadb.Collection,
        top_k: int,
    ) -> list[CandidateTrack]:
        """ANN retrieval from ChromaDB using cosine similarity."""
        raise NotImplementedError

    def _bm25_retrieve(
        self,
        query: str,
        tier: Literal["full", "metadata_only"],
        top_k: int,
    ) -> list[CandidateTrack]:
        """BM25 retrieval over descriptor text."""
        raise NotImplementedError

    def _rrf_fuse(
        self,
        dense_results: list[CandidateTrack],
        bm25_results: list[CandidateTrack],
        k: int | None = None,
    ) -> list[CandidateTrack]:
        """Reciprocal Rank Fusion of dense and BM25 result lists.

        scope.md: Hybrid BM25 + dense RRF fusion. Concrete ablation vs dense-only.
        """
        raise NotImplementedError

    def _detect_retrieval_collapse(
        self,
        candidates: list[CandidateTrack],
        query: str,
    ) -> bool:
        """Check if candidates are dominated by a single genre unexpectedly.

        failure_modes.md §2.2: If >80% of top-50 candidates share one genre and
        the query doesn't name that genre, log warning and set response.warning.
        """
        raise NotImplementedError

    def _check_query_vocabulary(self, query: str) -> list[str]:
        """Check for likely OOV tokens in the query.

        failure_modes.md §2.4: bge-base has poor representations for niche
        genre terms (hyperpop, djent, zeuhl, fourth-world). Log a warning
        when query tokens are likely OOV. Does not fix the failure, provides
        diagnostic signal.
        """
        raise NotImplementedError

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
        raise NotImplementedError
