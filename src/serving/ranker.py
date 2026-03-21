"""src/serving/ranker.py

Two-stage pipeline: retrieve → rerank.
ADR-005: Three decomposed commands; ranker orchestrates retrieve + rerank.
scope.md §two-stage retrieval: ANN top-50 candidates → cross-encoder top-10.
failure_modes.md §2.3: Rerank fallback on low score variance.

Import discipline (CLAUDE.md): serving/ imports from models/ and data/schema.py.
"""

from __future__ import annotations

import logging
import time

from src.data.schema import (
    AppConfig,
    CandidateTrack,
    RerankRequest,
    RerankResponse,
    RetrieveRequest,
    RetrieveResponse,
)
from src.models.reranker import Reranker
from src.serving.retriever import Retriever

logger = logging.getLogger(__name__)


class Ranker:
    """Orchestrates the two-stage retrieve → rerank pipeline.

    ADR-005: Decomposed into retrieve and rerank stages for per-stage
    latency profiling and independent testing.

    Initialized once at startup; stateless between requests (ADR-006 invariant 2).
    """

    def __init__(
        self,
        config: AppConfig,
        retriever: Retriever,
        reranker: Reranker,
    ) -> None:
        self.config = config
        self.retriever = retriever
        self.reranker = reranker

    def retrieve(self, request: RetrieveRequest) -> RetrieveResponse:
        """Stage 1: Dense + BM25 hybrid ANN retrieval.

        Delegates to Retriever.retrieve(). Exposed here so the Ranker can be
        used as a single entry point for the full pipeline.
        """
        raise NotImplementedError

    def rerank(self, request: RerankRequest) -> RerankResponse:
        """Stage 2: Cross-encoder reranking of retrieved candidates.

        failure_modes.md §2.3: If cross-encoder score std < variance_threshold,
        returns candidates in ANN order with reranker_used=False and
        warning="low_rerank_variance" in response.
        """
        raise NotImplementedError

    def retrieve_and_rerank(
        self,
        query: str,
        top_k_retrieve: int | None = None,
        top_k_rerank: int | None = None,
        index_tier: str = "auto",
    ) -> tuple[RetrieveResponse, RerankResponse]:
        """End-to-end pipeline: retrieve top-50 → rerank to top-10.

        scope.md §two-stage retrieval. Returns both stage responses so
        per-stage latency is observable.
        """
        raise NotImplementedError
