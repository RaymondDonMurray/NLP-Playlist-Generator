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
        return self.retriever.retrieve(request)

    def rerank(self, request: RerankRequest) -> RerankResponse:
        """Stage 2: Cross-encoder reranking of retrieved candidates.

        failure_modes.md §2.3: If cross-encoder score std < variance_threshold,
        returns candidates in ANN order with reranker_used=False and
        warning="low_rerank_variance" in response.

        UQ-1: If config.retrieval.use_reranker is False (ablation flag),
        returns candidates in input order without invoking the cross-encoder.
        """
        start_ms = time.monotonic()

        if not self.config.retrieval.use_reranker:
            truncated = request.candidates[: request.top_k]
            elapsed_ms = (time.monotonic() - start_ms) * 1000.0
            return RerankResponse(
                query=request.query,
                tracks=truncated,
                rerank_ms=elapsed_ms,
                reranker_used=False,
                warning=None,
            )

        ranked, meta = self.reranker.rerank(request.query, request.candidates)
        truncated = ranked[: request.top_k]
        elapsed_ms = (time.monotonic() - start_ms) * 1000.0
        warning = "low_rerank_variance" if not meta["reranker_used"] else None
        return RerankResponse(
            query=request.query,
            tracks=truncated,
            rerank_ms=elapsed_ms,
            reranker_used=meta["reranker_used"],
            warning=warning,
        )

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
        top_k_retrieve = top_k_retrieve or self.config.retrieval.ann_top_k
        top_k_rerank = top_k_rerank or self.config.retrieval.rerank_top_k

        retrieve_request = RetrieveRequest(
            query=query,
            top_k=top_k_retrieve,
            index_tier=index_tier,  # type: ignore[arg-type]
        )
        retrieve_response = self.retrieve(retrieve_request)

        rerank_request = RerankRequest(
            query=query,
            candidates=retrieve_response.candidates,
            top_k=top_k_rerank,
        )
        rerank_response = self.rerank(rerank_request)

        return retrieve_response, rerank_response
