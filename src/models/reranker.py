"""src/models/reranker.py

Cross-encoder reranker: ms-marco-MiniLM-L-6-v2.
ADR-001: Well-matched to natural language query → short-text reranking.
         Reranks 50 candidates in ~12ms on MPS.
failure_modes.md §2.3: Low score variance → fall back to ANN order.

Import discipline (CLAUDE.md): models/ imports from data/schema.py only.
"""

from __future__ import annotations

import logging

import numpy as np
from sentence_transformers import CrossEncoder

from src.data.schema import AppConfig, CandidateTrack

logger = logging.getLogger(__name__)


class Reranker:
    """Wrapper around ms-marco-MiniLM-L-6-v2 for cross-encoder reranking.

    ADR-001: ~12ms for 50 candidate pairs on M1 Max MPS backend.
    Initialized once at serving startup; stateless between requests (ADR-006).
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.model_name = config.retrieval.cross_encoder
        self._model: CrossEncoder | None = None

    def _load_model(self) -> CrossEncoder:
        """Lazy-load cross-encoder. Model name read from config (CLAUDE.md)."""
        logger.info("Loading cross-encoder model: %s", self.model_name)
        return CrossEncoder(self.model_name)

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[CandidateTrack],
    ) -> tuple[list[CandidateTrack], dict]:
        """Rerank candidates and return sorted list with metadata.

        failure_modes.md §2.3: If cross-encoder score std is below
        config.retrieval.rerank_variance_threshold, returns ANN order and
        sets reranker_used=False in metadata dict.

        Returns:
            (ranked_candidates, metadata) where metadata contains:
            - reranker_used: bool
            - score_std: float
            - reason: str (if reranker_used=False)
        """
        scores = self._score_pairs(query, candidates)
        score_std = float(np.std(scores))

        if score_std < self.config.retrieval.rerank_variance_threshold:
            logger.warning(
                "Low rerank variance (%.3f) for query '%s' — returning ANN order",
                score_std,
                query[:60],
            )
            return candidates, {
                "reranker_used": False,
                "reason": "low_variance",
                "score_std": score_std,
            }

        ranked = [
            candidate.model_copy(update={"rerank_score": float(score)})
            for candidate, score in sorted(
                zip(candidates, scores), key=lambda x: -x[1]
            )
        ]
        return ranked, {"reranker_used": True, "score_std": score_std}

    def _score_pairs(
        self,
        query: str,
        candidates: list[CandidateTrack],
    ) -> np.ndarray:
        """Score all (query, descriptor) pairs with the cross-encoder."""
        pairs = [(query, candidate.descriptor or "") for candidate in candidates]
        return self.model.predict(pairs)
