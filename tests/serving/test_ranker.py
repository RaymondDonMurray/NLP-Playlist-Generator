"""tests/serving/test_ranker.py

Tests for src/serving/ranker.py and src/models/reranker.py — reranking logic.
scope.md §CI/CD: Unit tests for ranker.py — mock ChromaDB, test scoring logic.
failure_modes.md §2.3: Low cross-encoder score variance → fallback to ANN order.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.data.schema import AppConfig, CandidateTrack, RerankRequest
from src.models.reranker import Reranker


class TestRerankerFallback:
    """failure_modes.md §2.3: Low variance → ANN order fallback."""

    def test_low_variance_returns_ann_order(self, config, fixture_candidates):
        """When score std < threshold, candidates returned in input (ANN) order."""
        reranker = Reranker(config=config)
        uniform_scores = np.array([0.5] * len(fixture_candidates))

        with patch.object(reranker, "_score_pairs", return_value=uniform_scores):
            ranked, meta = reranker.rerank("chill jazz", fixture_candidates)

        assert meta["reranker_used"] is False
        assert meta["reason"] == "low_variance"
        # Order should match input (ANN order)
        assert [c.track_id for c in ranked] == [c.track_id for c in fixture_candidates]

    def test_high_variance_uses_reranker(self, config, fixture_candidates):
        """When score std ≥ threshold, cross-encoder ranking is used."""
        reranker = Reranker(config=config)
        # High variance scores — one track clearly dominates
        scores = np.array([0.9, 0.1, 0.5, 0.2, 0.3])

        with patch.object(reranker, "_score_pairs", return_value=scores):
            ranked, meta = reranker.rerank("chill jazz", fixture_candidates)

        assert meta["reranker_used"] is True
        # Highest score (0.9, index 0) should be first
        assert ranked[0].track_id == fixture_candidates[0].track_id

    def test_rerank_scores_stored_on_candidates(self, config, fixture_candidates):
        """Reranked candidates have rerank_score populated."""
        reranker = Reranker(config=config)
        scores = np.array([0.9, 0.1, 0.5, 0.2, 0.3])

        with patch.object(reranker, "_score_pairs", return_value=scores):
            ranked, meta = reranker.rerank("chill jazz", fixture_candidates)

        if meta["reranker_used"]:
            assert ranked[0].rerank_score is not None

    def test_variance_threshold_from_config(self):
        """Variance threshold is read from config, not hardcoded."""
        config = AppConfig()
        config.retrieval.rerank_variance_threshold = 0.20   # high threshold
        reranker = Reranker(config=config)

        candidates = [
            CandidateTrack(track_id=f"t{i}", title="T", artist="A", genre_top="Jazz")
            for i in range(3)
        ]
        # Std of [0.5, 0.6, 0.55] ≈ 0.04 < 0.20 → should fall back
        scores = np.array([0.5, 0.6, 0.55])
        with patch.object(reranker, "_score_pairs", return_value=scores):
            _, meta = reranker.rerank("query", candidates)
        assert meta["reranker_used"] is False
