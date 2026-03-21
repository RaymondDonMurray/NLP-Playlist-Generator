"""tests/serving/test_retriever.py

Tests for src/serving/retriever.py — retrieval logic with mocked ChromaDB.
scope.md §CI/CD: Unit tests for retriever.py — mock ChromaDB, test scoring logic.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.data.schema import AppConfig, CandidateTrack, RetrieveRequest
from src.serving.retriever import Retriever


class TestRRFFusion:
    """Test Reciprocal Rank Fusion implementation."""

    def test_rrf_combines_both_lists(self, config, fixture_candidates):
        """RRF produces a result for every track in either input list."""
        retriever = Retriever(config=config, encoder=MagicMock())
        dense = fixture_candidates[:3]
        bm25 = fixture_candidates[2:]   # overlap at index 2

        fused = retriever._rrf_fuse(dense, bm25)
        all_ids = {c.track_id for c in dense + bm25}
        result_ids = {c.track_id for c in fused}
        assert result_ids == all_ids

    def test_rrf_top_result_in_both_lists_scores_highest(self, config):
        """A track at top of both dense and BM25 lists should score highest."""
        retriever = Retriever(config=config, encoder=MagicMock())
        shared_top = CandidateTrack(
            track_id="top_both", title="T", artist="A", genre_top="Jazz",
            dense_score=1.0,
        )
        dense = [shared_top] + [
            CandidateTrack(track_id=f"d{i}", title="T", artist="A", genre_top="Jazz")
            for i in range(4)
        ]
        bm25 = [shared_top] + [
            CandidateTrack(track_id=f"b{i}", title="T", artist="A", genre_top="Jazz")
            for i in range(4)
        ]
        fused = retriever._rrf_fuse(dense, bm25)
        assert fused[0].track_id == "top_both"


class TestRetrievalCollapseDetection:
    """failure_modes.md §2.2: Retrieval collapse detection."""

    def test_detects_single_genre_collapse(self, config):
        """Returns True when >80% of candidates share one genre unexpectedly."""
        retriever = Retriever(config=config, encoder=MagicMock())
        candidates = [
            CandidateTrack(track_id=f"t{i}", title="T", artist="A", genre_top="Jazz")
            for i in range(9)
        ] + [
            CandidateTrack(track_id="t9", title="T", artist="A", genre_top="Folk")
        ]
        # Query doesn't mention Jazz — should detect collapse
        assert retriever._detect_retrieval_collapse(candidates, "chill rainy afternoon") is True

    def test_no_collapse_when_genre_expected(self, config):
        """Returns False when the dominant genre is explicitly in the query."""
        retriever = Retriever(config=config, encoder=MagicMock())
        candidates = [
            CandidateTrack(track_id=f"t{i}", title="T", artist="A", genre_top="Jazz")
            for i in range(10)
        ]
        # Query mentions Jazz — collapse is expected
        assert retriever._detect_retrieval_collapse(candidates, "smooth jazz relaxing") is False

    def test_no_collapse_when_diverse(self, config, fixture_candidates):
        """Returns False for a naturally diverse candidate set."""
        retriever = Retriever(config=config, encoder=MagicMock())
        # fixture_candidates have different genres
        assert retriever._detect_retrieval_collapse(fixture_candidates, "chill music") is False


class TestIndexTierSelection:
    """ADR-006: Auto-routing by confidence score."""

    def test_auto_routes_to_full_above_threshold(self, config):
        """Confidence ≥ threshold → full_enrichment index."""
        retriever = Retriever(config=config, encoder=MagicMock(), index_tier="auto")
        with patch.object(retriever, "_compute_confidence", return_value=0.80):
            tier = retriever._select_index_tier([0.1] * 768)
        assert tier == "full"

    def test_auto_routes_to_metadata_below_threshold(self, config):
        """Confidence < threshold → metadata_only index."""
        retriever = Retriever(config=config, encoder=MagicMock(), index_tier="auto")
        with patch.object(retriever, "_compute_confidence", return_value=0.50):
            tier = retriever._select_index_tier([0.1] * 768)
        assert tier == "metadata_only"

    def test_explicit_tier_bypasses_routing(self, config):
        """Explicit index_tier="full" skips confidence computation."""
        retriever = Retriever(config=config, encoder=MagicMock(), index_tier="full")
        tier = retriever._select_index_tier([0.1] * 768)
        assert tier == "full"
