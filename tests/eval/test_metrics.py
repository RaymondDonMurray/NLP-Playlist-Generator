"""tests/eval/test_metrics.py

Tests for src/eval/metrics.py — NDCG@k, MRR, Precision@k, Novelty@k.
All metrics functions are pure; no mocking required.
"""

from __future__ import annotations

import pytest

from src.eval.metrics import ndcg_at_k, mrr, novelty_at_k, precision_at_k


class TestNDCGAtK:

    def test_perfect_ranking(self):
        """Perfect ranking (all relevant, top-ranked) → NDCG = 1.0."""
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        assert ndcg_at_k(retrieved, relevant, k=3) == pytest.approx(1.0)

    def test_no_relevant_results(self):
        """No relevant results retrieved → NDCG = 0.0."""
        retrieved = ["x", "y", "z"]
        relevant = {"a", "b"}
        assert ndcg_at_k(retrieved, relevant, k=3) == pytest.approx(0.0)

    def test_single_relevant_at_top(self):
        """One relevant result at top → NDCG > 0."""
        retrieved = ["a", "x", "y"]
        relevant = {"a"}
        assert ndcg_at_k(retrieved, relevant, k=3) > 0.0

    def test_relevant_at_bottom_scores_less(self):
        """Relevant result at bottom scores lower than at top."""
        relevant = {"a"}
        top_score = ndcg_at_k(["a", "x", "y"], relevant, k=3)
        bottom_score = ndcg_at_k(["x", "y", "a"], relevant, k=3)
        assert top_score > bottom_score

    def test_cutoff_k_respected(self):
        """Relevant results beyond cutoff k are ignored."""
        retrieved = ["x", "y", "z", "a"]   # "a" is at rank 4
        relevant = {"a"}
        assert ndcg_at_k(retrieved, relevant, k=3) == pytest.approx(0.0)


class TestMRR:

    def test_first_result_relevant(self):
        """First result relevant → MRR = 1.0."""
        assert mrr(["a", "b", "c"], {"a"}, k=10) == pytest.approx(1.0)

    def test_second_result_relevant(self):
        """Second result relevant → MRR = 0.5."""
        assert mrr(["x", "a", "b"], {"a"}, k=10) == pytest.approx(0.5)

    def test_no_relevant_results(self):
        """No relevant results → MRR = 0.0."""
        assert mrr(["x", "y", "z"], {"a"}, k=10) == pytest.approx(0.0)

    def test_relevant_beyond_k_ignored(self):
        """Relevant result at rank > k → MRR = 0.0."""
        assert mrr(["x", "a"], {"a"}, k=1) == pytest.approx(0.0)


class TestPrecisionAtK:

    def test_all_relevant(self):
        """All top-k results relevant → P@k = 1.0."""
        assert precision_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3) == pytest.approx(1.0)

    def test_half_relevant(self):
        """Half of top-k relevant → P@k = 0.5."""
        assert precision_at_k(["a", "x", "b", "y"], {"a", "b"}, k=4) == pytest.approx(0.5)

    def test_none_relevant(self):
        """No relevant results → P@k = 0.0."""
        assert precision_at_k(["x", "y", "z"], {"a"}, k=3) == pytest.approx(0.0)


class TestNoveltyAtK:

    def test_all_unpopular(self, track_popularity):
        """Very unpopular tracks → high novelty."""
        # Override popularity: all very unpopular
        popularity = {"t1": 0.01, "t2": 0.02}
        score = novelty_at_k(["t1", "t2"], popularity, k=2)
        assert score > 0.9

    def test_all_popular(self):
        """Very popular tracks → low novelty."""
        popularity = {"t1": 0.99, "t2": 0.98}
        score = novelty_at_k(["t1", "t2"], popularity, k=2)
        assert score < 0.1

    def test_missing_track_in_popularity(self):
        """Tracks not in popularity dict are handled gracefully."""
        popularity = {"t1": 0.5}
        # t2 is not in popularity — should not crash
        score = novelty_at_k(["t1", "t2_missing"], popularity, k=2)
        assert 0.0 <= score <= 1.0

    def test_cutoff_k_respected(self, track_popularity):
        """Only top-k tracks contribute to novelty score."""
        retrieved = list(track_popularity.keys())
        # k=1 should only consider the first track
        score_k1 = novelty_at_k(retrieved, track_popularity, k=1)
        expected = 1.0 - track_popularity[retrieved[0]]
        assert score_k1 == pytest.approx(expected, abs=0.01)
