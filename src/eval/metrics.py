"""src/eval/metrics.py

Retrieval evaluation metrics: NDCG@k, MRR, Precision@k, Novelty@k.
scope.md §evaluation suite: These four metrics are the required reporting set.
CLAUDE.md: eval/ imports from serving/ via the public Retriever interface only.

All functions are pure (no side effects, no I/O) and deterministic.
"""

from __future__ import annotations

import math


def ndcg_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int = 10,
) -> float:
    """Normalized Discounted Cumulative Gain at k.

    scope.md: Were the most relevant tracks ranked highest?
    Standard binary relevance: 1 if retrieved_id ∈ relevant_ids, else 0.

    Args:
        retrieved_ids: Ordered list of track IDs from the retrieval system.
        relevant_ids: Set of ground-truth relevant track IDs for this query.
        k: Cutoff rank.

    Returns:
        NDCG@k ∈ [0, 1].
    """
    raise NotImplementedError


def mrr(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int = 10,
) -> float:
    """Mean Reciprocal Rank.

    scope.md: Was there at least one relevant track near the top?
    Evaluates at position of first relevant result, up to k.

    Args:
        retrieved_ids: Ordered list of track IDs.
        relevant_ids: Ground-truth relevant IDs.
        k: Max rank to consider.

    Returns:
        MRR ∈ [0, 1]. 0.0 if no relevant track in top-k.
    """
    raise NotImplementedError


def precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int = 10,
) -> float:
    """Precision at k.

    scope.md: Did the top-k contain relevant tracks?

    Returns:
        P@k ∈ [0, 1].
    """
    raise NotImplementedError


def novelty_at_k(
    retrieved_ids: list[str],
    track_popularity: dict[str, float],
    k: int = 10,
) -> float:
    """Novelty at k — inverse popularity of retrieved tracks.

    scope.md: Anti-popularity-bias signal. Inverse of average popularity score
    of the top-k results. Higher = more novel (less popular) results.

    scope.md §Tier 1: Naive dense retrieval over-retrieves popular tracks
    because their descriptors are richer. Novelty@k measures this bias.

    Args:
        retrieved_ids: Ordered list of track IDs.
        track_popularity: Mapping of track_id → popularity score [0, 1].
                          Higher = more popular. From FMA play_count, normalized.
        k: Cutoff rank.

    Returns:
        Novelty@k ∈ [0, 1]. Higher is better (less popular = more novel).
    """
    raise NotImplementedError


def compute_all_metrics(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    track_popularity: dict[str, float],
    k: int = 10,
) -> dict[str, float]:
    """Compute all four metrics for a single query.

    Returns:
        dict with keys: ndcg_at_k, mrr, precision_at_k, novelty_at_k.
    """
    return {
        "ndcg_at_k":       ndcg_at_k(retrieved_ids, relevant_ids, k),
        "mrr":             mrr(retrieved_ids, relevant_ids, k),
        "precision_at_k":  precision_at_k(retrieved_ids, relevant_ids, k),
        "novelty_at_k":    novelty_at_k(retrieved_ids, track_popularity, k),
    }
