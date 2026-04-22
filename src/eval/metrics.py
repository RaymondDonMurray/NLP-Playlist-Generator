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
    if not relevant_ids:
        return 0.0

    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, track_id in enumerate(retrieved_ids[:k])
        if track_id in relevant_ids
    )

    ideal_hits = min(k, len(relevant_ids))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    return dcg / idcg if idcg > 0.0 else 0.0


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
    for i, track_id in enumerate(retrieved_ids[:k]):
        if track_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


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
    hits = sum(1 for t in retrieved_ids[:k] if t in relevant_ids)
    return hits / k


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
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    novelties = [1.0 - track_popularity.get(t, 0.5) for t in top_k]
    return sum(novelties) / len(novelties)


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
