"""src/eval/runner.py

Evaluation harness: loads queries.json, runs retrieval, writes results.
CLAUDE.md: eval/ is first-class code — not a notebook, not a one-off script.
           Runnable from: python scripts/run_eval.py --config config/config.yaml
           Deterministic: same config → same results.
           Logged to TensorBoard under a named run (ADR-004).

scope.md §evaluation suite: 50 queries, ~12 relevant tracks per query,
author-labeled (must be acknowledged in README).

Import discipline (CLAUDE.md): eval/ imports from serving/ via Retriever interface.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from src.data.schema import AppConfig, EvalQuery, EvalResult
from src.serving.ranker import Ranker
from src.serving.retriever import Retriever

logger = logging.getLogger(__name__)


def load_eval_queries(queries_path: Path) -> list[EvalQuery]:
    """Load and validate eval queries from queries.json.

    scope.md §evaluation: {query, relevant_ids, source, notes}
    Raises ValidationError (hard failure) if any query fails the EvalQuery schema.
    """
    raw = json.loads(queries_path.read_text())
    return [EvalQuery.model_validate(entry) for entry in raw]


def run_single_query(
    query: EvalQuery,
    retriever: Retriever,
    ranker: Ranker,
    config: AppConfig,
    track_popularity: dict[str, float],
    config_name: str,
) -> EvalResult:
    """Run retrieval for a single eval query and compute all metrics.

    CLAUDE.md: When reporting eval metrics, always specify which index tier
    was used. Do not mix results from both tiers in a single NDCG number.
    """
    from src.eval.metrics import compute_all_metrics

    retrieve_response, rerank_response = ranker.retrieve_and_rerank(
        query.query,
        top_k_retrieve=config.retrieval.ann_top_k,
        top_k_rerank=config.retrieval.rerank_top_k,
    )

    retrieved_ids = [t.track_id for t in rerank_response.tracks]
    relevant_ids = set(query.relevant_ids)

    metrics = compute_all_metrics(
        retrieved_ids=retrieved_ids,
        relevant_ids=relevant_ids,
        track_popularity=track_popularity,
        k=config.eval.ndcg_k,
    )

    warning = rerank_response.warning or retrieve_response.warning

    return EvalResult(
        query=query.query,
        config_name=config_name,
        ndcg_at_k=metrics["ndcg_at_k"],
        mrr=metrics["mrr"],
        precision_at_k=metrics["precision_at_k"],
        novelty_at_k=metrics["novelty_at_k"],
        index_tier=retrieve_response.index_tier_used,
        reranker_used=rerank_response.reranker_used,
        warning=warning,
    )


def run_eval(
    config: AppConfig,
    retriever: Retriever,
    ranker: Ranker,
    config_name: str,
    writer: SummaryWriter,
    queries_override: list[EvalQuery] | None = None,
    track_popularity: dict[str, float] | None = None,
) -> list[EvalResult]:
    """Run the full evaluation harness.

    Loads queries from config.eval.queries_path (or queries_override for
    smoke tests). Computes all four metrics per query. Logs aggregate metrics
    to TensorBoard. Writes results to config.eval.results_path.

    scope.md §evaluation: source="author_labeled" caveat must propagate
    to any result summary.
    ADR-004: Logged under eval/ TensorBoard scope.
    """
    queries = queries_override if queries_override is not None else load_eval_queries(
        Path(config.eval.queries_path)
    )
    pop = track_popularity or {}

    results: list[EvalResult] = []
    for i, query in enumerate(queries):
        logger.info(
            "Evaluating query %d/%d [%s]: '%s'",
            i + 1, len(queries), config_name, query.query[:60],
        )
        try:
            result = run_single_query(query, retriever, ranker, config, pop, config_name)
            results.append(result)
        except Exception as exc:
            logger.error("Query '%s' failed: %s", query.query[:60], exc)

    if results:
        mean_ndcg    = sum(r.ndcg_at_k      for r in results) / len(results)
        mean_mrr     = sum(r.mrr            for r in results) / len(results)
        mean_prec    = sum(r.precision_at_k for r in results) / len(results)
        mean_novelty = sum(r.novelty_at_k   for r in results) / len(results)

        logger.info(
            "Eval complete [%s]: NDCG@10=%.3f MRR=%.3f P@10=%.3f Novelty=%.3f",
            config_name, mean_ndcg, mean_mrr, mean_prec, mean_novelty,
        )

        # ADR-004: log aggregate metrics to TensorBoard under eval/ scope
        writer.add_scalar("eval/ndcg_at_10",      mean_ndcg)
        writer.add_scalar("eval/mrr",              mean_mrr)
        writer.add_scalar("eval/precision_at_10",  mean_prec)
        writer.add_scalar("eval/novelty_at_10",    mean_novelty)

    write_results(results, Path(config.eval.results_path))

    return results


def write_results(results: list[EvalResult], output_path: Path) -> None:
    """Write eval results to CSV.

    CLAUDE.md: eval/results.csv is the ablation table output.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(EvalResult.model_fields.keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(result.model_dump())


def check_regression(
    results: list[EvalResult],
    baseline_path: Path,
    threshold: float,
) -> bool:
    """Check if NDCG@10 has dropped more than threshold from baseline.

    scope.md §CI/CD eval job: Fails if NDCG@10 drops >5% from baseline
    stored in eval/baseline_metrics.json.

    Returns True if no regression (pass), False if regression detected (fail).
    """
    if not baseline_path.exists():
        logger.warning("No baseline found at '%s' — skipping regression check", baseline_path)
        return True

    if not results:
        return True

    baseline = json.loads(baseline_path.read_text())

    if "hybrid_rerank" in baseline:
        baseline_ndcg = baseline["hybrid_rerank"]["ndcg_at_k"]
    else:
        first_key = next(iter(baseline))
        baseline_ndcg = baseline[first_key]["ndcg_at_k"]

    current_mean_ndcg = sum(r.ndcg_at_k for r in results) / len(results)
    drop = baseline_ndcg - current_mean_ndcg

    if drop > threshold:
        logger.error(
            "Regression detected: NDCG@10 dropped %.3f (baseline %.3f → current %.3f)",
            drop, baseline_ndcg, current_mean_ndcg,
        )
        return False

    logger.info(
        "Regression check passed: NDCG@10 %.3f (baseline %.3f, threshold %.3f)",
        current_mean_ndcg, baseline_ndcg, threshold,
    )
    return True
