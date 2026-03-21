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
    raise NotImplementedError


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
    raise NotImplementedError


def run_eval(
    config: AppConfig,
    retriever: Retriever,
    ranker: Ranker,
    config_name: str,
    writer: SummaryWriter,
    queries_override: list[EvalQuery] | None = None,
) -> list[EvalResult]:
    """Run the full evaluation harness.

    Loads queries from config.eval.queries_path (or queries_override for
    smoke tests). Computes all four metrics per query. Logs aggregate metrics
    to TensorBoard. Writes results to config.eval.results_path.

    scope.md §evaluation: source="author_labeled" caveat must propagate
    to any result summary.
    ADR-004: Logged under eval/ TensorBoard scope.
    """
    raise NotImplementedError


def write_results(results: list[EvalResult], output_path: Path) -> None:
    """Write eval results to CSV.

    CLAUDE.md: eval/results.csv is the ablation table output.
    """
    raise NotImplementedError


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
    raise NotImplementedError
