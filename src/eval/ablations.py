"""src/eval/ablations.py

Orchestrates the 5-row ablation matrix.
scope.md §ablation table: 5 configurations, all four metrics, logged to TensorBoard.
CLAUDE.md: Every row in the README ablation table maps to a named, reproducible
           TensorBoard run. If the run doesn't exist, the number doesn't go in the table.
ADR-004: Run naming convention: {YYYYMMDD}_{config_hash_6char}_{short_description}

The 5 ablation configurations (CLAUDE.md §evaluation):
    1. dense_only      — bge-base ANN, no reranking, no BM25
    2. bm25_only       — BM25 over descriptor text, no dense
    3. hybrid_rrf      — Dense + BM25 fused via RRF
    4. hybrid_rerank   — Hybrid + cross-encoder reranking
    5. hybrid_rerank_expanded — Hybrid + reranking + LLM query expansion (Tier 1)

Import discipline (CLAUDE.md): eval/ imports from serving/ via Retriever interface.
"""

from __future__ import annotations

import dataclasses
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from src.data.schema import AppConfig
from src.eval.runner import run_eval
from src.serving.ranker import Ranker
from src.serving.retriever import Retriever

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class AblationConfig:
    """One row in the ablation matrix.

    Each config overrides specific retrieval parameters from the base AppConfig.
    scope.md §ablation table: 5 required configurations.
    """

    name: str                       # e.g. "dense_only", "hybrid_rerank"
    description: str
    use_dense: bool = True
    use_bm25: bool = True
    use_reranker: bool = True
    use_query_expansion: bool = False
    config_overrides: dict[str, Any] = dataclasses.field(default_factory=dict)


# The 5 required ablation configurations (CLAUDE.md)
ABLATION_CONFIGS: list[AblationConfig] = [
    AblationConfig(
        name="dense_only",
        description="bge-base ANN retrieval only — no BM25, no reranking",
        use_dense=True,
        use_bm25=False,
        use_reranker=False,
    ),
    AblationConfig(
        name="bm25_only",
        description="BM25 over descriptor text — no dense retrieval",
        use_dense=False,
        use_bm25=True,
        use_reranker=False,
    ),
    AblationConfig(
        name="hybrid_rrf",
        description="Dense + BM25 fused via Reciprocal Rank Fusion",
        use_dense=True,
        use_bm25=True,
        use_reranker=False,
    ),
    AblationConfig(
        name="hybrid_rerank",
        description="Hybrid RRF + cross-encoder reranking (ms-marco-MiniLM-L-6-v2)",
        use_dense=True,
        use_bm25=True,
        use_reranker=True,
    ),
    AblationConfig(
        name="hybrid_rerank_expanded",
        description="Hybrid + reranking + LLM query expansion via Mistral 7B (Tier 1)",
        use_dense=True,
        use_bm25=True,
        use_reranker=True,
        use_query_expansion=True,
    ),
]


def make_run_name(ablation: AblationConfig, config: AppConfig) -> str:
    """Generate a TensorBoard run name for this ablation.

    ADR-004: {YYYYMMDD}_{config_hash_6char}_{short_description}
    e.g. 20260322_a3f91c_hybrid_rerank
    """
    raise NotImplementedError


def apply_ablation_config(
    base_config: AppConfig,
    ablation: AblationConfig,
) -> AppConfig:
    """Return a modified AppConfig reflecting the ablation's overrides.

    Each ablation is a self-contained config modification — enables deterministic
    reproduction from config alone (ADR-006 invariant 3).
    """
    raise NotImplementedError


def run_ablation_matrix(
    base_config: AppConfig,
    ablation_configs: list[AblationConfig] | None = None,
    smoke_test: bool = False,
) -> dict[str, list]:
    """Run all ablation configurations and return a results summary.

    ADR-004: Each config runs under a separate named TensorBoard run.
    scope.md §ablation table: Results written to eval/results.csv.

    Args:
        base_config: Base AppConfig loaded from config.yaml.
        ablation_configs: Override list; defaults to ABLATION_CONFIGS.
        smoke_test: If True, runs only smoke_test_n_queries queries (CI mode).

    Returns:
        dict mapping config_name → list of EvalResult.
    """
    raise NotImplementedError


def print_ablation_table(results_by_config: dict[str, list]) -> None:
    """Print a markdown ablation table to stdout.

    scope.md: "This table is the engineering contribution. Run it. Put it in the README."
    Format matches scope.md §ablation table structure.
    """
    raise NotImplementedError
