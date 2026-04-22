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
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from torch.utils.tensorboard import SummaryWriter

from src.data.schema import AppConfig
from src.eval.runner import load_eval_queries, run_eval
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
    date = datetime.now().strftime("%Y%m%d")
    hash6 = hashlib.md5(ablation.name.encode()).hexdigest()[:6]
    return f"{date}_{hash6}_{ablation.name}"


def apply_ablation_config(
    base_config: AppConfig,
    ablation: AblationConfig,
) -> AppConfig:
    """Return a modified AppConfig reflecting the ablation's overrides.

    Each ablation is a self-contained config modification — enables deterministic
    reproduction from config alone (ADR-006 invariant 3).

    UQ-1: Sets use_dense, use_bm25, use_reranker on a copy of RetrievalConfig.
    The Retriever and Ranker read these flags to skip pipeline stages.
    """
    config_dict = base_config.model_dump()
    config_dict["retrieval"]["use_dense"]    = ablation.use_dense
    config_dict["retrieval"]["use_bm25"]     = ablation.use_bm25
    config_dict["retrieval"]["use_reranker"] = ablation.use_reranker

    for key, value in ablation.config_overrides.items():
        parts = key.split(".")
        node = config_dict
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = value

    return AppConfig.model_validate(config_dict)


def run_ablation_matrix(
    base_config: AppConfig,
    ablation_configs: list[AblationConfig] | None = None,
    smoke_test: bool = False,
    track_popularity: dict[str, float] | None = None,
) -> dict[str, list]:
    """Run all ablation configurations and return a results summary.

    ADR-004: Each config runs under a separate named TensorBoard run.
    scope.md §ablation table: Results written to eval/results.csv.

    Args:
        base_config: Base AppConfig loaded from config.yaml.
        ablation_configs: Override list; defaults to ABLATION_CONFIGS.
        smoke_test: If True, runs only smoke_test_n_queries queries (CI mode).
        track_popularity: Optional popularity dict for Novelty@k.

    Returns:
        dict mapping config_name → list of EvalResult.
    """
    from src.models.encoder import Encoder
    from src.models.reranker import Reranker

    configs = ablation_configs if ablation_configs is not None else ABLATION_CONFIGS

    queries = load_eval_queries(Path(base_config.eval.queries_path))
    if smoke_test:
        queries = queries[: base_config.eval.smoke_test_n_queries]

    # Build shared encoder once — model loading is expensive (ADR-001)
    encoder = Encoder(base_config)

    results_by_config: dict[str, list] = {}

    for ablation in configs:
        if ablation.use_query_expansion:
            logger.warning(
                "Skipping '%s' — query expansion not implemented (UQ-3)", ablation.name
            )
            continue

        run_name = make_run_name(ablation, base_config)
        logger.info("Running ablation '%s' (run: %s)", ablation.name, run_name)

        ablation_config = apply_ablation_config(base_config, ablation)

        retriever = Retriever(ablation_config, encoder)
        reranker  = Reranker(ablation_config)
        ranker    = Ranker(ablation_config, retriever, reranker)

        log_dir = str(Path(base_config.tensorboard.log_dir) / run_name)
        writer = SummaryWriter(log_dir=log_dir)

        try:
            results = run_eval(
                config=ablation_config,
                retriever=retriever,
                ranker=ranker,
                config_name=ablation.name,
                writer=writer,
                queries_override=queries,
                track_popularity=track_popularity,
            )
            results_by_config[ablation.name] = results
        finally:
            writer.close()

    return results_by_config


def print_ablation_table(results_by_config: dict[str, list]) -> None:
    """Print a markdown ablation table to stdout.

    scope.md: "This table is the engineering contribution. Run it. Put it in the README."
    Format matches scope.md §ablation table structure.
    """
    print("| Config | P@10 | NDCG@10 | MRR | Novelty |")
    print("|---|---|---|---|---|")

    for config_name, results in results_by_config.items():
        if not results:
            print(f"| {config_name} | — | — | — | — |")
            continue

        mean_prec    = sum(r.precision_at_k for r in results) / len(results)
        mean_ndcg    = sum(r.ndcg_at_k      for r in results) / len(results)
        mean_mrr     = sum(r.mrr            for r in results) / len(results)
        mean_novelty = sum(r.novelty_at_k   for r in results) / len(results)

        print(
            f"| {config_name} "
            f"| {mean_prec:.3f} "
            f"| {mean_ndcg:.3f} "
            f"| {mean_mrr:.3f} "
            f"| {mean_novelty:.3f} |"
        )
