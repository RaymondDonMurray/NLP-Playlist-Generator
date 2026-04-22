"""scripts/run_eval.py

Entry point: run the full ablation evaluation matrix.
scope.md §hourly timeline hours 16–18.
CLAUDE.md §evaluation: Runnable from single command; deterministic; logged to TensorBoard.
scope.md §CI/CD: --smoke-test runs only 10 queries for fast CI checks.

Usage:
    # Full eval
    python scripts/run_eval.py --config config/config.yaml

    # Smoke test (CI mode — 10 queries)
    python scripts/run_eval.py --config config/config.yaml --smoke-test

    # Single ablation config
    python scripts/run_eval.py --config config/config.yaml --config-name hybrid_rerank

    # Check regression against baseline
    python scripts/run_eval.py --config config/config.yaml --check-regression
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retrieval evaluation and ablation matrix")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run only smoke_test_n_queries queries (CI mode)",
    )
    parser.add_argument(
        "--config-name",
        default=None,
        help="Run only a specific ablation config (e.g. hybrid_rerank)",
    )
    parser.add_argument(
        "--check-regression",
        action="store_true",
        help="Fail with exit code 1 if NDCG@10 drops >threshold from baseline",
    )
    parser.add_argument(
        "--print-table",
        action="store_true",
        default=True,
        help="Print ablation table to stdout after eval",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import yaml
    from dotenv import load_dotenv
    load_dotenv()

    from src.data.schema import AppConfig
    with open(args.config) as f:
        config = AppConfig.model_validate(yaml.safe_load(f))

    logger.info("Eval queries: %s", config.eval.queries_path)
    logger.info("Results output: %s", config.eval.results_path)
    logger.info("Smoke test mode: %s", args.smoke_test)

    # Step 1: Load eval queries
    from src.eval.runner import load_eval_queries, check_regression
    queries = load_eval_queries(Path(config.eval.queries_path))
    logger.info("Loaded %d eval queries", len(queries))

    if args.smoke_test:
        queries = queries[: config.eval.smoke_test_n_queries]
        logger.info("Smoke test: using %d queries", len(queries))

    # Step 2: Build encoder once — model loading is expensive (ADR-001).
    # Retriever/Reranker/Ranker are built per-ablation to pick up config overrides.
    from src.models.encoder import Encoder
    encoder = Encoder(config)

    # Step 3: Load track popularity for Novelty@k
    # Normalized FMA play counts; fall back to uniform 0.5 if unavailable.
    track_popularity: dict[str, float] = {}
    catalog_path = Path(config.paths.processed_dir) / "catalog.parquet"
    if catalog_path.exists():
        try:
            import pandas as pd
            catalog = pd.read_parquet(catalog_path, columns=["track_id", "listens"])
            max_listens = catalog["listens"].max()
            if max_listens and max_listens > 0:
                track_popularity = dict(
                    zip(catalog["track_id"], catalog["listens"] / max_listens)
                )
                logger.info("Loaded popularity scores for %d tracks", len(track_popularity))
            else:
                logger.warning("All play counts are zero — using uniform popularity 0.5")
        except Exception as exc:
            logger.warning("Could not load popularity from catalog: %s — using 0.5", exc)
    else:
        logger.warning("catalog.parquet not found at '%s' — using uniform popularity 0.5", catalog_path)

    # Step 4: Run ablation matrix or a single named config
    from src.eval.ablations import (
        ABLATION_CONFIGS,
        apply_ablation_config,
        make_run_name,
        print_ablation_table,
        run_ablation_matrix,
    )
    from torch.utils.tensorboard import SummaryWriter
    from src.eval.runner import run_eval

    results_by_config: dict[str, list] = {}

    if args.config_name:
        # Single config mode
        matching = [a for a in ABLATION_CONFIGS if a.name == args.config_name]
        if not matching:
            logger.error(
                "Unknown --config-name '%s'. Valid names: %s",
                args.config_name,
                [a.name for a in ABLATION_CONFIGS],
            )
            sys.exit(1)

        ablation = matching[0]
        ablation_config = apply_ablation_config(config, ablation)
        run_name = make_run_name(ablation, config)
        log_dir = str(Path(config.tensorboard.log_dir) / run_name)
        writer = SummaryWriter(log_dir=log_dir)

        # Build components with ablation-specific config flags
        from src.models.reranker import Reranker
        from src.serving.retriever import Retriever
        from src.serving.ranker import Ranker

        abl_retriever = Retriever(ablation_config, encoder)
        abl_reranker  = Reranker(ablation_config)
        abl_ranker    = Ranker(ablation_config, abl_retriever, abl_reranker)

        try:
            results = run_eval(
                config=ablation_config,
                retriever=abl_retriever,
                ranker=abl_ranker,
                config_name=ablation.name,
                writer=writer,
                queries_override=queries,
                track_popularity=track_popularity,
            )
            results_by_config[ablation.name] = results
        finally:
            writer.close()
    else:
        # Full ablation matrix
        results_by_config = run_ablation_matrix(
            base_config=config,
            smoke_test=args.smoke_test,
            track_popularity=track_popularity,
        )

    # Steps 5–6 are handled inside run_eval / run_ablation_matrix
    # (CSV write + TensorBoard logging happen there)

    # Step 7: Print ablation table
    if args.print_table and results_by_config:
        print()
        print_ablation_table(results_by_config)

    # Step 8: Regression check
    if args.check_regression:
        all_results = [r for results in results_by_config.values() for r in results]
        passed = check_regression(
            results=all_results,
            baseline_path=Path(config.eval.baseline_metrics_path),
            threshold=config.eval.regression_threshold,
        )
        if not passed:
            sys.exit(1)


if __name__ == "__main__":
    main()
