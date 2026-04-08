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

    from src.data.schema import AppConfig
    import yaml
    with open(args.config) as f:
        config = AppConfig.model_validate(yaml.safe_load(f))

    logger.info("Eval queries: %s", config.eval.queries_path)
    logger.info("Results output: %s", config.eval.results_path)
    logger.info("Smoke test mode: %s", args.smoke_test)

    # Steps (not yet implemented):
    # 1. Load eval queries from queries.json
    # 2. Initialize Encoder, Retriever, Reranker, Ranker
    # 3. Load track popularity scores (from catalog.parquet) for Novelty@k
    # 4. Run ablation matrix (or single config if --config-name)
    # 5. Write results to eval/results.csv
    # 6. Log to TensorBoard (ADR-004)
    # 7. Print ablation table (--print-table)
    # 8. Check regression if --check-regression; exit 1 if fails

    raise NotImplementedError("run_eval.py not yet implemented")


if __name__ == "__main__":
    main()
