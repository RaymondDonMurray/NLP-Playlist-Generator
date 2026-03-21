"""scripts/build_index.py

Entry point: embed descriptors → bulk load → ChromaDB dual indexes.
scope.md §hourly timeline hours 10–12.
scope.md §vector index: Generate embeddings → save to parquet → single bulk load.
ADR-002: Dual ChromaDB indexes (full_enrichment, metadata_only).
failure_modes.md §4.2: Staging → validate → atomic promote pattern.

Usage:
    python scripts/build_index.py --config config/config.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ChromaDB dual indexes")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--skip-embedding",
        action="store_true",
        help="Load embeddings from embeddings.parquet instead of re-computing",
    )
    parser.add_argument(
        "--n-spot-checks",
        type=int,
        default=10,
        help="Number of spot-check queries for index validation (failure_modes.md §4.2)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from src.data.schema import AppConfig
    import yaml
    with open(args.config) as f:
        config = AppConfig.model_validate(yaml.safe_load(f))

    logger.info("Bi-encoder: %s", config.retrieval.bi_encoder)
    logger.info("ChromaDB path: %s", config.paths.chroma_dir)

    # Steps (not yet implemented):
    # 1. Load descriptors.parquet
    # 2. Split into full_enrichment (has lyrics) and metadata_only (no lyrics)
    # 3. Encode all descriptors with bge-base-en-v1.5 (Encoder.encode_descriptors)
    # 4. Save embeddings.parquet (avoid re-encoding on re-runs)
    # 5. Build staging index (failure_modes.md §4.2):
    #    a. full_enrichment collection
    #    b. metadata_only collection
    # 6. Cosine dedup within each collection (failure_modes.md §1.4)
    # 7. Spot-check validation
    # 8. Atomic promotion: staging → production

    raise NotImplementedError("build_index.py not yet implemented")


if __name__ == "__main__":
    main()
