"""scripts/build_catalog.py

Entry point: FMA + Spotify + Genius data pipeline → catalog.parquet + lyrics.parquet.
scope.md §hourly timeline hours 1–6.
ADR-002: Batch pipeline; strict Pydantic validation at ingest.
ADR-006: Single config.yaml loaded at startup via Pydantic AppConfig.

Usage:
    python scripts/build_catalog.py --config config/config.yaml [--dry-run]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Configure root logger before any library imports
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FMA+Spotify+Genius catalog")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate pipeline setup without running API calls",
    )
    parser.add_argument(
        "--max-tracks",
        type=int,
        default=None,
        help="Limit tracks processed (for development testing)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ADR-006: Load config at entry point — hard failure on misconfiguration
    from src.data.schema import AppConfig
    import yaml
    with open(args.config) as f:
        config = AppConfig.model_validate(yaml.safe_load(f))

    logger.info("Config loaded from %s", args.config)
    logger.info("Pipeline checkpointing every %d tracks", config.pipeline.checkpoint_interval)

    # Steps (not yet implemented):
    # 1. Load FMA metadata → fma_df
    # 2. Fetch Spotify audio features → spotify_df
    # 3. Fuzzy join FMA → Spotify (two-pass validation)
    # 4. Write suspicious matches to review CSV
    # 5. Crawl Genius lyrics
    # 6. Clean lyrics (ftfy + ASCII filter)
    # 7. Validate all records as CatalogRecord (hard failure on schema violation)
    # 8. Filter non-music tracks (speechiness, duration)
    # 9. Deduplicate on (artist, title)
    # 10. Save catalog.parquet + lyrics.parquet

    raise NotImplementedError("build_catalog.py not yet implemented")


if __name__ == "__main__":
    main()
