"""scripts/generate_descriptors.py

Entry point: Mistral 7B descriptor generation → descriptors.parquet.
scope.md §hourly timeline hours 7–9.
ADR-001: mistral:7b-instruct-q4_K_M via Ollama (~2s/track, ~3hr for 5K tracks).
ADR-002: Checkpointing every N=100 tracks; resume from latest checkpoint.
failure_modes.md §4.3: Hard 30s timeout; exponential backoff.
failure_modes.md §3.3: Prompt hash in checkpoint filenames; assert at merge.

Usage:
    python scripts/generate_descriptors.py --config config/config.yaml [--resume]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Mistral 7B track descriptors")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint (default behavior; flag is explicit confirmation)",
    )
    parser.add_argument(
        "--max-tracks",
        type=int,
        default=None,
        help="Limit tracks processed (for development testing)",
    )
    parser.add_argument(
        "--prompt-version",
        default=None,
        help="Override prompt version from config (for prompt iteration testing)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from src.data.schema import AppConfig
    import yaml
    with open(args.config) as f:
        config = AppConfig.model_validate(yaml.safe_load(f))

    if args.prompt_version:
        # Allow prompt version override for development; remind operator to
        # bump config.yaml before production run (ADR-006 invariant 1)
        logger.warning(
            "Prompt version overridden to '%s' via CLI flag. "
            "Update config.yaml enrichment.prompt_version before full batch run.",
            args.prompt_version,
        )

    logger.info("Ollama model: %s", config.enrichment.ollama_model)
    logger.info("Checkpoint every %d tracks", config.pipeline.checkpoint_interval)
    logger.info("Timeout: %ss | Max retries: %d", config.enrichment.ollama_timeout_s, config.enrichment.max_retries)

    # Steps (not yet implemented):
    # 1. Load catalog.parquet
    # 2. Find latest checkpoint (if --resume)
    # 3. Collect already-processed track_ids from checkpoint
    # 4. For each unprocessed track:
    #    a. build_prompt(track)
    #    b. generate_descriptor_with_retry()
    #    c. Checkpoint every N tracks (failure_modes.md §3.2)
    # 5. Merge checkpoints with prompt hash validation (failure_modes.md §3.3)
    # 6. Save descriptors.parquet

    raise NotImplementedError("generate_descriptors.py not yet implemented")


if __name__ == "__main__":
    main()
