"""scripts/generate_descriptors.py

Entry point: Mistral 7B descriptor generation → descriptors.parquet.
scope.md §hourly timeline hours 7–9.
ADR-001: Mistral 7B via the configured LLM backend (LM Studio / Ollama),
~2s/track, ~3hr for 5K tracks.
ADR-002: Checkpointing every N=100 tracks; resume from latest checkpoint.
failure_modes.md §4.3: Hard timeout; exponential backoff.
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

    import math
    from pathlib import Path

    import pandas as pd
    import yaml

    from src.data.schema import AppConfig
    from src.data.preprocess import validate_catalog_record
    from src.models.descriptor import PROMPT_VERSIONS, generate_descriptors_batch

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

    logger.info(
        "LLM backend: %s | model: %s | url: %s",
        config.enrichment.llm_backend, config.enrichment.llm_model, config.enrichment.llm_base_url,
    )
    logger.info("Checkpoint every %d tracks", config.pipeline.checkpoint_interval)
    logger.info("Timeout: %ss | Max retries: %d", config.enrichment.llm_timeout_s, config.enrichment.max_retries)

    processed_dir = Path(config.paths.processed_dir)
    checkpoints_dir = Path(config.paths.checkpoints_dir) / "descriptors"

    # --- Step 1: Load catalog.parquet ---
    catalog_path = processed_dir / "catalog.parquet"
    if not catalog_path.exists():
        logger.error("catalog.parquet not found at %s — run build_catalog.py first", catalog_path)
        raise FileNotFoundError(f"catalog.parquet not found at {catalog_path}")

    catalog_df = pd.read_parquet(catalog_path)
    logger.info("Loaded catalog: %d tracks", len(catalog_df))

    # --- Step 2: Load lyrics.parquet and merge on track_id ---
    lyrics_path = processed_dir / "lyrics.parquet"
    if lyrics_path.exists():
        lyrics_df = pd.read_parquet(lyrics_path)
        logger.info("Loaded lyrics: %d entries", len(lyrics_df))

        # Left join: keep all catalog tracks; add lyrics where available
        if "track_id" in lyrics_df.columns and "lyrics_text" in lyrics_df.columns:
            lyrics_subset = lyrics_df[["track_id", "lyrics_text"]].drop_duplicates("track_id")
            catalog_df = catalog_df.merge(lyrics_subset, on="track_id", how="left")
            logger.info(
                "Merged lyrics: %d/%d tracks now have lyrics_text",
                catalog_df["lyrics_text"].notna().sum(), len(catalog_df),
            )
        else:
            logger.warning(
                "lyrics.parquet missing expected columns (track_id, lyrics_text) — skipping merge"
            )
    else:
        logger.warning("lyrics.parquet not found at %s — proceeding without lyrics", lyrics_path)

    # --- Step 3: Validate and construct list[CatalogRecord] ---
    # Set lyrics_available based on whether lyrics_text is populated
    if "lyrics_text" in catalog_df.columns:
        catalog_df["lyrics_available"] = catalog_df["lyrics_text"].notna() & (catalog_df["lyrics_text"] != "")
    else:
        catalog_df["lyrics_available"] = False

    records = []
    n_invalid = 0
    for _, row in catalog_df.iterrows():
        raw = {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in row.to_dict().items()}
        try:
            record = validate_catalog_record(raw, config)
            records.append(record)
        except Exception as e:
            # Hard failure on schema violations (ADR-002) — but log and count to
            # give the operator a full picture rather than crashing on row 1 of 5K.
            logger.error("Schema validation failed for track %s: %s", raw.get("track_id"), e)
            n_invalid += 1

    if n_invalid > 0:
        raise RuntimeError(
            f"{n_invalid} catalog records failed schema validation. "
            f"Fix the catalog (run build_catalog.py) before generating descriptors."
        )

    logger.info("Validated %d catalog records", len(records))

    # --- Step 4: Apply --max-tracks limit ---
    if args.max_tracks is not None:
        records = records[: args.max_tracks]
        logger.info("--max-tracks %d: processing first %d records", args.max_tracks, len(records))

    # --- Step 5: Select prompt template ---
    prompt_version = args.prompt_version or config.enrichment.prompt_version
    if prompt_version in PROMPT_VERSIONS:
        prompt_template = PROMPT_VERSIONS[prompt_version]
        logger.info("Using prompt version '%s'", prompt_version)
    else:
        logger.warning(
            "Unknown prompt version '%s' — falling back to v1. Available: %s",
            prompt_version, list(PROMPT_VERSIONS.keys()),
        )
        prompt_template = PROMPT_VERSIONS["v1"]

    # --- Step 6: Generate descriptors with checkpointing/resume ---
    updated_records = generate_descriptors_batch(
        records=records,
        config=config,
        checkpoints_dir=checkpoints_dir,
        prompt_template=prompt_template,
    )

    # --- Step 7: Save descriptors.parquet ---
    rows = [
        {
            "track_id": r.track_id,
            "title": r.title,
            "artist": r.artist,
            "genre_top": r.genre_top,
            "descriptor": r.descriptor,
            "enrichment_tier": r.enrichment_tier,
            "lyrics_available": r.lyrics_available,
        }
        for r in updated_records
    ]
    output_df = pd.DataFrame(rows)
    output_path = processed_dir / "descriptors.parquet"
    output_df.to_parquet(output_path, index=False)

    n_full = (output_df["enrichment_tier"] == "full").sum()
    n_meta = (output_df["enrichment_tier"] == "metadata_only").sum()
    n_with_desc = output_df["descriptor"].notna().sum()
    coverage = n_with_desc / len(output_df) if len(output_df) > 0 else 0.0

    logger.info(
        "Saved %d records to %s — full: %d, metadata_only: %d, coverage: %.1f%%",
        len(output_df), output_path, n_full, n_meta, coverage * 100,
    )


if __name__ == "__main__":
    main()
