"""scripts/build_catalog.py

Entry point: FMA metadata + Essentia audio features + Genius lyrics → catalog.parquet
+ lyrics.parquet.
ADR-002 (amended 2026-06-12): Batch pipeline; audio features sourced from Essentia
(audio_features.parquet via extract_audio_features.py), replacing the EchoNest join.
Strict Pydantic validation at ingest. ADR-006: single config.yaml at startup.

Prereq: run scripts/extract_audio_features.py first to produce audio_features.parquet.

Usage:
    python scripts/build_catalog.py --config config/config.yaml
    python scripts/build_catalog.py --config config/config.yaml --max-tracks 100
    python scripts/build_catalog.py --config config/config.yaml --skip-lyrics

Outputs (written to config.paths.processed_dir):
    catalog.parquet  — track metadata + Essentia features (no lyrics text, no descriptors)
    lyrics.parquet   — track_id + cleaned lyrics_text (joined by track_id)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
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
    parser.add_argument(
        "--skip-lyrics",
        action="store_true",
        help="Skip Genius lyrics crawl (faster; lyrics.parquet will be empty)",
    )
    parser.add_argument(
        "--lyrics-sample",
        type=int,
        default=None,
        help="Crawl in waves until M tracks actually HAVE lyrics, spread across the "
             "listen-count range (most→least popular). Keeps trying regardless of the "
             "Genius hit rate until M succeed or the catalog is exhausted. Omit to "
             "crawl all vocal tracks once.",
    )
    parser.add_argument(
        "--skip-spotify",
        action="store_true",
        help="DEPRECATED no-op — Spotify ID annotation was removed in the Essentia pipeline "
             "(ADR-002 amendment); spotify_id is no longer populated.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load .env before any API clients are initialized (ADR-006: keys in .env only)
    from dotenv import load_dotenv
    load_dotenv()

    # ADR-006: Load config at entry point — hard failure on misconfiguration
    import yaml
    from src.data.schema import AppConfig

    with open(args.config) as f:
        config = AppConfig.model_validate(yaml.safe_load(f))

    processed_dir = Path(config.paths.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Config loaded from %s", args.config)
    logger.info("Output directory: %s", processed_dir)
    if args.max_tracks:
        logger.info("Development mode: capped at %d tracks", args.max_tracks)

    # -----------------------------------------------------------------------
    # Step 1: Load FMA metadata → fma_df
    # -----------------------------------------------------------------------
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    from src.data.ingest import FMAClient, GeniusClient, GeniusQuotaExceeded
    from src.data.preprocess import (
        clean_lyrics,
        dedup_catalog,
        essentia_features_to_record,
        filter_non_music_tracks,
    )

    logger.info("Step 1/5 — Loading FMA metadata")
    fma_client = FMAClient(config)
    fma_records = list(fma_client.load_tracks())

    # --max-tracks is applied later (Step 3, listens-aware) against tracks that
    # actually survived the audio join — not as an arbitrary head slice here.
    fma_df = pd.DataFrame(fma_records)
    logger.info("Loaded %d English FMA tracks", len(fma_df))

    if args.dry_run:
        logger.info("--dry-run: stopping after FMA load. First 3 rows:\n%s", fma_df.head(3).to_string())
        return

    # -----------------------------------------------------------------------
    # Step 2: Load Essentia audio features + join to FMA metadata
    # ADR-002 amendment (2026-06-12): replaces the EchoNest join. Features come
    # from scripts/extract_audio_features.py (audio_features.parquet), keyed by
    # integer FMA track_id, lifting the 13K-track coverage ceiling to all of FMA.
    # -----------------------------------------------------------------------
    logger.info("Step 2/5 — Loading Essentia audio features + joining to FMA metadata")
    audio_path = processed_dir / "audio_features.parquet"
    if not audio_path.exists():
        logger.error(
            "audio_features.parquet not found at %s — run scripts/extract_audio_features.py first",
            audio_path,
        )
        sys.exit(1)

    audio_df = pd.read_parquet(audio_path)
    # audio_features uses integer track_id; fma_df uses "fma_000002" strings
    fma_df["track_num"] = fma_df["track_id"].apply(lambda tid: int(tid[4:]))
    fma_meta_cols = fma_df[["track_num", "title", "artist", "genre_top", "duration_ms", "listens"]]
    merged_df = audio_df.merge(
        fma_meta_cols, left_on="track_id", right_on="track_num", how="inner",
        validate="many_to_one",  # each FMA track_num is unique; guards against dup metadata rows
    )
    logger.info(
        "Audio feature join: %d/%d extracted tracks matched FMA metadata",
        len(merged_df), len(audio_df),
    )
    if merged_df.empty:
        logger.error("No extracted tracks matched FMA metadata — check track_id alignment")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Step 3: Filter non-music + deduplicate, then build records via the adapter
    # failure_modes §1.3: speechiness > max, duration < 30s. NOTE: the speechiness
    # proxy is now AudioSet-derived and NOT EchoNest-calibrated — speechiness_max
    # needs re-tuning (ADR-002 amendment); the filter mechanism is unchanged.
    # CLAUDE.md: schema violations at ingest are hard failures (adapter raises).
    # -----------------------------------------------------------------------
    logger.info("Step 3/5 — Filtering non-music tracks and deduplicating")
    merged_df = filter_non_music_tracks(merged_df, config)
    merged_df = dedup_catalog(merged_df)

    # --max-tracks is listens-aware: keep the N most-played tracks that survived the
    # audio join + filtering, so a capped catalog is the popular core (easier eval
    # labeling) rather than an arbitrary slice of FMA load order.
    if args.max_tracks:
        merged_df = merged_df.sort_values("listens", ascending=False).head(args.max_tracks)
        logger.info("--max-tracks: kept top %d tracks by listens", len(merged_df))

    logger.info("Building %d CatalogRecords via the Essentia adapter", len(merged_df))
    validated_records = []
    for _, row in merged_df.iterrows():
        row_dict = row.to_dict()
        # the merged row carries both audio-feature and FMA-metadata keys
        record = essentia_features_to_record(row_dict, row_dict)  # raises on schema violation
        validated_records.append(record)

    catalog_df = pd.DataFrame([r.model_dump(exclude={"embedding", "lyrics_text"}) for r in validated_records])
    logger.info("Schema validation passed: %d clean catalog records", len(catalog_df))

    # -----------------------------------------------------------------------
    # Step 4: Crawl Genius lyrics (async) — skip instrumentals (HF flag)
    # ADR-002 amendment: instrumental tracks skip the crawl entirely (saves time;
    # hardens failure_modes §2.1 — no hallucinated themes for instrumentals).
    # GeniusClient handles the guarded multi-result fallback + 429 backoff.
    # -----------------------------------------------------------------------
    lyrics_rows: list[dict] = []
    lyrics_df = pd.DataFrame(columns=["track_id", "lyrics_text"])
    lyrics_path = processed_dir / "lyrics.parquet"

    if args.skip_lyrics:
        logger.info("Step 4/5 — Skipping lyrics crawl (--skip-lyrics)")
    else:
        vocal_records = [r for r in validated_records if not r.instrumental]
        n_instrumental = len(validated_records) - len(vocal_records)
        genius_client = GeniusClient(config)

        def _persist_lyrics(rows: list[dict]) -> None:
            """Atomically checkpoint accumulated lyrics to disk (deduped on track_id).

            Called after every crawl wave/chunk so an interrupted run (Ctrl-C, crash,
            or a rate-limit window we choose not to wait out) resumes from here instead
            of re-burning Genius quota from scratch — directly addressing the lost-lyrics
            problem when we're merely hitting the API limit.
            """
            if not rows:
                return
            df = pd.DataFrame(rows).drop_duplicates(subset="track_id", keep="first")
            tmp = lyrics_path.with_suffix(".parquet.tmp")
            df.to_parquet(tmp, index=False)
            os.replace(tmp, lyrics_path)

        # Resume: reuse any lyrics already fetched in a prior (possibly interrupted)
        # run so a restart never re-requests tracks we already have. This is what
        # keeps a stop/restart from perpetually reopening the same 429 window.
        done_ids: set[str] = set()
        if lyrics_path.exists():
            prior = pd.read_parquet(lyrics_path)
            lyrics_rows.extend(prior[["track_id", "lyrics_text"]].to_dict("records"))
            done_ids = set(prior["track_id"])
            logger.info(
                "Resuming lyrics crawl: %d tracks already have lyrics on disk (will be skipped)",
                len(done_ids),
            )
        vocal_records = [r for r in vocal_records if r.track_id not in done_ids]

        def _crawl_and_clean(records: list) -> list[dict]:
            """Crawl Genius for these records; return cleaned {track_id, lyrics_text} rows."""
            pairs = [(r.artist, r.title) for r in records]
            raw_map = asyncio.run(
                genius_client.batch_get_lyrics(pairs, concurrency=config.pipeline.genius_concurrency)
            )
            rows: list[dict] = []
            for rec in records:
                raw = raw_map.get(f"{rec.artist}|{rec.title}")
                if raw is None:
                    continue
                cleaned = clean_lyrics(raw, min_tokens=config.pipeline.min_lyrics_tokens)
                if cleaned is not None:
                    rows.append({"track_id": rec.track_id, "lyrics_text": cleaned})
            return rows

        if args.lyrics_sample is not None:
            # Target M tracks that actually HAVE lyrics, spread across the listen-count
            # range (most→least popular). Crawl in waves: each wave pulls a popularity-
            # spread of not-yet-tried candidates sized to the remaining shortfall, so we
            # keep going until M succeed (or the catalog is exhausted) regardless of the
            # Genius hit rate. Misses in one wave are backfilled by the next.
            target = args.lyrics_sample
            ranked = sorted(vocal_records, key=lambda r: r.listens, reverse=True)
            tried: set[str] = set()
            logger.info(
                "Step 4/5 — Crawling until %d tracks have lyrics, spread across %d vocal "
                "tracks (%d instrumentals skipped)", target, len(ranked), n_instrumental,
            )
            wave = 0
            while len(lyrics_rows) < target:
                pool = [r for r in ranked if r.track_id not in tried]
                if not pool:
                    logger.warning(
                        "Exhausted vocal catalog at %d/%d lyrics — Genius coverage too low "
                        "to reach the target", len(lyrics_rows), target,
                    )
                    break
                need = target - len(lyrics_rows)
                k = min(len(pool), need)
                picks = sorted({int(i) for i in np.linspace(0, len(pool) - 1, k).round()})
                batch = [pool[i] for i in picks]
                tried.update(r.track_id for r in batch)
                wave += 1
                try:
                    new_rows = _crawl_and_clean(batch)
                except GeniusQuotaExceeded as e:
                    logger.error(
                        "Stopping lyrics crawl — %s. %d lyrics checkpointed to %s; "
                        "rerun after the reset to resume where this left off.",
                        e, len(lyrics_rows), lyrics_path,
                    )
                    break
                lyrics_rows.extend(new_rows)
                _persist_lyrics(lyrics_rows)  # checkpoint each wave — interrupt-safe
                logger.info(
                    "  wave %d: crawled %d, +%d lyrics (%d/%d)",
                    wave, len(batch), len(new_rows), len(lyrics_rows), target,
                )
        else:
            logger.info(
                "Step 4/5 — Crawling Genius lyrics for %d tracks (%d instrumentals skipped, "
                "%d already done)", len(vocal_records), n_instrumental, len(done_ids),
            )
            # Chunk the full crawl so progress is checkpointed periodically rather than
            # only at the very end — a Ctrl-C now costs at most one chunk, not everything.
            chunk = config.pipeline.checkpoint_interval
            for i in range(0, len(vocal_records), chunk):
                batch = vocal_records[i : i + chunk]
                try:
                    new_rows = _crawl_and_clean(batch)
                except GeniusQuotaExceeded as e:
                    logger.error(
                        "Stopping lyrics crawl — %s. %d lyrics checkpointed to %s; "
                        "rerun after the reset to resume where this left off.",
                        e, len(lyrics_rows), lyrics_path,
                    )
                    break
                lyrics_rows.extend(new_rows)
                _persist_lyrics(lyrics_rows)  # checkpoint each chunk — interrupt-safe
                logger.info(
                    "  crawled %d/%d, +%d lyrics (%d total)",
                    min(i + chunk, len(vocal_records)), len(vocal_records),
                    len(new_rows), len(lyrics_rows),
                )

        if lyrics_rows:
            lyrics_df = pd.DataFrame(lyrics_rows).drop_duplicates(subset="track_id", keep="first")
        logger.info("Lyrics obtained: %d tracks", len(lyrics_rows))

    # Resolve lyrics_available from the authoritative lyrics.parquet on disk. On a
    # --skip-lyrics rerun this reuses prior coverage instead of zeroing the flag;
    # after a crawl it reflects the rows just written.
    if args.skip_lyrics and lyrics_path.exists():
        existing = pd.read_parquet(lyrics_path, columns=["track_id"])
        tracks_with_lyrics = set(existing["track_id"])
        logger.info("Reusing lyrics_available from existing %s (%d tracks)", lyrics_path, len(tracks_with_lyrics))
    else:
        tracks_with_lyrics = set(lyrics_df["track_id"]) if not lyrics_df.empty else set()
    catalog_df["lyrics_available"] = catalog_df["track_id"].isin(tracks_with_lyrics)

    # -----------------------------------------------------------------------
    # Write outputs
    # scope.md: Save catalog.parquet + lyrics.parquet before any enrichment
    # -----------------------------------------------------------------------
    catalog_path = processed_dir / "catalog.parquet"
    catalog_df.to_parquet(catalog_path, index=False)
    logger.info("Wrote %s (%d tracks)", catalog_path, len(catalog_df))

    if not args.skip_lyrics:
        lyrics_df.to_parquet(lyrics_path, index=False)
        logger.info("Wrote %s (%d tracks with lyrics)", lyrics_path, len(lyrics_df))

    logger.info(
        "Pipeline complete. Next step: python scripts/generate_descriptors.py --config %s",
        args.config,
    )


if __name__ == "__main__":
    main()
