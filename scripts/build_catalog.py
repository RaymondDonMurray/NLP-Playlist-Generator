"""scripts/build_catalog.py

Entry point: FMA + Spotify + Genius data pipeline → catalog.parquet + lyrics.parquet.
scope.md §hourly timeline hours 1–6.
ADR-002: Batch pipeline; strict Pydantic validation at ingest.
ADR-006: Single config.yaml loaded at startup via Pydantic AppConfig.

Usage:
    python scripts/build_catalog.py --config config/config.yaml
    python scripts/build_catalog.py --config config/config.yaml --max-tracks 100
    python scripts/build_catalog.py --config config/config.yaml --dry-run

Outputs (written to config.paths.processed_dir):
    catalog.parquet  — track metadata + audio features (no lyrics text, no descriptors)
    lyrics.parquet   — track_id + cleaned lyrics_text (joined by track_id)
    suspicious_matches.csv — SUSPICIOUS fuzzy matches for manual review (failure_modes §1.1)
"""

from __future__ import annotations

import argparse
import asyncio
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
    parser.add_argument(
        "--skip-lyrics",
        action="store_true",
        help="Skip Genius lyrics crawl (faster; lyrics.parquet will be empty)",
    )
    parser.add_argument(
        "--skip-spotify",
        action="store_true",
        help="Skip Spotify ID annotation (avoids rate limits; spotify_id not used in retrieval or eval)",
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
    import pandas as pd
    from tqdm import tqdm
    from src.data.ingest import FMAClient, SpotifyClient, GeniusClient
    from src.data.preprocess import (
        clean_lyrics,
        dedup_catalog,
        filter_non_music_tracks,
        validate_catalog_record,
        validate_audio_features,
        validate_match,
    )

    logger.info("Step 1/6 — Loading FMA metadata")
    fma_client = FMAClient(config)
    fma_records = list(fma_client.load_tracks())

    if args.max_tracks:
        fma_records = fma_records[: args.max_tracks]

    fma_df = pd.DataFrame(fma_records)
    logger.info("Loaded %d English FMA tracks", len(fma_df))

    if args.dry_run:
        logger.info("--dry-run: stopping after FMA load. First 3 rows:\n%s", fma_df.head(3).to_string())
        return

    # -----------------------------------------------------------------------
    # Step 2: Join FMA audio features from echonest.csv (local, no API call)
    #
    # The Spotify audio-features endpoint (/v1/audio-features) was restricted
    # for new apps in November 2024. FMA's echonest.csv provides identical
    # features (tempo, energy, valence, acousticness, danceability, speechiness)
    # for ~13K tracks, computed from the same EchoNest/Spotify technology.
    # -----------------------------------------------------------------------
    logger.info("Step 2/6 — Loading FMA audio features from echonest.csv (local)")
    features_df = fma_client.load_audio_features()

    # echonest.csv uses integer track_id; fma_df uses "fma_000002" strings
    fma_df["track_num"] = fma_df["track_id"].apply(lambda tid: int(tid[4:]))
    features_df.index = features_df.index.astype(int)

    # Validate audio features before joining (failure_modes §1.3: sentinel values)
    feature_cols = ["acousticness", "danceability", "energy", "speechiness", "tempo", "valence"]
    for col in feature_cols:
        if col in features_df.columns:
            features_df[col] = features_df[col].apply(
                lambda v, c=col: validate_audio_features({c: v}).get(c)
            )

    # Inner join: only keep tracks that have EchoNest audio features
    matched_df = (
        fma_df.set_index("track_num")
        .join(features_df, how="inner")
        .reset_index(drop=True)
    )
    # Restore track_id column (was preserved through the join)
    logger.info(
        "Audio feature join: %d/%d FMA tracks have EchoNest features",
        len(matched_df), len(fma_df),
    )

    if matched_df.empty:
        logger.error("No tracks have EchoNest audio features — check echonest.csv")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Step 3: Search Spotify for spotify_id (annotation only; no audio features)
    #
    # Spotify search still works fine. We use it purely to annotate tracks with
    # their spotify_id for reference. Two-pass fuzzy validation prevents saving
    # incorrect IDs (failure_modes §1.1).
    # -----------------------------------------------------------------------
    matched_df["spotify_id"] = None  # populated below if --skip-spotify not set

    if args.skip_spotify:
        logger.info("Step 3/6 — Skipping Spotify ID annotation (--skip-spotify)")
    else:
        logger.info("Step 3/6 — Searching Spotify for spotify_id (reference annotation)")
        spotify_client = SpotifyClient(config)
        spotify_id_map: dict[str, str] = {}  # track_id → spotify_id (HIGH quality only)
        spotify_suspicious: list[dict] = []

        for row in tqdm(matched_df.itertuples(), total=len(matched_df), desc="Spotify search", unit="track"):
            result = spotify_client.search_track(row.artist, row.title)  # type: ignore[attr-defined]
            if result is None:
                continue
            artists = result.get("artists") or []
            spotify_title = result.get("name", "")
            spotify_artist = artists[0].get("name", "") if artists else ""

            quality = validate_match(
                fma_title=row.title,  # type: ignore[attr-defined]
                fma_artist=row.artist,  # type: ignore[attr-defined]
                spotify_title=spotify_title,
                spotify_artist=spotify_artist,
                title_threshold=config.pipeline.fuzzy_match_threshold,
                artist_high_threshold=config.pipeline.artist_match_threshold,
                artist_suspicious_threshold=config.pipeline.suspicious_artist_threshold,
            )
            if quality == "high":
                spotify_id_map[row.track_id] = result["id"]  # type: ignore[attr-defined]
            elif quality == "suspicious":
                spotify_suspicious.append({
                    "track_id": row.track_id,  # type: ignore[attr-defined]
                    "fma_title": row.title, "fma_artist": row.artist,  # type: ignore[attr-defined]
                    "spotify_title": spotify_title, "spotify_artist": spotify_artist,
                    "spotify_id": result["id"],
                })

        matched_df["spotify_id"] = matched_df["track_id"].map(spotify_id_map)
        logger.info(
            "Spotify annotation: %d HIGH, %d SUSPICIOUS, %d no match (out of %d tracks)",
            len(spotify_id_map), len(spotify_suspicious),
            len(matched_df) - len(spotify_id_map) - len(spotify_suspicious), len(matched_df),
        )

        if spotify_suspicious:
            suspicious_path = processed_dir / "suspicious_matches.csv"
            pd.DataFrame(spotify_suspicious).to_csv(suspicious_path, index=False)
            logger.warning(
                "MANUAL REVIEW REQUIRED: %d suspicious Spotify ID matches written to %s",
                len(spotify_suspicious), suspicious_path,
            )

    # -----------------------------------------------------------------------
    # Step 4: Filter non-music tracks + deduplicate
    # failure_modes §1.3: speechiness > 0.5, duration < 30s
    # scope.md §preprocessing step 4
    # -----------------------------------------------------------------------
    logger.info("Step 4/6 — Filtering non-music tracks and deduplicating")
    matched_df = filter_non_music_tracks(matched_df, config)
    matched_df = dedup_catalog(matched_df)

    # -----------------------------------------------------------------------
    # Validate every record as CatalogRecord (hard failure on schema violation)
    # CLAUDE.md: Schema violations at ingest = hard failures — raise, don't continue
    # -----------------------------------------------------------------------
    logger.info("Validating %d records against CatalogRecord schema", len(matched_df))
    validated_records = []
    for _, row in matched_df.iterrows():
        record = validate_catalog_record(row.to_dict(), config)  # raises on schema violation
        validated_records.append(record)

    catalog_df = pd.DataFrame([r.model_dump(exclude={"embedding", "lyrics_text"}) for r in validated_records])
    logger.info("Schema validation passed: %d clean catalog records", len(catalog_df))

    # -----------------------------------------------------------------------
    # Step 6: Crawl Genius lyrics (async, concurrency=5)
    # scope.md: ~5hr crawl for 5K tracks; 60-70% coverage on indie tracks
    # -----------------------------------------------------------------------
    lyrics_rows: list[dict] = []

    if args.skip_lyrics:
        logger.info("Step 6/6 — Skipping lyrics crawl (--skip-lyrics)")
    else:
        logger.info(
            "Step 6/6 — Crawling Genius lyrics for %d tracks "
            "(concurrency=5; this will take a while for large catalogs)",
            len(validated_records),
        )
        genius_client = GeniusClient(config)
        track_pairs = [(r.artist, r.title) for r in validated_records]

        raw_lyrics_map = asyncio.run(
            genius_client.batch_get_lyrics(track_pairs, concurrency=5)
        )

        for record in tqdm(validated_records, desc="Cleaning lyrics", unit="track"):
            key = f"{record.artist}|{record.title}"
            raw = raw_lyrics_map.get(key)
            if raw is None:
                continue
            cleaned = clean_lyrics(raw, min_tokens=config.pipeline.min_lyrics_tokens)
            if cleaned is not None:
                lyrics_rows.append({"track_id": record.track_id, "lyrics_text": cleaned})

        lyrics_df = pd.DataFrame(lyrics_rows) if lyrics_rows else pd.DataFrame(columns=["track_id", "lyrics_text"])
        coverage = len(lyrics_rows) / len(validated_records) * 100 if validated_records else 0
        logger.info("Lyrics coverage: %d/%d tracks (%.1f%%)", len(lyrics_rows), len(validated_records), coverage)

        # Mark lyrics_available in catalog_df
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
        lyrics_path = processed_dir / "lyrics.parquet"
        lyrics_df.to_parquet(lyrics_path, index=False)
        logger.info("Wrote %s (%d tracks with lyrics)", lyrics_path, len(lyrics_df))

    logger.info(
        "Pipeline complete. Next step: python scripts/generate_descriptors.py --config %s",
        args.config,
    )


if __name__ == "__main__":
    main()
