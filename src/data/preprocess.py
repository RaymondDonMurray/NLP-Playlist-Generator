"""src/data/preprocess.py

Preprocessing pipeline: fuzzy join, lyrics cleaning, audio feature validation,
Pydantic schema validation, deduplication.

ADR-002: Mixed validation — strict on catalog schema, lenient on enrichment quality.
failure_modes.md: All defensive patterns for data quality issues live here.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Literal

import pandas as pd

from src.data.schema import (
    AppConfig,
    CatalogRecord,
    FEATURE_VALID_RANGES,
    FuzzyMatchResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fuzzy join: FMA → Spotify
# failure_modes.md §1.1: Two-pass artist+title validation
# scope.md §preprocessing step 1: rapidfuzz threshold ≥ 85
# ---------------------------------------------------------------------------

def validate_match(
    fma_title: str,
    fma_artist: str,
    spotify_title: str,
    spotify_artist: str,
    title_threshold: int,
    artist_high_threshold: int,
    artist_suspicious_threshold: int,
) -> Literal["high", "suspicious", "reject"]:
    """Two-pass fuzzy match quality classification.

    failure_modes.md §1.1: A single title-only threshold confidently matches
    "Blue (Da Ba Dee)" by Eiffel 65 to "Blue" by LeAnn Rimes.
    Two-pass validation uses both title AND artist similarity.

    Args:
        fma_title: Track title from FMA catalog.
        fma_artist: Artist name from FMA catalog.
        spotify_title: Track title from Spotify search result.
        spotify_artist: Artist name from Spotify search result.
        title_threshold: Minimum title similarity for any match (default 85).
        artist_high_threshold: Artist similarity for HIGH quality match (default 70).
        artist_suspicious_threshold: Artist similarity for SUSPICIOUS match (default 40).

    Returns:
        "high"       — title ≥ threshold AND artist ≥ high_threshold
        "suspicious" — title ≥ threshold AND artist ≥ suspicious_threshold
        "reject"     — title below threshold, or artist below suspicious_threshold
    """
    from rapidfuzz import fuzz

    title_sim = fuzz.token_sort_ratio(fma_title, spotify_title)
    artist_sim = fuzz.token_sort_ratio(fma_artist, spotify_artist)

    if title_sim >= title_threshold and artist_sim >= artist_high_threshold:
        return "high"
    elif title_sim >= title_threshold and artist_sim >= artist_suspicious_threshold:
        return "suspicious"
    else:
        return "reject"


def fuzzy_join_fma_spotify(
    fma_df: pd.DataFrame,
    spotify_df: pd.DataFrame,
    config: AppConfig,
    suspicious_csv_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join FMA tracks to Spotify search results via two-pass rapidFuzz validation.

    Args:
        fma_df: FMA tracks with at minimum columns: track_id, title, artist.
        spotify_df: Spotify search results indexed (or with a column) by track_id.
            Must contain: spotify_id, spotify_title, spotify_artist, plus any
            audio features (tempo, energy, valence, acousticness, danceability,
            loudness, speechiness, duration_ms).
        config: AppConfig with pipeline thresholds.
        suspicious_csv_path: If provided, write SUSPICIOUS matches here for manual
            review before running the 3-hour LLM descriptor batch.

    Returns:
        (matched_df, suspicious_df) where matched_df contains only HIGH-quality
        matches with FMA and Spotify columns merged. REJECT matches are silently
        dropped.

    failure_modes.md §1.1 operational note: Spot-check 50 HIGH and 50 SUSPICIOUS
    matches manually before running descriptor generation. A 3-hour LLM batch
    seeded from bad joins cannot be recovered cheaply.
    """
    title_threshold = config.pipeline.fuzzy_match_threshold
    artist_high_threshold = config.pipeline.artist_match_threshold
    artist_suspicious_threshold = config.pipeline.suspicious_artist_threshold

    # Index spotify_df by track_id for O(1) lookup
    if "track_id" in spotify_df.columns:
        spotify_indexed = spotify_df.set_index("track_id")
    else:
        spotify_indexed = spotify_df

    high_rows: list[dict] = []
    suspicious_rows: list[dict] = []
    n_reject = 0

    for _, fma_row in fma_df.iterrows():
        track_id = fma_row["track_id"]

        if track_id not in spotify_indexed.index:
            logger.debug("No Spotify result for FMA track %s — skipping", track_id)
            n_reject += 1
            continue

        spotify_row = spotify_indexed.loc[track_id]
        spotify_title = str(spotify_row.get("spotify_title", ""))
        spotify_artist = str(spotify_row.get("spotify_artist", ""))

        quality = validate_match(
            fma_title=str(fma_row["title"]),
            fma_artist=str(fma_row["artist"]),
            spotify_title=spotify_title,
            spotify_artist=spotify_artist,
            title_threshold=title_threshold,
            artist_high_threshold=artist_high_threshold,
            artist_suspicious_threshold=artist_suspicious_threshold,
        )

        merged_row = {**fma_row.to_dict(), **spotify_row.to_dict(), "match_quality": quality}

        if quality == "high":
            high_rows.append(merged_row)
        elif quality == "suspicious":
            suspicious_rows.append(merged_row)
        else:
            n_reject += 1

    matched_df = pd.DataFrame(high_rows) if high_rows else pd.DataFrame()
    suspicious_df = pd.DataFrame(suspicious_rows) if suspicious_rows else pd.DataFrame()

    logger.info(
        "Fuzzy join complete: %d HIGH, %d SUSPICIOUS, %d REJECT out of %d FMA tracks",
        len(high_rows), len(suspicious_rows), n_reject, len(fma_df),
    )

    if not suspicious_df.empty and suspicious_csv_path is not None:
        suspicious_df.to_csv(suspicious_csv_path, index=False)
        logger.warning(
            "Wrote %d SUSPICIOUS matches to %s — manual review required before "
            "running descriptor generation (failure_modes.md §1.1)",
            len(suspicious_df), suspicious_csv_path,
        )

    return matched_df, suspicious_df


# ---------------------------------------------------------------------------
# Lyrics cleaning
# failure_modes.md §1.2: ftfy + ASCII filter; treat <20 tokens as missing
# scope.md §preprocessing step 2
# ---------------------------------------------------------------------------

def clean_lyrics(raw: str, min_tokens: int = 20, max_tokens: int = 2000) -> str | None:
    """Clean raw Genius lyrics.

    Strips Genius page metadata artifacts, structural annotations ([Verse 1],
    [Chorus]), fixes Unicode encoding issues via ftfy, removes non-ASCII
    characters, and normalizes whitespace. Caps output at max_tokens words to
    reject scraped non-lyrics pages (e.g. wrong Genius search result).

    Returns None if the result is under min_tokens — treats garbage as missing
    rather than passing it downstream to the LLM.

    failure_modes.md §1.2: A descriptor from "سئم\\xa0تعبت\\xa0weariness"
    produces incoherent embeddings that silently poison the index.

    Args:
        raw: Raw lyrics string from Genius (may contain HTML artifacts).
        min_tokens: Minimum whitespace-split token count. Results below this
            are treated as missing (default 20).
        max_tokens: Hard cap on output token count. Results exceeding this
            after cleaning are almost certainly wrong Genius search results
            rather than song lyrics (default 2000; typical song is 200–800).
    """
    import ftfy
    import re

    text = ftfy.fix_text(raw)

    # Strip Genius page metadata that leaks into the lyrics container:
    # "7 ContributorsSong Title Lyrics" prefix appears on every scraped page.
    text = re.sub(r"^\d+\s+Contributors?\s*", "", text)       # "7 Contributors"
    text = re.sub(r"^[^\n]{0,200}?\bLyrics\b\s*", "", text)  # "Song Title Lyrics" heading

    text = re.sub(r"\[.*?\]", "", text)           # strip [Verse 1], [Chorus], etc.
    text = re.sub(r"\xa0", " ", text)              # non-breaking spaces
    text = re.sub(r"[^\x00-\x7F]+", " ", text)    # strip non-ASCII characters
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()

    if len(tokens) < min_tokens:
        return None   # treat as missing — don't pass garbage to the LLM

    if len(tokens) > max_tokens:
        return None   # almost certainly a wrong Genius result, not song lyrics

    return text


# ---------------------------------------------------------------------------
# Audio feature validation
# failure_modes.md §1.3: Spotify sentinel values (tempo: 0.0, loudness: 0.0)
# ---------------------------------------------------------------------------

def validate_audio_features(features: dict, valid_ranges: dict | None = None) -> dict:
    """Null out Spotify audio features that fall outside valid ranges.

    failure_modes.md §1.3: Spotify returns sentinel 0.0 for unanalyzed tracks.
    A tempo: 0.0 in a prompt produces a descriptor claiming "no discernible
    pulse" — accurate for the sentinel, wrong for the actual track.

    Args:
        features: Raw Spotify audio feature dict.
        valid_ranges: Override for FEATURE_VALID_RANGES. Defaults to module-level
            constant from schema.py.

    Returns:
        Cleaned dict with out-of-range or None values set to None for known keys.
        Unknown keys are passed through unchanged.
    """
    ranges = valid_ranges if valid_ranges is not None else FEATURE_VALID_RANGES
    clean: dict = {}

    for k, v in features.items():
        if k in ranges:
            lo, hi = ranges[k]
            if v is not None:
                try:
                    fv = float(v)
                    clean[k] = fv if lo <= fv <= hi else None
                except (TypeError, ValueError):
                    clean[k] = None
            else:
                clean[k] = None
        else:
            clean[k] = v

    return clean


def filter_non_music_tracks(df: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
    """Drop podcasts/comedy (speechiness > 0.5) and intros/outros (< 30s).

    failure_modes.md §1.3: Hard filters at ingest stage.
    Gracefully skips columns that are absent (e.g., when Spotify features haven't
    been merged yet).
    """
    initial = len(df)

    if "speechiness" in df.columns:
        mask = df["speechiness"].isna() | (df["speechiness"] <= config.pipeline.speechiness_max)
        n_dropped = int((~mask).sum())
        df = df[mask].copy()
        logger.info("Dropped %d tracks with speechiness > %.1f (podcasts/comedy)", n_dropped, config.pipeline.speechiness_max)

    if "duration_ms" in df.columns:
        mask = df["duration_ms"].isna() | (df["duration_ms"] >= config.pipeline.duration_ms_min)
        n_dropped = int((~mask).sum())
        df = df[mask].copy()
        logger.info("Dropped %d tracks shorter than %d ms (intros/outros)", n_dropped, config.pipeline.duration_ms_min)

    logger.info("filter_non_music_tracks: %d → %d tracks", initial, len(df))
    return df


# ---------------------------------------------------------------------------
# Schema validation
# CLAUDE.md: Schema violations at ingest = hard failures (raise immediately).
# ---------------------------------------------------------------------------

def validate_catalog_record(raw: dict, config: AppConfig) -> CatalogRecord:
    """Parse and validate a raw track dict into a CatalogRecord.

    Applies audio feature range validation (nulling sentinels) before schema
    parsing. Does NOT catch ValidationError — schema violations at ingest are
    hard failures that must propagate to stop the pipeline immediately.

    ADR-002: Strict validation — a Pydantic schema violation at ingest causes
    a hard stop, preventing a bad fuzzy join from polluting 3 hours of LLM
    generation. Do not catch ValidationError here; let it propagate.

    Args:
        raw: Raw merged dict with FMA + Spotify fields.
        config: AppConfig (unused directly; reserved for future threshold overrides).

    Returns:
        Validated CatalogRecord with sentinel audio features nulled.

    Raises:
        pydantic.ValidationError: If required fields are missing or have wrong types.
    """
    import math

    # pandas.DataFrame.to_dict() emits float('nan') for missing values;
    # Pydantic's Optional[str] fields reject nan. Normalize NaN → None first.
    normalized = {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in raw.items()}

    # Null out sentinel audio feature values before schema validation
    audio_fields = {k: normalized.get(k) for k in FEATURE_VALID_RANGES if k in normalized}
    clean_audio = validate_audio_features(audio_fields)

    merged = {**normalized, **clean_audio}
    # Hard failure — ValidationError intentionally not caught
    return CatalogRecord.model_validate(merged)


# ---------------------------------------------------------------------------
# Essentia → CatalogRecord adapter (ADR-002 amendment 2026-06-12)
# Bridges the extracted audio_features.parquet rows + FMA metadata into the
# canonical CatalogRecord, replacing the EchoNest join.
# ---------------------------------------------------------------------------

def essentia_features_to_record(audio_row: dict, fma_meta: dict) -> CatalogRecord:
    """Build a CatalogRecord from an Essentia audio-feature row + FMA metadata.

    Maps tempo_bpm→tempo, energy→energy, passes the categorical danceability and
    key/genres/moods/instruments/speechiness straight through, and carries the HF
    `instrumental` flag (which drives lyrics-skip + null lyrical_themes downstream).
    valence/acousticness have no Essentia equivalent and are left None.

    Lyrics are NOT set here (crawled separately); lyrics_available stays False.
    Schema violations propagate (ADR-002: hard failure at ingest).

    Args:
        audio_row: One row of audio_features.parquet — keys: track_id (int),
            tempo_bpm, key, danceability, energy, genres, moods, instruments,
            speechiness, instrumental.
        fma_meta: FMA metadata for the track — title, artist, genre_top, duration_ms,
            listens.

    Returns:
        Validated CatalogRecord.
    """
    def _f(v) -> float | None:
        return None if v is None else float(v)

    def _list(v) -> list[str]:
        return [str(x) for x in v] if v is not None else []

    track_id_int = int(audio_row["track_id"])
    return CatalogRecord(
        track_id=f"fma_{track_id_int:06d}",
        title=str(fma_meta["title"]),
        artist=str(fma_meta["artist"]),
        genre_top=str(fma_meta.get("genre_top") or "Unknown"),
        listens=int(fma_meta.get("listens") or 0),
        tempo=_f(audio_row.get("tempo_bpm")),
        energy=_f(audio_row.get("energy")),
        danceability=audio_row.get("danceability"),
        key=audio_row.get("key"),
        genres=_list(audio_row.get("genres")),
        moods=_list(audio_row.get("moods")),
        instruments=_list(audio_row.get("instruments")),
        speechiness=_f(audio_row.get("speechiness")),
        duration_ms=fma_meta.get("duration_ms"),
        instrumental=bool(audio_row.get("instrumental", False)),
    )


# ---------------------------------------------------------------------------
# Checkpoint utilities
# failure_modes.md §3.2 and §3.3
# ADR-002: Parquet checkpointing every N tracks with atomic manifest
# ---------------------------------------------------------------------------

def prompt_version_hash(prompt_template: str) -> str:
    """Return 8-char MD5 hash of the prompt template.

    failure_modes.md §3.3: Tag every checkpoint filename with this hash.
    Assert all checkpoints share the same hash at merge time.
    """
    return hashlib.md5(prompt_template.encode()).hexdigest()[:8]


def write_checkpoint(df: pd.DataFrame, path: Path, manifest_path: Path) -> None:
    """Write checkpoint parquet with atomic manifest update.

    failure_modes.md §3.2: Atomic write via temp file + os.replace() is safe
    on macOS and Linux. Validates row count on read to detect partial writes.

    The manifest is a JSON file mapping checkpoint filenames to their expected
    row count and column list. Written atomically (temp → replace) so a
    mid-write crash never leaves a half-written manifest.
    """
    df.to_parquet(path, index=False)

    manifest: dict = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())

    manifest[path.name] = {"row_count": len(df), "columns": list(df.columns)}

    # Atomic write: temp file then os.replace (safe on macOS + Linux)
    tmp = manifest_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    os.replace(str(tmp), str(manifest_path))

    logger.debug("Wrote checkpoint %s (%d rows)", path.name, len(df))


def load_checkpoint(path: Path, manifest_path: Path) -> pd.DataFrame | None:
    """Load checkpoint parquet, validating against manifest row count.

    Returns None if the checkpoint is absent from the manifest, the file is
    missing, or the row count does not match the manifest entry.

    failure_modes.md §3.2: Silent row drops from parquet corruption cause
    incorrect resume logic — always validate against the manifest.
    """
    if not manifest_path.exists():
        logger.error("Checkpoint manifest not found at %s", manifest_path)
        return None

    manifest = json.loads(manifest_path.read_text())
    expected = manifest.get(path.name)

    if not expected:
        logger.error("No manifest entry for checkpoint %s", path.name)
        return None

    if not path.exists():
        logger.error("Checkpoint file not found: %s", path)
        return None

    try:
        df = pd.read_parquet(path)
    except Exception as e:
        logger.error("Failed to read checkpoint %s: %s", path.name, e)
        return None

    if len(df) != expected["row_count"]:
        logger.error(
            "Checkpoint %s corrupted: expected %d rows, got %d",
            path.name, expected["row_count"], len(df),
        )
        return None

    return df


def merge_checkpoints(checkpoints_dir: Path, prompt_hash: str) -> pd.DataFrame:
    """Merge all checkpoint parquets matching the given prompt hash.

    failure_modes.md §3.3: Raises RuntimeError if checkpoints from a different
    prompt version are present — prevents merging incompatible schemas that
    would produce DataFrames with 40% NaN in newly-added columns.

    Args:
        checkpoints_dir: Directory containing desc_{hash}_{batch}.parquet files.
        prompt_hash: 8-char MD5 hash from prompt_version_hash(). Only files
            matching desc_{prompt_hash}_*.parquet are merged.

    Returns:
        Merged DataFrame deduplicated on track_id.

    Raises:
        RuntimeError: If checkpoints from a different prompt version exist.
        FileNotFoundError: If no matching checkpoints are found.
    """
    files = sorted(checkpoints_dir.glob(f"desc_{prompt_hash}_*.parquet"))

    # Fail fast if mixed prompt versions are present
    wrong_version = [
        f for f in checkpoints_dir.glob("desc_*.parquet")
        if prompt_hash not in f.name
    ]
    if wrong_version:
        raise RuntimeError(
            f"Found {len(wrong_version)} checkpoints from a different prompt version. "
            f"Delete or archive them before merging: {[f.name for f in wrong_version]}"
        )

    if not files:
        raise FileNotFoundError(
            f"No checkpoints found in {checkpoints_dir} for prompt hash '{prompt_hash}'. "
            f"Expected pattern: desc_{prompt_hash}_*.parquet"
        )

    dfs = [pd.read_parquet(f) for f in files]
    merged = pd.concat(dfs, ignore_index=True)
    before = len(merged)
    merged = merged.drop_duplicates(subset="track_id")

    logger.info(
        "Merged %d checkpoint files: %d total rows → %d after track_id dedup",
        len(files), before, len(merged),
    )
    return merged


# ---------------------------------------------------------------------------
# Deduplication
# failure_modes.md §1.4: Post-join dedup on (artist, track_name)
# scope.md §preprocessing step 4
# ---------------------------------------------------------------------------

def dedup_catalog(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate on (artist, title) post-matching.

    scope.md §preprocessing step 4: Keeps first occurrence of each
    (artist, title) pair. FMA can have the same recording under multiple
    track IDs (live vs. studio, remaster vs. original).
    """
    before = len(df)
    df = df.drop_duplicates(subset=["artist", "title"]).copy()
    logger.info(
        "dedup_catalog: %d → %d tracks (removed %d duplicates on artist+title)",
        before, len(df), before - len(df),
    )
    return df
