"""scripts/extract_audio_features.py

Entry point: stream FMA audio from HuggingFace, extract Essentia audio features,
write incremental checkpointed parquet → audio_features.parquet.
ADR-002 amendment (2026-06-12): replaces EchoNest coverage (13K) with extracted
features over the full FMA catalog (106K).

Usage:
    python scripts/extract_audio_features.py --config config/config.yaml
    python scripts/extract_audio_features.py --config config/config.yaml --max-tracks 50
    python scripts/extract_audio_features.py --config config/config.yaml --skip-download

Output (config.paths.processed_dir):
    audio_features.parquet — merged per-track features (track_id + schema)
Checkpoints (config.paths.checkpoints_dir/audio_features/):
    audio_{idx:04d}.parquet + manifest.json  (resume-safe, atomic)

Note: HuggingFace streaming has no random access — a resumed run still streams
past already-processed tracks (skipped cheaply via track_id). For the full 10K+
run on a VM, downloading shards is faster; streaming is the zero-setup default.
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
    parser = argparse.ArgumentParser(description="Extract Essentia audio features from FMA")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--max-tracks", type=int, default=None, help="Cap extracted tracks (dev)")
    parser.add_argument("--models-dir", default=None, help="Override config.audio.models_dir")
    parser.add_argument("--skip-download", action="store_true", help="Assume models already present")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing checkpoints")
    return parser.parse_args()


def _resolve_track_id(record: dict, url_map: dict[str, int]) -> int | None:
    """Recover the FMA integer track_id for a streamed record.

    Primary key: the `url` field → raw_tracks.csv map (exact, ~99%). Fallback:
    the zero-padded digits in the audio filename (FMA names files NNNNNN.mp3).
    """
    url = record.get("url")
    if url is not None and str(url) in url_map:
        return url_map[str(url)]

    audio = record.get("audio")
    path = audio.get("path") if isinstance(audio, dict) else None
    if path:
        stem = Path(str(path)).stem
        digits = "".join(ch for ch in stem if ch.isdigit())
        if digits:
            return int(digits)
    return None


def main() -> None:
    args = parse_args()

    import yaml
    from src.data.schema import AppConfig

    with open(args.config) as f:
        config = AppConfig.model_validate(yaml.safe_load(f))
    audio_cfg = config.audio

    import pandas as pd
    from datasets import Audio, load_dataset

    from src.data.audio_features import FeatureExtractors, ensure_models, load_audio_bytes
    from src.data.ingest import FMAClient
    from src.data.preprocess import load_checkpoint, write_checkpoint

    models_dir = Path(args.models_dir) if args.models_dir else Path(audio_cfg.models_dir)
    checkpoints_dir = Path(config.paths.checkpoints_dir) / "audio_features"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = checkpoints_dir / "manifest.json"
    processed_dir = Path(config.paths.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: models present ---
    if args.skip_download:
        logger.info("Skipping model download (--skip-download)")
    else:
        logger.info("Step 1/5 — Ensuring Essentia models in %s", models_dir)
        ensure_models(models_dir)

    # --- Step 2: url → track_id join map ---
    logger.info("Step 2/5 — Building url→track_id map from raw_tracks.csv")
    url_map = FMAClient(config).load_url_to_track_id()

    # --- Step 3: resume — collect already-processed track_ids ---
    processed_ids: set[int] = set()
    existing = sorted(checkpoints_dir.glob("audio_*.parquet"))
    if args.no_resume:
        logger.info("Step 3/5 — Resume disabled (--no-resume)")
    else:
        for ckpt in existing:
            df = load_checkpoint(ckpt, manifest_path)
            if df is not None and "track_id" in df.columns:
                processed_ids.update(int(t) for t in df["track_id"].tolist())
        logger.info(
            "Step 3/5 — Resume: %d tracks already processed across %d checkpoints",
            len(processed_ids), len(existing),
        )

    # --- Step 4: load models once + stream dataset ---
    logger.info("Step 4/5 — Loading models and opening dataset stream")
    extractor = FeatureExtractors(audio_cfg, models_dir=models_dir)

    ds = load_dataset(audio_cfg.hf_dataset, split="train", streaming=True)
    # decode=False → raw bytes; Essentia MonoLoader decodes+resamples (no torchcodec).
    ds = ds.cast_column("audio", Audio(decode=False))

    n_extracted = n_fail = n_unresolved = n_skipped = 0
    batch_rows: list[dict] = []
    batch_idx = len(existing)
    interval = audio_cfg.checkpoint_interval

    def flush() -> None:
        nonlocal batch_rows, batch_idx
        if not batch_rows:
            return
        path = checkpoints_dir / f"audio_{batch_idx:04d}.parquet"
        write_checkpoint(pd.DataFrame(batch_rows), path, manifest_path)
        logger.info("Checkpoint %04d written (%d tracks)", batch_idx, len(batch_rows))
        batch_rows = []
        batch_idx += 1

    for record in ds:
        track_id = _resolve_track_id(record, url_map)
        if track_id is None:
            n_unresolved += 1
            continue
        if track_id in processed_ids:
            n_skipped += 1
            continue

        try:
            data = record["audio"].get("bytes")
            if not data:
                logger.warning("Track %d has no audio bytes — skipping", track_id)
                n_fail += 1
                continue
            arr = load_audio_bytes(data, audio_cfg.sample_rate)
            if arr.size < audio_cfg.sample_rate:  # < 1s of audio — unusable
                logger.warning("Track %d too short (%d samples) — skipping", track_id, arr.size)
                n_fail += 1
                continue
            feats = extractor.extract_track(arr, track_id)
        except Exception as e:  # soft failure: log, count, continue (never abort the batch)
            logger.warning("Extraction failed for track %d: %s", track_id, e)
            n_fail += 1
            continue

        batch_rows.append(feats)
        processed_ids.add(track_id)
        n_extracted += 1

        if len(batch_rows) >= interval:
            flush()
        if args.max_tracks is not None and n_extracted >= args.max_tracks:
            logger.info("Reached --max-tracks=%d", args.max_tracks)
            break

    flush()

    logger.info(
        "Extraction done: %d extracted, %d failed, %d unresolved (no track_id), %d skipped (resume)",
        n_extracted, n_fail, n_unresolved, n_skipped,
    )

    # --- Step 5: merge checkpoints → audio_features.parquet ---
    logger.info("Step 5/5 — Merging checkpoints")
    all_ckpts = sorted(checkpoints_dir.glob("audio_*.parquet"))
    frames = [df for c in all_ckpts if (df := load_checkpoint(c, manifest_path)) is not None]
    if not frames:
        logger.warning("No checkpoints to merge — nothing written.")
        return
    merged = pd.concat(frames, ignore_index=True).drop_duplicates(subset="track_id")
    out_path = processed_dir / "audio_features.parquet"
    merged.to_parquet(out_path, index=False)
    logger.info("Wrote %s (%d tracks)", out_path, len(merged))


if __name__ == "__main__":
    main()
