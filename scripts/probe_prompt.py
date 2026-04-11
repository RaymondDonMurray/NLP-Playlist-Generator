"""scripts/probe_prompt.py

Lightweight prompt iteration tool. Runs N tracks through a prompt version and
prints a compliance report — no checkpointing, no output file, no side effects.

Use this to validate a new prompt before committing to a full batch run.

Usage:
    # Test 10 tracks with default mix (lyrics + no-lyrics) using v2 prompt
    python scripts/probe_prompt.py --prompt-version v2 --n 10

    # Test only tracks with lyrics
    python scripts/probe_prompt.py --prompt-version v2 --n 20 --with-lyrics-only

    # Compare two prompt versions on the same sample (fixed seed)
    python scripts/probe_prompt.py --prompt-version v1 --n 10 --seed 42
    python scripts/probe_prompt.py --prompt-version v2 --n 10 --seed 42

    # Show full descriptor output (verbose)
    python scripts/probe_prompt.py --prompt-version v2 --n 5 --verbose

Intended workflow:
    1. Edit prompt in descriptor.py
    2. probe_prompt.py  →  validate cheaply (seconds, not hours)
    3. Iterate on prompt until compliance looks good
    4. generate_descriptors.py  →  full batch run
    5. Audit descriptors.parquet directly for final coverage check
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=logging.WARNING,  # suppress library noise during probing
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe a descriptor prompt on N tracks and report compliance."
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--prompt-version",
        default="v2",
        help="Prompt version to test (default: v2). Available: v1, v2.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of tracks to run (default: 10).",
    )
    parser.add_argument(
        "--with-lyrics-only",
        action="store_true",
        help="Sample only tracks that have lyrics.",
    )
    parser.add_argument(
        "--no-lyrics-only",
        action="store_true",
        help="Sample only tracks without lyrics.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling (default: random).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the full rendered descriptor for each track.",
    )
    return parser.parse_args()


def _compliance(descriptor_str: str | None, has_lyrics: bool) -> dict[str, bool]:
    """Check which required fields are present in the rendered descriptor string."""
    if descriptor_str is None:
        return {f: False for f in ("mood", "tempo_feel", "genre", "lyrical_themes", "narrative")}
    s = descriptor_str
    return {
        "mood":          "Mood:" in s,
        "tempo_feel":    "Tempo feel:" in s,
        "genre":         "Genre:" in s,
        "lyrical_themes": "Lyrical themes:" in s if has_lyrics else True,  # not required for instrumentals
        "narrative":     len(s.split(". ")) >= 3,  # rough proxy for a narrative sentence
    }


def main() -> None:
    import httpx
    import pandas as pd
    import yaml

    from src.data.preprocess import validate_catalog_record
    from src.data.schema import AppConfig
    from src.models.descriptor import PROMPT_VERSIONS, generate_descriptor

    args = parse_args()

    if args.with_lyrics_only and args.no_lyrics_only:
        print("ERROR: --with-lyrics-only and --no-lyrics-only are mutually exclusive.")
        sys.exit(1)

    # --- Load config ---
    with open(args.config) as f:
        config = AppConfig.model_validate(yaml.safe_load(f))

    # --- Resolve prompt ---
    if args.prompt_version not in PROMPT_VERSIONS:
        print(f"ERROR: Unknown prompt version '{args.prompt_version}'. Available: {list(PROMPT_VERSIONS.keys())}")
        sys.exit(1)
    prompt_template = PROMPT_VERSIONS[args.prompt_version]

    # --- Load catalog + lyrics ---
    processed_dir = Path(config.paths.processed_dir)

    catalog_df = pd.read_parquet(processed_dir / "catalog.parquet")

    lyrics_path = processed_dir / "lyrics.parquet"
    if lyrics_path.exists():
        lyrics_df = pd.read_parquet(lyrics_path)[["track_id", "lyrics_text"]].drop_duplicates("track_id")
        catalog_df = catalog_df.merge(lyrics_df, on="track_id", how="left")
    else:
        catalog_df["lyrics_text"] = None

    catalog_df["lyrics_available"] = catalog_df["lyrics_text"].notna() & (catalog_df["lyrics_text"] != "")

    # --- Filter by lyrics flag ---
    if args.with_lyrics_only:
        catalog_df = catalog_df[catalog_df["lyrics_available"]]
        print(f"Filtered to {len(catalog_df)} tracks with lyrics.")
    elif args.no_lyrics_only:
        catalog_df = catalog_df[~catalog_df["lyrics_available"]]
        print(f"Filtered to {len(catalog_df)} tracks without lyrics.")

    if len(catalog_df) == 0:
        print("ERROR: No tracks available after filtering.")
        sys.exit(1)

    # --- Sample N tracks ---
    rng = random.Random(args.seed)
    n = min(args.n, len(catalog_df))
    sample_df = catalog_df.sample(n=n, random_state=args.seed if args.seed is not None else rng.randint(0, 2**31))

    # --- Build CatalogRecord list ---
    records = []
    for _, row in sample_df.iterrows():
        raw = {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in row.to_dict().items()}
        try:
            records.append(validate_catalog_record(raw, config))
        except Exception as e:
            print(f"  [SKIP] {raw.get('track_id')}: schema error — {e}")

    print(f"\n{'='*70}")
    print(f"  Probe: prompt={args.prompt_version}  n={len(records)}  seed={args.seed}")
    print(f"{'='*70}\n")

    # --- Run generation ---
    results: list[dict] = []
    with httpx.Client() as client:
        for i, record in enumerate(records, 1):
            print(f"[{i:>2}/{len(records)}] {record.track_id}  {record.artist} — {record.title}", end="  ")
            descriptor = generate_descriptor(record, client, config, prompt_template)

            if descriptor is None:
                print("FAILED")
                results.append({"record": record, "descriptor": None, "descriptor_str": None})
                continue

            descriptor_str = descriptor.to_descriptor_string()
            comp = _compliance(descriptor_str, record.lyrics_available)
            flags = "".join(
                ("✓" if ok else "✗") + k[0].upper()
                for k, ok in comp.items()
            )
            print(f"[{flags}]")

            if args.verbose:
                print(f"    {descriptor_str}\n")

            results.append({"record": record, "descriptor": descriptor, "descriptor_str": descriptor_str})

    # --- Compliance report ---
    total = len(results)
    succeeded = [r for r in results if r["descriptor"] is not None]
    failed = total - len(succeeded)

    print(f"\n{'='*70}")
    print(f"  Results: {len(succeeded)}/{total} succeeded  ({failed} failed)")
    print(f"{'='*70}")

    if not succeeded:
        print("No successful descriptors — cannot compute compliance.\n")
        return

    field_hits: dict[str, int] = {"mood": 0, "tempo_feel": 0, "genre": 0, "lyrical_themes": 0, "narrative": 0}
    lyric_tracks = [r for r in succeeded if r["record"].lyrics_available]

    for r in succeeded:
        comp = _compliance(r["descriptor_str"], r["record"].lyrics_available)
        for field, ok in comp.items():
            if ok:
                field_hits[field] += 1

    # lyrical_themes compliance denominator = only tracks with lyrics
    lyric_n = len(lyric_tracks)

    print(f"\n  Format compliance ({len(succeeded)} descriptors):\n")
    for field in ("mood", "tempo_feel", "genre"):
        pct = field_hits[field] / len(succeeded) * 100
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"  {field:<16} [{bar}] {field_hits[field]}/{len(succeeded)} ({pct:.0f}%)")

    if lyric_n > 0:
        lt_hits = sum(
            1 for r in lyric_tracks
            if "Lyrical themes:" in (r["descriptor_str"] or "")
        )
        pct = lt_hits / lyric_n * 100
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"  {'lyrical_themes':<16} [{bar}] {lt_hits}/{lyric_n} lyric tracks ({pct:.0f}%)")
    else:
        print(f"  lyrical_themes   [n/a — no lyric tracks in sample]")

    # Confidence distribution
    conf_counts: dict[str, int] = {"high": 0, "medium": 0, "low": 0}
    for r in succeeded:
        if r["descriptor"]:
            conf_counts[r["descriptor"].confidence] += 1
    print(f"\n  Confidence:  high={conf_counts['high']}  medium={conf_counts['medium']}  low={conf_counts['low']}")
    print()


if __name__ == "__main__":
    main()
