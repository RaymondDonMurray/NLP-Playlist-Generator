"""scripts/label_candidates.py

Surface candidate tracks to hand-label for a given eval query.

This is a LABELING AID, not part of the retrieval system. It is deliberately
decoupled from src/serving/retriever.py: it uses BM25 over descriptor text plus
an optional genre filter — no bge-base embeddings, no ChromaDB. Labeling with
the dense retriever would make the eval circular (testing the system against its
own output). Use this to find candidates to *judge*; you decide relevance.

scope.md §evaluation suite: ~12 author-labeled relevant tracks per query.

Usage:
    python scripts/label_candidates.py --query "chill rainy afternoon jazz with some melancholy"
    python scripts/label_candidates.py --query "modal jazz" --genre Jazz --top-k 40
    python scripts/label_candidates.py --query "warm lo-fi hip hop" --ids-only
    python scripts/label_candidates.py --query "modal jazz" --instrumental
"""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap
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
    parser = argparse.ArgumentParser(
        description="Surface candidate tracks to hand-label for an eval query (BM25, not the retriever)"
    )
    parser.add_argument("--query", required=True, help="Eval query text to find candidates for")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--genre",
        default=None,
        help="Optional genre_top filter (case-insensitive) applied before BM25 scoring",
    )
    parser.add_argument("--top-k", type=int, default=30, help="Number of candidates to surface")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Print the full descriptor for each candidate instead of a 160-char snippet",
    )
    parser.add_argument(
        "--instruments",
        nargs="+",
        default=None,
        metavar="INSTRUMENT",
        help="Filter to tracks tagged with these Essentia instruments (from catalog.parquet), "
        "e.g. --instruments guitar acousticguitar. Matches a track if it has ANY of them "
        "(use --instruments-all to require ALL).",
    )
    parser.add_argument(
        "--instruments-all",
        action="store_true",
        help="With --instruments, require a track to have ALL listed instruments, not any.",
    )
    parser.add_argument(
        "--instrumental",
        action="store_true",
        help="Filter to instrumental tracks only (no lyrics), using the `instrumental` flag "
        "from catalog.parquet.",
    )
    parser.add_argument(
        "--ids-only",
        action="store_true",
        help="Print only track_ids as a JSON array (ready to paste into queries.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import json

    import pandas as pd
    import yaml
    from rank_bm25 import BM25Okapi

    from src.data.schema import AppConfig

    with open(args.config) as f:
        config = AppConfig.model_validate(yaml.safe_load(f))

    descriptors_path = Path(config.paths.processed_dir) / "descriptors.parquet"
    if not descriptors_path.exists():
        logger.error(
            "descriptors.parquet not found at %s — run generate_descriptors.py first",
            descriptors_path,
        )
        raise FileNotFoundError(f"descriptors.parquet not found at {descriptors_path}")

    df = pd.read_parquet(descriptors_path)
    logger.info("Loaded %d tracks from %s", len(df), descriptors_path)

    # Only tracks with a descriptor can be scored by BM25 over descriptor text.
    df = df[df["descriptor"].notna() & (df["descriptor"] != "")].copy()

    # Optional genre pre-filter — narrows the candidate pool before scoring.
    if args.genre is not None:
        before = len(df)
        df = df[df["genre_top"].str.lower() == args.genre.lower()].copy()
        logger.info("Genre filter '%s': %d → %d tracks", args.genre, before, len(df))

    # Optional instrument pre-filter. The `instruments` column lives in
    # catalog.parquet (Essentia Instrument-Jamendo tags), not descriptors.parquet,
    # so we merge it in by track_id before filtering.
    if args.instruments is not None:
        catalog_path = Path(config.paths.processed_dir) / "catalog.parquet"
        if not catalog_path.exists():
            raise FileNotFoundError(
                f"catalog.parquet not found at {catalog_path} — needed for --instruments filter"
            )
        catalog_df = pd.read_parquet(catalog_path, columns=["track_id", "instruments"])
        df = df.merge(catalog_df, on="track_id", how="left")

        wanted = {i.lower() for i in args.instruments}

        def has_instruments(cell: object) -> bool:
            if cell is None or (hasattr(cell, "__len__") and len(cell) == 0):
                return False
            track_instruments = {str(x).lower() for x in cell}
            if args.instruments_all:
                return wanted.issubset(track_instruments)
            return bool(wanted & track_instruments)

        before = len(df)
        df = df[df["instruments"].apply(has_instruments)].copy()
        mode = "ALL" if args.instruments_all else "ANY"
        logger.info(
            "Instrument filter (%s of %s): %d → %d tracks",
            mode, sorted(wanted), before, len(df),
        )

    # Optional instrumental-only pre-filter. The `instrumental` flag lives in
    # catalog.parquet (HF FMA flag), not descriptors.parquet, so merge it in by
    # track_id before filtering.
    if args.instrumental:
        if "instrumental" not in df.columns:
            catalog_path = Path(config.paths.processed_dir) / "catalog.parquet"
            if not catalog_path.exists():
                raise FileNotFoundError(
                    f"catalog.parquet not found at {catalog_path} — needed for --instrumental filter"
                )
            catalog_df = pd.read_parquet(catalog_path, columns=["track_id", "instrumental"])
            df = df.merge(catalog_df, on="track_id", how="left")

        before = len(df)
        df = df[df["instrumental"].fillna(False).astype(bool)].copy()
        logger.info("Instrumental filter: %d → %d tracks", before, len(df))

    if df.empty:
        logger.warning("No candidate tracks after filtering — nothing to label.")
        return

    # BM25 over descriptor text. Whitespace tokenization mirrors the retriever's
    # BM25 (Retriever._build_bm25_indexes) so candidate surfacing is comparable.
    corpus = df["descriptor"].tolist()
    tokenized_corpus = [doc.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    scores = bm25.get_scores(args.query.lower().split())
    df = df.assign(_bm25=scores).sort_values("_bm25", ascending=False)

    top = df.head(args.top_k)

    if args.ids_only:
        print(json.dumps(top["track_id"].tolist(), indent=2))
        return

    print(f"\nQuery: {args.query!r}")
    if args.genre:
        print(f"Genre filter: {args.genre}")
    print(f"Top {len(top)} BM25 candidates (judge relevance yourself):\n")

    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        print(f"{rank:>3}. [{row['track_id']}] {row['title']} — {row['artist']}")
        print(f"     genre={row['genre_top']} | tier={row['enrichment_tier']} | bm25={row['_bm25']:.2f}")
        if args.instruments is not None and row.get("instruments") is not None:
            print(f"     instruments={', '.join(str(x) for x in row['instruments'])}")
        if args.full:
            body = textwrap.fill(str(row["descriptor"]), width=100, initial_indent="     ",
                                 subsequent_indent="     ")
            print(f"{body}\n")
        else:
            snippet = textwrap.shorten(str(row["descriptor"]), width=160, placeholder="…")
            print(f"     {snippet}\n")

    print("Copy the track_ids of genuinely relevant tracks into relevant_ids in eval/queries.json.")
    print("Re-run with --ids-only to get a paste-ready JSON array of all surfaced ids.")


if __name__ == "__main__":
    main()
