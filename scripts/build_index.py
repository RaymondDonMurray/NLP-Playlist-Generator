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
import random
import shutil
import sys
from pathlib import Path

import chromadb
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Collection names — must match src/serving/retriever.py
COLLECTION_FULL = "full_enrichment"
COLLECTION_META_ONLY = "metadata_only"

# ChromaDB add() hard limit per batch (empirically ~5000; use 4000 to be safe)
CHROMA_BATCH_SIZE = 4000


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


def _build_chroma_metadata(row: pd.Series) -> dict:
    """Build a ChromaDB-safe metadata dict from a DataFrame row.

    ChromaDB rejects None values in metadata; only include non-null fields.
    Mirrors CatalogRecord.chroma_metadata property logic (schema.py).
    """
    meta: dict = {
        "track_id":        str(row["track_id"]),
        "title":           str(row.get("title", "")),
        "artist":          str(row.get("artist", "")),
        "genre_top":       str(row.get("genre_top", "Unknown")),
        "enrichment_tier": str(row.get("enrichment_tier", "metadata_only")),
        "lyrics_available": bool(row.get("lyrics_available", False)),
    }
    optional_fields = ["spotify_id", "descriptor", "tempo", "energy", "valence", "acousticness"]
    for field in optional_fields:
        val = row.get(field)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            meta[field] = val
    return meta


def _populate_collection(
    collection: chromadb.Collection,
    df: pd.DataFrame,
    embedding_map: dict[str, list[float]],
) -> int:
    """Bulk-add records to a ChromaDB collection in safe-sized batches.

    scope.md §vector index: Single bulk load after all embeddings are ready.
    ChromaDB add() has a per-batch limit; split into CHROMA_BATCH_SIZE batches.

    Returns the number of records successfully added.
    """
    ids: list[str] = []
    embeddings: list[list[float]] = []
    documents: list[str] = []
    metadatas: list[dict] = []

    skipped = 0
    for _, row in df.iterrows():
        track_id = str(row["track_id"])
        emb = embedding_map.get(track_id)
        if emb is None:
            logger.warning("No embedding for track %s — skipping from index", track_id)
            skipped += 1
            continue

        descriptor = row.get("descriptor")
        if descriptor is None or (isinstance(descriptor, float) and np.isnan(descriptor)):
            descriptor = ""
        doc = str(descriptor)

        ids.append(track_id)
        embeddings.append(emb)
        documents.append(doc)
        metadatas.append(_build_chroma_metadata(row))

    if skipped:
        logger.warning("Skipped %d records with missing embeddings", skipped)

    n_added = 0
    for batch_start in range(0, len(ids), CHROMA_BATCH_SIZE):
        batch_end = batch_start + CHROMA_BATCH_SIZE
        collection.add(
            ids=ids[batch_start:batch_end],
            embeddings=embeddings[batch_start:batch_end],
            documents=documents[batch_start:batch_end],
            metadatas=metadatas[batch_start:batch_end],
        )
        n_added += len(ids[batch_start:batch_end])
        logger.info(
            "Added %d/%d records to '%s'", n_added, len(ids), collection.name
        )

    return n_added


def _dedup_collection(
    collection: chromadb.Collection,
    threshold: float,
) -> int:
    """Remove near-duplicate tracks from a ChromaDB collection.

    failure_modes.md §1.4: Query each track against its nearest neighbors.
    Remove pairs with cosine similarity > threshold, keeping the first seen.

    Returns the number of records removed.
    """
    result = collection.get(include=["embeddings"])  # type: ignore[arg-type]
    ids: list[str] = result["ids"]
    raw_embeddings = result.get("embeddings")
    embeddings: list[list[float]] = raw_embeddings if raw_embeddings is not None else []  # type: ignore[assignment]

    if not ids:
        return 0

    logger.info("Dedup: checking %d records in '%s'", len(ids), collection.name)

    # Query top-2 (self + nearest neighbor) for all records
    # ChromaDB with cosine space: distance = 1 - similarity
    dist_threshold = 1.0 - threshold

    seen: set[str] = set()
    to_remove: set[str] = set()

    # Batch the queries to avoid memory issues on large catalogs
    query_batch_size = 500
    for batch_start in range(0, len(ids), query_batch_size):
        batch_ids = ids[batch_start:batch_start + query_batch_size]
        batch_embs = embeddings[batch_start:batch_start + query_batch_size]

        query_results = collection.query(
            query_embeddings=batch_embs,
            n_results=min(3, len(ids)),
            include=["distances"],
        )

        raw_ids = query_results.get("ids") or []
        raw_dists = query_results.get("distances") or []

        for i, own_id in enumerate(batch_ids):
            if own_id in to_remove:
                continue
            seen.add(own_id)

            neighbor_ids: list[str] = raw_ids[i] if i < len(raw_ids) else []
            neighbor_dists: list[float] = raw_dists[i] if i < len(raw_dists) else []

            for nid, dist in zip(neighbor_ids, neighbor_dists):
                if nid == own_id:
                    continue
                if nid in to_remove or nid in seen:
                    continue
                if dist <= dist_threshold:
                    to_remove.add(nid)

    if to_remove:
        logger.info(
            "Dedup removing %d near-duplicates from '%s' (threshold=%.3f)",
            len(to_remove), collection.name, threshold,
        )
        collection.delete(ids=list(to_remove))

    return len(to_remove)


def _spot_check(
    collection: chromadb.Collection,
    n: int,
    embedding_map: dict[str, list[float]],
) -> None:
    """Validate collection by querying a random sample of tracks.

    failure_modes.md §4.2: Assert non-empty results for each spot-check query.
    Raises AssertionError if any query returns empty results.
    """
    result = collection.get(include=[])
    ids: list[str] = result["ids"]

    if not ids:
        logger.warning(
            "Collection '%s' is empty — skipping spot check", collection.name
        )
        return

    sample_size = min(n, len(ids))
    sample_ids = random.sample(ids, sample_size)

    for sid in sample_ids:
        emb = embedding_map.get(sid)
        if emb is None:
            logger.warning("Spot check: no embedding for %s — skipping", sid)
            continue
        query_result = collection.query(
            query_embeddings=[emb],
            n_results=5,
            include=["metadatas"],
        )
        assert len(query_result["ids"][0]) > 0, (
            f"Spot check failed: no results for track {sid} in '{collection.name}'"
        )

    logger.info(
        "Spot check passed for '%s' (%d/%d queries)",
        collection.name, sample_size, len(ids),
    )


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        from src.data.schema import AppConfig
        config = AppConfig.model_validate(yaml.safe_load(f))

    logger.info("Bi-encoder: %s", config.retrieval.bi_encoder)
    logger.info("ChromaDB path: %s", config.paths.chroma_dir)

    processed_dir = Path(config.paths.processed_dir)
    chroma_root = Path(config.paths.chroma_dir)
    staging_path = chroma_root / "staging"
    production_path = chroma_root / "production"
    backup_path = chroma_root / "production_backup"

    descriptors_path = processed_dir / "descriptors.parquet"
    catalog_path = processed_dir / "catalog.parquet"
    embeddings_path = processed_dir / "embeddings.parquet"

    # ---- Step 1: Load descriptors.parquet ----
    if not descriptors_path.exists():
        raise FileNotFoundError(
            f"descriptors.parquet not found at {descriptors_path}. "
            "Run 'python scripts/generate_descriptors.py' first."
        )
    desc_df = pd.read_parquet(descriptors_path)
    logger.info("Loaded descriptors: %d rows", len(desc_df))

    # Merge with catalog to enrich Chroma metadata with audio features.
    # Only merge columns not already present in desc_df to avoid conflicts.
    if catalog_path.exists():
        catalog_df = pd.read_parquet(catalog_path)
        merge_cols = ["track_id"] + [
            c for c in catalog_df.columns if c not in desc_df.columns
        ]
        desc_df = desc_df.merge(catalog_df[merge_cols], on="track_id", how="left")
        logger.info("Merged with catalog: %d columns total", len(desc_df.columns))
    else:
        logger.warning(
            "catalog.parquet not found at %s — proceeding with descriptor-only metadata",
            catalog_path,
        )

    # ---- Step 2: Split into full_enrichment and metadata_only ----
    has_descriptor = desc_df["descriptor"].notna() & (desc_df["descriptor"] != "")
    full_mask = (desc_df["enrichment_tier"] == "full") & has_descriptor
    full_df = desc_df[full_mask].reset_index(drop=True)

    # metadata_only: all tracks not in full_enrichment, but must have *some*
    # text to embed. Tracks with no descriptor and no metadata are dropped.
    meta_df = desc_df[~full_mask].reset_index(drop=True)

    logger.info(
        "Split: %d full_enrichment, %d metadata_only candidates",
        len(full_df), len(meta_df),
    )

    # ---- Step 3 & 4: Encode descriptors (or load from parquet) ----
    if args.skip_embedding:
        if not embeddings_path.exists():
            raise FileNotFoundError(
                f"--skip-embedding set but embeddings.parquet not found at {embeddings_path}"
            )
        emb_df = pd.read_parquet(embeddings_path)
        logger.info("Loaded pre-computed embeddings: %d rows", len(emb_df))
        embedding_map: dict[str, list[float]] = {
            str(row["track_id"]): list(row["embedding"])
            for _, row in emb_df.iterrows()
        }
    else:
        from src.models.encoder import Encoder
        encoder = Encoder(config)

        # Build the combined list of (track_id, text_to_embed).
        # For metadata_only tracks with no descriptor, synthesize minimal text
        # from title/artist/genre so they remain retrievable by keyword.
        all_df = pd.concat([full_df, meta_df], ignore_index=True)
        texts_to_embed: list[str] = []
        for _, row in all_df.iterrows():
            desc = row.get("descriptor")
            if desc is None or (isinstance(desc, float) and np.isnan(desc)) or str(desc).strip() == "":
                # Synthesise minimal embedding text from available metadata
                title = row.get("title", "")
                artist = row.get("artist", "")
                genre = row.get("genre_top", "Unknown")
                desc = f"{title} by {artist}. Genre: {genre}."
            texts_to_embed.append(str(desc))

        logger.info(
            "Encoding %d descriptors with '%s'",
            len(texts_to_embed), config.retrieval.bi_encoder,
        )
        embeddings: list[list[float]] = encoder.encode_descriptors(
            texts_to_embed, batch_size=64, show_progress=True
        )

        # Save to parquet for re-use on subsequent runs (--skip-embedding)
        emb_df = pd.DataFrame({
            "track_id": all_df["track_id"].tolist(),
            "embedding": embeddings,
        })
        embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        emb_df.to_parquet(embeddings_path, index=False)
        logger.info("Saved embeddings to %s", embeddings_path)

        embedding_map = {
            str(tid): emb
            for tid, emb in zip(all_df["track_id"].tolist(), embeddings)
        }

    # ---- Step 5: Build staging ChromaDB collections ----
    # failure_modes.md §4.2: Never write to production directly.
    if staging_path.exists():
        logger.info("Removing existing staging directory: %s", staging_path)
        shutil.rmtree(staging_path)
    staging_path.mkdir(parents=True, exist_ok=True)

    logger.info("Building staging index at %s", staging_path)
    staging_client = chromadb.PersistentClient(path=str(staging_path))

    # Collections created with cosine space (UQ-4 coupling: retriever assumes cosine)
    full_collection = staging_client.create_collection(
        name=COLLECTION_FULL,
        metadata={"hnsw:space": "cosine"},
    )
    meta_collection = staging_client.create_collection(
        name=COLLECTION_META_ONLY,
        metadata={"hnsw:space": "cosine"},
    )

    n_full_added = _populate_collection(full_collection, full_df, embedding_map)
    n_meta_added = _populate_collection(meta_collection, meta_df, embedding_map)
    logger.info(
        "Staging populated — full_enrichment: %d, metadata_only: %d",
        n_full_added, n_meta_added,
    )

    # ---- Step 6: Cosine dedup within each collection ----
    # failure_modes.md §1.4: Near-identical embeddings inflate Precision@10.
    dedup_threshold = config.pipeline.dedup_similarity_threshold
    n_removed_full = _dedup_collection(full_collection, threshold=dedup_threshold)
    n_removed_meta = _dedup_collection(meta_collection, threshold=dedup_threshold)
    logger.info(
        "Dedup complete — removed %d from full_enrichment, %d from metadata_only",
        n_removed_full, n_removed_meta,
    )

    # ---- Step 7: Spot-check validation ----
    # failure_modes.md §4.2: Assert non-empty results before promoting to production.
    _spot_check(full_collection, n=args.n_spot_checks, embedding_map=embedding_map)
    _spot_check(meta_collection, n=args.n_spot_checks, embedding_map=embedding_map)

    # Release ChromaDB handles before moving directories (avoids SQLite file locks)
    final_full_count = full_collection.count()
    final_meta_count = meta_collection.count()
    del full_collection, meta_collection, staging_client

    # ---- Step 8: Atomic promotion: staging → production ----
    # failure_modes.md §4.2: Backup existing production, then shutil.move staging.
    if production_path.exists():
        if backup_path.exists():
            shutil.rmtree(backup_path)
        logger.info("Backing up production → %s", backup_path)
        shutil.move(str(production_path), str(backup_path))

    logger.info("Promoting staging → production")
    shutil.move(str(staging_path), str(production_path))

    logger.info(
        "Index built successfully — full_enrichment: %d, metadata_only: %d (path=%s)",
        final_full_count, final_meta_count, production_path,
    )


if __name__ == "__main__":
    main()
