"""src/models/descriptor.py

Mistral 7B descriptor generation via the configured local LLM backend
(LM Studio / Ollama — see src/models/llm_client.py).
ADR-001: Mistral 7B chosen for descriptor quality over speed.
ADR-002: ~2s/track; batch with parquet checkpointing every N tracks.
failure_modes.md §4.3: Hard timeout; exponential backoff; no retry on ValidationError.
failure_modes.md §2.1: Pydantic enforces null lyrical_themes for instrumentals.

Output is constrained at the decoder via a JSON schema derived from
TrackDescriptor; extract_json + Pydantic validation remain the gatekeeper.

Import discipline (CLAUDE.md): models/ imports from data/schema.py only
(llm_client.py is same-layer).
"""

from __future__ import annotations

import logging
from pathlib import Path

import httpx
from pydantic import ValidationError

from src.data.schema import AppConfig, CatalogRecord, TrackDescriptor
from src.models.llm_client import LLMClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template
# scope.md §enrichment layer: structured prompting to produce canonical format.
# ADR-006 invariant 1: changing the template forces a new prompt_version tag.
# ---------------------------------------------------------------------------

DESCRIPTOR_PROMPT_V1 = """\
You are a music librarian generating structured track descriptors for a music retrieval system.

Given the following track metadata, generate a descriptor in valid JSON format.

Track metadata:
- Title: {title}
- Artist: {artist}
- Genre: {genre_top}
- Tempo: {tempo}
- Energy: {energy} (0=low, 1=high)
- Valence: {valence} (0=sad/negative, 1=happy/positive)
- Acousticness: {acousticness} (0=electronic, 1=acoustic)
- Danceability: {danceability} (0=low, 1=high)
- Has lyrics: {has_lyrics}
{lyrics_section}

Generate a JSON object with EXACTLY these fields:
{{
  "mood": "<2-4 adjectives describing the emotional character>",
  "tempo_feel": "<subjective tempo: very slow | slow | moderate | upbeat | fast | frenetic>",
  "genre": "<specific genre descriptor, more specific than the genre tag if possible>",
  "lyrical_themes": <"<themes as comma-separated phrases>" or null if has_lyrics is false>,
  "narrative": "<1-2 sentences describing the sonic character and feel>",
  "has_lyrics": <true or false>,
  "confidence": "<high | medium | low>"
}}

Rules:
- Set lyrical_themes to null if has_lyrics is false. Do NOT invent lyrical themes for instrumentals.
- Set confidence to "low" if you are uncertain about the genre or mood.
- The narrative should evoke the listening experience, not describe the metadata.
- Respond with ONLY the JSON object, no explanation.
"""

# ---------------------------------------------------------------------------
# V2: Tightened lyrical_themes enforcement.
# Root cause of V1 failure: lyrical_themes was treated as optional even for
# lyric-bearing tracks. V2 makes it rule #1 and explicitly marks it REQUIRED.
# ---------------------------------------------------------------------------

DESCRIPTOR_PROMPT_V2 = """\
You are a music librarian generating structured track descriptors for a music retrieval system.

Given the following track metadata, generate a descriptor in valid JSON format.

Track metadata:
- Title: {title}
- Artist: {artist}
- Genre: {genre_top}
- Tempo: {tempo}
- Energy: {energy} (0=low, 1=high)
- Valence: {valence} (0=sad/negative, 1=happy/positive)
- Acousticness: {acousticness} (0=electronic, 1=acoustic)
- Danceability: {danceability} (0=low, 1=high)
- Has lyrics: {has_lyrics}
{lyrics_section}

Generate a JSON object with EXACTLY these fields:
{{
  "mood": "<2-4 adjectives describing the emotional character>",
  "tempo_feel": "<subjective tempo: very slow | slow | moderate | upbeat | fast | frenetic>",
  "genre": "<specific genre descriptor, more specific than the genre tag if possible>",
  "lyrical_themes": "<comma-separated lyrical themes — REQUIRED non-null string when has_lyrics is true, null ONLY when has_lyrics is false>",
  "narrative": "<1-2 sentences evoking the sonic character and feel>",
  "has_lyrics": <true or false>,
  "confidence": "<high | medium | low>"
}}

RULES — all are mandatory:
1. lyrical_themes: When has_lyrics is true you MUST provide a non-null string summarising the lyrical content \
(e.g. "loss and longing, late-night isolation, nostalgia"). Use the lyrics excerpt above. \
Only set lyrical_themes to null when has_lyrics is false.
2. Do NOT invent lyrical themes for instrumental tracks (has_lyrics=false). Set lyrical_themes to null.
3. narrative: Write 1-2 evocative sentences about the listening experience. Do not restate the metadata.
4. confidence: Set to "low" when uncertain about mood, genre, or lyrical content.
5. Respond with ONLY the JSON object — no explanation, no markdown code fences.
"""

# ---------------------------------------------------------------------------
# V3: Essentia-feature schema (ADR-002 amendment 2026-06-12).
# Drops valence/acousticness (no Essentia equivalent); adds key, Essentia genre
# tags, moods, and instruments; danceability is categorical. The OUTPUT schema
# (TrackDescriptor) is unchanged, so the decoder json_schema is unaffected.
# ---------------------------------------------------------------------------

DESCRIPTOR_PROMPT_V3 = """\
You are a music librarian generating structured track descriptors for a music retrieval system.

Given the following track metadata, generate a descriptor in valid JSON format.

Track metadata:
- Title: {title}
- Artist: {artist}
- Genre: {genre_top}
- Detected genres: {genres}
- Key: {key}
- Tempo: {tempo}
- Energy: {energy} (0=low, 1=high)
- Danceability: {danceability}
- Detected moods: {moods}
- Detected instruments: {instruments}
- Has lyrics: {has_lyrics}
{lyrics_section}

Generate a JSON object with EXACTLY these fields:
{{
  "mood": "<2-4 adjectives describing the emotional character>",
  "tempo_feel": "<subjective tempo: very slow | slow | moderate | upbeat | fast | frenetic>",
  "genre": "<specific genre descriptor, more specific than the genre tag if possible>",
  "lyrical_themes": "<comma-separated lyrical themes — REQUIRED non-null string when has_lyrics is true, null ONLY when has_lyrics is false>",
  "narrative": "<1-2 sentences evoking the sonic character and feel>",
  "has_lyrics": <true or false>,
  "confidence": "<high | medium | low>"
}}

RULES — all are mandatory:
1. lyrical_themes: When has_lyrics is true you MUST provide a non-null string summarising the lyrical content. \
Only set lyrical_themes to null when has_lyrics is false.
2. Do NOT invent lyrical themes for instrumental tracks (has_lyrics=false). Set lyrical_themes to null.
3. Use the detected genres, moods, and instruments to ground mood, genre, and narrative — do not restate them verbatim.
4. confidence: Set to "low" when uncertain about mood, genre, or lyrical content.
5. Respond with ONLY the JSON object — no explanation, no markdown code fences.
"""

LYRICS_SECTION_TEMPLATE = "- Lyrics excerpt (first ~200 tokens):\n{lyrics}"

PROMPT_VERSIONS: dict[str, str] = {
    "v1": DESCRIPTOR_PROMPT_V1,   # legacy (EchoNest schema; valence/acousticness)
    "v2": DESCRIPTOR_PROMPT_V2,   # legacy (EchoNest schema; valence/acousticness)
    "v3": DESCRIPTOR_PROMPT_V3,   # current — Essentia schema
}


def build_prompt(track: CatalogRecord, prompt_template: str = DESCRIPTOR_PROMPT_V3) -> str:
    """Render the descriptor prompt for a single track.

    Uses CatalogRecord.prompt_features to ensure no None values reach the LLM.
    failure_modes.md §3.1: prompt_features converts nullable fields to "unknown X"
    strings; the string ": None" must never appear in output. Targets prompt v3
    (Essentia schema); legacy v1/v2 reference valence/acousticness placeholders no
    longer supplied by prompt_features and are retained for history only.
    """
    pf = track.prompt_features

    if track.lyrics_available and track.lyrics_text is not None:
        words = track.lyrics_text.split()
        excerpt = " ".join(words[:200])
        lyrics_section = LYRICS_SECTION_TEMPLATE.format(lyrics=excerpt)
    else:
        lyrics_section = ""

    return prompt_template.format(
        title=track.title,
        artist=track.artist,
        genre_top=track.genre_top,
        genres=pf["genres"],
        key=pf["key"],
        tempo=pf["tempo"],
        energy=pf["energy"],
        danceability=pf["danceability"],
        moods=pf["moods"],
        instruments=pf["instruments"],
        has_lyrics="true" if track.lyrics_available else "false",
        lyrics_section=lyrics_section,
    )


def extract_json(response_text: str) -> str:
    """Extract the JSON object from the LLM response text.

    Handles three cases:
    1. Bare JSON object.
    2. JSON wrapped in ```json ... ``` code fences.
    3. JSON with surrounding prose ("Here is the descriptor: {...} Done.").

    Uses rfind('}') not find('}') to avoid truncating nested objects.
    """
    import re

    text = response_text.strip()

    # Strip markdown code fences (```json ... ``` or ``` ... ```)
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = text.replace("```", "")

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end < start:
        return text  # return as-is; downstream ValidationError will surface the issue

    return text[start : end + 1]


# ---------------------------------------------------------------------------
# Descriptor generation with retry
# failure_modes.md §4.3
# ---------------------------------------------------------------------------

def generate_descriptor(
    track: CatalogRecord,
    client: httpx.Client,
    config: AppConfig,
    prompt_template: str = DESCRIPTOR_PROMPT_V3,
) -> TrackDescriptor | None:
    """Generate a descriptor for a single track via the configured LLM backend.

    Transport + retry (timeout backoff, 503 wait, fail-fast on dead process)
    live in LLMClient (failure_modes.md §4.3). Output is constrained by a JSON
    schema derived from TrackDescriptor; the response is still run through
    extract_json + Pydantic validation here.

    Returns None on:
    - transport failure (LLMClient.complete returns None), or
    - ValidationError — schema mismatch is a prompt/model issue, not transient,
      so it is NOT retried (failure_modes.md §4.3).
    """
    prompt = build_prompt(track, prompt_template)
    llm = LLMClient(config)

    raw_response = llm.complete(
        prompt,
        client,
        json_schema=TrackDescriptor.model_json_schema(),
        context=track.track_id,
    )
    if raw_response is None:
        return None

    try:
        extracted = extract_json(raw_response)
        return TrackDescriptor.model_validate_json(extracted)
    except ValidationError as e:
        # Schema failure: retrying produces the same bad output (failure_modes.md §4.3)
        logger.warning(
            "Bad descriptor schema for track %s (prompt issue, not retrying): %s",
            track.track_id, e,
        )
        return None


# ---------------------------------------------------------------------------
# Batch generation with checkpointing
# ADR-002: Checkpoint every N=100 tracks; resume from latest checkpoint.
# ---------------------------------------------------------------------------

def generate_descriptors_batch(
    records: list[CatalogRecord],
    config: AppConfig,
    checkpoints_dir: Path,
    prompt_template: str = DESCRIPTOR_PROMPT_V3,
) -> list[CatalogRecord]:
    """Generate descriptors for all records with parquet checkpointing and resume.

    Algorithm:
    1. Compute prompt_hash for checkpoint naming (failure_modes.md §3.3).
    2. Scan checkpoints_dir for existing checkpoint files; collect processed
       track_ids (resume logic — skips already-done tracks on restart).
    3. Iterate unprocessed records in batches of checkpoint_interval.
    4. Call generate_descriptor() per track; None results set enrichment_tier
       to metadata_only and are still recorded in the checkpoint.
    5. Checkpoint every N tracks (failure_modes.md §3.2: atomic manifest).
    6. Merge all checkpoints; update and return records.

    ADR-002: Checkpoint every config.pipeline.checkpoint_interval tracks.
    failure_modes.md §3.3: Checkpoint filenames include prompt hash; mismatched
    versions raise RuntimeError at merge time.
    """
    import pandas as pd
    from src.data.preprocess import (
        load_checkpoint,
        merge_checkpoints,
        prompt_version_hash,
        write_checkpoint,
    )

    # --- Preflight: verify the LLM backend is reachable before the batch ---
    LLMClient(config).health_check()

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = checkpoints_dir / "manifest.json"
    prompt_hash = prompt_version_hash(prompt_template)

    logger.info("Prompt hash: %s | Checkpoints dir: %s", prompt_hash, checkpoints_dir)

    # --- Resume: collect already-processed track IDs from valid checkpoints ---
    processed_ids: set[str] = set()
    existing_ckpts = sorted(checkpoints_dir.glob(f"desc_{prompt_hash}_*.parquet"))
    for ckpt_path in existing_ckpts:
        df = load_checkpoint(ckpt_path, manifest_path)
        if df is not None and "track_id" in df.columns:
            processed_ids.update(df["track_id"].tolist())

    logger.info(
        "Resume: %d tracks already processed from %d checkpoint files",
        len(processed_ids), len(existing_ckpts),
    )

    unprocessed = [r for r in records if r.track_id not in processed_ids]
    logger.info(
        "Generating descriptors for %d tracks (%d skipped as already processed)",
        len(unprocessed), len(records) - len(unprocessed),
    )

    # --- Batch generation ---
    n_success = 0
    n_fail = 0
    batch_rows: list[dict] = []
    batch_idx = len(existing_ckpts)  # continue checkpoint numbering after existing files
    interval = config.pipeline.checkpoint_interval

    with httpx.Client() as client:
        for i, record in enumerate(unprocessed):
            descriptor = generate_descriptor(record, client, config, prompt_template)

            if descriptor is not None:
                # confidence: low → metadata_only even if lyrics present (failure_modes.md §2.1)
                tier: str = "full" if (descriptor.confidence != "low" and record.lyrics_available) else "metadata_only"
                batch_rows.append({
                    "track_id": record.track_id,
                    "descriptor": descriptor.to_descriptor_string(),
                    "enrichment_tier": tier,
                })
                n_success += 1
            else:
                logger.warning("Descriptor failed for track %s — marking metadata_only", record.track_id)
                batch_rows.append({
                    "track_id": record.track_id,
                    "descriptor": None,
                    "enrichment_tier": "metadata_only",
                })
                n_fail += 1

            # Write checkpoint every N tracks
            if len(batch_rows) >= interval:
                ckpt_path = checkpoints_dir / f"desc_{prompt_hash}_{batch_idx:04d}.parquet"
                write_checkpoint(pd.DataFrame(batch_rows), ckpt_path, manifest_path)
                logger.info(
                    "Checkpoint %04d written (%d tracks; %d/%d total processed)",
                    batch_idx, len(batch_rows), i + 1, len(unprocessed),
                )
                batch_rows = []
                batch_idx += 1

        # Write any remaining tracks as final checkpoint
        if batch_rows:
            ckpt_path = checkpoints_dir / f"desc_{prompt_hash}_{batch_idx:04d}.parquet"
            write_checkpoint(pd.DataFrame(batch_rows), ckpt_path, manifest_path)
            logger.info("Final checkpoint %04d written (%d tracks)", batch_idx, len(batch_rows))

    # --- TensorBoard pipeline metrics (ADR-004) ---
    total_attempted = n_success + n_fail
    coverage_rate = n_success / total_attempted if total_attempted > 0 else 0.0
    retry_rate = n_fail / total_attempted if total_attempted > 0 else 0.0

    logger.info(
        "Descriptor generation done: %d/%d succeeded (coverage %.1f%%, failure %.1f%%)",
        n_success, total_attempted, coverage_rate * 100, retry_rate * 100,
    )

    try:
        import hashlib as _hashlib
        from datetime import datetime
        from torch.utils.tensorboard.writer import SummaryWriter

        ts = datetime.now().strftime("%Y%m%d")
        hash6 = _hashlib.md5(prompt_template.encode()).hexdigest()[:6]
        run_name = f"{ts}_{hash6}_descriptor_generation"
        writer = SummaryWriter(log_dir=f"runs/{run_name}")
        writer.add_scalar("pipeline/coverage_rate", coverage_rate, global_step=0)
        writer.add_scalar("pipeline/retry_rate", retry_rate, global_step=0)
        writer.add_text("pipeline/prompt_version", config.enrichment.prompt_version, global_step=0)
        writer.close()
        logger.info("TensorBoard metrics logged to runs/%s", run_name)
    except Exception as tb_err:
        logger.warning("TensorBoard logging failed (non-fatal): %s", tb_err)

    # --- Merge all checkpoints and update records ---
    try:
        merged_df = merge_checkpoints(checkpoints_dir, prompt_hash)
    except FileNotFoundError:
        logger.warning("No checkpoints found to merge — returning records unchanged.")
        return records

    descriptor_map: dict[str, dict] = {}
    for _, row in merged_df.iterrows():
        tid = row.get("track_id")
        if tid and pd.notna(tid):
            desc_val = row.get("descriptor")
            descriptor_map[str(tid)] = {
                "descriptor": str(desc_val) if desc_val is not None and pd.notna(desc_val) else None,
                "enrichment_tier": row.get("enrichment_tier", "metadata_only"),
            }

    updated: list[CatalogRecord] = []
    for record in records:
        info = descriptor_map.get(record.track_id)
        if info is not None:
            updated.append(record.model_copy(update={
                "descriptor": info["descriptor"],
                "enrichment_tier": info["enrichment_tier"],
            }))
        else:
            updated.append(record)

    return updated
