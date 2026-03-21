"""src/models/descriptor.py

Mistral 7B descriptor generation via Ollama.
ADR-001: mistral:7b-instruct-q4_K_M chosen for descriptor quality over speed.
ADR-002: ~2s/track; batch with parquet checkpointing every N tracks.
failure_modes.md §4.3: Hard 30s timeout; exponential backoff; no retry on ValidationError.
failure_modes.md §2.1: Pydantic enforces null lyrical_themes for instrumentals.

Import discipline (CLAUDE.md): models/ imports from data/schema.py only.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import httpx
from pydantic import ValidationError

from src.data.schema import AppConfig, CatalogRecord, TrackDescriptor

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

LYRICS_SECTION_TEMPLATE = "- Lyrics excerpt (first ~200 tokens):\n{lyrics}"


def build_prompt(track: CatalogRecord, prompt_template: str = DESCRIPTOR_PROMPT_V1) -> str:
    """Render the descriptor prompt for a single track.

    Uses CatalogRecord.prompt_features to ensure no None values reach the LLM.
    failure_modes.md §3.1: prompt_features property converts nullable fields
    to "unknown X" strings.
    """
    raise NotImplementedError


def extract_json(response_text: str) -> str:
    """Extract the JSON object from Ollama response text.

    Handles cases where the model wraps the JSON in markdown code fences.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Descriptor generation with retry
# failure_modes.md §4.3
# ---------------------------------------------------------------------------

def generate_descriptor(
    track: CatalogRecord,
    client: httpx.Client,
    config: AppConfig,
    prompt_template: str = DESCRIPTOR_PROMPT_V1,
) -> TrackDescriptor | None:
    """Generate a descriptor for a single track via Ollama.

    Retry logic:
    - httpx.TimeoutException → exponential backoff (1s, 2s, 4s)
    - HTTP 503 → wait 10s, retry
    - ValidationError → return None immediately (prompt issue, not transient)
    - Other HTTP errors → return None immediately

    failure_modes.md §4.3: Hard 30s timeout on every request. A hang per
    track on 5K tracks at 30s = 40+ hours of blocked pipeline.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Batch generation with checkpointing
# ADR-002: Checkpoint every N=100 tracks; resume from latest checkpoint.
# ---------------------------------------------------------------------------

def generate_descriptors_batch(
    records: list[CatalogRecord],
    config: AppConfig,
    checkpoints_dir: Path,
    prompt_template: str = DESCRIPTOR_PROMPT_V1,
) -> list[CatalogRecord]:
    """Generate descriptors for all records with parquet checkpointing.

    Resume logic: loads latest checkpoint, collects processed track_ids,
    skips them. Final step merges and deduplicates all checkpoint files.

    ADR-002: Checkpoint every config.pipeline.checkpoint_interval tracks.
    failure_modes.md §3.3: Checkpoint filenames include prompt hash.
    """
    raise NotImplementedError
