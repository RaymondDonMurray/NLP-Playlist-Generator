"""src/serving/explainer.py

Per-track explanation generation via Mistral 7B.
ADR-005: explain command is async/offline — not on the critical retrieval path.
scope.md §playlist output: one-line explanation of why each track matches query.
ADR-001: Same Mistral 7B model as descriptor generation via Ollama.
failure_modes.md §4.3: Hard timeout; exponential backoff patterns apply here too.

Import discipline (CLAUDE.md): serving/ imports from models/ and data/schema.py.
"""

from __future__ import annotations

import logging
import time
from time import perf_counter

import httpx

from src.data.schema import (
    AppConfig,
    CandidateTrack,
    ExplainRequest,
    ExplainResponse,
)

logger = logging.getLogger(__name__)

EXPLANATION_PROMPT_TEMPLATE = """\
A user searched for: "{query}"

Track: "{title}" by {artist}
Genre: {genre}
Descriptor: {descriptor}

In one sentence, explain why this track matches the user's search. Be specific about
which aspects of the track align with the query. Do not start with "This track".
"""


class Explainer:
    """Generates per-track natural language explanations via Mistral 7B.

    ADR-005: The explain command invokes Mistral 7B and is treated as an
    async/offline operation — not on the critical retrieval path.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def _build_explanation_prompt(
        self,
        query: str,
        track: CandidateTrack,
    ) -> str:
        """Build the per-track explanation prompt.

        Handles None descriptor and genre_top with safe fallbacks so the LLM
        never receives None values (failure_modes.md §3.1).
        """
        return EXPLANATION_PROMPT_TEMPLATE.format(
            query=query,
            title=track.title,
            artist=track.artist,
            genre=track.genre_top if track.genre_top is not None else "Unknown",
            descriptor=track.descriptor if track.descriptor is not None else "[no descriptor available]",
        )

    def explain_track(
        self,
        query: str,
        track: CandidateTrack,
        client: httpx.Client,
    ) -> str | None:
        """Generate a one-line explanation for why a track matches a query.

        Applies the same retry/timeout logic as descriptor generation
        (failure_modes.md §4.3): 30s hard timeout, exponential backoff on
        transient failures, no retry on schema errors.

        Returns None on failure (timeout exhausted, HTTP error). A missing
        explanation is a soft failure — does not break the response.
        """
        prompt = self._build_explanation_prompt(query, track)
        max_retries = self.config.enrichment.max_retries

        for attempt in range(max_retries):
            try:
                response = client.post(
                    f"{self.config.enrichment.ollama_base_url}/api/generate",
                    json={
                        "model": self.config.enrichment.ollama_model,
                        "prompt": prompt,
                        "stream": False,
                    },
                    timeout=self.config.enrichment.ollama_timeout_s,
                )
                response.raise_for_status()
                explanation = response.json()["response"].strip()
                return explanation if explanation else None

            except httpx.TimeoutException:
                logger.warning(
                    "Ollama timeout explaining track %s (attempt %d/%d)",
                    track.track_id, attempt + 1, max_retries,
                )
                time.sleep(2 ** attempt)

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 503:
                    logger.warning(
                        "Ollama unavailable while explaining track %s — waiting 10s",
                        track.track_id,
                    )
                    time.sleep(10)
                else:
                    logger.error(
                        "Unexpected HTTP error explaining track %s: %s",
                        track.track_id, e,
                    )
                    return None

        logger.error("Exhausted retries explaining track %s", track.track_id)
        return None

    def explain(self, request: ExplainRequest) -> ExplainResponse:
        """Generate explanations for all tracks in the request.

        Creates one httpx.Client reused across all tracks. Tracks with failed
        explanations have explanation=None — a missing explanation is a soft
        failure that does not raise.
        """
        start = perf_counter()
        tracks_with_explanations: list[CandidateTrack] = []

        with httpx.Client() as client:
            for track in request.tracks:
                explanation = self.explain_track(request.query, track, client)
                if explanation is None:
                    logger.warning(
                        "Failed to generate explanation for track %s", track.track_id,
                    )
                tracks_with_explanations.append(
                    track.model_copy(update={"explanation": explanation})
                )

        elapsed_ms = (perf_counter() - start) * 1000
        return ExplainResponse(
            query=request.query,
            tracks=tracks_with_explanations,
            explain_ms=elapsed_ms,
        )
