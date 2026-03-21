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
        """Build the per-track explanation prompt."""
        raise NotImplementedError

    def explain_track(
        self,
        query: str,
        track: CandidateTrack,
        client: httpx.Client,
    ) -> str | None:
        """Generate a one-line explanation for why a track matches a query.

        Returns None on failure (timeout, schema error). Explanation is a
        soft-failure context — a missing explanation does not break the response.
        """
        raise NotImplementedError

    def explain(self, request: ExplainRequest) -> ExplainResponse:
        """Generate explanations for all tracks in the request.

        Tracks with failed explanations have explanation=None.
        """
        raise NotImplementedError
