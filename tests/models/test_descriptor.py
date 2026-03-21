"""tests/models/test_descriptor.py

Tests for src/models/descriptor.py — prompt building and output validation.
scope.md §CI/CD: Descriptor generation — test prompt formatting + Pydantic
output validation on 5 fixture tracks.
"""

from __future__ import annotations

import pytest

from src.data.schema import CatalogRecord, TrackDescriptor
from src.models.descriptor import DESCRIPTOR_PROMPT_V1, build_prompt, extract_json


class TestBuildPrompt:
    """Test prompt rendering for the 5 fixture tracks."""

    def test_prompt_contains_track_title(self, fixture_tracks):
        """Each prompt includes the track title."""
        for track in fixture_tracks:
            prompt = build_prompt(track)
            assert track.title in prompt

    def test_prompt_contains_artist(self, fixture_tracks):
        """Each prompt includes the artist name."""
        for track in fixture_tracks:
            prompt = build_prompt(track)
            assert track.artist in prompt

    def test_prompt_no_none_values(self, fixture_tracks):
        """No raw None appears in the rendered prompt (failure_modes.md §3.1)."""
        for track in fixture_tracks:
            prompt = build_prompt(track)
            # "None" should not appear as a literal value in the prompt
            assert ": None" not in prompt
            assert "= None" not in prompt

    def test_prompt_instrumental_no_lyrics_section(self, fixture_tracks):
        """Instrumental tracks (has_lyrics=False) don't include lyrics excerpt."""
        instrumental = fixture_tracks[1]   # fma_000002 — no lyrics
        assert instrumental.lyrics_available is False
        prompt = build_prompt(instrumental)
        assert "Lyrics excerpt" not in prompt

    def test_prompt_lyric_track_includes_excerpt(self, fixture_tracks):
        """Tracks with lyrics include the lyrics excerpt in the prompt."""
        lyric_track = fixture_tracks[0]   # fma_000001 — has lyrics
        assert lyric_track.lyrics_available is True
        prompt = build_prompt(lyric_track)
        assert "Lyrics excerpt" in prompt

    def test_prompt_unknown_features_for_null_audio(self):
        """Null audio features render as 'unknown X' in the prompt."""
        track = CatalogRecord(
            track_id="test_null",
            title="Test",
            artist="Artist",
            tempo=None,
            energy=None,
        )
        prompt = build_prompt(track)
        assert "unknown tempo" in prompt
        assert "unknown energy" in prompt


class TestExtractJson:
    """Test JSON extraction from Ollama response text."""

    def test_extracts_bare_json(self):
        """Plain JSON object is extracted directly."""
        response = '{"mood": "calm", "genre": "jazz", "confidence": "high"}'
        extracted = extract_json(response)
        assert '"mood"' in extracted
        assert '"calm"' in extracted

    def test_strips_markdown_code_fences(self):
        """JSON wrapped in ```json ... ``` is extracted."""
        response = '```json\n{"mood": "calm"}\n```'
        extracted = extract_json(response)
        assert "{" in extracted
        assert "```" not in extracted

    def test_returns_json_portion_only(self):
        """Only the JSON object is returned, not surrounding text."""
        response = "Here is the descriptor:\n{\"mood\": \"calm\"}\nDone."
        extracted = extract_json(response)
        assert extracted.strip().startswith("{")
        assert extracted.strip().endswith("}")
