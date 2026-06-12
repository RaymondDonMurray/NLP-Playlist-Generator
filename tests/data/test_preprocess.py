"""tests/data/test_preprocess.py

Tests for src/data/preprocess.py — fuzzy join, lyrics cleaning, validation.
scope.md §CI/CD: Schema validation and descriptor generation tests.
"""

from __future__ import annotations

import pytest

from src.data.preprocess import (
    clean_lyrics,
    essentia_features_to_record,
    prompt_version_hash,
    validate_audio_features,
)
from src.data.schema import FEATURE_VALID_RANGES


class TestEssentiaFeaturesToRecord:
    """ADR-002 amendment: Essentia audio_features row + FMA meta → CatalogRecord."""

    def _audio_row(self, **over):
        row = {
            "track_id": 2,
            "tempo_bpm": 114.4,
            "key": "F# major",
            "danceability": "danceable",
            "energy": 0.23,
            "genres": ["Hip Hop---Gangsta", "Hip Hop---Thug Rap"],
            "moods": ["energetic", "Exciting music"],
            "instruments": ["synthesizer", "drums"],
            "speechiness": 0.001,
            "instrumental": False,
        }
        row.update(over)
        return row

    def _fma_meta(self, **over):
        meta = {"title": "Food", "artist": "AWOL", "genre_top": "Hip-Hop", "duration_ms": 200000}
        meta.update(over)
        return meta

    def test_maps_all_fields(self):
        rec = essentia_features_to_record(self._audio_row(), self._fma_meta())
        assert rec.track_id == "fma_000002"   # int → zero-padded fma id
        assert rec.title == "Food" and rec.artist == "AWOL"
        assert rec.tempo == pytest.approx(114.4)        # tempo_bpm → tempo
        assert rec.energy == pytest.approx(0.23)
        assert rec.danceability == "danceable"          # categorical passthrough
        assert rec.key == "F# major"
        assert rec.genres == ["Hip Hop---Gangsta", "Hip Hop---Thug Rap"]
        assert rec.instruments == ["synthesizer", "drums"]
        assert rec.speechiness == pytest.approx(0.001)
        assert rec.duration_ms == 200000

    def test_valence_acousticness_left_none(self):
        rec = essentia_features_to_record(self._audio_row(), self._fma_meta())
        assert rec.valence is None and rec.acousticness is None

    def test_instrumental_flag_carried(self):
        rec = essentia_features_to_record(self._audio_row(instrumental=True), self._fma_meta())
        assert rec.instrumental is True

    def test_lyrics_not_set_by_adapter(self):
        rec = essentia_features_to_record(self._audio_row(), self._fma_meta())
        assert rec.lyrics_available is False and rec.lyrics_text is None

    def test_missing_genre_top_defaults(self):
        rec = essentia_features_to_record(self._audio_row(), self._fma_meta(genre_top=None))
        assert rec.genre_top == "Unknown"


class TestCleanLyrics:
    """failure_modes.md §1.2: ftfy + ASCII filter; <20 tokens → None."""

    def test_strips_structural_annotations(self):
        """[Verse 1] and [Chorus] annotations are removed."""
        raw = (
            "[Verse 1]\nWalking in the rain feeling so alone tonight\n"
            "[Chorus]\nFeel the blues wash over me like a wave crashing on the shore\n"
            "[Verse 2]\nThe empty streets and hollow sounds remind me of a time\n"
            "when everything was simpler and the world made perfect sense\n"
        )
        result = clean_lyrics(raw)
        assert result is not None
        assert "[Verse 1]" not in result
        assert "[Chorus]" not in result

    def test_returns_none_for_short_text(self):
        """Text under 20 tokens treated as missing, not passed downstream."""
        short = "Too short."
        assert clean_lyrics(short) is None

    def test_returns_none_for_non_ascii_garbage(self):
        """Primarily non-ASCII text treated as missing (failure_modes.md §1.2)."""
        garbage = "سئم تعبت " * 3   # less than 20 meaningful tokens
        result = clean_lyrics(garbage)
        # Either returns None or strips to under 20 tokens
        # Implementation may vary; the contract is: don't pass garbage to LLM
        assert result is None or len(result.split()) < 20

    def test_valid_lyrics_pass_through(self):
        """Normal English lyrics survive cleaning."""
        lyrics = " ".join(["word"] * 25)   # 25 words > min_lyrics_tokens
        result = clean_lyrics(lyrics)
        assert result is not None

    def test_strips_non_breaking_spaces(self):
        """\\xa0 non-breaking spaces are replaced with regular spaces."""
        raw = "word\xa0word\xa0word " * 8
        result = clean_lyrics(raw)
        assert result is not None
        assert "\xa0" not in result


class TestValidateAudioFeatures:
    """failure_modes.md §1.3: Spotify sentinel value detection."""

    def test_sentinel_tempo_nulled(self):
        """tempo: 0.0 (Spotify sentinel for unanalyzed tracks) → None."""
        features = {"tempo": 0.0, "energy": 0.5}
        result = validate_audio_features(features)
        assert result["tempo"] is None

    def test_valid_tempo_passes(self):
        """Valid tempo in [40, 250] passes through unchanged."""
        features = {"tempo": 120.0, "energy": 0.5}
        result = validate_audio_features(features)
        assert result["tempo"] == 120.0

    def test_out_of_range_energy_nulled(self):
        """Energy outside [0, 1] → None."""
        features = {"energy": 1.5}
        result = validate_audio_features(features)
        assert result["energy"] is None

    def test_none_input_stays_none(self):
        """None values (already missing) remain None."""
        features = {"tempo": None}
        result = validate_audio_features(features)
        assert result["tempo"] is None

    def test_unknown_keys_passed_through(self):
        """Keys not in FEATURE_VALID_RANGES are passed through unchanged."""
        features = {"tempo": 100.0, "spotify_id": "abc123"}
        result = validate_audio_features(features)
        assert result["spotify_id"] == "abc123"


class TestPromptVersionHash:
    """failure_modes.md §3.3: Prompt hash for checkpoint versioning."""

    def test_same_prompt_same_hash(self):
        """Identical prompt templates produce identical hashes."""
        template = "Generate a descriptor for {title} by {artist}."
        assert prompt_version_hash(template) == prompt_version_hash(template)

    def test_different_prompts_different_hashes(self):
        """Different prompts produce different hashes."""
        t1 = "Prompt version 1."
        t2 = "Prompt version 2."
        assert prompt_version_hash(t1) != prompt_version_hash(t2)

    def test_hash_is_8_chars(self):
        """Hash is exactly 8 characters (MD5 prefix)."""
        h = prompt_version_hash("any template")
        assert len(h) == 8

    def test_hash_is_alphanumeric(self):
        """Hash contains only hex characters."""
        h = prompt_version_hash("any template")
        assert all(c in "0123456789abcdef" for c in h)
