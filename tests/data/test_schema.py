"""tests/data/test_schema.py

Tests for src/data/schema.py — Pydantic models and data contracts.
scope.md §CI/CD: Schema validation — assert enriched catalog parquet matches
expected columns/dtypes.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.data.schema import (
    AppConfig,
    CatalogRecord,
    EvalQuery,
    RetrieveRequest,
    TrackDescriptor,
)


class TestCatalogRecord:

    def test_valid_full_enrichment_record(self, fixture_tracks):
        """A fully enriched record (with lyrics) passes validation."""
        record = fixture_tracks[0]
        assert record.track_id == "fma_000001"
        assert record.enrichment_tier == "full"
        assert record.lyrics_available is True

    def test_metadata_only_record_no_lyrics(self, fixture_tracks):
        """A metadata-only record (no lyrics) is valid."""
        record = fixture_tracks[1]
        assert record.enrichment_tier == "metadata_only"
        assert record.lyrics_available is False

    def test_prompt_features_never_returns_none(self, fixture_tracks):
        """prompt_features must not contain None values (failure_modes.md §3.1)."""
        for track in fixture_tracks:
            features = track.prompt_features
            for key, val in features.items():
                assert val is not None, f"prompt_features[{key}] is None for {track.track_id}"
                assert isinstance(val, str), f"prompt_features[{key}] is not a string"

    def test_prompt_features_unknown_when_null(self):
        """Null audio features become 'unknown X' strings in prompt_features."""
        record = CatalogRecord(track_id="t1", title="T", artist="A", tempo=None, energy=None)
        features = record.prompt_features
        assert "unknown" in features["tempo"].lower()
        assert "unknown" in features["energy"].lower()

    def test_chroma_metadata_excludes_none(self, fixture_tracks):
        """chroma_metadata must not contain None values (ChromaDB rejects them)."""
        for track in fixture_tracks:
            meta = track.chroma_metadata
            for key, val in meta.items():
                assert val is not None, f"chroma_metadata[{key}] is None for {track.track_id}"

    def test_missing_required_field_raises(self):
        """Missing required fields (track_id, title, artist) raise ValidationError."""
        with pytest.raises(ValidationError):
            CatalogRecord(title="T", artist="A")   # missing track_id


class TestTrackDescriptor:

    def test_valid_descriptor_with_lyrics(self):
        """Valid descriptor for a track with lyrics passes validation."""
        desc = TrackDescriptor(
            mood="melancholy, introspective",
            tempo_feel="slow",
            genre="piano jazz",
            lyrical_themes="late-night loneliness, rain",
            narrative="Sparse piano; feels like rain at 2am.",
            has_lyrics=True,
            confidence="high",
        )
        assert desc.lyrical_themes is not None

    def test_valid_instrumental_descriptor(self):
        """Instrumental descriptor with null lyrical_themes passes."""
        desc = TrackDescriptor(
            mood="calm, meditative",
            tempo_feel="moderate",
            genre="ambient",
            lyrical_themes=None,
            narrative="Layered pads creating a contemplative atmosphere.",
            has_lyrics=False,
            confidence="high",
        )
        assert desc.lyrical_themes is None

    def test_hallucinated_lyrical_themes_rejected(self):
        """failure_modes.md §2.1: lyrical_themes must be None when has_lyrics=False."""
        with pytest.raises(ValidationError, match="lyrical_themes"):
            TrackDescriptor(
                mood="calm",
                tempo_feel="slow",
                genre="ambient",
                lyrical_themes="themes of heartbreak",   # hallucinated — must fail
                narrative="Quiet instrumental piece.",
                has_lyrics=False,
                confidence="high",
            )

    def test_to_descriptor_string_format(self):
        """Descriptor string follows canonical format."""
        desc = TrackDescriptor(
            mood="melancholy",
            tempo_feel="slow",
            genre="jazz",
            lyrical_themes="loneliness",
            narrative="Quiet piano.",
            has_lyrics=True,
            confidence="high",
        )
        s = desc.to_descriptor_string()
        assert s.startswith("Mood:")
        assert "Tempo feel:" in s
        assert "Genre:" in s
        assert "Lyrical themes:" in s
        assert "Quiet piano." in s


class TestRetrieveRequest:

    def test_valid_request(self):
        req = RetrieveRequest(query="chill rainy afternoon jazz")
        assert req.query == "chill rainy afternoon jazz"
        assert req.top_k == 50
        assert req.index_tier == "auto"

    def test_empty_query_rejected(self):
        """failure_modes.md §4.4: Empty queries must be rejected."""
        with pytest.raises(ValidationError):
            RetrieveRequest(query="   ")

    def test_query_too_long_rejected(self):
        with pytest.raises(ValidationError):
            RetrieveRequest(query="x" * 501)

    def test_invalid_top_k_rejected(self):
        with pytest.raises(ValidationError):
            RetrieveRequest(query="jazz", top_k=-1)
        with pytest.raises(ValidationError):
            RetrieveRequest(query="jazz", top_k=201)


class TestEvalQuery:

    def test_valid_eval_query(self):
        q = EvalQuery(
            query="chill rainy afternoon jazz",
            relevant_ids=["fma_000001", "fma_000002"],
            source="author_labeled",
        )
        assert q.source == "author_labeled"
        assert len(q.relevant_ids) == 2

    def test_source_must_be_author_labeled(self):
        """source field enforces author_labeled — CLAUDE.md §evaluation."""
        with pytest.raises(ValidationError):
            EvalQuery(
                query="test",
                relevant_ids=["fma_001"],
                source="auto_generated",   # not allowed
            )


class TestAppConfig:

    def test_default_config_is_valid(self):
        """Default AppConfig with all defaults is valid."""
        config = AppConfig()
        assert config.retrieval.confidence_threshold == 0.65
        assert config.pipeline.fuzzy_match_threshold == 85
        assert config.enrichment.llm_backend == "lmstudio"
        assert config.enrichment.llm_model == "mistral-7b-instruct-v0.3"

    def test_config_override(self):
        """Config values can be overridden for testing."""
        config = AppConfig()
        config.retrieval.confidence_threshold = 0.70
        assert config.retrieval.confidence_threshold == 0.70
