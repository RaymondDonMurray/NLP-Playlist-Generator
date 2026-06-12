"""tests/conftest.py

Shared pytest fixtures for the test suite.
scope.md §CI/CD: unit tests for retriever.py, ranker.py (mock ChromaDB, test scoring).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

from src.data.schema import (
    AppConfig,
    CandidateTrack,
    CatalogRecord,
    EvalQuery,
    TrackDescriptor,
)


# ---------------------------------------------------------------------------
# Config fixture — loads test config with overrides for fast tests
# ---------------------------------------------------------------------------

@pytest.fixture
def config() -> AppConfig:
    """Minimal AppConfig suitable for unit tests.

    Overrides: small top-k values, short timeouts, test data paths.
    """
    return AppConfig()   # defaults from schema; override in individual tests as needed


# ---------------------------------------------------------------------------
# Catalog record fixtures (5 fixture tracks — scope.md §CI/CD)
# ---------------------------------------------------------------------------

FIXTURE_TRACKS = [
    {
        "track_id": "fma_000001",
        "title": "Rainy Day Blues",
        "artist": "Test Artist A",
        "genre_top": "Jazz",
        "tempo": 72.0,
        "energy": 0.31,
        "valence": 0.18,
        "acousticness": 0.89,
        "danceability": "non-danceable",
        "lyrics_available": True,
        "lyrics_text": "Walking in the rain, feeling so alone. "
                       "The blues have found me here at home. "
                       "Empty streets and empty souls tonight.",
        "enrichment_tier": "full",
    },
    {
        "track_id": "fma_000002",
        "title": "Sunrise Drive",
        "artist": "Test Artist B",
        "genre_top": "Electronic",
        "tempo": 128.0,
        "energy": 0.85,
        "valence": 0.72,
        "acousticness": 0.05,
        "danceability": "danceable",
        "lyrics_available": False,
        "enrichment_tier": "metadata_only",
    },
    {
        "track_id": "fma_000003",
        "title": "Mountain Echo",
        "artist": "Test Artist C",
        "genre_top": "Folk",
        "tempo": 95.0,
        "energy": 0.52,
        "valence": 0.61,
        "acousticness": 0.76,
        "danceability": "danceable",
        "lyrics_available": True,
        "lyrics_text": "The mountain sings a song so old. "
                       "Through valleys deep and forests cold. "
                       "A wandering heart finds its way home.",
        "enrichment_tier": "full",
    },
    {
        "track_id": "fma_000004",
        "title": "City Pulse",
        "artist": "Test Artist D",
        "genre_top": "Hip-Hop",
        "tempo": 95.0,
        "energy": 0.78,
        "valence": 0.55,
        "acousticness": 0.12,
        "danceability": "danceable",
        "lyrics_available": True,
        "lyrics_text": "Concrete jungle, neon lights. "
                       "Moving through the city nights. "
                       "Every corner tells a story here.",
        "enrichment_tier": "full",
    },
    {
        "track_id": "fma_000005",
        "title": "Quiet Piano",
        "artist": "Test Artist E",
        "genre_top": "Classical",
        "tempo": 58.0,
        "energy": 0.22,
        "valence": 0.35,
        "acousticness": 0.95,
        "danceability": "non-danceable",
        "lyrics_available": False,
        "enrichment_tier": "metadata_only",
    },
]


@pytest.fixture
def fixture_tracks() -> list[CatalogRecord]:
    """5 fixture CatalogRecord instances for testing."""
    return [CatalogRecord(**t) for t in FIXTURE_TRACKS]


@pytest.fixture
def fixture_candidates() -> list[CandidateTrack]:
    """5 fixture CandidateTrack instances with mock scores."""
    return [
        CandidateTrack(
            track_id=t["track_id"],
            title=t["title"],
            artist=t["artist"],
            genre_top=t["genre_top"],
            descriptor=f"Mood: test. Genre: {t['genre_top']}. Narrative: test track.",
            dense_score=0.9 - i * 0.1,
        )
        for i, t in enumerate(FIXTURE_TRACKS)
    ]


@pytest.fixture
def fixture_eval_queries() -> list[EvalQuery]:
    """Small set of eval queries for unit testing the eval harness."""
    return [
        EvalQuery(
            query="chill rainy afternoon jazz",
            relevant_ids=["fma_000001"],
            source="author_labeled",
        ),
        EvalQuery(
            query="energetic electronic driving",
            relevant_ids=["fma_000002"],
            source="author_labeled",
        ),
        EvalQuery(
            query="acoustic folk introspective",
            relevant_ids=["fma_000003"],
            source="author_labeled",
        ),
    ]


@pytest.fixture
def track_popularity() -> dict[str, float]:
    """Mock popularity scores for Novelty@k testing."""
    return {
        "fma_000001": 0.8,
        "fma_000002": 0.6,
        "fma_000003": 0.3,
        "fma_000004": 0.5,
        "fma_000005": 0.2,
    }
