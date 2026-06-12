"""tests/data/test_audio_features.py

Tests for src/data/audio_features.py pure transforms + the FMA url→track_id
join resolver. The Essentia model path (FeatureExtractors.extract_track) needs
the heavy dependency + model files and is validated separately against real
audio; these tests cover the deterministic logic and are CI-safe.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.audio_features import (
    AUDIOSET_MOOD_CLASSES,
    SPEECH_CLASSES,
    _resolve_indices,
    bin_danceability,
    combine_key,
    normalize_energy,
    speechiness_from_activations,
    topk_from_indices,
    topk_labels,
)


class TestSignalTransforms:
    def test_bin_danceability_above_threshold(self):
        assert bin_danceability(1.5, threshold=1.0) == "danceable"

    def test_bin_danceability_at_threshold_is_danceable(self):
        assert bin_danceability(1.0, threshold=1.0) == "danceable"

    def test_bin_danceability_below_threshold(self):
        assert bin_danceability(0.4, threshold=1.0) == "non-danceable"

    def test_combine_key(self):
        assert combine_key("G", "minor") == "G minor"
        assert combine_key("Eb", "major") == "Eb major"

    def test_normalize_energy_clamps_high(self):
        assert normalize_energy(50.0, norm=10.0) == 1.0

    def test_normalize_energy_clamps_negative(self):
        assert normalize_energy(-3.0, norm=10.0) == 0.0

    def test_normalize_energy_scales(self):
        assert normalize_energy(5.0, norm=10.0) == 0.5

    def test_normalize_energy_zero_norm_safe(self):
        assert normalize_energy(5.0, norm=0.0) == 0.0


class TestTopKLabels:
    def test_mean_pools_and_orders(self):
        labels = ["a", "b", "c", "d"]
        # two patches; class 'c' (idx 2) wins on mean, then 'a' (idx 0)
        preds = np.array([[0.9, 0.1, 0.95, 0.0], [0.7, 0.2, 0.99, 0.1]])
        assert topk_labels(preds, labels, k=2) == ["c", "a"]

    def test_k_caps_results(self):
        labels = ["a", "b", "c"]
        preds = np.array([[0.1, 0.2, 0.3]])
        assert topk_labels(preds, labels, k=2) == ["c", "b"]

    def test_handles_fewer_labels_than_classes(self):
        # defensive: out-of-range indices are dropped, not crashed on
        labels = ["a", "b"]
        preds = np.array([[0.1, 0.2, 0.9]])  # idx 2 has no label
        assert topk_labels(preds, labels, k=3) == ["b", "a"]


class TestAudioSetTransforms:
    def test_speechiness_sums_named_indices(self):
        # 4 classes; speech idx [0,2]; two patches
        activations = np.array([[0.4, 0.1, 0.3, 0.2], [0.2, 0.0, 0.1, 0.0]])
        # mean = [0.3, 0.05, 0.2, 0.1]; sum of idx 0,2 = 0.5
        assert speechiness_from_activations(activations, [0, 2]) == pytest.approx(0.5)

    def test_speechiness_empty_idx(self):
        assert speechiness_from_activations(np.array([[0.5, 0.5]]), []) == 0.0

    def test_topk_from_indices_restricts_to_subset(self):
        activations = np.array([[0.9, 0.1, 0.8, 0.7]])
        # only consider idx 1 and 3; idx 3 (0.7) > idx 1 (0.1)
        idx_labels = [(1, "sad"), (3, "happy")]
        assert topk_from_indices(activations, idx_labels, k=1) == ["happy"]

    def test_topk_from_indices_empty(self):
        assert topk_from_indices(np.array([[0.5]]), [], k=2) == []

    def test_resolve_indices_skips_missing(self):
        labels = ["Speech", "Music", "Conversation"]
        resolved = _resolve_indices(labels, ["Speech", "Conversation", "Nope"])
        assert resolved == [(0, "Speech"), (2, "Conversation")]

    def test_class_name_lists_are_distinct(self):
        # guard against accidental overlap between the two AudioSet subsets
        assert set(SPEECH_CLASSES).isdisjoint(set(AUDIOSET_MOOD_CLASSES))


class TestUrlToTrackIdMap:
    """FMAClient.load_url_to_track_id — the HF dataset join key."""

    def _make_client(self, tmp_path: Path, rows: list[dict]):
        from src.data.ingest import FMAClient
        from src.data.schema import AppConfig

        fma_dir = tmp_path / "fma_metadata"
        fma_dir.mkdir(parents=True)
        pd.DataFrame(rows).to_csv(fma_dir / "raw_tracks.csv", index=False)
        config = AppConfig.model_validate({"paths": {"raw_dir": str(tmp_path)}})
        return FMAClient(config)

    def test_basic_mapping(self, tmp_path):
        client = self._make_client(
            tmp_path,
            [
                {"track_id": 2, "track_url": "http://fma/AWOL/Food"},
                {"track_id": 5, "track_url": "http://fma/AWOL/Electric"},
            ],
        )
        m = client.load_url_to_track_id()
        assert m == {"http://fma/AWOL/Food": 2, "http://fma/AWOL/Electric": 5}

    def test_null_urls_skipped(self, tmp_path):
        client = self._make_client(
            tmp_path,
            [
                {"track_id": 2, "track_url": "http://fma/a"},
                {"track_id": 3, "track_url": None},
            ],
        )
        m = client.load_url_to_track_id()
        assert m == {"http://fma/a": 2}

    def test_duplicate_urls_dropped_entirely(self, tmp_path):
        # an ambiguous url maps to two tracks → unsafe key, drop both
        client = self._make_client(
            tmp_path,
            [
                {"track_id": 2, "track_url": "http://fma/dup"},
                {"track_id": 9, "track_url": "http://fma/dup"},
                {"track_id": 4, "track_url": "http://fma/unique"},
            ],
        )
        m = client.load_url_to_track_id()
        assert m == {"http://fma/unique": 4}


class TestScriptTrackIdResolver:
    """_resolve_track_id in scripts/extract_audio_features.py (url + filename fallback)."""

    @staticmethod
    def _load_resolver():
        path = Path(__file__).resolve().parents[2] / "scripts" / "extract_audio_features.py"
        spec = importlib.util.spec_from_file_location("extract_audio_features", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod._resolve_track_id

    def test_url_primary_key(self):
        resolve = self._load_resolver()
        rec = {"url": "http://fma/Food", "audio": {"path": "whatever.mp3"}}
        assert resolve(rec, {"http://fma/Food": 2}) == 2

    def test_filename_fallback_when_url_missing(self):
        resolve = self._load_resolver()
        rec = {"url": "http://fma/unknown", "audio": {"path": "/data/000142.mp3"}}
        assert resolve(rec, {}) == 142

    def test_unresolved_returns_none(self):
        resolve = self._load_resolver()
        rec = {"url": "http://fma/unknown", "audio": {"path": "no-digits.mp3"}}
        assert resolve(rec, {}) is None
