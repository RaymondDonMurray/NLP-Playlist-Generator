"""src/data/audio_features.py

Essentia audio-feature extraction — replaces EchoNest coverage on FMA
(ADR-002 amendment 2026-06-12). Two model passes per track plus signal algos:

  - signal algos (no ML): RhythmExtractor2013 (BPM), KeyExtractor (key+scale),
    Danceability (binned), DynamicComplexity (energy proxy).
  - discogs-effnet embeddings → genre / mood / instrument classifier heads.
  - YAMNet (AudioSet, 521 classes) → speechiness (sum of speech-class mass) and
    moods-audioset (top-k of the AudioSet music-mood subtree).

Output schema per track (see extract_track):
  track_id:int, tempo_bpm:float, key:str, danceability:str, energy:float,
  genres:list[str], moods:list[str], instruments:list[str], speechiness:float

ML models are loaded ONCE into FeatureExtractors and reused across all tracks.
essentia is imported lazily so the pure helpers and the config/registry are
importable (and unit-testable) without the heavy dependency installed.

Import discipline (CLAUDE.md): data/ is the lowest layer; imports compute.py
(same layer) and stdlib/numpy only.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.data.compute import get_compute_device
from src.data.schema import AudioExtractionConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model registry. I/O node names are MODEL FACTS (from each model's metadata
# JSON), not tunables — they live here, not in config.yaml. Verified Phase 0.
# ---------------------------------------------------------------------------

_MODELS_BASE = "https://essentia.upf.edu/models"


@dataclass(frozen=True)
class EmbeddingSpec:
    name: str
    graph: str          # .pb filename
    output: str         # graph output node yielding the embedding
    url: str            # download source


@dataclass(frozen=True)
class HeadSpec:
    name: str
    graph: str          # .pb filename (TensorflowPredict2D head)
    labels: str         # .json filename (carries "classes")
    input: str          # graph input node
    output: str         # graph output node yielding predictions
    url: str            # .pb download source
    labels_url: str     # .json download source


EFFNET = EmbeddingSpec(
    name="discogs-effnet",
    graph="discogs-effnet-bs64-1.pb",
    output="PartitionedCall:1",
    url=f"{_MODELS_BASE}/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb",
)

# Heads that consume discogs-effnet embeddings. Keyed by output field name.
HEAD_SPECS: dict[str, HeadSpec] = {
    "genres": HeadSpec(
        name="genre_discogs400",
        graph="genre_discogs400-discogs-effnet-1.pb",
        labels="genre_discogs400-discogs-effnet-1.json",
        input="serving_default_model_Placeholder",
        output="PartitionedCall:0",
        url=f"{_MODELS_BASE}/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.pb",
        labels_url=f"{_MODELS_BASE}/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.json",
    ),
    "moods": HeadSpec(
        name="mtg_jamendo_moodtheme",
        graph="mtg_jamendo_moodtheme-discogs-effnet-1.pb",
        labels="mtg_jamendo_moodtheme-discogs-effnet-1.json",
        input="model/Placeholder",
        output="model/Sigmoid",
        url=f"{_MODELS_BASE}/classification-heads/mtg_jamendo_moodtheme/mtg_jamendo_moodtheme-discogs-effnet-1.pb",
        labels_url=f"{_MODELS_BASE}/classification-heads/mtg_jamendo_moodtheme/mtg_jamendo_moodtheme-discogs-effnet-1.json",
    ),
    "instruments": HeadSpec(
        name="mtg_jamendo_instrument",
        graph="mtg_jamendo_instrument-discogs-effnet-1.pb",
        labels="mtg_jamendo_instrument-discogs-effnet-1.json",
        input="model/Placeholder",
        output="model/Sigmoid",
        url=f"{_MODELS_BASE}/classification-heads/mtg_jamendo_instrument/mtg_jamendo_instrument-discogs-effnet-1.pb",
        labels_url=f"{_MODELS_BASE}/classification-heads/mtg_jamendo_instrument/mtg_jamendo_instrument-discogs-effnet-1.json",
    ),
}

# YAMNet classifies audio directly into 521 AudioSet classes (no shared embedding).
YAMNET = HeadSpec(
    name="audioset-yamnet",
    graph="audioset-yamnet-1.pb",
    labels="audioset-yamnet-1.json",
    input="melspectrogram",
    output="activations",
    url=f"{_MODELS_BASE}/audio-event-recognition/yamnet/audioset-yamnet-1.pb",
    labels_url=f"{_MODELS_BASE}/audio-event-recognition/yamnet/audioset-yamnet-1.json",
)

# AudioSet speech-family classes → speechiness proxy (Option A). Resolved to
# indices by name at load time (robust to ontology ordering).
SPEECH_CLASSES = [
    "Speech",
    "Child speech, kid speaking",
    "Conversation",
    "Narration, monologue",
    "Babbling",
    "Speech synthesizer",
]
# AudioSet music-mood subtree → moods-audioset.
# NB: YAMNet exposes 521 of AudioSet's 527 classes — "Funny music" is among the
# dropped ones, so it is intentionally excluded here (it would only warn + skip).
AUDIOSET_MOOD_CLASSES = [
    "Happy music",
    "Sad music",
    "Tender music",
    "Exciting music",
    "Angry music",
    "Scary music",
]


# ---------------------------------------------------------------------------
# Pure transforms (no essentia, no I/O) — unit-tested directly.
# ---------------------------------------------------------------------------

def bin_danceability(value: float, threshold: float) -> str:
    """Bin Essentia's continuous Danceability into the categorical the prompt wants."""
    return "danceable" if value >= threshold else "non-danceable"


def combine_key(key: str, scale: str) -> str:
    """Combine KeyExtractor's (key, scale) into one string, e.g. 'G minor'."""
    return f"{key} {scale}".strip()


def normalize_energy(dynamic_complexity: float, norm: float) -> float:
    """Map DynamicComplexity (dB-ish) to a clamped 0-1 energy proxy."""
    if norm <= 0:
        return 0.0
    return float(min(max(dynamic_complexity / norm, 0.0), 1.0))


def topk_labels(predictions: np.ndarray, labels: list[str], k: int) -> list[str]:
    """Mean-pool per-patch predictions over time, return the top-k class labels."""
    mean = np.asarray(predictions, dtype=float).mean(axis=0)
    if mean.ndim != 1:
        mean = mean.ravel()
    order = np.argsort(mean)[::-1][:k]
    return [labels[i] for i in order if i < len(labels)]


def speechiness_from_activations(activations: np.ndarray, speech_idx: list[int]) -> float:
    """Speechiness = summed probability mass of the AudioSet speech classes."""
    if not speech_idx:
        return 0.0
    mean = np.asarray(activations, dtype=float).mean(axis=0)
    return float(mean[speech_idx].sum())


def topk_from_indices(
    activations: np.ndarray, idx_labels: list[tuple[int, str]], k: int
) -> list[str]:
    """Top-k labels restricted to a named index subset (e.g. AudioSet moods)."""
    if not idx_labels:
        return []
    mean = np.asarray(activations, dtype=float).mean(axis=0)
    ranked = sorted(idx_labels, key=lambda pair: mean[pair[0]], reverse=True)[:k]
    return [label for _, label in ranked]


def _resolve_indices(labels: list[str], wanted: list[str]) -> list[tuple[int, str]]:
    """Map wanted class names to (index, name) pairs, skipping any not present."""
    out: list[tuple[int, str]] = []
    for name in wanted:
        if name in labels:
            out.append((labels.index(name), name))
        else:
            logger.warning("AudioSet class not found in labels: %r", name)
    return out


# ---------------------------------------------------------------------------
# Model download / cache
# ---------------------------------------------------------------------------

def _silence_essentia_warnings() -> None:
    """Quiet Essentia's C++ warning logger. MonoLoader emits a benign
    'No network created, or last created network has been deleted' warning once
    per decoded track, which floods the extraction run. Idempotent.
    """
    import essentia

    essentia.log.warningActive = False


def load_audio_bytes(data: bytes, sample_rate: int, suffix: str = ".mp3") -> np.ndarray:
    """Decode raw (e.g. mp3) audio bytes to a mono float32 array at sample_rate.

    Uses Essentia's MonoLoader, which decodes (via ffmpeg/avcodec) and resamples
    natively — avoiding a torchcodec/soundfile dependency for the streamed bytes.
    MonoLoader needs a file path, so the bytes are written to a short-lived temp
    file. essentia is imported lazily (heavy dependency, offline pipeline only).
    """
    import essentia.standard as es

    _silence_essentia_warnings()

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as fh:
            fh.write(data)
            tmp_path = fh.name
        return es.MonoLoader(filename=tmp_path, sampleRate=sample_rate, resampleQuality=4)()
    finally:
        if tmp_path is not None:
            os.unlink(tmp_path)


def ensure_models(models_dir: Path) -> None:
    """Download all required Essentia model + label files to models_dir if absent.

    Idempotent: existing files are left untouched (resume-friendly). Downloads
    are atomic (temp file → rename) so an interrupted fetch never leaves a
    half-written .pb that would crash graph loading.
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[str, str]] = [(EFFNET.graph, EFFNET.url)]
    for spec in (*HEAD_SPECS.values(), YAMNET):
        jobs.append((spec.graph, spec.url))
        jobs.append((spec.labels, spec.labels_url))

    for filename, url in jobs:
        dest = models_dir / filename
        if dest.exists():
            continue
        logger.info("Downloading model file %s", filename)
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        urllib.request.urlretrieve(url, tmp)  # noqa: S310 (trusted essentia.upf.edu host)
        tmp.replace(dest)


# ---------------------------------------------------------------------------
# Feature extractor — loads every model once, reused across all tracks.
# ---------------------------------------------------------------------------

class FeatureExtractors:
    """Holds all loaded Essentia algorithms/models. Construct once per run."""

    def __init__(self, config: AudioExtractionConfig, models_dir: Path | None = None) -> None:
        self.config = config
        self.sample_rate = config.sample_rate
        md = Path(models_dir) if models_dir is not None else Path(config.models_dir)
        self.models_dir = md
        self.device = get_compute_device()
        logger.info("Loading Essentia models from %s (device=%s)", md, self.device)

        import essentia.standard as es  # lazy: heavy dependency, offline pipeline only

        _silence_essentia_warnings()

        # Signal algorithms (CPU C++ DSP; no GPU path).
        self._rhythm = es.RhythmExtractor2013(method="multifeature")
        self._key = es.KeyExtractor()
        self._dance = es.Danceability()
        self._dyncomplex = es.DynamicComplexity()

        # Shared embedding model.
        self._effnet = es.TensorflowPredictEffnetDiscogs(
            graphFilename=str(md / EFFNET.graph), output=EFFNET.output
        )

        # Effnet-fed classifier heads: name → (model, labels, top_k).
        topk = {
            "genres": config.genre_top_k,
            "moods": config.mood_top_k,
            "instruments": config.instrument_top_k,
        }
        self._heads: dict[str, tuple[object, list[str], int]] = {}
        for key, spec in HEAD_SPECS.items():
            model = es.TensorflowPredict2D(
                graphFilename=str(md / spec.graph), input=spec.input, output=spec.output
            )
            labels = json.loads((md / spec.labels).read_text())["classes"]
            self._heads[key] = (model, labels, topk[key])

        # YAMNet (AudioSet) — speechiness + moods-audioset from one pass.
        self._yamnet = es.TensorflowPredictVGGish(
            graphFilename=str(md / YAMNET.graph), input=YAMNET.input, output=YAMNET.output
        )
        yamnet_labels = json.loads((md / YAMNET.labels).read_text())["classes"]
        self._speech_idx = [i for i, _ in _resolve_indices(yamnet_labels, SPEECH_CLASSES)]
        self._audioset_mood_idx = _resolve_indices(yamnet_labels, AUDIOSET_MOOD_CLASSES)

    def extract_track(self, audio: np.ndarray, track_id: int) -> dict:
        """Extract the full feature schema for one 16 kHz mono track.

        Raises on a hard model failure — the caller (script) decides whether to
        skip the track and continue. Audio must already be resampled to
        config.sample_rate (mono float32).
        """
        cfg = self.config

        # --- signal features ---
        bpm = float(self._rhythm(audio)[0])
        key, scale, _strength = self._key(audio)
        dance_value = float(self._dance(audio)[0])
        dynamic_complexity = float(self._dyncomplex(audio)[0])

        # --- effnet embedding + heads ---
        embeddings = self._effnet(audio)
        labels_out: dict[str, list[str]] = {}
        for field_name, (model, labels, k) in self._heads.items():
            preds = model(embeddings)
            labels_out[field_name] = topk_labels(preds, labels, k)

        # --- YAMNet (AudioSet): speechiness + moods-audioset ---
        activations = self._yamnet(audio)
        speechiness = speechiness_from_activations(activations, self._speech_idx)
        audioset_moods = topk_from_indices(
            activations, self._audioset_mood_idx, cfg.audioset_mood_top_k
        )

        # Merge mtg-jamendo moods with AudioSet moods (dedup, preserve order).
        moods = list(dict.fromkeys([*labels_out["moods"], *audioset_moods]))

        return {
            "track_id": int(track_id),
            "tempo_bpm": bpm,
            "key": combine_key(key, scale),
            "danceability": bin_danceability(dance_value, cfg.danceability_threshold),
            "energy": normalize_energy(dynamic_complexity, cfg.energy_norm),
            "genres": labels_out["genres"],
            "moods": moods,
            "instruments": labels_out["instruments"],
            "speechiness": speechiness,
        }
