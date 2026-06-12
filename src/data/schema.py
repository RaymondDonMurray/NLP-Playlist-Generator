"""src/data/schema.py

Single source of truth for all Pydantic data models.
ADR-006: All data contracts defined here; no model definitions elsewhere.
CLAUDE.md: Schema violations at ingest = hard failure. At enrichment = soft failure (return None).
failure_modes.md §3.1: Centralized missing-value contract via prompt_features property.
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Audio feature valid ranges (failure_modes.md §1.3: Spotify sentinel values)
# Loaded from config at runtime; defaults mirror config.yaml values.
# ---------------------------------------------------------------------------
FEATURE_VALID_RANGES: dict[str, tuple[float, float]] = {
    "tempo":        (40.0,  250.0),
    "loudness":     (-60.0,   0.0),
    "energy":       (0.0,     1.0),
    "valence":      (0.0,     1.0),
    "acousticness": (0.0,     1.0),
    "speechiness":  (0.0,     1.0),
    # NOTE: danceability is no longer a numeric range-checked feature — the Essentia
    # pipeline (ADR-002 amendment) emits it as a categorical string, so it is excluded
    # here and passed through validate_audio_features untouched.
}


# ---------------------------------------------------------------------------
# Core catalog record (scope.md §enriched track record schema)
# ---------------------------------------------------------------------------

class CatalogRecord(BaseModel):
    """One enriched track record. Hard schema contract at pipeline ingest."""

    track_id: str
    title: str
    artist: str
    genre_top: str = "Unknown"

    # Audio features. Numeric ones (tempo/energy/speechiness) are nullable; sentinels
    # nulled at ingest (failure_modes.md §1.3). The Essentia pipeline (ADR-002 amendment
    # 2026-06-12) supplies tempo/energy/speechiness/key/genres/moods/instruments and a
    # categorical danceability; valence/acousticness have no Essentia equivalent and are
    # retained only as legacy nullable fields (always None under the Essentia pipeline).
    spotify_id: str | None = None
    tempo:        float | None = None
    energy:       float | None = None
    valence:      float | None = None      # legacy (EchoNest); None under Essentia
    acousticness: float | None = None      # legacy (EchoNest); None under Essentia
    danceability: str | None = None        # categorical: "danceable" | "non-danceable"
    loudness:     float | None = None
    speechiness:  float | None = None
    duration_ms:  int   | None = None
    key:          str | None = None        # e.g. "G minor" (Essentia KeyExtractor)
    genres:       list[str] = Field(default_factory=list)       # Essentia Genre-Discogs400 top-k
    moods:        list[str] = Field(default_factory=list)       # mtg-jamendo + AudioSet moods
    instruments:  list[str] = Field(default_factory=list)       # mtg-jamendo instruments
    instrumental: bool = False             # HF FMA flag; drives lyrics-skip + null themes

    # Lyrics enrichment
    lyrics_text:     str | None = None
    lyrics_available: bool = False

    # Descriptor enrichment
    descriptor:       str | None = None
    enrichment_tier:  Literal["full", "metadata_only"] = "metadata_only"

    # Embedding (populated after encoder pass; not stored in Chroma metadata)
    embedding: list[float] | None = None

    @property
    def prompt_features(self) -> dict[str, str]:
        """Prompt-safe feature dict — never passes None to the LLM.

        failure_modes.md §3.1: The LLM receives "unknown tempo" rather than None
        or an empty string. The latter causes the model to invent a value.
        Drives prompt v3 (ADR-002 amendment): valence/acousticness dropped; key,
        genres, moods, instruments added; danceability is categorical.
        """
        def _join(items: list[str], label: str) -> str:
            return ", ".join(items) if items else f"unknown {label}"

        return {
            "tempo":        f"{self.tempo:.0f} BPM" if self.tempo is not None else "unknown tempo",
            "energy":       f"{self.energy:.2f}"    if self.energy is not None else "unknown energy",
            "danceability": self.danceability if self.danceability else "unknown danceability",
            "key":          self.key if self.key else "unknown key",
            "genres":       _join(self.genres, "genres"),
            "moods":        _join(self.moods, "moods"),
            "instruments":  _join(self.instruments, "instruments"),
        }

    @property
    def chroma_metadata(self) -> dict:
        """Serializable metadata payload for ChromaDB storage.

        Excludes embedding (stored separately) and None values (ChromaDB
        rejects None metadata values).
        """
        meta = {
            "track_id":       self.track_id,
            "title":          self.title,
            "artist":         self.artist,
            "genre_top":      self.genre_top,
            "enrichment_tier": self.enrichment_tier,
            "lyrics_available": self.lyrics_available,
        }
        optional = {
            "spotify_id": self.spotify_id,
            "descriptor":  self.descriptor,
            "tempo":       self.tempo,
            "energy":      self.energy,
            "valence":     self.valence,
            "acousticness": self.acousticness,
        }
        meta.update({k: v for k, v in optional.items() if v is not None})
        return meta


# ---------------------------------------------------------------------------
# Descriptor output model (models/descriptor.py produces this)
# failure_modes.md §2.1: Pydantic enforces null lyrical_themes for instrumentals
# ---------------------------------------------------------------------------

class TrackDescriptor(BaseModel):
    """Structured LLM output for descriptor generation.

    Produced by Mistral 7B; validated before writing to catalog.
    failure_modes.md §2.1: lyrical_themes must be None when has_lyrics=False.
    """

    mood: str
    tempo_feel: str
    genre: str
    has_lyrics: bool
    lyrical_themes: str | None = None   # None for instrumentals
    narrative: str
    confidence: Literal["high", "medium", "low"]

    @field_validator("confidence", mode= "before")
    @classmethod
    def normalize_confidence_case(cls, v: str) -> str: 
        """Normalize LLM confidence value to lowercase. 
        
        Mistral occasionally returns "High" instead of "high", etc. Normalize before the `Literal` check fires. 
        """

        if isinstance(v, str):
            return v.lower()
        return v

    @field_validator("lyrical_themes", mode="before")
    @classmethod
    def no_themes_without_lyrics(cls, v: str | None, info) -> str | None:
        """Reject hallucinated lyrical themes for instrumental tracks."""
        has_lyrics = info.data.get("has_lyrics")
        if has_lyrics is False and v is not None:
            raise ValueError(
                "Cannot have lyrical_themes for an instrumental track (has_lyrics=False). "
                "The LLM hallucinated lyrical content — set lyrical_themes to null."
            )
        return v

    def to_descriptor_string(self) -> str:
        """Render descriptor as the canonical flat string stored in ChromaDB."""
        parts = [
            f"Mood: {self.mood}.",
            f"Tempo feel: {self.tempo_feel}.",
            f"Genre: {self.genre}.",
        ]
        if self.lyrical_themes:
            parts.append(f"Lyrical themes: {self.lyrical_themes}.")
        parts.append(self.narrative)
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Fuzzy match result (data/preprocess.py)
# failure_modes.md §1.1: Two-pass artist+title validation
# ---------------------------------------------------------------------------

class MatchQuality(str):
    HIGH       = "high"
    SUSPICIOUS = "suspicious"
    REJECT     = "reject"


class FuzzyMatchResult(BaseModel):
    """Result of a single FMA→Spotify fuzzy join attempt."""

    fma_track_id: str
    spotify_id:   str | None = None
    title_sim:    float
    artist_sim:   float
    quality:      Literal["high", "suspicious", "reject"]


# ---------------------------------------------------------------------------
# Retrieval request/response models (serving/cli.py, serving/retriever.py)
# failure_modes.md §4.4: All CLI inputs validated through Pydantic before inference.
# ---------------------------------------------------------------------------

class RetrieveRequest(BaseModel):
    """Validated input to the retrieve CLI command."""

    query: str
    top_k: int = Field(default=50, ge=1, le=200)
    index_tier: Literal["auto", "full", "metadata_only"] = "auto"

    @field_validator("query", mode="before")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Query cannot be empty.")
        if len(v) > 500:
            raise ValueError(f"Query too long ({len(v)} chars, max 500).")
        return v


class CandidateTrack(BaseModel):
    """A single candidate track in a retrieval result."""

    track_id:    str
    title:       str
    artist:      str
    genre_top:   str
    descriptor:  str | None = None
    dense_score: float | None = None
    bm25_score:  float | None = None
    rrf_score:   float | None = None
    rerank_score: float | None = None
    explanation: str | None = None


class RetrieveResponse(BaseModel):
    """Schema-validated JSON response from the retrieve command.

    failure_modes.md §4.4: The warning field is the designated channel for
    soft failures (retrieval collapse, low rerank variance, OOV tokens).
    """

    query:            str
    candidates:       list[CandidateTrack]
    index_tier_used:  Literal["full", "metadata_only"]
    retrieval_ms:     float
    reranker_used:    bool
    warning:          str | None = None


class RerankRequest(BaseModel):
    """Validated input to the rerank CLI command."""

    query:      str
    candidates: list[CandidateTrack]
    top_k:      int = Field(default=10, ge=1, le=50)

    @field_validator("query", mode="before")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Query cannot be empty.")
        return v


class RerankResponse(BaseModel):
    """Schema-validated JSON response from the rerank command."""

    query:        str
    tracks:       list[CandidateTrack]
    rerank_ms:    float
    reranker_used: bool
    warning:      str | None = None


class ExplainRequest(BaseModel):
    """Validated input to the explain CLI command."""

    query:  str
    tracks: list[CandidateTrack]

    @field_validator("query", mode="before")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Query cannot be empty.")
        return v


class ExplainResponse(BaseModel):
    """Schema-validated JSON response from the explain command."""

    query:       str
    tracks:      list[CandidateTrack]   # candidates with explanation field populated
    explain_ms:  float


# ---------------------------------------------------------------------------
# Evaluation dataset models (eval/runner.py)
# scope.md §evaluation suite
# ---------------------------------------------------------------------------

class EvalQuery(BaseModel):
    """One entry in eval/queries.json."""

    query:        str
    relevant_ids: list[str]
    source:       Literal["author_labeled"] = "author_labeled"
    notes:        str | None = None


class EvalResult(BaseModel):
    """Per-query eval result row written to eval/results.csv."""

    query:        str
    config_name:  str
    ndcg_at_k:    float
    mrr:          float
    precision_at_k: float
    novelty_at_k: float
    index_tier:   Literal["full", "metadata_only"]
    reranker_used: bool
    warning:      str | None = None


# ---------------------------------------------------------------------------
# Config model — loaded from config.yaml at each entry point
# ADR-006: Pydantic validation at startup; any missing/misconfigured key = hard failure
# ---------------------------------------------------------------------------

class PipelineConfig(BaseModel):
    checkpoint_interval: int = 100
    fuzzy_match_threshold: int = 85
    artist_match_threshold: int = 70
    suspicious_artist_threshold: int = 40
    max_lyrics_tokens: int = 200
    min_lyrics_tokens: int = 20
    speechiness_max: float = 0.5
    duration_ms_min: int = 30000
    dedup_similarity_threshold: float = 0.98
    genius_max_results: int = 3      # Genius fallback: try up to N search hits for lyrics


class EnrichmentConfig(BaseModel):
    # Backend-selectable LLM transport (see src/models/llm_client.py).
    #   "lmstudio" → OpenAI-compatible server, MLX models (local Apple-Silicon dev)
    #   "ollama"   → native Ollama API (Docker/Linux sidecar; MLX unavailable there)
    llm_backend: Literal["lmstudio", "ollama"] = "lmstudio"
    # Must match the model id the server exposes (GET /v1/models for LM Studio).
    llm_model: str = "mistral-7b-instruct-v0.3"
    llm_base_url: str = "http://localhost:1234/v1"
    # 60s (raised from 30): LM Studio JIT cold-load can exceed 30s on first call.
    llm_timeout_s: float = 60.0
    max_retries: int = 3
    prompt_version: str = "v1"
    descriptor_version: str = "v1"


class RetrievalConfig(BaseModel):
    bi_encoder: str = "BAAI/bge-base-en-v1.5"
    cross_encoder: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ann_top_k: int = 50
    rerank_top_k: int = 10
    confidence_threshold: float = 0.65
    rerank_variance_threshold: float = 0.05
    collapse_genre_threshold: float = 0.8
    bm25_weight: float = 0.5
    rrf_k: int = 60
    # Ablation flags (UQ-1): set by apply_ablation_config to toggle pipeline stages
    use_dense: bool = True
    use_bm25: bool = True
    use_reranker: bool = True


class TrainingConfig(BaseModel):
    output_dir: str = "checkpoints/biencoder"
    num_epochs: int = 10
    batch_size: int = 32
    hard_negative_weight: float = 0.3
    in_batch_weight: float = 0.7
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    save_total_limit: int = 3
    hard_negative_mine_top_k: int = 20


class EvalConfig(BaseModel):
    queries_path: str = "eval/queries.json"
    results_path: str = "eval/results.csv"
    baseline_metrics_path: str = "eval/baseline_metrics.json"
    ndcg_k: int = 10
    mrr_k: int = 10
    precision_k: int = 10
    novelty_k: int = 10
    early_stop_patience: int = 3
    early_stop_min_delta: float = 0.005
    smoke_test_n_queries: int = 10
    regression_threshold: float = 0.05


class TensorBoardConfig(BaseModel):
    log_dir: str = "runs/"
    log_every_n_steps: int = 50


class AudioExtractionConfig(BaseModel):
    """Essentia audio-feature extraction (ADR-002 amendment 2026-06-12).

    Replaces EchoNest coverage by extracting features directly from FMA audio
    streamed from HuggingFace. Model I/O node names live in
    src/data/audio_features.py (model facts, not tunables); only
    quality-affecting knobs live here.
    """

    hf_dataset: str = "benjamin-paine/free-music-archive-large"
    sample_rate: int = 16000             # Essentia TF models (effnet/yamnet) expect 16 kHz mono
    models_dir: str = "checkpoints/essentia_models"
    checkpoint_interval: int = 200       # tracks per parquet checkpoint (resume granularity)
    genre_top_k: int = 3
    mood_top_k: int = 3                  # top-k from mtg-jamendo moodtheme head
    instrument_top_k: int = 4
    audioset_mood_top_k: int = 2         # top-k from YAMNet AudioSet music-mood classes
    # Essentia Danceability returns ~[0,3] (higher = more danceable); threshold bins
    # it to the categorical the prompt expects. Coarse — tune on a labeled sample.
    danceability_threshold: float = 1.0
    # DynamicComplexity (dB-ish) → 0-1 energy proxy via divide-and-clamp. Coarse proxy.
    energy_norm: float = 10.0


class PathsConfig(BaseModel):
    data_dir: str = "data/"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    chroma_dir: str = "chroma_db/"
    checkpoints_dir: str = "checkpoints/"


class AppConfig(BaseModel):
    """Root config model. Loaded from config.yaml at every entry point."""

    pipeline:    PipelineConfig    = Field(default_factory=PipelineConfig)
    enrichment:  EnrichmentConfig  = Field(default_factory=EnrichmentConfig)
    retrieval:   RetrievalConfig   = Field(default_factory=RetrievalConfig)
    training:    TrainingConfig    = Field(default_factory=TrainingConfig)
    eval:        EvalConfig        = Field(default_factory=EvalConfig)
    tensorboard: TensorBoardConfig = Field(default_factory=TensorBoardConfig)
    paths:       PathsConfig       = Field(default_factory=PathsConfig)
    audio:       AudioExtractionConfig = Field(default_factory=AudioExtractionConfig)
