"""src/models/encoder.py

bge-base-en-v1.5 inference wrapper for embedding queries and descriptors.
ADR-001: BAAI/bge-base-en-v1.5 chosen for BEIR quality + fine-tuning path.
ADR-001: ~35ms per query on M1 Max MPS backend.
scope.md: Embed descriptors → save to parquet → single bulk load into Chroma.

Import discipline (CLAUDE.md): models/ imports from data/schema.py only.
"""

from __future__ import annotations

import logging

import torch
from sentence_transformers import SentenceTransformer

from src.data.schema import AppConfig

logger = logging.getLogger(__name__)


class Encoder:
    """Wrapper around bge-base-en-v1.5 for encoding queries and descriptors.

    ADR-001: MPS backend on M1 Max. ~35ms/query, ~430MB model footprint.
    Initialized once at serving startup; stateless between requests (ADR-006).
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.model_name = config.retrieval.bi_encoder
        self._model: SentenceTransformer | None = None

    def _load_model(self) -> SentenceTransformer:
        """Lazy-load model to MPS if available, else CPU.

        Loads from config.retrieval.bi_encoder — never hardcoded (CLAUDE.md).
        """
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = "mps"
        else:
            device = "cpu"
        logger.info("Loading encoder model '%s' on device '%s'", self.model_name, device)
        return SentenceTransformer(self.model_name, device=device)

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def encode_query(self, query: str) -> list[float]:
        """Encode a single query string to a float embedding vector.

        bge-base uses query instruction prefix for asymmetric retrieval.
        ADR-001: Asymmetric bi-encoder training maps naturally to
        query→descriptor retrieval structure.
        """
        prefixed = "Represent this sentence for searching relevant passages: " + query
        vector = self.model.encode([prefixed], normalize_embeddings=True)
        return vector[0].tolist()

    def encode_descriptors(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> list[list[float]]:
        """Encode a list of descriptor strings in batches.

        scope.md §vector index: Generate all embeddings first → save to parquet
        → single bulk load into Chroma (avoid incremental writes on M1 macOS).
        """
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
        )
        return [v.tolist() for v in vectors]

    def encode_batch(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        """General-purpose batch encoding without query instruction prefix."""
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
        )
        return [v.tolist() for v in vectors]
