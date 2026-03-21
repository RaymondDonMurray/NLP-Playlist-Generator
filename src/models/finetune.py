"""src/models/finetune.py

Contrastive fine-tuning of bge-base-en-v1.5. Tier 2 stretch goal.
ADR-003: sentence-transformers Trainer API (not custom PyTorch loop).
ADR-003: Combined in-batch + hard negative sampling, NDCG@10 early stopping.
ADR-003: MPS backend, batch size 32–64, ~2 hours for 10 epochs.
failure_modes.md §4.1: save_strategy="epoch", load_best_model_at_end=True.

Import discipline (CLAUDE.md): models/ imports from data/schema.py only.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.data.schema import AppConfig, EvalQuery

logger = logging.getLogger(__name__)


def mine_hard_negatives(
    queries: list[EvalQuery],
    catalog_descriptors: list[str],
    catalog_ids: list[str],
    encoder,
    top_k: int = 20,
) -> list[tuple[str, str, str]]:
    """Mine hard negatives for contrastive training.

    ADR-003: Hard negatives = same genre, different mood, mined via bge-base
    cosine similarity before fine-tuning. Forces the model to learn fine-grained
    musical distinctions (e.g., "melancholy acoustic guitar" vs "energetic
    acoustic guitar").

    Returns list of (query, positive_descriptor, hard_negative_descriptor) triples.
    """
    raise NotImplementedError


def build_training_dataset(
    queries: list[EvalQuery],
    catalog_records: list,
    config: AppConfig,
    encoder,
) -> tuple[list, list]:
    """Build train and validation datasets for contrastive fine-tuning.

    ADR-003: 80/20 train/val split on the 50-query eval set.
    Val set = held-out 20% (10 queries) for NDCG@10 early stopping.

    Returns (train_examples, val_examples) in sentence-transformers format.
    """
    raise NotImplementedError


def train(
    config: AppConfig,
    train_examples: list,
    val_examples: list,
    val_queries: list[EvalQuery],
    tensorboard_run_name: str,
) -> Path:
    """Fine-tune bge-base-en-v1.5 on music-specific (query, descriptor) pairs.

    ADR-003 decisions:
    - sentence-transformers Trainer API (not custom PyTorch loop)
    - MultipleNegativesRankingLoss for in-batch negatives (weight 0.7)
    - Triplet margin loss for hard negatives (weight 0.3)
    - NDCG@10 early stopping (patience=3, min_delta=0.005)
    - save_strategy="epoch", load_best_model_at_end=True (failure_modes.md §4.1)
    - resume_from_checkpoint=True for crash recovery

    Returns path to the best checkpoint.
    """
    raise NotImplementedError
