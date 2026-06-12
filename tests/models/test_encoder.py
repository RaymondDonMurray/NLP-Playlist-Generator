"""tests/models/test_encoder.py

Tests for src/models/encoder.py — device selection, lazy loading,
encoding shapes and types, instruction prefix, and normalization.

Acceptance criterion (T01):
    encode_query('test') returns list[float] of length 768.

All tests mock SentenceTransformer so no model download is required.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Stub sentence_transformers ONLY when it is genuinely not installed, so the
# suite still runs in a minimal CI env. When the real package is present we must
# NOT clobber it — this module is collected before tests/serving/, and a stub
# left in sys.modules would break `from sentence_transformers import CrossEncoder`
# in reranker.py. The device tests below patch src.models.encoder.SentenceTransformer
# directly, so they pass either way.
if importlib.util.find_spec("sentence_transformers") is None and "sentence_transformers" not in sys.modules:
    _st_stub = types.ModuleType("sentence_transformers")
    _st_stub.SentenceTransformer = MagicMock
    sys.modules["sentence_transformers"] = _st_stub

from src.data.schema import AppConfig  # noqa: E402
from src.models.encoder import Encoder  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BGE_DIM = 768  # bge-base-en-v1.5 output dimension


def _make_mock_model(n_texts: int = 1) -> MagicMock:
    """Return a mock SentenceTransformer that yields unit-norm float32 vectors."""
    mock = MagicMock()
    mock.encode.side_effect = lambda texts, **kwargs: np.ones(
        (len(texts), BGE_DIM), dtype=np.float32
    ) / np.sqrt(BGE_DIM)
    return mock


# ---------------------------------------------------------------------------
# TestDeviceSelection
# ---------------------------------------------------------------------------

class TestDeviceSelection:
    """_load_model picks the correct device based on hardware availability."""

    def test_cuda_preferred_when_available(self, config):
        """CUDA is chosen first when torch.cuda.is_available() is True."""
        encoder = Encoder(config)
        with (
            patch("src.models.encoder.torch.cuda.is_available", return_value=True),
            patch("src.models.encoder.SentenceTransformer") as mock_st,
        ):
            mock_st.return_value = _make_mock_model()
            encoder._load_model()
            mock_st.assert_called_once_with(config.retrieval.bi_encoder, device="cuda")

    def test_mps_chosen_when_cuda_absent_and_mps_available(self, config):
        """MPS is chosen when CUDA is absent and both MPS checks pass."""
        encoder = Encoder(config)
        with (
            patch("src.models.encoder.torch.cuda.is_available", return_value=False),
            patch("src.models.encoder.torch.backends.mps.is_available", return_value=True),
            patch("src.models.encoder.torch.backends.mps.is_built", return_value=True),
            patch("src.models.encoder.SentenceTransformer") as mock_st,
        ):
            mock_st.return_value = _make_mock_model()
            encoder._load_model()
            mock_st.assert_called_once_with(config.retrieval.bi_encoder, device="mps")

    def test_cpu_fallback_when_mps_not_built(self, config):
        """CPU is used when MPS is available but not built (e.g., non-Apple build)."""
        encoder = Encoder(config)
        with (
            patch("src.models.encoder.torch.cuda.is_available", return_value=False),
            patch("src.models.encoder.torch.backends.mps.is_available", return_value=True),
            patch("src.models.encoder.torch.backends.mps.is_built", return_value=False),
            patch("src.models.encoder.SentenceTransformer") as mock_st,
        ):
            mock_st.return_value = _make_mock_model()
            encoder._load_model()
            mock_st.assert_called_once_with(config.retrieval.bi_encoder, device="cpu")

    def test_cpu_fallback_when_no_accelerator(self, config):
        """CPU is used when neither CUDA nor MPS is present."""
        encoder = Encoder(config)
        with (
            patch("src.models.encoder.torch.cuda.is_available", return_value=False),
            patch("src.models.encoder.torch.backends.mps.is_available", return_value=False),
            patch("src.models.encoder.torch.backends.mps.is_built", return_value=False),
            patch("src.models.encoder.SentenceTransformer") as mock_st,
        ):
            mock_st.return_value = _make_mock_model()
            encoder._load_model()
            mock_st.assert_called_once_with(config.retrieval.bi_encoder, device="cpu")

    def test_model_name_read_from_config(self, config):
        """Model name is read from config.retrieval.bi_encoder, never hardcoded."""
        encoder = Encoder(config)
        with (
            patch("src.models.encoder.torch.cuda.is_available", return_value=False),
            patch("src.models.encoder.torch.backends.mps.is_available", return_value=False),
            patch("src.models.encoder.torch.backends.mps.is_built", return_value=False),
            patch("src.models.encoder.SentenceTransformer") as mock_st,
        ):
            mock_st.return_value = _make_mock_model()
            encoder._load_model()
            args, kwargs = mock_st.call_args
            assert args[0] == config.retrieval.bi_encoder


# ---------------------------------------------------------------------------
# TestLazyLoading
# ---------------------------------------------------------------------------

class TestLazyLoading:
    """The model property triggers _load_model only on first access."""

    def test_model_not_loaded_at_init(self, config):
        """_load_model is not called during __init__."""
        with patch("src.models.encoder.SentenceTransformer") as mock_st:
            Encoder(config)
            mock_st.assert_not_called()

    def test_model_loaded_on_first_property_access(self, config):
        """Accessing .model calls _load_model exactly once."""
        encoder = Encoder(config)
        mock_inner = _make_mock_model()
        with patch.object(encoder, "_load_model", return_value=mock_inner) as mock_load:
            _ = encoder.model
            mock_load.assert_called_once()

    def test_model_cached_after_first_access(self, config):
        """Subsequent .model accesses do not trigger additional loads."""
        encoder = Encoder(config)
        mock_inner = _make_mock_model()
        with patch.object(encoder, "_load_model", return_value=mock_inner) as mock_load:
            _ = encoder.model
            _ = encoder.model
            _ = encoder.model
            mock_load.assert_called_once()


# ---------------------------------------------------------------------------
# TestEncodeQuery
# ---------------------------------------------------------------------------

class TestEncodeQuery:
    """encode_query returns a properly shaped, typed, normalized list[float]."""

    @pytest.fixture
    def encoder_with_mock(self, config):
        encoder = Encoder(config)
        encoder._model = _make_mock_model()
        return encoder

    def test_returns_list(self, encoder_with_mock):
        """Return type is list, not np.ndarray or any other type."""
        result = encoder_with_mock.encode_query("chill rainy jazz")
        assert isinstance(result, list)

    def test_returns_list_of_floats(self, encoder_with_mock):
        """Every element of the returned list is a Python float."""
        result = encoder_with_mock.encode_query("test")
        assert all(isinstance(v, float) for v in result)

    def test_length_equals_bge_dimension(self, encoder_with_mock):
        """Returned vector has length 768 (bge-base-en-v1.5 output dim)."""
        result = encoder_with_mock.encode_query("test")
        assert len(result) == BGE_DIM

    def test_instruction_prefix_prepended(self, config):
        """The bge asymmetric prefix is prepended to the query before encoding."""
        encoder = Encoder(config)
        encoder._model = _make_mock_model()
        encoder.encode_query("chill jazz")
        call_args = encoder._model.encode.call_args
        encoded_texts = call_args[0][0]  # first positional arg is the list of texts
        assert len(encoded_texts) == 1
        assert encoded_texts[0].startswith(
            "Represent this sentence for searching relevant passages: "
        )
        assert encoded_texts[0].endswith("chill jazz")

    def test_normalize_embeddings_true(self, config):
        """encode_query calls model.encode with normalize_embeddings=True."""
        encoder = Encoder(config)
        encoder._model = _make_mock_model()
        encoder.encode_query("test")
        call_kwargs = encoder._model.encode.call_args[1]
        assert call_kwargs.get("normalize_embeddings") is True

    def test_result_is_unit_norm(self, config):
        """The returned vector has magnitude approximately 1.0 (normalized)."""
        encoder = Encoder(config)
        # Use a real unit-norm vector rather than a mock magnitude check
        unit_vec = np.zeros(BGE_DIM, dtype=np.float32)
        unit_vec[0] = 1.0
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([unit_vec])
        encoder._model = mock_model
        result = encoder.encode_query("test")
        magnitude = sum(v * v for v in result) ** 0.5
        assert abs(magnitude - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# TestEncodeDescriptors
# ---------------------------------------------------------------------------

class TestEncodeDescriptors:
    """encode_descriptors returns list[list[float]] of correct shape."""

    @pytest.fixture
    def encoder_with_mock(self, config):
        encoder = Encoder(config)
        encoder._model = _make_mock_model()
        return encoder

    def test_returns_list_of_lists(self, encoder_with_mock):
        """Return type is list[list[...]], not np.ndarray."""
        result = encoder_with_mock.encode_descriptors(["desc one", "desc two"])
        assert isinstance(result, list)
        assert all(isinstance(row, list) for row in result)

    def test_inner_elements_are_floats(self, encoder_with_mock):
        """Each inner element is a Python float."""
        result = encoder_with_mock.encode_descriptors(["desc"])
        assert all(isinstance(v, float) for v in result[0])

    def test_output_count_matches_input(self, encoder_with_mock):
        """One embedding vector is returned per input descriptor."""
        texts = ["a", "b", "c", "d", "e"]
        result = encoder_with_mock.encode_descriptors(texts)
        assert len(result) == len(texts)

    def test_each_vector_has_bge_dimension(self, encoder_with_mock):
        """Every returned vector has length 768."""
        result = encoder_with_mock.encode_descriptors(["x", "y"])
        assert all(len(v) == BGE_DIM for v in result)

    def test_batch_size_forwarded(self, config):
        """The batch_size argument is passed through to model.encode."""
        encoder = Encoder(config)
        encoder._model = _make_mock_model()
        encoder.encode_descriptors(["desc"], batch_size=32)
        call_kwargs = encoder._model.encode.call_args[1]
        assert call_kwargs.get("batch_size") == 32

    def test_show_progress_forwarded(self, config):
        """show_progress maps to show_progress_bar in model.encode."""
        encoder = Encoder(config)
        encoder._model = _make_mock_model()
        encoder.encode_descriptors(["desc"], show_progress=False)
        call_kwargs = encoder._model.encode.call_args[1]
        assert call_kwargs.get("show_progress_bar") is False

    def test_normalize_embeddings_true(self, config):
        """encode_descriptors calls model.encode with normalize_embeddings=True."""
        encoder = Encoder(config)
        encoder._model = _make_mock_model()
        encoder.encode_descriptors(["desc"])
        call_kwargs = encoder._model.encode.call_args[1]
        assert call_kwargs.get("normalize_embeddings") is True

    def test_no_instruction_prefix(self, config):
        """Descriptors are encoded as-is, without the query instruction prefix."""
        encoder = Encoder(config)
        encoder._model = _make_mock_model()
        texts = ["Mood: melancholy. Genre: Jazz."]
        encoder.encode_descriptors(texts)
        encoded_texts = encoder._model.encode.call_args[0][0]
        assert encoded_texts[0] == texts[0]


# ---------------------------------------------------------------------------
# TestEncodeBatch
# ---------------------------------------------------------------------------

class TestEncodeBatch:
    """encode_batch is a general-purpose encoder with no prefix and no progress bar."""

    @pytest.fixture
    def encoder_with_mock(self, config):
        encoder = Encoder(config)
        encoder._model = _make_mock_model()
        return encoder

    def test_returns_list_of_lists(self, encoder_with_mock):
        """Return type is list[list[...]]."""
        result = encoder_with_mock.encode_batch(["text a", "text b"])
        assert isinstance(result, list)
        assert all(isinstance(row, list) for row in result)

    def test_inner_elements_are_floats(self, encoder_with_mock):
        """Each inner element is a Python float."""
        result = encoder_with_mock.encode_batch(["text"])
        assert all(isinstance(v, float) for v in result[0])

    def test_output_count_matches_input(self, encoder_with_mock):
        """One vector per input text."""
        texts = ["a", "b", "c"]
        result = encoder_with_mock.encode_batch(texts)
        assert len(result) == len(texts)

    def test_each_vector_has_bge_dimension(self, encoder_with_mock):
        """Every vector has length 768."""
        result = encoder_with_mock.encode_batch(["text"])
        assert all(len(v) == BGE_DIM for v in result)

    def test_no_instruction_prefix(self, config):
        """Texts are encoded as-is, without any instruction prefix."""
        encoder = Encoder(config)
        encoder._model = _make_mock_model()
        texts = ["some raw text"]
        encoder.encode_batch(texts)
        encoded_texts = encoder._model.encode.call_args[0][0]
        assert encoded_texts[0] == texts[0]

    def test_normalize_embeddings_true(self, config):
        """encode_batch calls model.encode with normalize_embeddings=True."""
        encoder = Encoder(config)
        encoder._model = _make_mock_model()
        encoder.encode_batch(["text"])
        call_kwargs = encoder._model.encode.call_args[1]
        assert call_kwargs.get("normalize_embeddings") is True

    def test_batch_size_forwarded(self, config):
        """The batch_size argument is passed through to model.encode."""
        encoder = Encoder(config)
        encoder._model = _make_mock_model()
        encoder.encode_batch(["text"], batch_size=16)
        call_kwargs = encoder._model.encode.call_args[1]
        assert call_kwargs.get("batch_size") == 16
