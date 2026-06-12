"""tests/data/test_compute.py

Tests for src/data/compute.py — Essentia/TF compute-device fallback chain.

The fallback (CUDA → Metal → CPU) is validated by mocking the two probe
helpers, so the suite runs on any host (no GPU, no tensorflow-metal required)
and is CI-safe. get_compute_device() is lru_cache'd, so every test clears the
cache first to get a fresh resolution.
"""

from __future__ import annotations

import pytest

from src.data import compute
from src.data.compute import get_compute_device


@pytest.fixture(autouse=True)
def _clear_device_cache():
    """get_compute_device is memoised — reset it before and after each test."""
    get_compute_device.cache_clear()
    yield
    get_compute_device.cache_clear()


class TestFallbackChain:
    """CUDA → Metal → CPU resolution order."""

    def test_cuda_wins_when_available(self, monkeypatch):
        monkeypatch.setattr(compute, "_cuda_available", lambda: True)
        monkeypatch.setattr(compute, "_metal_available", lambda: True)
        assert get_compute_device() == "cuda"

    def test_metal_when_no_cuda(self, monkeypatch):
        monkeypatch.setattr(compute, "_cuda_available", lambda: False)
        monkeypatch.setattr(compute, "_metal_available", lambda: True)
        assert get_compute_device() == "metal"

    def test_cpu_when_neither(self, monkeypatch):
        monkeypatch.setattr(compute, "_cuda_available", lambda: False)
        monkeypatch.setattr(compute, "_metal_available", lambda: False)
        assert get_compute_device() == "cpu"

    def test_result_is_memoised(self, monkeypatch):
        """The chain runs once; later calls return the cached device."""
        calls = {"n": 0}

        def _probe():
            calls["n"] += 1
            return False

        monkeypatch.setattr(compute, "_cuda_available", _probe)
        monkeypatch.setattr(compute, "_metal_available", lambda: False)

        first = get_compute_device()
        second = get_compute_device()
        assert first == second == "cpu"
        assert calls["n"] == 1  # probe not re-run on the second call


class TestCudaProbe:
    """_cuda_available: CUDA_VISIBLE_DEVICES handling + graceful TF-absent path."""

    def test_disabled_via_visible_devices(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
        assert compute._cuda_available() is False

    def test_empty_visible_devices_disables(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
        assert compute._cuda_available() is False

    def test_returns_false_when_tensorflow_absent(self, monkeypatch):
        """No standalone tensorflow (the Apple-Silicon / essentia-only case)."""
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        import builtins

        real_import = builtins.__import__

        def _no_tf(name, *args, **kwargs):
            if name == "tensorflow" or name.startswith("tensorflow."):
                raise ImportError("No module named 'tensorflow'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _no_tf)
        assert compute._cuda_available() is False


class TestMetalProbe:
    """_metal_available: requires arm64 mac AND the tensorflow-metal plugin."""

    def test_false_on_non_apple_silicon(self, monkeypatch):
        monkeypatch.setattr(compute.platform, "machine", lambda: "x86_64")
        monkeypatch.setattr(compute.platform, "processor", lambda: "i386")
        assert compute._metal_available() is False

    def test_false_when_metal_plugin_missing(self, monkeypatch):
        monkeypatch.setattr(compute.platform, "machine", lambda: "arm64")
        monkeypatch.setattr(compute, "_distribution_installed", lambda name: False)
        assert compute._metal_available() is False

    def test_true_when_arm64_and_plugin_present(self, monkeypatch):
        monkeypatch.setattr(compute.platform, "machine", lambda: "arm64")
        monkeypatch.setattr(compute, "_distribution_installed", lambda name: True)
        assert compute._metal_available() is True


class TestRealEnvironment:
    """Smoke test against the actual host — must resolve to a valid device."""

    def test_returns_valid_device(self):
        assert get_compute_device() in {"cuda", "metal", "cpu"}
