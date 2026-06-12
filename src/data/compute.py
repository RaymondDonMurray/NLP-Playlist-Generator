"""src/data/compute.py

Compute-device detection for the Essentia/TensorFlow audio-feature extraction
pipeline. Resolves a single device via the fallback chain CUDA → Metal → CPU
and logs the choice once at startup.

This is the *Essentia/TF compute device* — deliberately distinct from
`config.enrichment.llm_backend`, which selects the LLM transport (LM Studio /
Ollama). The two are different axes; do not conflate "device" here with
"backend" there.

Platform note (verified Phase 0): on Apple Silicon, essentia-tensorflow bundles
a CPU-only libtensorflow and exposes no standalone `tensorflow` module, so this
resolves to "cpu" locally regardless of whether tensorflow-metal is installed.
The CUDA branch is for a future Linux GPU/CPU-parallel extraction host, where a
standalone TensorFlow with GPU support is present.

Import discipline (CLAUDE.md): data/ is the lowest layer. This module imports
only the standard library (+ TensorFlow lazily, if present).
"""

from __future__ import annotations

import functools
import logging
import os
import platform
from importlib import metadata
from typing import Literal

logger = logging.getLogger(__name__)

ComputeDevice = Literal["cuda", "metal", "cpu"]


def _distribution_installed(dist_name: str) -> bool:
    """True if a pip distribution is installed (it need not be an importable module).

    tensorflow-metal registers a PluggableDevice rather than a top-level module,
    so it cannot be detected with importlib.util.find_spec — query the installed
    distribution metadata instead.
    """
    try:
        metadata.version(dist_name)
        return True
    except metadata.PackageNotFoundError:
        return False


def _cuda_available() -> bool:
    """Probe for a usable CUDA GPU via TensorFlow.

    Honours an explicit disable via CUDA_VISIBLE_DEVICES=-1 (or empty). On
    success, enables GPU memory growth so the process does not pre-allocate the
    whole VRAM pool — important on shared/multi-tenant VMs.
    """
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible is not None and cuda_visible.strip() in {"", "-1"}:
        logger.debug("CUDA disabled via CUDA_VISIBLE_DEVICES=%r", cuda_visible)
        return False

    try:
        import tensorflow as tf
    except Exception as e:
        # No standalone tensorflow (e.g. essentia-tensorflow only) — not a CUDA host.
        logger.debug("TensorFlow import failed during CUDA probe: %s", e)
        return False

    try:
        gpus = tf.config.list_physical_devices("GPU")
    except Exception as e:
        logger.debug("tf.config.list_physical_devices('GPU') failed: %s", e)
        return False

    if not gpus:
        return False

    # Prevent TF from grabbing all VRAM up front (fails on shared VMs).
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:
            # Raised if the GPU is already initialised — non-fatal; the env var
            # set above still governs allocation behaviour.
            logger.debug("set_memory_growth failed for %s: %s", gpu, e)

    logger.debug("CUDA probe: %d GPU(s) visible", len(gpus))
    return True


def _metal_available() -> bool:
    """Apple-Silicon Metal path: arm64 macOS AND the tensorflow-metal plugin.

    platform.machine() == 'arm64' is the reliable Apple-Silicon signal;
    platform.processor() == 'arm' is kept only as a secondary check because it
    can return 'i386'/'' under Rosetta.
    """
    is_apple_silicon = platform.machine() == "arm64" or platform.processor() == "arm"
    if not is_apple_silicon:
        return False
    return _distribution_installed("tensorflow-metal")


@functools.lru_cache(maxsize=1)
def get_compute_device() -> ComputeDevice:
    """Resolve the Essentia/TF compute device once: CUDA → Metal → CPU.

    Memoised: the fallback chain runs (and logs the selection) only on the first
    call; later calls return the cached result. The returned string is
    informational for the extraction pipeline (startup logging, batch-size
    choices) — Essentia's TF algorithms run on whichever device the bundled
    libtensorflow supports.

    Returns:
        "cuda", "metal", or "cpu".
    """
    if _cuda_available():
        device: ComputeDevice = "cuda"
    elif _metal_available():
        device = "metal"
    else:
        device = "cpu"

    logger.info("Compute device selected: %s", device)
    return device
