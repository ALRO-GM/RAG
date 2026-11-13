"""Where the models run and in what precision.

Copied in part from ``src/adapters/outbound/torch_runtime.py`` (the
transcriber never imports ``src``): models run in float16 on CUDA and
in float32 elsewhere. :func:`release` gives the GPU memory back once a
job ends, so the speech and voice models are not resident between
jobs.
"""

import gc
from typing import Any, Dict

from transcriber.config import Device
from transcriber.domain.errors import TranscriberError

__all__ = [
    "Device", "HALF_PRECISION_DEVICES", "ModelLoadError", "model_kwargs",
    "pick_device", "release",
]

HALF_PRECISION_DEVICES = ("cuda",)
"""Devices where models run in float16.

Measured under ``src`` on the RTX 3070 Ti with bge-base (SciFact, 300
queries): 3x faster than float32 and the same top-1 for every query.
CPUs keep float32 (float16 is slow there). Not yet measured for the
transcriber's models.
"""


class ModelLoadError(TranscriberError):
    """A model cannot be loaded (not downloaded, or not that kind)."""


def pick_device(preferred: Device) -> str:
    """Return the torch device to load models on.

    ``auto`` is ``cuda`` when a CUDA device is usable, else ``cpu``.
    """
    if preferred != "auto":
        return preferred
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def model_kwargs(device: str) -> Dict[str, Any]:
    """Keyword arguments for loading a transformer on ``device``."""
    if device not in HALF_PRECISION_DEVICES:
        return {}
    import torch

    return {"dtype": torch.float16}


def release(device: str) -> None:
    """Collect freed models and return cached GPU memory to the driver."""
    gc.collect()
    if device in HALF_PRECISION_DEVICES:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
