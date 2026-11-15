"""Where the models run and in what precision.

Copied in part from ``src/adapters/outbound/torch_runtime.py`` (the
transcriber never imports ``src``): models run in float16 on CUDA and
in float32 elsewhere. :func:`release` gives the GPU memory back once a
job ends, so the speech and voice models are not resident between
jobs; :func:`empty_cuda_cache` is how (#117).
"""

import gc
from typing import Any, Dict, Optional

from transcriber.config import Device
from transcriber.domain.errors import ModelLoadError

__all__ = [
    "Device", "HALF_PRECISION_DEVICES", "ModelLoadError",
    "empty_cuda_cache", "model_kwargs", "pick_device", "release",
]

HALF_PRECISION_DEVICES = ("cuda",)
"""Devices where models run in float16.

Measured under ``src`` on the RTX 3070 Ti with bge-base (SciFact, 300
queries): 3x faster than float32 and the same top-1 for every query.
CPUs keep float32 (float16 is slow there). Not yet measured for the
transcriber's models.
"""


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

        empty_cuda_cache(torch)


def empty_cuda_cache(torch: Optional[Any]) -> None:
    """Give PyTorch's cached GPU memory back to the driver.

    ``torch.cuda.empty_cache`` returns only the cached segments that
    hold no live block. PyTorch keeps cuBLAS's workspace (8.5 MB per
    stream on this GPU) for the life of the process, and it is carved
    from whatever segment was free when the first matrix product ran:
    transformers loads a model's weights into one segment the size of
    the model, so once the model is freed that small workspace kept the
    whole segment (5.97 GB with Qwen3-8B at 4 bits, measured on the RTX
    3070 Ti, #117) reserved. The workspaces are dropped first (they are
    made again, small, on the next product). Nothing without CUDA.
    """
    if torch is None or not torch.cuda.is_available():
        return
    clear = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)
    if callable(clear):
        clear()
    torch.cuda.empty_cache()
