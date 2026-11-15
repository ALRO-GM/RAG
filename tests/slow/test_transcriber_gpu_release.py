"""The model for names leaves the GPU after each use (#117), real weights.

Loads the local model for names the way the service does (Qwen3-8B at
4 bits when it fits, else Qwen3-4B), reads one sentence, frees it with
the service's :class:`TorchGpuMemory`, and does it all again: after
each round PyTorch's allocated and reserved memory are back at the
baseline taken before the first load. Needs CUDA and the cached
weights (``HF_HUB_OFFLINE=1``); skipped when the GPU has no room now.
"""

from pathlib import Path

import pytest

from transcriber.adapters.outbound.local_name_model import LocalNameModel
from transcriber.adapters.outbound.torch_gpu_memory import TorchGpuMemory
from transcriber.domain.errors import ModelLoadError
from transcriber.domain.speaker_names import SYSTEM_PROMPT

pytestmark = pytest.mark.slow

CACHE = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-4B"

TOLERANCE = 16 * 2**20
"""Bytes allowed above the baseline (a few small CUDA buffers)."""


@pytest.mark.skipif(not CACHE.exists(), reason="Qwen3-4B is not cached")
def test_two_rounds_of_names_give_the_memory_back() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    model = LocalNameModel(wait_seconds=60.0, device="cuda")
    gpu = TorchGpuMemory(model.release)
    gpu.free()
    allocated = torch.cuda.memory_allocated(0)
    reserved = torch.cuda.memory_reserved(0)
    used = []
    for _ in range(2):
        try:
            reply = model.complete(
                SYSTEM_PROMPT, "Hola, soy Marta y coordino el proyecto.", 32
            )
        except ModelLoadError as exc:
            gpu.free()
            pytest.skip(f"the model cannot run now: {exc}")
        assert isinstance(reply, str)
        used.append(torch.cuda.memory_allocated(0))
        gpu.free()
        assert torch.cuda.memory_allocated(0) <= allocated + TOLERANCE
        assert torch.cuda.memory_reserved(0) <= reserved + TOLERANCE
    assert all(u > allocated + 2**30 for u in used)  # it really loaded
