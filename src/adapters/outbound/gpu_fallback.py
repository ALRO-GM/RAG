"""Running out of GPU memory: recognise it and give the memory back (#110).

The workstation's GPU is shared with other processes, so a model that
fitted when the analysis started can find the card full a minute later.
The analysis' GPU models (the act tagger, the embeddings, the local
judge) then retry on the CPU and say so in their step's note
(:data:`CPU_FALLBACK_NOTE`) instead of failing the run.
"""

import gc

CUDA_OOM_MESSAGE = "CUDA out of memory"
"""What PyTorch's error says when the CUDA allocator gives up."""

CPU_FALLBACK_NOTE = (
    "CUDA ran out of memory: {what} ran on the CPU instead (slower)."
)
"""The step's note when a model moved to the CPU; ``{what}`` names it."""


def is_cuda_oom(exc: BaseException) -> bool:
    """Tell whether ``exc`` is the CUDA allocator running out of memory.

    PyTorch raises ``torch.OutOfMemoryError`` (a ``RuntimeError``);
    some libraries re-raise it as a plain ``RuntimeError`` with the
    same message, which counts too.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a dependency
        return False
    kind = getattr(torch, "OutOfMemoryError", None)
    if isinstance(kind, type) and isinstance(exc, kind):
        return True
    return isinstance(exc, RuntimeError) and CUDA_OOM_MESSAGE in str(exc)


def free_cuda_memory() -> None:
    """Drop unreferenced tensors and return PyTorch's cached GPU memory."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except (ImportError, RuntimeError):  # pragma: no cover - best effort
        pass


def cpu_fallback_note(what: str) -> str:
    """Return the note of a model that moved to the CPU."""
    return CPU_FALLBACK_NOTE.format(what=what)
