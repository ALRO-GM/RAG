"""GPU memory adapter: frees the models and tells who holds the GPU.

Implements :class:`~transcriber.application.ports.GpuMemory` (#117).
:meth:`TorchGpuMemory.free` unloads every model of the service (the
composition root passes its ``release_models``), collects what was
freed and returns PyTorch's cached GPU memory to the driver; the job
runner calls it after every job, so an idle service holds no model.
:meth:`TorchGpuMemory.holders` lists the processes on the GPU (from
``nvidia-smi``) for the message of a stage that ran out of memory twice.

Torch is never imported here: if no model loaded it, this process holds
no GPU memory, and there is nothing to ask it.
"""

import gc
import logging
import os
import subprocess
import sys
from pathlib import PurePath
from types import ModuleType
from typing import Callable, List, Optional, Sequence

from transcriber.adapters.outbound.torch_runtime import empty_cuda_cache

LOG = logging.getLogger(__name__)

OUT_OF_MEMORY_MARKERS = (
    "CUDA out of memory",
    "CUBLAS_STATUS_ALLOC_FAILED",
    "CUDA error: out of memory",
)
"""Texts of the errors CUDA gives when the GPU has no memory left."""

NVIDIA_SMI = (
    "nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
    "--format=csv,noheader,nounits",
)
"""Lists the processes on the GPU: pid, program, MiB used."""

GB = 1e9


def _torch() -> Optional[ModuleType]:
    """Return torch if a model already imported it, else ``None``."""
    return sys.modules.get("torch")


def run_nvidia_smi() -> str:
    """Return what :data:`NVIDIA_SMI` prints ("" if it cannot run)."""
    try:
        done = subprocess.run(
            NVIDIA_SMI, capture_output=True, text=True, timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return done.stdout if done.returncode == 0 else ""


def gpu_processes(listing: str, own_pid: int) -> List[str]:
    """Return ``"<program> (pid <n>): <MiB> MiB"`` per line of ``listing``.

    Only the program's file name is kept (not its arguments); this
    process is called ``this service``. Lines that do not parse are
    skipped.
    """
    found: List[str] = []
    for line in listing.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3 or not parts[0].isdigit():
            continue
        pid, used = int(parts[0]), parts[-1]
        words = ",".join(parts[1:-1]).split()
        program = PurePath(words[0]).name if words else "?"
        who = "this service" if pid == own_pid else program
        found.append(f"{who} (pid {pid}): {used} MiB")
    return found


class TorchGpuMemory:
    """The GPU memory seen through PyTorch and ``nvidia-smi``.

    Args:
        release: Unloads every model of the service.
        listing: Returns the ``nvidia-smi`` listing of the GPU's
            processes (tests replace it).
        torch_module: Returns torch when it is in use (tests replace
            it).
        own_pid: This process's id.
    """

    def __init__(
        self,
        release: Callable[[], None] = lambda: None,
        listing: Callable[[], str] = run_nvidia_smi,
        torch_module: Callable[[], Optional[ModuleType]] = _torch,
        own_pid: Optional[int] = None,
    ) -> None:
        """Keep the callbacks; nothing is measured yet."""
        self._release = release
        self._listing = listing
        self._torch_module = torch_module
        self._own_pid = os.getpid() if own_pid is None else own_pid

    def out_of_memory(self, error: BaseException) -> bool:
        """Tell whether ``error`` (or what caused it) is CUDA out of memory.

        A model that fails to load because of it raises another error
        caused by it, so the whole chain is looked at.
        """
        torch = self._torch_module()
        kinds: Sequence[type] = ()
        if torch is not None:
            kinds = tuple(
                k for k in (
                    getattr(torch, "OutOfMemoryError", None),
                    getattr(getattr(torch, "cuda", None),
                            "OutOfMemoryError", None),
                ) if isinstance(k, type)
            )
        seen = set()
        current: Optional[BaseException] = error
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if isinstance(current, tuple(kinds)) or any(
                marker in str(current) for marker in OUT_OF_MEMORY_MARKERS
            ):
                return True
            current = current.__cause__ or current.__context__
        return False

    def free(self) -> None:
        """Unload the models, collect them, return the cached memory."""
        debug = LOG.isEnabledFor(logging.DEBUG)
        if debug:
            LOG.debug("GPU memory before freeing: %s", self.usage())
        try:
            self._release()
        finally:
            gc.collect()
            empty_cuda_cache(self._torch_module())
            if debug:
                LOG.debug("GPU memory after freeing: %s", self.usage())

    def usage(self) -> str:
        """Return this process's allocated and reserved GPU memory, and
        the GPU's free and total memory."""
        torch = self._torch_module()
        if torch is None or not torch.cuda.is_available():
            return "no CUDA in use"
        try:
            free, total = torch.cuda.mem_get_info(0)
            allocated = torch.cuda.memory_allocated(0)
            reserved = torch.cuda.memory_reserved(0)
        except RuntimeError as exc:  # a broken CUDA context
            return f"unknown ({exc})"
        return (
            f"this process {allocated / GB:.2f} GB allocated,"
            f" {reserved / GB:.2f} GB reserved; GPU {free / GB:.2f} of"
            f" {total / GB:.2f} GB free"
        )

    def holders(self) -> str:
        """Return the processes on the GPU and this process's memory."""
        processes = gpu_processes(self._listing(), self._own_pid)
        listed = "; ".join(processes) if processes else (
            "the processes cannot be listed"
        )
        return f"{listed} ({self.usage()})"
