"""Free GPU memory, read from ``nvidia-smi`` (LV-8).

Implements :class:`~transcriber.application.ports.FreeGpuMemory`. The
card's own count (``nvidia-smi --query-gpu=memory.free``) sees every
process, which is what the live session must wait for; when the tool
is missing it falls back to ``torch.cuda.mem_get_info`` if torch is
already imported (it is never imported here just for this). ``None``
means no GPU could be read.
"""

import subprocess
import sys
from typing import Callable, Optional, Sequence

QUERY: Sequence[str] = (
    "nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits",
)
MIB = 1024 * 1024


def run_query() -> str:
    """Return what :data:`QUERY` prints (``""`` if it cannot run)."""
    try:
        done = subprocess.run(
            list(QUERY), capture_output=True, text=True, timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return done.stdout if done.returncode == 0 else ""


def torch_free_mib() -> Optional[int]:
    """Return CUDA's free memory via torch, if torch is loaded."""
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    try:  # pragma: no cover - needs a CUDA device
        if not torch.cuda.is_available():
            return None
        free, _total = torch.cuda.mem_get_info()
        return int(free // MIB)
    except Exception:  # pragma: no cover - driver errors
        return None


def parse_free(listing: str) -> Optional[int]:
    """Return the free MiB of the first GPU in ``listing``, if any."""
    for line in listing.splitlines():
        text = line.strip()
        if text.isdigit():
            return int(text)
    return None


class NvidiaSmiFreeMemory:
    """Free memory of the first GPU."""

    def __init__(
        self,
        query: Callable[[], str] = run_query,
        fallback: Callable[[], Optional[int]] = torch_free_mib,
    ) -> None:
        """Keep the query (tests give a fake) and the fallback."""
        self._query = query
        self._fallback = fallback

    def free_mib(self) -> Optional[int]:
        """Return the free memory of the first GPU in MiB, or ``None``."""
        free = parse_free(self._query())
        return free if free is not None else self._fallback()
