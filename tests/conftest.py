"""Shared fixtures: sample documents, in-memory chunks and a tiny corpus."""

import signal
from pathlib import Path
from types import FrameType
from typing import Dict, Iterator, List, Optional

import pytest

from src.domain.models import Chunk
from tests.fakes import make_chunk
from tests.samples import CORPUS, write_corpus


@pytest.fixture()
def corpus_files() -> Dict[str, str]:
    """Return a fresh copy of the sample corpus ``{path: text}``."""
    return dict(CORPUS)


@pytest.fixture()
def chunks() -> List[Chunk]:
    """Return a handful of distinct chunks about different topics."""
    return [
        make_chunk("a.py", 0, "def load_lora_adapter(name): pass",
                   "def load_lora_adapter", "python"),
        make_chunk("a.py", 40, "class KVCacheManager: allocate blocks",
                   "class KVCacheManager", "python"),
        make_chunk("b.md", 0, "FP8 quantization reduces memory usage",
                   "Serving > Quantization"),
        make_chunk("b.md", 50, "Speculative decoding with a draft model",
                   "Serving > Speculative"),
    ]


@pytest.fixture()
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write the sample corpus under ``tmp_path`` and chdir into it."""
    write_corpus(tmp_path)
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture(autouse=True)
def isolated_gpu_lease(
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    """Point ``XDG_RUNTIME_DIR`` at an empty directory for every test.

    The GPU lease lives in ``$XDG_RUNTIME_DIR/gpu-lease``: a test (or a
    subprocess it starts, which inherits the environment) must never read
    the live lease of a real session (#200). Tests that need a lease
    still pass their own ``environ``.
    """
    runtime = tmp_path_factory.mktemp("xdg-runtime")
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
    yield


@pytest.fixture()
def hard_deadline() -> Iterator[None]:
    """Fail the test (instead of hanging the run) after 60 seconds.

    Uses ``SIGALRM`` in the main thread; a no-op where it is missing.
    """
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def expired(signum: int, frame: Optional[FrameType]) -> None:
        raise TimeoutError("test exceeded its 60 s hard deadline")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.alarm(60)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
