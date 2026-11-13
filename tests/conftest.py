"""Shared fixtures: sample documents, in-memory chunks and a tiny corpus."""

from pathlib import Path
from typing import Dict, List

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
