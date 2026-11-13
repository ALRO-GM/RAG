"""Sample corpus documents and helpers shared by the test suite."""

import json
from pathlib import Path
from typing import Dict

PY_FILE = '''"""Module docstring."""

import os


def load_lora_adapter(name: str) -> str:
    """Load a LoRA adapter dynamically."""
    return os.path.join("adapters", name)


class KVCacheManager:
    """Manage the paged KV cache blocks."""

    def allocate_blocks(self, n: int) -> int:
        """Allocate n blocks."""
        return n
'''

MD_FILE = """# Serving

Intro text about the OpenAI compatible server.

## LoRA

Use the `/v1/load_lora_adapter` endpoint to load adapters at runtime.

## Quantization

FP8 quantization reduces memory usage.
"""

TXT_FILE = """Speculative decoding uses a draft model to propose tokens.

The target model verifies the proposed tokens in a single pass.
"""

RAW = "data/raw/proj"
PY_PATH = f"{RAW}/lora.py"
MD_PATH = f"{RAW}/docs/serving.md"
TXT_PATH = f"{RAW}/notes.txt"

CORPUS: Dict[str, str] = {
    PY_PATH: PY_FILE,
    MD_PATH: MD_FILE,
    TXT_PATH: TXT_FILE,
}


def write_json(path: Path, data: object) -> str:
    """Write ``data`` as JSON to ``path`` and return the path as text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def write_corpus(root: Path) -> None:
    """Write the sample corpus under ``root``.

    An ignored ``.json`` file and a hidden directory are added so that
    extension and hidden-directory filtering can be checked.
    """
    for rel, text in CORPUS.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    (root / RAW / "config.json").write_text("{}", encoding="utf-8")
    hidden = root / RAW / ".hidden"
    hidden.mkdir()
    (hidden / "secret.py").write_text("x = 1\n", encoding="utf-8")
