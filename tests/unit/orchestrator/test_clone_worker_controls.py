"""The cloning worker turns a request's Piper controls into flags (#211)."""

import importlib.util
from pathlib import Path
from types import ModuleType

WORKER = Path(__file__).resolve().parents[3] / "lab" / "runners" / \
    "voice_clone.py"


def load() -> ModuleType:
    """Import the worker's file (it is not a package)."""
    spec = importlib.util.spec_from_file_location("voice_clone_worker",
                                                  WORKER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_controls_become_piper_flags() -> None:
    flags = load().piper_flags
    assert flags({}) == ()
    assert flags({"length_scale": 1.0}) == (), "Piper's own default"
    assert flags({"length_scale": 1.4, "noise_w": 0.5, "text": "x",
                  "noise_scale": 0.4, "sentence_silence": 0.2}) == (
        "--length-scale", "1.4", "--noise-scale", "0.4",
        "--noise-w-scale", "0.5", "--sentence-silence", "0.2")
    assert flags({"length_scale": True, "noise_w": "x"}) == ()
