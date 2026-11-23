"""The cloning worker cleans up when its input ends (#225)."""

import importlib.util
import io
import sys
from pathlib import Path
from typing import List

import pytest

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "lab" / "runners" / "voice_clone.py"


class Engine:
    """A fake engine that records being closed."""

    sample_rate = 24000

    def __init__(self, events: List[str]) -> None:
        """Keep the log."""
        self.events = events

    def close(self) -> None:
        """Record the close."""
        self.events.append("closed")


def test_the_worker_closes_its_engine_when_input_ends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = importlib.util.spec_from_file_location("voice_clone_w", PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    events: List[str] = []
    monkeypatch.setattr(module, "Chatterbox",
                        lambda *a, **k: Engine(events), raising=False)
    monkeypatch.setattr(module, "emit", lambda doc: None)
    monkeypatch.setattr(module.signal, "signal", lambda *a: None)
    monkeypatch.setattr(sys, "argv", ["w", "--engine", "chatterbox"])
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    monkeypatch.setattr(sys, "stdout", sys.stdout)
    assert module.main() == 0
    assert events == ["closed"]
