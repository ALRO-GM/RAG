"""The lab runners refuse a step that would never end.

The runners are scripts of the lab's own environments
(``lab/runners``); their helpers that need only numpy are loaded here
by file, as the runners load ``_common`` themselves.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator, List

import numpy as np
import pytest

RUNNERS = Path(__file__).resolve().parents[3] / "lab" / "runners"


def load(name: str) -> ModuleType:
    """Load ``lab/runners/<name>.py`` as a module of its own."""
    spec = importlib.util.spec_from_file_location(
        f"lab_runner_{name}", RUNNERS / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def keep_path() -> Iterator[None]:
    """Restore ``sys.path`` and ``_common`` after a runner is loaded."""
    path = list(sys.path)
    common = sys.modules.get("_common")
    yield
    sys.path[:] = path
    if common is None:
        sys.modules.pop("_common", None)
    else:
        sys.modules["_common"] = common


class TestPieces:
    def test_a_piece_below_one_sample_is_refused(self) -> None:
        common = load("_common")
        calls: List[Any] = []
        for piece in (0, -160):
            with pytest.raises(common.RunnerError) as caught:
                common.pieces(np.zeros(32000, dtype=np.float32), piece,
                              16000, calls.append)
            assert caught.value.code == "bad_input"
        assert calls == []

    def test_pieces_cover_the_audio(self) -> None:
        common = load("_common")
        samples = np.linspace(-0.5, 0.5, 40000).astype(np.float32)
        out = common.pieces(samples, 16000, 16000, lambda chunk: chunk)
        assert np.allclose(out, samples, atol=1e-5)


class TestWindows:
    @pytest.mark.parametrize("length, hop", [
        (1.5, 0.0), (1.5, 0.005), (1.5, -0.75), (0.005, 0.75),
    ])
    def test_a_step_or_window_below_one_frame_is_refused(
        self, keep_path: None, length: float, hop: float
    ) -> None:
        speechbrain = load("speechbrain")
        mask = np.ones(500, dtype=bool)
        with pytest.raises(speechbrain.RunnerError) as caught:
            speechbrain._windows(mask, length, hop)
        assert caught.value.code == "bad_input"

    def test_windows_step_through_speech(self, keep_path: None) -> None:
        speechbrain = load("speechbrain")
        mask = np.zeros(500, dtype=bool)
        mask[100:400] = True
        spans = speechbrain._windows(mask, 1.5, 0.75)
        assert spans[0] == (100, 250)
        assert spans[1] == (175, 325)
        assert spans[-1] == (250, 400)
