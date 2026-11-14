"""The lab runners: pieces, cross-fades, and steps that would never end.

The runners are scripts of the lab's own environments
(``lab/runners``); their helpers that need only numpy are loaded here
by file, as the runners load ``_common`` themselves.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
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

    @pytest.mark.parametrize("length", [
        1, 800, 1599, 16000, 16001, 48000, 48000 + 1600, 48000 + 1601,
        123457,
    ])
    def test_pieces_reassemble_to_the_original_length(
        self, length: int
    ) -> None:
        common = load("_common")
        samples = np.full(length, 0.25, dtype=np.float32)
        seen: List[int] = []

        def shorter(chunk: Any) -> Any:
            """Lose the last 10 samples, as a model with a hop may."""
            seen.append(len(chunk))
            return chunk[:-10]

        out = common.pieces(samples, 16000, 16000, shorter)
        assert out.shape == samples.shape
        assert all(n <= 16000 + 16000 for n in seen)

    def test_pieces_cross_fade_linearly(self) -> None:
        common = load("_common")
        rate, piece = 1000, 3000
        samples = np.zeros(10000, dtype=np.float32)
        starts: List[int] = []

        def constant(chunk: Any) -> Any:
            """Answer the piece's number everywhere (a fake enhancer)."""
            starts.append(len(starts))
            return np.full(len(chunk), float(len(starts)), np.float32)

        out = common.pieces(samples, piece, rate, constant)
        # pieces [0, 4000), [3000, 7000), [6000, 10000): 1 s overlaps
        assert len(starts) == 3
        assert np.allclose(out[:3000], 1.0)
        assert np.allclose(out[4000:6000], 2.0)
        assert np.allclose(out[7000:], 3.0)
        for first, value in ((3000, 1.0), (6000, 2.0)):
            fade = out[first:first + 1000]
            assert np.allclose(fade, value + np.linspace(0.0, 1.0, 1000),
                               atol=1e-5)
            assert np.all(np.diff(fade) > 0)
        assert np.all((out >= 1.0 - 1e-6) & (out <= 3.0 + 1e-6))


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


class FakeDeepFilterNet:
    """``df.enhance`` without a model: it halves what it is given."""

    def __init__(self) -> None:
        """Record the length of every call."""
        self.lengths: List[int] = []
        self.module = ModuleType("df.enhance")
        self.module.init_df = self.init_df  # type: ignore[attr-defined]
        self.module.enhance = self.enhance  # type: ignore[attr-defined]

    def init_df(self, **kwargs: Any) -> Any:
        """Return a model, a state at 48 kHz and nothing."""
        model = SimpleNamespace()
        model.to = lambda device: model
        model.eval = lambda: model
        return model, SimpleNamespace(sr=lambda: 48000), None

    def enhance(self, model: Any, state: Any, audio: Any,
                atten_lim_db: Any = None) -> Any:
        """Halve ``audio`` (a ``1 x samples`` tensor)."""
        self.lengths.append(int(audio.shape[-1]))
        return audio * 0.5


class TestDeepFilterNet:
    def test_the_recording_is_enhanced_in_pieces(
        self, keep_path: None, monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        soundfile = pytest.importorskip("soundfile")
        pytest.importorskip("torch")
        fake = FakeDeepFilterNet()
        monkeypatch.setitem(sys.modules, "df", ModuleType("df"))
        monkeypatch.setitem(sys.modules, "df.enhance", fake.module)
        monkeypatch.setitem(sys.modules, "torchaudio.backend.common",
                            ModuleType("torchaudio.backend.common"))
        enhance = load("enhance")
        rate = 16000
        times = np.arange(75 * rate) / rate
        samples = (0.4 * np.sin(2 * np.pi * 220 * times)).astype(np.float32)
        source = tmp_path / "in.wav"
        soundfile.write(str(source), samples, rate, subtype="PCM_16")
        out = tmp_path / "out"
        out.mkdir()
        context = enhance.Context(
            "denoise", "deepfilternet3", source, out,
            {"piece_seconds": 30.0}, "cpu",
        )
        fields = enhance.deepfilternet(context)
        assert fields == {"audio": "audio.wav", "sample_rate": 16000}
        assert context.notes == []
        # pieces start at 0, 30 and 60 s and reach 1 s into the next
        assert fake.lengths == [31 * 48000, 31 * 48000, 15 * 48000]
        written, written_rate = soundfile.read(str(out / "audio.wav"))
        assert written_rate == rate and len(written) == len(samples)
        middle = slice(rate, len(samples) - rate)
        assert np.allclose(written[middle], 0.5 * samples[middle],
                           atol=2e-3)
