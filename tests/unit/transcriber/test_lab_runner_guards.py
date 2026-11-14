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


RATE = 16000


def recording(tmp_path: Path, speech: bool) -> Path:
    """Write 6 s of synthetic audio: faint noise, with a tone if speech.

    The noise alone stands for a recording with no speaker in it (the
    CALLHOME calls of #48); the tone from 2 to 4 s for a voice.
    """
    soundfile = pytest.importorskip("soundfile")
    rng = np.random.default_rng(48)
    samples = (0.001 * rng.standard_normal(6 * RATE)).astype(np.float32)
    if speech:
        times = np.arange(2 * RATE) / RATE
        samples[2 * RATE:4 * RATE] += (
            0.3 * np.sin(2 * np.pi * 200 * times)
        ).astype(np.float32)
    path = tmp_path / ("speech.wav" if speech else "silence.wav")
    soundfile.write(str(path), samples, RATE, subtype="PCM_16")
    return path


def active_seconds(path: Path) -> List[bool]:
    """Tell, second by second, whether ``path`` is loud (a fake VAD)."""
    import soundfile

    samples, rate = soundfile.read(str(path), dtype="float32")
    seconds = samples[: len(samples) // rate * rate].reshape(-1, rate)
    return [bool(v) for v in np.sqrt((seconds ** 2).mean(axis=1)) > 0.01]


class FakeTurn:
    """A ``pyannote.core.Segment``: start and end in seconds."""

    def __init__(self, start: float, end: float) -> None:
        """Keep the bounds."""
        self.start = start
        self.end = end


class FakeAnnotation:
    """A ``pyannote.core.Annotation`` of labelled turns."""

    def __init__(self, turns: List[Any]) -> None:
        """Keep ``(start, end, label)`` triples."""
        self.turns = turns

    def itertracks(self, yield_label: bool = False) -> Iterator[Any]:
        """Yield ``(turn, track, label)`` as pyannote does."""
        for start, end, label in self.turns:
            yield FakeTurn(start, end), "_", label


class FakeDiariZen:
    """DiariZen's pipeline without a model, down to its reconstruction.

    A chunk is one second with two local speakers; the first is active
    when the second is loud, the other never is. Inactive local
    speakers get cluster -2 and ``reconstruct`` sizes its array with
    ``max(cluster) + 1`` clusters, as the fork does: with no speaker
    anywhere, numpy refuses a negative dimension.
    """

    @classmethod
    def from_pretrained(cls, repo: str) -> "FakeDiariZen":
        """Return a fresh pipeline (no weights)."""
        return cls()

    def __call__(self, wav: str, sess_name: Any = None) -> FakeAnnotation:
        """Diarize ``wav``: segment, cluster, reconstruct."""
        active = active_seconds(Path(wav))
        hard_clusters = np.array(
            [[0 if loud else -2, -2] for loud in active], dtype=np.int8
        )
        result: FakeAnnotation = self.reconstruct(None, hard_clusters, None)
        return result

    def reconstruct(self, segmentations: Any, hard_clusters: Any,
                    count: Any) -> FakeAnnotation:
        """Turn the chunks of cluster 0 into turns of speaker 0."""
        np.zeros((len(hard_clusters), 1, int(np.max(hard_clusters)) + 1))
        return FakeAnnotation([
            (float(n), float(n + 1), 0)
            for n, row in enumerate(hard_clusters) if row[0] == 0
        ])


class TestDiariZenWithoutSpeech:
    @pytest.fixture()
    def diarizen(
        self, keep_path: None, monkeypatch: pytest.MonkeyPatch
    ) -> Any:
        """The runner, with the fake pipeline and no torchaudio shims."""
        inference = ModuleType("diarizen.pipelines.inference")
        inference.DiariZenPipeline = (  # type: ignore[attr-defined]
            FakeDiariZen
        )
        for name in ("diarizen", "diarizen.pipelines"):
            monkeypatch.setitem(sys.modules, name, ModuleType(name))
        monkeypatch.setitem(sys.modules, inference.__name__, inference)
        module = load("diarizen")
        monkeypatch.setattr(module, "_shims", lambda: None)
        monkeypatch.setattr(module._common, "hf_revision",
                            lambda repo: "rev")
        return module

    def run(self, diarizen: Any, source: Path) -> Any:
        """Diarize ``source`` with the runner; return context and fields."""
        out = source.parent / "out"
        out.mkdir(exist_ok=True)
        context = diarizen.Context(
            "diarize", "diarizen-wavlm-large-s80-md-v2", source, out, {},
            "cpu",
        )
        return context, diarizen.diarize(context)

    def test_the_fork_fails_on_a_recording_without_speech(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match="negative dimensions"):
            FakeDiariZen()(str(recording(tmp_path, speech=False)))

    def test_no_speech_gives_no_segments_and_a_note(
        self, diarizen: Any, tmp_path: Path
    ) -> None:
        context, fields = self.run(diarizen, recording(tmp_path, False))
        assert fields == {"segments": [], "overlap": []}
        assert diarizen.NO_SPEECH in context.notes

    def test_speech_takes_the_pipeline_path_unchanged(
        self, diarizen: Any, tmp_path: Path
    ) -> None:
        source = recording(tmp_path, speech=True)
        expected = [
            {"start": turn.start, "end": turn.end, "speaker": f"spk{label}"}
            for turn, _, label in FakeDiariZen()(str(source)).itertracks(
                yield_label=True
            )
        ]
        context, fields = self.run(diarizen, source)
        assert [s["start"] for s in expected] == [2.0, 3.0]
        assert fields["segments"] == expected
        assert diarizen.NO_SPEECH not in context.notes


class FakeClusteringDiarizer:
    """NeMo's clustering diarizer without models, VAD to RTTM.

    The VAD writes one manifest line per second (``duration`` 0 for a
    quiet one), then aborts as NeMo does when no line has speech;
    otherwise every loud second is ``speaker_0``. ``fail_after_vad``
    raises another ``ValueError`` once the VAD found speech, to check
    that it is not taken for silence.
    """

    fail_after_vad = False

    def __init__(self, cfg: Any) -> None:
        """Keep the configuration (a plain dict here)."""
        self.cfg = cfg

    def diarize(self) -> None:
        """Run the fake VAD, then write the RTTM."""
        import json

        params = self.cfg["diarizer"]
        out = Path(params["out_dir"])
        line = Path(params["manifest_filepath"]).read_text().splitlines()[0]
        active = active_seconds(Path(json.loads(line)["audio_filepath"]))
        vad = out / "vad_outputs"
        vad.mkdir(parents=True)
        (vad / "vad_out.json").write_text("".join(
            json.dumps({"uniq_id": "audio", "offset": float(n),
                        "duration": 1.0 if loud else 0.0}) + "\n"
            for n, loud in enumerate(active)
        ))
        if not any(active):
            raise ValueError("All files present in manifest contains"
                             " silence, aborting next steps")
        if self.fail_after_vad:
            raise ValueError("something else went wrong")
        rttm = out / "pred_rttms"
        rttm.mkdir()
        (rttm / "audio.rttm").write_text("".join(
            f"SPEAKER audio 1 {n:.3f} 1.000 <NA> <NA> speaker_0 <NA> <NA>\n"
            for n, loud in enumerate(active) if loud
        ))


class TestNemoClusteringWithoutSpeech:
    @pytest.fixture()
    def nemo(self, keep_path: None, monkeypatch: pytest.MonkeyPatch) -> Any:
        """The runner, with the fake diarizer and a dict for OmegaConf."""
        models = ModuleType("nemo.collections.asr.models")
        models.ClusteringDiarizer = (  # type: ignore[attr-defined]
            FakeClusteringDiarizer
        )
        for name in ("nemo", "nemo.collections", "nemo.collections.asr"):
            monkeypatch.setitem(sys.modules, name, ModuleType(name))
        monkeypatch.setitem(sys.modules, models.__name__, models)
        omegaconf = ModuleType("omegaconf")
        omegaconf.OmegaConf = SimpleNamespace(  # type: ignore[attr-defined]
            create=lambda value: value
        )
        monkeypatch.setitem(sys.modules, "omegaconf", omegaconf)
        monkeypatch.setattr(FakeClusteringDiarizer, "fail_after_vad", False)
        return load("nemo")

    def run(self, nemo: Any, source: Path) -> Any:
        """Diarize ``source`` with the runner; return context and fields."""
        out = source.parent / "out"
        out.mkdir(exist_ok=True)
        work = source.parent / "work"
        work.mkdir(exist_ok=True)
        context = nemo.Context("diarize", "nemo-clustering", source, out,
                               {}, "cpu")
        return context, nemo._clustered(context, work)

    def test_no_speech_gives_no_segments_and_a_note(
        self, nemo: Any, tmp_path: Path
    ) -> None:
        context, fields = self.run(nemo, recording(tmp_path, False))
        assert fields == {"segments": [], "overlap": []}
        assert context.notes == [nemo.NO_SPEECH]

    def test_speech_is_diarized_as_before(
        self, nemo: Any, tmp_path: Path
    ) -> None:
        context, fields = self.run(nemo, recording(tmp_path, True))
        assert fields["segments"] == [
            {"start": 2.0, "end": 3.0, "speaker": "speaker_0"},
            {"start": 3.0, "end": 4.0, "speaker": "speaker_0"},
        ]
        assert context.notes == []

    def test_another_failure_after_speech_is_raised(
        self, nemo: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(FakeClusteringDiarizer, "fail_after_vad", True)
        with pytest.raises(ValueError, match="something else"):
            self.run(nemo, recording(tmp_path, True))

    def test_speech_found_reads_the_vad_manifest(
        self, nemo: Any, tmp_path: Path
    ) -> None:
        manifest = tmp_path / "vad_out.json"
        assert nemo.speech_found(manifest)  # the VAD did not finish
        manifest.write_text('{"duration": 0}\n\n{"duration": 0.0}\n')
        assert not nemo.speech_found(manifest)
        manifest.write_text('{"duration": 0}\n{"duration": 0.5}\n')
        assert nemo.speech_found(manifest)
