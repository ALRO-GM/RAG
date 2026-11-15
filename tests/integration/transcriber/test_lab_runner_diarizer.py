"""The lab's diarizers as the transcription's diarizer (TR-18)."""

import wave
from pathlib import Path
from typing import List

import numpy as np
import pytest

from tests.integration.transcriber.lab_sandbox import make_lab
from transcriber.adapters.outbound.lab_runner_diarizer import (
    LabRunnerDiarizer,
    write_wav,
)
from transcriber.adapters.outbound.subprocess_lab_tool import (
    FileSystemLabEnvironment,
    JsonCatalogue,
    SubprocessLabTool,
)
from transcriber.application.ports import Audio
from transcriber.bootstrap import (
    DIARIZE_DIR,
    build_diarizer,
    diarizer_problems,
)
from transcriber.config import LabSettings, TranscriberSettings
from transcriber.domain.diarization import BUILTIN_RULE, DiarizerRule
from transcriber.domain.errors import LabStepError


def audio(seconds: float = 4.0) -> Audio:
    """A quiet tone of ``seconds`` at 16 kHz."""
    t = np.arange(int(seconds * 16000), dtype=np.float32) / 16000
    return Audio((0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32),
                 16000)


@pytest.fixture()
def lab(tmp_path: Path) -> LabSettings:
    """A lab with the fake runner."""
    return make_lab(tmp_path / "lab")


def diarizer(
    lab: LabSettings, work: Path, calls: List[str],
    environ: "dict[str, str] | None" = None,
) -> LabRunnerDiarizer:
    """The adapter over the sandbox; ``calls`` records ``before_run``."""
    env = environ if environ is not None else {"HOME": str(work / "home")}
    environment = FileSystemLabEnvironment(
        lab.envs_dir, lab.runners_dir, env
    )
    return LabRunnerDiarizer(
        SubprocessLabTool(environment, env, timeout=10.0), environment,
        JsonCatalogue(lab.catalogue), work / "diarize",
        lambda: calls.append("released"),
    )


class TestLabRunnerDiarizer:
    def test_runs_the_runner_and_cleans_up(
        self, lab: LabSettings, tmp_path: Path
    ) -> None:
        calls: List[str] = []
        adapter = diarizer(lab, tmp_path, calls)
        assert adapter.problem("fake-diarize") is None
        found = adapter.diarize("fake-diarize", audio(), 2)
        assert [s.speaker for s in found.segments] == ["A", "B"]
        assert found.model == "fake-diarize"
        assert found.licence == "MIT" and found.commercial is True
        assert found.version == "fake==1"
        assert calls == ["released"]
        # the audio and the runner's output are gone
        assert list((tmp_path / "diarize").iterdir()) == []

    def test_a_relative_work_dir_reaches_the_runner(
        self, lab: LabSettings, tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # The default data directory is relative; the runner starts in
        # its own workspace and must still find the audio.
        monkeypatch.chdir(tmp_path)
        env = {"HOME": str(tmp_path / "home")}
        environment = FileSystemLabEnvironment(
            lab.envs_dir, lab.runners_dir, env
        )
        adapter = LabRunnerDiarizer(
            SubprocessLabTool(environment, env, timeout=10.0), environment,
            JsonCatalogue(lab.catalogue), Path("data/diarize"),
        )
        found = adapter.diarize("fake-diarize", audio(), None)
        assert [s.speaker for s in found.segments] == ["A", "B"]

    def test_licence_terms_come_from_the_catalogue(
        self, lab: LabSettings, tmp_path: Path
    ) -> None:
        found = diarizer(lab, tmp_path, []).diarize(
            "fake-split", audio(), None
        )
        assert found.commercial is False

    def test_problems(self, lab: LabSettings, tmp_path: Path) -> None:
        adapter = diarizer(lab, tmp_path, [])
        assert "No diarize model" in str(adapter.problem("nope"))
        assert "runs in the service" in str(adapter.problem("builtin"))
        assert "not installed" in str(adapter.problem("missing-model"))
        assert "gated" in str(adapter.problem("fake-gated"))
        with_token = diarizer(
            lab, tmp_path, [],
            {"HOME": str(tmp_path / "home"), "HF_TOKEN": "hf_x"},
        )
        assert with_token.problem("fake-gated") is None

    def test_a_failing_runner_raises(
        self, lab: LabSettings, tmp_path: Path
    ) -> None:
        adapter = diarizer(lab, tmp_path, [])
        with pytest.raises(LabStepError):
            adapter.diarize("fake-crash", audio(), None)
        assert list((tmp_path / "diarize").iterdir()) == []

    def test_write_wav(self, tmp_path: Path) -> None:
        path = tmp_path / "a.wav"
        write_wav(path, audio(1.0))
        with wave.open(str(path), "rb") as reader:
            assert (reader.getnchannels(), reader.getsampwidth(),
                    reader.getframerate(), reader.getnframes()) == (
                1, 2, 16000, 16000)


class TestComposition:
    def test_fake_runtime_uses_the_builtin_grouping(
        self, tmp_path: Path
    ) -> None:
        settings = TranscriberSettings(data_dir=tmp_path, runtime="fake")
        assert build_diarizer(settings, {}) == (None, BUILTIN_RULE)

    def test_a_builtin_only_rule_needs_no_diarizer(
        self, tmp_path: Path
    ) -> None:
        settings = TranscriberSettings(
            data_dir=tmp_path,
            diarizer=DiarizerRule(model="builtin", by_speakers={}),
        )
        assert build_diarizer(settings, {}) == (None, BUILTIN_RULE)

    def test_real_runtime_runs_the_rule_models(
        self, lab: LabSettings, tmp_path: Path
    ) -> None:
        rule = DiarizerRule(
            model="fake-diarize", by_speakers={2: "missing-model"}
        )
        settings = TranscriberSettings(
            data_dir=tmp_path / "data", lab=lab, diarizer=rule,
        )
        built, got = build_diarizer(settings, {"HOME": str(tmp_path)})
        assert isinstance(built, LabRunnerDiarizer) and got == rule
        problems = diarizer_problems(built, rule)
        assert list(problems) == ["missing-model"]
        built.diarize("fake-diarize", audio(), None)
        assert (tmp_path / "data" / DIARIZE_DIR).is_dir()
        assert diarizer_problems(None, rule) == {}
        assert diarizer_problems(
            built, DiarizerRule(model="builtin", by_speakers={})
        ) == {}

    def test_default_settings_pick_diarizen_and_nemotron(self) -> None:
        rule = TranscriberSettings().diarizer
        assert rule.model_for(None) == "diarizen-wavlm-large-s80-md-v2"
        assert rule.model_for(2) == "nemotron-3-diarization"
