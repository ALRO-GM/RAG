"""The lab's outbound adapters: storage, subprocess runner, environment."""

import json
import os
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pytest

from tests.integration.transcriber.lab_sandbox import make_lab
from transcriber.adapters.outbound.fs_lab_repository import (
    FileSystemLabRepository,
    check_wav,
    read_wav,
)
from transcriber.adapters.outbound.subprocess_lab_tool import (
    READY_MARKER,
    FileSystemLabEnvironment,
    JsonCatalogue,
    OsProcessTable,
    SubprocessLabTool,
    hub_dir,
    parse_result,
    read_token,
    token_file,
)
from transcriber.application.lab_ports import ToolRequest
from transcriber.application.ports import Audio
from transcriber.config import LabSettings
from transcriber.domain.errors import (
    InvalidInputError,
    LabStepError,
    NotFoundError,
    StorageError,
)
from transcriber.domain.lab import (
    LabRequest,
    LabRun,
    ReferenceLabel,
    SpeakerMapping,
    StageOutcome,
)

NOW = datetime(2026, 11, 14, tzinfo=timezone.utc)
KEY = "a" * 40


def audio(seconds: float = 2.0, rate: int = 16000) -> Audio:
    """A 220 Hz tone."""
    t = np.arange(int(seconds * rate)) / rate
    return Audio((0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32),
                 rate)


def write_wav(path: Path, rate: int = 16000, channels: int = 1) -> None:
    """A short silent WAV."""
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes(b"\x00\x00" * channels * 160)


class TestRepository:
    def test_original_audio(self, tmp_path: Path) -> None:
        repo = FileSystemLabRepository(tmp_path)
        assert repo.original_key("s1") is None
        key = repo.store_original("s1", audio())
        assert repo.store_original("s1", audio()) == key
        assert repo.original_key("s1") == key
        loaded = repo.load_audio("s1", key)
        assert loaded.sample_rate == 16000
        assert loaded.duration == pytest.approx(2.0)
        assert np.abs(loaded.samples - audio().samples).max() < 1e-3
        playback = repo.playback("s1", key)
        assert playback.size == 44 + 2 * 32000
        playback.close()
        resampled = repo.store_original("s2", audio(1.0, 8000))
        assert repo.load_audio("s2", resampled).duration == pytest.approx(1)
        with pytest.raises(NotFoundError):
            repo.audio_path("s1", "b" * 40)
        with pytest.raises(InvalidInputError):
            repo.audio_path("s1", "../x")
        with pytest.raises(InvalidInputError):
            repo.store_original("../s", audio())
        repo.audio_path("s1", key).unlink()
        assert repo.original_key("s1") is None

    def test_stages_are_kept_whole(self, tmp_path: Path) -> None:
        repo = FileSystemLabRepository(tmp_path)
        work = repo.new_workspace("s1")
        (work / "runner.log").write_text("log")
        kept = repo.keep("s1", KEY, StageOutcome(
            stage="diarize", model="z", seconds=1.0,
        ), work)
        assert kept.model == "z" and not work.exists()
        assert repo.outcome("s1", KEY) == kept
        stage = tmp_path / "lab" / "s1" / "stages" / KEY
        assert (stage / "runner.log").read_text() == "log"
        again = repo.new_workspace("s1")
        repo.keep("s1", KEY, kept, again)  # already there: dropped
        assert not again.exists()
        assert repo.outcome("s1", "b" * 40) is None
        (stage / "outcome.json").write_text("{\"stage\": 3}")
        with pytest.raises(StorageError):
            repo.outcome("s1", KEY)

    def test_a_denoiser_output_becomes_audio(self, tmp_path: Path) -> None:
        repo = FileSystemLabRepository(tmp_path)
        work = repo.new_workspace("s1")
        write_wav(work / "audio.wav")
        kept = repo.keep("s1", KEY, StageOutcome(stage="denoise",
                                                 model="dn"), work)
        assert kept.audio_key == KEY
        assert repo.load_audio("s1", KEY).duration == pytest.approx(0.01)
        repo.audio_path("s1", KEY).unlink()
        assert repo.outcome("s1", KEY) is None  # its audio is gone

    @pytest.mark.parametrize("rate, channels", [(48000, 1), (16000, 2)])
    def test_a_denoiser_must_write_16k_mono(
        self, tmp_path: Path, rate: int, channels: int
    ) -> None:
        repo = FileSystemLabRepository(tmp_path)
        work = repo.new_workspace("s1")
        write_wav(work / "audio.wav", rate, channels)
        with pytest.raises(LabStepError, match="16 kHz mono"):
            repo.keep("s1", KEY, StageOutcome(stage="denoise", model="d"),
                      work)
        with pytest.raises(LabStepError, match="no readable WAV"):
            check_wav(work / "missing.wav")
        repo.discard("s1", work)
        assert not work.exists()

    def test_runs_cancel_results_mappings_labels(
        self, tmp_path: Path
    ) -> None:
        repo = FileSystemLabRepository(tmp_path)
        older = LabRun(id="1" * 32, session_id="s1", created_at=NOW,
                       request=LabRequest())
        newer = older.model_copy(update={
            "id": "2" * 32, "created_at": NOW.replace(hour=5),
        })
        repo.save_run(older)
        repo.save_run(newer)
        (tmp_path / "lab" / "s1" / "runs" / "notes.json").write_text("{}")
        assert [r.id for r in repo.runs("s1")] == [newer.id, older.id]
        assert repo.run("s1", "3" * 32) is None
        with pytest.raises(InvalidInputError):
            repo.run("s1", "../../x")
        assert not repo.cancel_requested("s1", older.id)
        repo.request_cancel("s1", older.id)
        assert repo.cancel_requested("s1", older.id)
        repo.clear_cancel("s1", older.id)
        repo.clear_cancel("s1", older.id)
        assert not repo.cancel_requested("s1", older.id)
        assert repo.results("s1", older.id) is None
        repo.save_results("s1", older.id, {"a": 1})
        assert repo.results("s1", older.id) == {"a": 1}
        mapping = [SpeakerMapping(label="A", speaker="ana",
                                  method="voiceprint", similarity=0.9)]
        assert repo.mapping("s1", KEY) is None
        repo.save_mapping("s1", KEY, mapping)
        assert repo.mapping("s1", KEY) == mapping
        assert repo.labels("s1") == []
        labels = [ReferenceLabel(start=0, end=1, kind="nobody")]
        repo.save_labels("s1", labels)
        assert repo.labels("s1") == labels
        (tmp_path / "lab" / "s1" / "labels.json").write_text("[{}]")
        with pytest.raises(StorageError):
            repo.labels("s1")
        path = tmp_path / "lab" / "s1" / "runs" / f"{older.id}.json"
        path.write_text("{}")
        with pytest.raises(StorageError):
            repo.run("s1", older.id)
        (tmp_path / "lab" / "s1" / "runs" / f"{older.id}.results.json") \
            .write_text("[]")
        with pytest.raises(StorageError):
            repo.results("s1", older.id)
        (tmp_path / "lab" / "s1" / "original.json").write_text("{}")
        with pytest.raises(StorageError):
            repo.original_key("s1")

    def test_read_wav_mixes_channels_down(self, tmp_path: Path) -> None:
        path = tmp_path / "x.wav"
        write_wav(path, 16000, 2)
        assert read_wav(path).samples.shape == (160,)
        with wave.open(str(path), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(1)
            writer.setframerate(8000)
            writer.writeframes(b"\x80" * 10)
        with pytest.raises(StorageError):
            read_wav(path)


@pytest.fixture()
def sandbox(tmp_path: Path) -> LabSettings:
    """A lab with the fake runner."""
    return make_lab(tmp_path / "lab")


def tool_and_request(
    sandbox: LabSettings, tmp_path: Path, model: str, stage: str,
    environ: Optional[Dict[str, str]] = None, **params: Any,
) -> "tuple[SubprocessLabTool, ToolRequest]":
    """The subprocess tool over the sandbox and a request for ``model``."""
    env = {"HOME": str(tmp_path / "home")} if environ is None else environ
    environment = FileSystemLabEnvironment(
        sandbox.envs_dir, sandbox.runners_dir, env
    )
    catalogue = JsonCatalogue(sandbox.catalogue).load()
    entry = catalogue.get(model)
    assert entry is not None
    source = tmp_path / "in.wav"
    if not source.exists():
        write_wav(source)
    work = tmp_path / f"work-{model}-{len(list(tmp_path.iterdir()))}"
    work.mkdir()
    request = ToolRequest(
        entry=entry, stage=stage, params=params,  # type: ignore[arg-type]
        audio=source, workspace=work, load=lambda: audio(),
    )
    return SubprocessLabTool(environment, env, timeout=5.0), request


class TestSubprocessTool:
    def test_each_stage(self, sandbox: LabSettings, tmp_path: Path) -> None:
        tool, request = tool_and_request(sandbox, tmp_path, "fake-diarize",
                                         "diarize", num_speakers=2)
        outcome = tool.run(request, lambda: False)
        assert [s.speaker for s in outcome.segments] == ["A", "B"]
        assert outcome.model_version == "fake==1"
        assert (request.workspace / "runner.log").exists()
        tool, request = tool_and_request(sandbox, tmp_path, "fake-asr",
                                         "recognise", language="es")
        assert tool.run(request, lambda: False).language == "es"
        tool, request = tool_and_request(sandbox, tmp_path, "fake-denoise",
                                         "denoise")
        tool.run(request, lambda: False)
        check_wav(request.workspace / "audio.wav")
        tool.release()

    def test_the_command_has_no_shell_and_no_token(
        self, sandbox: LabSettings, tmp_path: Path
    ) -> None:
        tool, request = tool_and_request(sandbox, tmp_path, "fake-diarize",
                                         "diarize")
        argv = tool.command(request)
        assert argv[2:4] == ["diarize", "--model"]
        assert json.loads(argv[-1]) == {}
        assert all("token" not in part.lower() for part in argv)

    def test_the_token_goes_only_to_gated_models(
        self, sandbox: LabSettings, tmp_path: Path
    ) -> None:
        environ = {"HOME": str(tmp_path), "HF_TOKEN": "hf_secret_value",
                   "AWS_SECRET": "x", "HF_HOME": str(tmp_path / "hf")}
        tool, request = tool_and_request(sandbox, tmp_path, "fake-env",
                                         "diarize", environ)
        notes = tool.run(request, lambda: False).notes
        assert "HF_TOKEN" in notes and "HF_HOME" in notes
        assert "AWS_SECRET" not in notes
        open_env = tool.environment(gated=False)
        assert "HF_TOKEN" not in open_env and "AWS_SECRET" not in open_env

    def test_a_missing_token_is_reported(
        self, sandbox: LabSettings, tmp_path: Path
    ) -> None:
        tool, request = tool_and_request(sandbox, tmp_path, "fake-gated",
                                         "diarize")
        with pytest.raises(LabStepError) as error:
            tool.run(request, lambda: False)
        assert error.value.code == "needs_token"

    def test_a_crash_is_reported_without_private_paths(
        self, sandbox: LabSettings, tmp_path: Path
    ) -> None:
        tool, request = tool_and_request(sandbox, tmp_path, "fake-crash",
                                         "diarize")
        with pytest.raises(LabStepError) as error:
            tool.run(request, lambda: False)
        assert error.value.code == "failed"
        assert "code 1" in str(error.value)
        assert str(request.audio) not in str(error.value)
        assert "<in.wav>" in str(error.value)

    def test_time_out_and_cancel_kill_the_runner(
        self, sandbox: LabSettings, tmp_path: Path
    ) -> None:
        tool, request = tool_and_request(sandbox, tmp_path, "fake-slow",
                                         "diarize")
        ticks = iter(range(0, 1000, 10))
        slow = SubprocessLabTool(
            FileSystemLabEnvironment(sandbox.envs_dir, sandbox.runners_dir,
                                     {}),
            {}, timeout=15.0, clock=lambda: float(next(ticks)),
        )
        with pytest.raises(LabStepError) as timed_out:
            slow.run(request, lambda: False)
        assert timed_out.value.code == "timeout"
        tool, request = tool_and_request(sandbox, tmp_path, "fake-slow",
                                         "diarize")
        with pytest.raises(LabStepError) as cancelled:
            tool.run(request, lambda: True)
        assert cancelled.value.code == "cancelled"

    def test_an_environment_not_ready(self, tmp_path: Path) -> None:
        sandbox = make_lab(tmp_path / "lab", ready=False)
        tool, request = tool_and_request(sandbox, tmp_path, "fake-diarize",
                                         "diarize")
        with pytest.raises(LabStepError) as error:
            tool.run(request, lambda: False)
        assert error.value.code == "not_installed"

    def test_results_are_checked(
        self, sandbox: LabSettings, tmp_path: Path
    ) -> None:
        _, request = tool_and_request(sandbox, tmp_path, "fake-asr",
                                      "recognise")
        base = {"ok": True, "stage": "recognise", "model": "fake-asr"}
        parsed = parse_result({**base, "words": [
            {"start": 1, "end": 0.5, "text": "x", "probability": 1.5},
        ]}, request)
        word = parsed.words[0]
        assert (word.end, word.probability) == (1.0, 1.0)
        for bad in ({**base, "model": "other"},
                    {**base, "words": [{"start": 1}]},
                    {**base, "segments": [{"start": 2, "end": 1,
                                           "speaker": "a"}]},
                    {**base, "segments": [{"start": 0, "end": 1}]}):
            with pytest.raises(LabStepError):
                parse_result(bad, request)

    def test_odd_runner_answers(
        self, sandbox: LabSettings, tmp_path: Path
    ) -> None:
        script = sandbox.runners_dir / "fake.py"
        original = script.read_text()
        for body, code in (
            ("import sys, json, os\nout = sys.argv[sys.argv.index('--out')"
             " + 1]\nopen(os.path.join(out, 'result.json'), 'w')"
             ".write('[1]')\n", "failed"),
            ("import sys, json, os\nout = sys.argv[sys.argv.index('--out')"
             " + 1]\nopen(os.path.join(out, 'result.json'), 'w').write("
             "json.dumps({'ok': True, 'stage': 'diarize', 'model':"
             " 'fake-diarize'}))\nsys.exit(4)\n", "failed"),
            ("import sys\nsys.exit(4)\n", "out_of_memory"),
        ):
            script.write_text(body)
            tool, request = tool_and_request(sandbox, tmp_path,
                                             "fake-diarize", "diarize")
            with pytest.raises(LabStepError) as error:
                tool.run(request, lambda: False)
            assert error.value.code == code
        script.write_text(original)


class TestEnvironment:
    def test_family_states(self, sandbox: LabSettings,
                           tmp_path: Path) -> None:
        environment = FileSystemLabEnvironment(
            sandbox.envs_dir, sandbox.runners_dir, {}
        )
        assert environment.family_problem("fake") is None
        assert "no runner" in str(environment.family_problem("nothing"))
        assert "not installed" in str(environment.family_problem("absent"))
        (sandbox.envs_dir / "fake" / READY_MARKER).unlink()
        assert "not verified" in str(environment.family_problem("fake"))

    def test_token_lookup(self, tmp_path: Path) -> None:
        home = {"HOME": str(tmp_path)}
        assert read_token(home) is None
        assert token_file(home) == tmp_path / ".cache/huggingface/token"
        token_file(home).parent.mkdir(parents=True)
        token_file(home).write_text("hf_from_file\n")
        assert read_token(home) == "hf_from_file"
        assert read_token({**home, "HF_TOKEN": "hf_env"}) == "hf_env"
        assert token_file({"HF_HOME": "/h"}) == Path("/h/token")
        token_file(home).write_text("  ")
        assert read_token(home) is None
        environment = FileSystemLabEnvironment(tmp_path, tmp_path,
                                               {"HF_TOKEN": "x"})
        assert environment.token_present()

    def test_weights_in_the_cache(self, tmp_path: Path) -> None:
        environ = {"HF_HUB_CACHE": str(tmp_path / "hub")}
        environment = FileSystemLabEnvironment(tmp_path, tmp_path, environ)
        assert environment.weights_cached("org/model") is False
        snapshot = tmp_path / "hub" / "models--org--model" / "snapshots"
        (snapshot / "abc").mkdir(parents=True)
        assert environment.weights_cached("org/model") is True
        assert environment.weights_cached("https://x/y") is None
        assert environment.weights_cached("plain") is None
        assert hub_dir({"HF_HOME": "/h"}) == Path("/h/hub")
        assert hub_dir({"HOME": "/u"}) == Path("/u/.cache/huggingface/hub")


class TestCatalogueFile:
    def test_reads_and_rereads(self, sandbox: LabSettings) -> None:
        source = JsonCatalogue(sandbox.catalogue)
        first = source.load()
        assert source.load() is first
        data = json.loads(sandbox.catalogue.read_text())
        data["entries"] = data["entries"][:2]
        sandbox.catalogue.write_text(json.dumps(data))
        os.utime(sandbox.catalogue, (1, 1))
        assert len(source.load().entries) == 2

    def test_errors(self, tmp_path: Path) -> None:
        with pytest.raises(StorageError, match="Cannot read"):
            JsonCatalogue(tmp_path / "none.json").load()
        bad = tmp_path / "bad.json"
        bad.write_text('{"schema": "lab.catalogue.v1", "entries": [{}]}')
        with pytest.raises(StorageError, match="invalid"):
            JsonCatalogue(bad).load()

    def test_the_repository_catalogue_is_valid(self) -> None:
        settings = LabSettings()
        if settings.catalogue.exists():
            catalogue = JsonCatalogue(settings.catalogue).load()
            stages = {e.stage for e in catalogue.entries}
            assert stages == {"denoise", "diarize", "recognise"}
            for entry in catalogue.entries:
                if entry.family != "project":
                    runner = settings.runners_dir / f"{entry.family}.py"
                    assert runner.is_file(), entry.id


def test_processes() -> None:
    table = OsProcessTable()
    assert table.alive(table.current())
    assert not table.alive(2 ** 22 + 12345)
