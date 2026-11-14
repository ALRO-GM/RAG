"""Parts of ``transcriber serve`` run in-process: the access code, the
settings from the environment and the progress lines."""

import io
import json
import time
from pathlib import Path
from typing import Iterator, List

import pytest
from pydantic import ValidationError

from tests.integration.transcriber.audio import sound_file, tone
from transcriber.adapters.inbound import serve as serve_module
from transcriber.adapters.inbound.cli import (
    clock_time,
    main,
    slug,
    transcript_lines,
)
from transcriber.adapters.inbound.serve import (
    WEB_DIR,
    ProgressLines,
    dev_settings,
)
from transcriber.adapters.outbound.fs_access_code import (
    AccessCodeError,
    code_file,
    kept_access_code,
    new_access_code,
)
from transcriber.application.ports import RecordedAudio, TunnelStatus
from transcriber.bootstrap import (
    TranscriberContainer,
    access_code,
    build_container,
)
from transcriber.config import TranscriberSettings, load_settings

WAIT = 30.0


class TestAccessCode:
    def test_a_fresh_code_is_long_random_and_url_safe(self) -> None:
        codes = {new_access_code() for _ in range(20)}
        assert len(codes) == 20
        for code in codes:
            assert len(code) >= 32
            assert set(code) <= set(
                "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
                "0123456789-_"
            )

    def test_a_kept_code_is_made_once_with_mode_600(
        self, tmp_path: Path
    ) -> None:
        path = code_file(tmp_path / "data")
        first = kept_access_code(path)
        assert path.stat().st_mode & 0o777 == 0o600
        assert kept_access_code(path) == first
        settings = TranscriberSettings(data_dir=tmp_path / "data")
        assert access_code(settings, keep=True) == first
        assert access_code(settings, keep=False) != first
        renewed = access_code(settings, keep=True, renew=True)
        assert renewed != first
        assert access_code(settings, keep=True) == renewed
        assert path.stat().st_mode & 0o777 == 0o600

    @pytest.mark.parametrize("mode", [0o644, 0o640, 0o620, 0o604])
    def test_a_kept_code_others_can_read_is_refused(
        self, tmp_path: Path, mode: int
    ) -> None:
        path = tmp_path / "access_code"
        kept_access_code(path)
        path.chmod(mode)
        with pytest.raises(AccessCodeError, match="readable by others"):
            kept_access_code(path)

    @pytest.mark.parametrize("content", ["short", "x" * 40 + " y", ""])
    def test_a_kept_file_without_a_valid_code_is_refused(
        self, tmp_path: Path, content: str
    ) -> None:
        path = tmp_path / "access_code"
        path.write_text(content)
        path.chmod(0o600)
        with pytest.raises(AccessCodeError, match="valid code"):
            kept_access_code(path)


class TestSettingsFromTheEnvironment:
    def test_no_variables_give_the_defaults(self) -> None:
        assert load_settings({}) == TranscriberSettings()

    def test_variables_win_over_the_file(self, tmp_path: Path) -> None:
        config = tmp_path / "settings.json"
        config.write_text(json.dumps({
            "data_dir": "from-file", "runtime": "real",
            "tunnel": {"remote_port": 9100, "public_host": "x.test"},
            "api": {"port": 9001},
        }))
        settings = load_settings({
            "TRANSCRIBER_CONFIG": str(config),
            "TRANSCRIBER_RUNTIME": "fake",
            "TRANSCRIBER_DEVICE": "cpu",
        })
        assert settings.data_dir == Path("from-file")
        assert settings.runtime == "fake" and settings.device == "cpu"
        assert settings.tunnel.remote_port == 9100
        assert settings.tunnel.public_host == "x.test"
        assert settings.api.port == 9001
        assert load_settings({
            "TRANSCRIBER_CONFIG": str(config),
            "TRANSCRIBER_DATA_DIR": "from-env",
        }).data_dir == Path("from-env")

    def test_bad_settings_are_refused(self, tmp_path: Path) -> None:
        config = tmp_path / "settings.json"
        config.write_text("[1, 2]")
        with pytest.raises(ValueError, match="JSON object"):
            load_settings({"TRANSCRIBER_CONFIG": str(config)})
        with pytest.raises(ValidationError):
            load_settings({"TRANSCRIBER_RUNTIME": "quantum"})
        with pytest.raises(OSError):
            load_settings({"TRANSCRIBER_CONFIG": str(tmp_path / "none")})


class FakeTunnel:
    """A tunnel whose status the test sets."""

    def __init__(self) -> None:
        self.current = TunnelStatus(state="connecting")

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def status(self) -> TunnelStatus:
        return self.current


@pytest.fixture()
def container(tmp_path: Path) -> Iterator[TranscriberContainer]:
    """A fake-runtime container; its job runner is closed afterwards."""
    built = build_container(TranscriberSettings(
        data_dir=tmp_path / "data", runtime="fake",
    ))
    yield built
    built.runner.close(WAIT)


def wait_done(container: TranscriberContainer, session_id: str) -> None:
    """Wait until the session's job ended."""
    deadline = time.monotonic() + WAIT
    while time.monotonic() < deadline:
        state = container.transcription_status(session_id).state.value
        if state in ("done", "failed"):
            return
        time.sleep(0.02)
    raise AssertionError("the job did not end")


class TestProgressLines:
    def test_changes_become_lines(
        self, container: TranscriberContainer
    ) -> None:
        lines: List[str] = []
        tunnel = FakeTunnel()
        progress = ProgressLines(container, tunnel, lines.append,
                                 now=lambda: "12:00:00")
        wav = sound_file(tone(2.0), 16000, "wav")
        container.import_recording("old", "es", RecordedAudio(wav, "wav"))
        progress.poll()  # first poll: only the tunnel and active jobs
        assert lines == ["12:00:00  tunnel  connecting"]
        progress.poll()
        assert len(lines) == 1  # nothing changed

        tunnel.current = TunnelStatus(state="reconnecting", reason="down")
        container.open_session("new", "es")
        progress.poll()
        assert lines[1:] == [
            "12:00:00  tunnel  reconnecting (down)",
            "12:00:00  new  recording",
        ]

        container.queue_transcription("old")
        wait_done(container, "old")
        progress.poll()
        assert lines[-1] == "12:00:00  old  done"
        container.queue_transcription("old")  # the same state again...
        wait_done(container, "old")
        progress.poll()  # ...but a new run
        assert lines[-1] == "12:00:00  old  done"
        assert len(lines) == 5

    def test_a_running_job_shows_its_stage(
        self, container: TranscriberContainer
    ) -> None:
        lines: List[str] = []
        progress = ProgressLines(container, None, lines.append,
                                 now=lambda: "t")
        snapshot = ("running", "transcribe", 4, "2 voiceprints skipped",
                    None)
        assert progress._line("s1", snapshot, None) == (
            "t  s1  running  transcribe 40%  (2 voiceprints skipped)"
        )
        assert progress._line(
            "s1", ("failed", "transcribe", 4, None, None), "decode: bad"
        ) == "t  s1  failed  decode: bad"


class TestFormatting:
    def test_slug(self) -> None:
        assert slug("Ana María") == "ana-maria"
        assert slug("  ¡Hola!  ") == "hola"
        assert slug("ñ" * 10) == "n" * 10
        assert slug("日本") == ""
        assert len(slug("a b " * 40)) <= 48

    def test_clock_time(self) -> None:
        assert clock_time(0) == "00:00:00"
        assert clock_time(3725.9) == "01:02:05"
        assert clock_time(-1) == "00:00:00"

    def test_transcript_lines(
        self, container: TranscriberContainer
    ) -> None:
        sample = sound_file(tone(30.0, hz=220.0), 16000, "wav")
        container.enrol_speaker(
            "ana", "Ana", "es", RecordedAudio(sample, "wav"), True, "c-1",
        )
        talk = sound_file(tone(6.0, hz=880.0), 16000, "wav")
        container.import_recording("s", "es", RecordedAudio(talk, "wav"))
        container.queue_transcription("s")
        wait_done(container, "s")
        transcript = container.show_transcript("s")
        lines = transcript_lines(transcript)
        assert len(lines) == len(transcript.utterances)
        first = transcript.utterances[0]
        assert lines[0] == f"[00:00:00] {first.speaker}: {first.text}"
        later = transcript_lines(transcript, lead=100.0)
        assert all(line.startswith("[00:00:00]") for line in later)


class TestDevPages:
    def test_the_pages_are_served_from_the_web_folder(self) -> None:
        settings = dev_settings(TranscriberSettings())
        assert settings.api.static_dir == WEB_DIR
        assert WEB_DIR.parent.name == "inbound"

    def test_missing_pages_are_a_usage_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(serve_module, "WEB_DIR", tmp_path / "none")
        err = io.StringIO()
        code = main(
            ["serve", "--no-tunnel", "--dev-pages"],
            environ={"TRANSCRIBER_RUNTIME": "fake",
                     "TRANSCRIBER_DATA_DIR": str(tmp_path / "data")},
            out=io.StringIO(), err=err,
        )
        assert code == 2 and "no pages" in err.getvalue()
