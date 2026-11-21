"""The Sortformer worker adapter over a stand-in worker (LV-6).

The stand-in (``fake_sortformer_worker.py``) speaks the worker's JSON
lines protocol without NeMo, so the process handling, the threads and
the failures are the real ones; no GPU, no model.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import numpy as np
import pytest

from transcriber.adapters.outbound.fs_voiceprint_repository import (
    FileSystemVoiceprintRepository,
)
from transcriber.adapters.outbound.sortformer_stream_diarizer import (
    SortformerStreamDiarizer,
    worker_command,
)
from transcriber.application.live import LiveOptions
from transcriber.bootstrap import live_attributor, live_naming
from transcriber.config import LabSettings, TranscriberSettings
from transcriber.domain.errors import DiarizerError
from transcriber.domain.live import LiveSpeaker
from transcriber.domain.models import ConsentRecord, Voiceprint
from transcriber.domain.voices import FloatArray

WORKER = Path(__file__).with_name("fake_sortformer_worker.py")
RATE = 16000


def diarizer(mode: str, tmp_path: Path) -> SortformerStreamDiarizer:
    return SortformerStreamDiarizer(
        [sys.executable, str(WORKER), mode],
        {"HOME": str(tmp_path), "SECRET": "x"}, log=tmp_path / "w.log",
    )


def speech(seconds: float, loud: bool = True) -> FloatArray:
    return np.full(int(seconds * RATE), 0.5 if loud else 0.0,
                   dtype=np.float32)


def test_frames_come_back_in_order_and_the_worker_stops(
    tmp_path: Path,
) -> None:
    d = diarizer("ok", tmp_path)
    assert (d.model_name, d.frame_seconds, d.slots) == (
        "sortformer-streaming-v2", 0.08, 4,
    )
    env = d.environment()
    assert "SECRET" not in env and env["HOME"] == str(tmp_path)
    d.start()
    d.feed(speech(0.4))
    d.feed(speech(0.4, loud=False))
    first, rows = d.frames(until=0.8, timeout=10.0)
    assert first == 0 and len(rows) == 10
    assert [r[0] for r in rows] == [0.9] * 5 + [0.0] * 5
    assert d.ready is not None and d.ready["frame_seconds"] == 0.08
    d.feed(speech(0.1))
    d.finish()
    first, rows = d.frames(until=0.9, timeout=10.0)
    assert first == 10 and len(rows) == 2
    assert d.frames() == (12, [])
    d.close()
    d.close()
    assert (tmp_path / "w.log").exists()


def test_a_new_stream_starts_a_new_worker(tmp_path: Path) -> None:
    d = diarizer("ok", tmp_path)
    d.start()
    d.feed(speech(0.16))
    assert d.frames(0.16, 10.0)[0] == 0
    d.start()
    d.feed(speech(0.16))
    first, rows = d.frames(0.16, 10.0)
    assert first == 0 and len(rows) == 2
    d.close()


@pytest.mark.parametrize("mode,message", [
    ("error", "model broke"),
    ("crash", r"exited \(code 3\)"),
    ("order", "out of order"),
    ("noise", "unreadable answer"),
])
def test_worker_failures_become_diarizer_errors(
    tmp_path: Path, mode: str, message: str
) -> None:
    d = diarizer(mode, tmp_path)
    d.start()
    d.feed(speech(0.4))
    with pytest.raises(DiarizerError, match=message):
        for _ in range(50):
            d.frames(until=0.4, timeout=0.2)
    d.close()


def test_a_worker_that_will_not_stop_is_killed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import transcriber.adapters.outbound.sortformer_stream_diarizer as mod

    monkeypatch.setattr(mod, "STOP_SECONDS", 0.5)
    d = diarizer("slow", tmp_path)
    d.start()
    d.feed(speech(0.08))
    d.frames(0.08, 10.0)
    d.close()


def test_a_missing_worker_is_reported(tmp_path: Path) -> None:
    d = SortformerStreamDiarizer(
        worker_command(tmp_path / "nemo" / "bin" / "python",
                       tmp_path / "w.py", "cuda"),
        {},
    )
    with pytest.raises(DiarizerError, match="not installed"):
        d.start()
    with pytest.raises(DiarizerError):
        d.feed(speech(0.1))
    d.finish()
    d.close()
    assert worker_command(Path("p"), Path("s"), "cpu") == [
        "p", "s", "--device", "cpu",
    ]


def test_an_interpreter_that_cannot_run_is_reported(tmp_path: Path) -> None:
    script = tmp_path / "w.py"
    script.write_text("")
    d = SortformerStreamDiarizer([str(tmp_path), str(script)], {})
    with pytest.raises(DiarizerError, match="did not start"):
        d.start()


def enrol(repository: FileSystemVoiceprintRepository, ident: str,
          model: str, name: str = "") -> None:
    at = datetime(2026, 11, 21, tzinfo=timezone.utc)
    repository.save(
        Voiceprint(speaker_id=ident, model=model, vector=[1.0, 0.0],
                   consent_version="c1", enrolled_at=at),
        ConsentRecord(speaker_id=ident, name=name or None,
                      consent_version="c1", text_version="t1",
                      given_at=at),
    )


def test_the_real_attributor_is_built_from_the_settings(
    tmp_path: Path,
) -> None:
    settings = TranscriberSettings(
        data_dir=tmp_path, lab=LabSettings(envs_dir=tmp_path / "envs"),
    )
    repository = FileSystemVoiceprintRepository(
        tmp_path, settings.voice_model
    )
    enrol(repository, "ana", settings.voice_model, "Ana")
    enrol(repository, "luis", settings.voice_model)
    make = live_attributor(
        settings, {}, repository,
        LiveOptions(device="cpu", diarize=True, enrolled=["ana"]),
    )
    mono = make([LiveSpeaker("agente", role="agent")])
    assert mono.model_name == "sortformer-streaming-v2"
    roster: List[LiveSpeaker] = mono.roster()
    assert [s.id for s in roster] == ["agente", "S1", "S2", "S3", "ana"]
    assert roster[-1].name == "Ana" and roster[-1].enrolled
    mono.start()
    assert mono.problem is not None and "not installed" in mono.problem
    assert mono.finish() == []


def test_naming_needs_someone_enrolled(tmp_path: Path) -> None:
    from transcriber.adapters.outbound.fake_models import FakeVoiceEncoder

    repository = FileSystemVoiceprintRepository(tmp_path, "fake/voice")
    assert live_naming(repository, FakeVoiceEncoder(), None) is None
