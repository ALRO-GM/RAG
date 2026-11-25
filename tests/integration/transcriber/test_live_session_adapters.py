"""Outbound adapters of the live sessions: the JSONL live log, the free
GPU memory reading, the contract check, the fake live models and the
wiring of the composition root (LV-7, LV-8)."""

import json
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional

import numpy as np
import pytest

from tests.unit.transcriber.live_fakes import bursts
from transcriber.adapters.outbound import gpu_free_memory
from transcriber.adapters.outbound.fake_models import FakeAsrStream, FakeVad
from transcriber.adapters.outbound.fs_transcript_publisher import (
    DirectoryTranscriptPublisher,
)
from transcriber.adapters.outbound.gpu_free_memory import (
    NvidiaSmiFreeMemory,
    parse_free,
    run_query,
    torch_free_mib,
)
from transcriber.adapters.outbound.jsonl_live_log import JsonlLiveLog
from transcriber.adapters.outbound.jsonschema_event_check import (
    JsonSchemaEventCheck,
    build_event_check,
)
from transcriber.bootstrap import CONTRACTS_DIR, build_live_sessions
from transcriber.config import TranscriberSettings
from transcriber.domain.live import LiveEventStream

AT = datetime(2026, 11, 21, 10, 0, tzinfo=timezone.utc)


def test_the_live_log_keeps_events_and_transcript_privately(
    tmp_path: Path,
) -> None:
    log = JsonlLiveLog(tmp_path)
    log.append("s-1", {"seq": 0, "text": "Hola ñ"})
    log.append("s-1", {"seq": 1})
    log.save_transcript("s-1", {"schema": "transcript.v1"})
    log.close("s-1")
    log.close("s-1")
    log.append("s-1", {"seq": 2})
    log.close("s-1")
    folder = log.directory("s-1")
    lines = (folder / "events.jsonl").read_text("utf-8").splitlines()
    assert [json.loads(line)["seq"] for line in lines] == [0, 1, 2]
    assert "Hola ñ" in lines[0]
    assert json.loads((folder / "transcript.json").read_text()) == {
        "schema": "transcript.v1"
    }
    assert stat.S_IMODE(folder.stat().st_mode) == 0o700
    for name in ("events.jsonl", "transcript.json"):
        assert stat.S_IMODE((folder / name).stat().st_mode) == 0o600
    assert sorted(p.name for p in folder.iterdir()) == [
        "events.jsonl", "transcript.json"
    ]


def test_the_attribution_numbers_are_kept_privately(tmp_path: Path) -> None:
    log = JsonlLiveLog(tmp_path)
    log.save_attribution("s-2", [{"event": "check", "heard": 5.1},
                                 {"event": "slots", "slots": []}])
    path = log.directory("s-2") / "attribution.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["event"] for r in rows] == ["check", "slots"]
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert sorted(p.name for p in log.directory("s-2").iterdir()) == [
        "attribution.jsonl"]


def test_free_memory_is_read_from_nvidia_smi_or_torch() -> None:
    assert parse_free("header\n 7012 \n100\n") == 7012
    assert parse_free("") is None
    assert NvidiaSmiFreeMemory(lambda: "5000\n", lambda: 1).free_mib() == 5000
    assert NvidiaSmiFreeMemory(lambda: "", lambda: 1234).free_mib() == 1234
    assert NvidiaSmiFreeMemory(lambda: "", lambda: None).free_mib() is None


def test_the_query_survives_a_missing_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gpu_free_memory, "QUERY", ("no-such-tool-here",))
    assert run_query() == ""
    monkeypatch.setattr(
        gpu_free_memory, "QUERY",
        (sys.executable, "-c", "print(4096)"),
    )
    assert run_query().strip() == "4096"
    monkeypatch.setattr(
        gpu_free_memory, "QUERY", (sys.executable, "-c", "exit(3)")
    )
    assert run_query() == ""


def test_torch_is_only_asked_when_already_loaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    assert torch_free_mib() is None
    fake = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake)
    assert torch_free_mib() is None


def test_the_contract_check_finds_broken_events(tmp_path: Path) -> None:
    check = JsonSchemaEventCheck(CONTRACTS_DIR / "live.event.v1.schema.json")
    events = LiveEventStream("s")
    good = events.warning(AT, 0.0, "asr_behind", "late")
    assert check.problem(good) is None
    broken = dict(good, type="final")
    assert "utterance" in str(check.problem(broken))
    assert check.problem(dict(good, seq=-1)) is not None
    missing, why = build_event_check(tmp_path)
    assert missing is None and "live.event.v1" in str(why)


def test_without_jsonschema_there_is_no_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "jsonschema", None)
    check, why = build_event_check(CONTRACTS_DIR)
    assert check is None and why == "jsonschema is not installed"


def test_the_fake_live_models_hear_loud_slots() -> None:
    vad = FakeVad()
    assert vad.probability(np.full(512, 0.5, dtype=np.float32)) == 0.9
    assert vad.probability(np.zeros(512, dtype=np.float32)) == 0.05
    assert vad.probability(np.zeros(0, dtype=np.float32)) == 0.05
    vad.reset()
    asr = FakeAsrStream(None)
    asr.warm_up()
    assert asr.warmed == 1
    audio = bursts([(0.0, 0.8)], 1.6, 0.5)
    final = asr.decode(audio, True)
    assert final.text == "palabra0 palabra1."
    assert [w.text for w in final.words] == ["palabra0", "palabra1."]
    assert final.language == "es" and final.confidence == 0.9
    partial = asr.decode(audio, False)
    assert partial.text == "palabra0 palabra1" and partial.words == []
    silent = asr.decode(np.zeros(16000, dtype=np.float32), True)
    assert silent.text == "" and silent.confidence is None


@pytest.mark.parametrize("runtime,device,expected", [
    ("fake", "auto", "cpu"), ("real", "cpu", "cpu"),
    ("real", "auto", "cuda"),
])
def test_the_composition_root_wires_the_live_sessions(
    tmp_path: Path, runtime: Any, device: Any, expected: str,
) -> None:
    settings = TranscriberSettings(
        data_dir=tmp_path, runtime=runtime, device=device
    )
    live = build_live_sessions(
        settings, {"XDG_RUNTIME_DIR": str(tmp_path)},
        DirectoryTranscriptPublisher(tmp_path / "pub"), None,
    )
    ports: Any = live._p
    assert live._device == expected
    assert ports.check is not None and ports.log is not None
    real: Optional[bool] = runtime == "real"
    assert (ports.lease is not None) is real
    assert (ports.memory is not None) is real


def test_a_missing_check_is_logged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import transcriber.bootstrap as bootstrap

    monkeypatch.setattr(bootstrap, "CONTRACTS_DIR", tmp_path / "none")
    live = build_live_sessions(
        TranscriberSettings(data_dir=tmp_path, runtime="fake"), {},
        DirectoryTranscriptPublisher(tmp_path), None,
    )
    assert live._p.check is None
    assert "not contract-checked" in caplog.text


def test_the_session_record_and_the_pending_sessions(
    tmp_path: Path,
) -> None:
    """#264: ``session.json`` is private; only sessions with a record and
    no ``handed`` mark are pending; a line cut by a crash is skipped."""
    log = JsonlLiveLog(tmp_path)
    assert log.pending() == []
    log.append("a", {"seq": 0})
    log.append("old", {"seq": 0})  # kept before #264: no record
    log.save_session("a", {"session_id": "a", "state": "running"})
    log.save_session("b", {"session_id": "b", "state": "ended"})
    log.save_transcript("b", {"schema": "transcript.v1"})
    log.close("a")
    with (log.directory("a") / "events.jsonl").open("a") as stream:
        stream.write('{"seq": 1, "cut')
    assert log.pending() == ["a", "b"]
    assert log.has_transcript("b") and not log.has_transcript("a")
    log.mark_handed("b")
    assert log.pending() == ["a"]
    handed = log.directory("b") / "handed"
    assert stat.S_IMODE(handed.stat().st_mode) == 0o600
    assert log.read_events("a") == [{"seq": 0}]
    assert log.read_session("a") == {"session_id": "a", "state": "running"}
    assert log.read_session("old") is None
    assert log.read_transcript("b") == {"schema": "transcript.v1"}
    assert log.read_transcript("a") is None
    path = log.directory("a") / "session.json"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    path.write_text("[1]")
    with pytest.raises(ValueError):
        log.read_session("a")
    (log.directory("b") / "transcript.json").write_text("[1]")
    with pytest.raises(ValueError):
        log.read_transcript("b")
    (tmp_path / "live" / "stray-file").write_text("x")
    assert log.pending() == ["a"]


def test_the_live_marker_is_private_and_gone_when_nothing_is_on(
    tmp_path: Path,
) -> None:
    import os

    from transcriber.adapters.outbound.fs_live_marker import (
        FileLiveMarker,
        marker_path,
    )

    path = marker_path({"XDG_RUNTIME_DIR": str(tmp_path)})
    assert path == tmp_path / "ragputin" / "live-active.json"
    assert marker_path({}).name == "live-active.json"
    marker = FileLiveMarker(path)
    marker.mark([])
    assert not path.exists()
    marker.mark([{"id": "live-1", "state": "running", "source": "lost"}])
    document = json.loads(path.read_text())
    assert document["pid"] == os.getpid()
    assert document["sessions"][0]["source"] == "lost"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    marker.mark([])
    assert not path.exists()


def test_the_live_marker_survives_writers_on_many_threads(
    tmp_path: Path,
) -> None:
    import threading

    from transcriber.adapters.outbound.fs_live_marker import FileLiveMarker

    marker = FileLiveMarker(tmp_path / "ragputin" / "live-active.json")
    errors: List[Exception] = []

    def write(n: int) -> None:
        try:
            for k in range(50):
                marker.mark([{"id": f"s{n}", "state": "running",
                              "source": str(k)}])
        except Exception as exc:  # noqa: BLE001 - collected
            errors.append(exc)

    threads = [threading.Thread(target=write, args=(n,)) for n in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    document = json.loads(marker.path.read_text())
    assert len(document["sessions"]) == 1
    assert list(marker.path.parent.glob(".*.tmp")) == []
