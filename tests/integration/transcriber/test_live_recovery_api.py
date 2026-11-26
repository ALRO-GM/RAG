"""Recovering a live session's lost audio over the real API (#265).

Fake runtime, temporary directories: a live session hears speech, loses
its source for a while (the page says ``gap``: silence on the server),
hears more and ends. The page then uploads the audio it kept for the
gap; the job transcribes it with the batch pipeline of its own
repository (never «Sesiones»), merges it into the live transcript,
publishes it and deletes the audio. Also the store on disk (private
modes, WAV assembly) and the routes' refusals. Invented tones only.
"""

import json
import stat
import time
import wave
import io
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tests.unit.transcriber.live_fakes import bursts, chunks_of, pcm
from transcriber.adapters.inbound.http_api import create_app
from transcriber.adapters.inbound.recovery_worker import RecoveryWorker
from transcriber.adapters.outbound.fs_recovery_store import (
    FileSystemRecoveryStore,
)
from transcriber.bootstrap import TranscriberContainer, build_container
from transcriber.config import TranscriberSettings
from transcriber.domain.errors import InvalidInputError
from transcriber.domain.live_recovery import RecoveryJob

CODE = "a-long-enough-access-code"
AUTH = {"X-Access-Code": CODE}
WAIT = 30.0
S = 16000


@pytest.fixture()
def container(tmp_path: Path) -> Iterator[TranscriberContainer]:
    """A fake-runtime container; its runner is closed afterwards."""
    built = build_container(TranscriberSettings(
        data_dir=tmp_path / "data", runtime="fake"))
    yield built
    built.runner.close(WAIT)


@pytest.fixture()
def client(container: TranscriberContainer) -> TestClient:
    """A client that sends the access code."""
    return TestClient(create_app(container, CODE), headers=AUTH)


def heard(seconds: float, at: float = 0.2, length: float = 2.0) -> bytes:
    """``seconds`` of mono PCM with one tone the fake models hear."""
    return pcm(bursts([(at, at + length)], seconds, 0.5))


def live_session_with_a_gap(client: TestClient) -> None:
    """3 s heard, 5 s the page could not send (silence here), 3 s."""
    answer = client.post("/api/live/sessions", json={
        "session_id": "iv-1", "handoff": {"transcript_v1": True},
        "buffer_audio": True, "buffer_max_hours": 48})
    assert answer.status_code == 201, answer.text
    with client.websocket_connect("/api/live/iv-1/audio") as ws:
        ws.receive_json()
        for message in chunks_of(heard(3.0), 3200):
            ws.send_bytes(message)
        ws.send_text(json.dumps({"type": "gap", "samples": 5 * S}))
        for message in chunks_of(heard(3.0), 3200):
            ws.send_bytes(message)
        ws.send_text(json.dumps({"type": "stop"}))
        while ws.receive_json().get("type") != "stopped":
            pass


def test_the_lost_minutes_come_back_into_the_transcript(
    client: TestClient, container: TranscriberContainer,
) -> None:
    live_session_with_a_gap(client)
    assert container.live is not None
    assert container.live.get("iv-1").wait_handed_over(WAIT)
    folder = container.settings.data_dir / "live" / "iv-1"
    record = json.loads((folder / "session.json").read_text())
    assert record["silences"] == [[3 * S, 8 * S]]
    assert record["started_at"] and record["request"]["buffer_audio"]
    assert record["received_samples"] == 11 * S
    before = json.loads((folder / "transcript.json").read_text())
    kept = heard(5.0, at=1.0, length=2.5)  # what the page kept for the gap
    for n, piece in enumerate(chunks_of(kept, 2 * S * 2)):
        answer = client.put(
            f"/api/live/sessions/iv-1/recovery/chunks/{n}",
            params={"offset": 3 * S + n * 2 * S}, content=piece)
        assert answer.status_code == 200, answer.text
    store = container.settings.data_dir / "live-recovery" / "iv-1"
    chunk = store / "audio" / "0.pcm"
    assert stat.S_IMODE(chunk.stat().st_mode) == 0o600
    assert stat.S_IMODE(store.stat().st_mode) == 0o700
    closed = client.post("/api/live/sessions/iv-1/recovery/close",
                         json={"count": 3})
    assert closed.status_code == 200 and closed.json()["state"] == "queued"
    recovery = container.live_recovery
    assert recovery is not None
    deadline = time.monotonic() + WAIT
    while recovery.status("iv-1")["state"] not in ("done", "failed"):
        assert time.monotonic() < deadline
        recovery.tick()
        time.sleep(0.05)
    view = client.get("/api/live/sessions/iv-1/recovery").json()
    assert view["state"] == "done", view
    assert 0 < view["seconds"] <= 5.0 and view["audio_deleted"]
    after = json.loads((folder / "transcript.json").read_text())
    assert len(after["utterances"]) > len(before["utterances"])
    gap = [u for u in after["utterances"] if 3 <= u["start"] < 8]
    assert gap and all(u["speaker"] == "S1" for u in gap)
    published = json.loads((container.settings.published()
                            / "iv-1.transcript.v1.json").read_text())
    assert published == after
    assert not (store / "audio").exists() and not (store / "text").exists()
    batch = container.settings.data_dir / "live-recovery-batch" / "sessions"
    assert not batch.exists() or list(batch.iterdir()) == []
    assert container.sessions.ids() == []  # never in «Sesiones»
    listed = client.get("/api/live/recoveries").json()["recoveries"]
    assert [r["session_id"] for r in listed] == ["iv-1"]
    assert client.delete("/api/live/sessions/iv-1/recovery"
                         ).status_code == 204
    assert client.get("/api/live/sessions/iv-1/recovery").status_code == 404


def test_the_routes_refuse_what_they_must(
    client: TestClient, container: TranscriberContainer,
) -> None:
    second = b"\x01\x00" * S
    unknown = client.put("/api/live/sessions/nobody/recovery/chunks/0",
                         params={"offset": 0}, content=second)
    assert unknown.status_code == 404
    bad = client.put("/api/live/sessions/b%2Fc/recovery/chunks/0",
                     params={"offset": 0}, content=second)
    assert bad.status_code in (400, 404)
    assert client.put("/api/live/sessions/x/recovery/chunks/0",
                      content=second).status_code == 422  # no offset
    big = b"\x00" * (container.settings.live.recovery.chunk_max_bytes + 2)
    too_big = client.put("/api/live/sessions/x/recovery/chunks/0",
                         params={"offset": 0}, content=big)
    assert too_big.status_code == 413
    assert client.post("/api/live/sessions/x/recovery/close",
                       content=b"nope").status_code == 400
    assert client.post("/api/live/sessions/x/recovery/close",
                       json={"count": "3"}).status_code == 400
    assert client.post("/api/live/sessions/x/recovery/close",
                       json={"count": 1}).status_code == 404
    anonymous = TestClient(create_app(container, CODE))
    assert anonymous.get("/api/live/recoveries").status_code == 401


def test_the_store_assembles_one_channel_inside_the_spans(
    tmp_path: Path,
) -> None:
    store = FileSystemRecoveryStore(tmp_path)
    left = np.full(S, 1000, dtype="<i2")
    right = np.full(S, -2000, dtype="<i2")
    stereo = np.stack([left, right], axis=1).reshape(-1).tobytes()
    store.put_chunk("iv-1", 0, stereo)
    store.put_chunk("iv-1", 1, stereo)
    chunks = {"0": [S, S], "1": [3 * S, S]}
    data = store.wav("iv-1", chunks, (S, 4 * S),
                     [(S, int(1.5 * S)), (3 * S, 10 * S)], 2, 1)
    with wave.open(io.BytesIO(data)) as found:
        assert found.getnchannels() == 1 and found.getframerate() == S
        samples = np.frombuffer(found.readframes(found.getnframes()),
                                dtype="<i2")
    assert len(samples) == 3 * S
    assert set(samples[: S // 2]) == {-2000}
    assert set(samples[S // 2: 2 * S]) == {0}  # outside spans / no chunk
    assert set(samples[2 * S:]) == {-2000}
    mixed = store.wav("iv-1", chunks, (S, 2 * S), [(S, 2 * S)], 2, None)
    assert store.missing("iv-1", ["0", "1", "7"]) == ["7"]
    with wave.open(io.BytesIO(mixed)) as found:
        assert set(np.frombuffer(found.readframes(S), dtype="<i2")) == {-500}
    assert store.wav("iv-1", {}, (0, 0), [], 1, None)[:4] == b"RIFF"
    assert store.audio_bytes() == 2 * len(stereo)
    job = RecoveryJob("iv-1", "queued", "2026-11-26T10:00:00+00:00",
                      "2026-11-27T10:00:00+00:00")
    store.save(job)
    assert store.load("iv-1") == job and store.ids() == ["iv-1"]
    assert stat.S_IMODE((tmp_path / "live-recovery" / "iv-1"
                         / "recovery.json").stat().st_mode) == 0o600
    store.save_text("iv-1", "live", {"a": 1})
    assert store.read_text("iv-1", "live") == {"a": 1}
    assert store.read_text("iv-1", "part-0") is None
    with pytest.raises(ValueError):
        store.save_text("iv-1", "../x", {})
    with pytest.raises(InvalidInputError):
        store.load("../etc")
    assert store.delete_audio("iv-1") == 2 and store.delete_audio("iv-1") == 0
    store.delete_texts("iv-1")
    store.delete("iv-1")
    assert store.ids() == [] and store.load("iv-1") is None
    (tmp_path / "live-recovery" / "iv-2").mkdir()
    (tmp_path / "live-recovery" / "iv-2" / "recovery.json").write_text("[]")
    with pytest.raises(ValueError):
        store.load("iv-2")


def test_the_worker_sweeps_then_ticks_and_survives_errors() -> None:
    calls = []

    class Recovery:
        def sweep(self) -> int:
            calls.append("sweep")
            return 0

        def tick(self) -> int:
            calls.append("tick")
            if len(calls) == 2:
                raise RuntimeError("broken")
            return 0

    now = [0.0]
    worker = RecoveryWorker(Recovery(), 0.01, 100.0,  # type: ignore[arg-type]
                            monotonic=lambda: now[0])
    worker.once()
    worker.once()
    now[0] = 200.0
    worker.once()
    assert calls == ["sweep", "tick", "tick", "sweep", "tick"]
    worker.start()
    worker.start()
    time.sleep(0.05)
    worker.stop()
