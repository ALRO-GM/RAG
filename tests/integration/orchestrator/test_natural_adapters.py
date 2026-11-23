"""The natural voices' adapters (#214): the worker's streaming line
protocol (a stand-in worker, no model), the engines on disk, the GPU
gate over the lease file and the wiring of ``serve``."""

import argparse
import json
import os
import sys
import textwrap
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Generator, Iterator, List, Optional, Tuple, cast

import pytest
from fastapi.testclient import TestClient

from orchestrator.adapters.inbound import cli
from orchestrator.adapters.inbound.http_server import create_app
from orchestrator.adapters.outbound.fakes import FakeTts
from orchestrator.adapters.outbound.gpu_gate import (
    HOLDER,
    LeaseGpuGate,
    lease_directory,
    nvidia_free_mib,
    process_alive,
)
from orchestrator.adapters.outbound.subprocess_speech import (
    REFERENCE,
    WorkerEngines,
    WorkerSpeech,
    _piece,
)
from orchestrator.application.natural_voice import NaturalWiring
from orchestrator.application.ports import CancelToken, TtsUnavailable
from orchestrator.domain.voices import EngineVoice
from tests.unit.orchestrator.test_voicebot_sessions import bot_sessions

POCKET = EngineVoice("pocket", "cpu", voice="lola")
QWEN3 = EngineVoice("qwen3", "cuda", reference="/r.wav",
                    reference_text="Hola.")
FAKE_WORKER = textwrap.dedent('''
    import base64, json, sys, threading, time, queue
    print("loading", file=sys.stderr, flush=True)
    print("not json", flush=True)
    print("[1]", flush=True)
    print(json.dumps({"ready": True, "engine": "fake",
                      "sample_rate": 24000}), flush=True)
    requests, cancelled = queue.Queue(), set()

    def read():
        for line in sys.stdin:
            msg = json.loads(line)
            if "cancel" in msg:
                cancelled.add(msg["cancel"])
            else:
                requests.put(msg)
        requests.put(None)

    threading.Thread(target=read, daemon=True).start()
    pcm = base64.b64encode(b"\\x01\\x00" * 240).decode()
    while True:
        req = requests.get()
        if req is None:
            break
        text, ident = req["text"], req["id"]
        if text == "DIE.":
            sys.exit(2)
        if text == "FAIL.":
            print(json.dumps({"id": ident, "error": "boom"}), flush=True)
            continue
        print(json.dumps({"id": "other", "pcm": pcm,
                          "sample_rate": 24000}), flush=True)
        print(json.dumps({"id": ident, "pcm": "!!", "sample_rate": 24000}),
              flush=True)
        n = 200 if text.startswith("LONG") else 3
        if text.startswith("REF"):
            n = 1 if req.get("reference_text") == "Hola." else 2
        for i in range(n):
            if ident in cancelled:
                break
            print(json.dumps({"id": ident, "pcm": pcm,
                              "sample_rate": 24000}), flush=True)
            time.sleep(0.01)
        if text == "STALL.":
            time.sleep(5)
        print(json.dumps({"id": ident, "done": True,
                          "cancelled": ident in cancelled}), flush=True)
''')


@pytest.fixture
def worker(tmp_path: Path) -> Iterator[WorkerSpeech]:
    script = tmp_path / "worker.py"
    script.write_text(FAKE_WORKER, encoding="utf-8")
    speech = WorkerSpeech([sys.executable, str(script)], load_timeout=5.0,
                          timeout=1.0, stop_timeout=0.2)
    yield speech
    speech.close()


def test_the_worker_streams_a_sentence_piece_by_piece(
    worker: WorkerSpeech,
) -> None:
    assert not worker.loaded
    worker.warm(POCKET)
    assert worker.loaded
    pieces = list(worker.stream("Hola a todos.", POCKET, "es",
                                CancelToken()))
    assert pieces == [(b"\x01\x00" * 240, 24000)] * 3
    assert len(list(worker.stream("REF.", QWEN3, "es", CancelToken()))) == 1
    with pytest.raises(TtsUnavailable, match="boom"):
        list(worker.stream("FAIL.", POCKET, "es", CancelToken()))
    with pytest.raises(TtsUnavailable, match="stopped"):
        list(worker.stream("DIE.", POCKET, "es", CancelToken()))
    assert not worker.loaded
    assert list(worker.stream("Otra vez.", POCKET, "es", CancelToken())), \
        "a dead worker is started again"


def test_a_cancelled_sentence_stops_and_the_next_one_is_clean(
    worker: WorkerSpeech,
) -> None:
    cancel = CancelToken()
    seen = []
    for piece in worker.stream("LONG sentence.", POCKET, "es", cancel):
        seen.append(piece)
        if len(seen) == 2:
            cancel.cancel()
    assert 2 <= len(seen) < 200
    assert len(list(worker.stream("Hola.", POCKET, "es",
                                  CancelToken()))) == 3
    stream = cast(Generator[Tuple[bytes, int], None, None], worker.stream(
        "LONG again.", POCKET, "es", CancelToken()))
    next(stream)
    stream.close()  # the caller stopped reading: cancelled and drained
    assert len(list(worker.stream("Hola.", POCKET, "es",
                                  CancelToken()))) == 3


def test_a_stalled_or_missing_worker_is_unavailable(
    worker: WorkerSpeech, tmp_path: Path,
) -> None:
    with pytest.raises(TtsUnavailable, match="in time"):
        list(worker.stream("STALL.", POCKET, "es", CancelToken()))
    missing = WorkerSpeech([str(tmp_path / "no-python")])
    with pytest.raises(TtsUnavailable, match="cannot run"):
        missing.warm(POCKET)
    silent = WorkerSpeech([sys.executable, "-c",
                           "import time; time.sleep(3)"], load_timeout=0.3,
                          stop_timeout=0.2)
    with pytest.raises(TtsUnavailable, match="in time"):
        silent.warm(POCKET)
    silent.close()
    assert _piece({"pcm": "AAA=", "sample_rate": 0}) is None
    assert _piece({"pcm": "", "sample_rate": 24000}) is None


def install(home: Path, engines: List[str]) -> None:
    for engine in engines:
        python = home / f"{engine}-venv" / "bin" / "python"
        python.parent.mkdir(parents=True)
        python.write_text("", encoding="utf-8")


def test_engines_are_found_shared_on_the_cpu_and_stopped_on_the_gpu(
    tmp_path: Path,
) -> None:
    home = tmp_path / "natural"
    engines = WorkerEngines(home)
    assert engines.installed() == frozenset()
    assert engines.reference() is None
    install(home, ["pocket", "qwen3"])
    (home / REFERENCE).parent.mkdir(parents=True)
    (home / REFERENCE).write_bytes(b"RIFF")
    assert engines.installed() == {"pocket", "qwen3"}
    reference = engines.reference()
    assert reference is not None and reference[0].endswith(".wav")
    cpu = engines.get(POCKET)
    assert engines.get(POCKET) is cpu, "the CPU engine is shared"
    gpu = engines.get(QWEN3)
    assert gpu is not engines.get(QWEN3), "a GPU engine is the caller's"
    assert isinstance(cpu, WorkerSpeech) and isinstance(gpu, WorkerSpeech)
    assert cpu.command[2:] == ["--engine", "pocket", "--device", "cpu",
                               "--voice", "lola"]
    assert gpu.command[-4:] == ["--warm-reference", "/r.wav",
                                "--warm-reference-text", "Hola."]
    assert cpu.env is not None and cpu.env["HF_HUB_OFFLINE"] == "1"
    assert cpu.env["HF_HOME"] == str(home / "hf")
    assert "PYTHONPATH" not in cpu.env
    closed: List[WorkerSpeech] = []
    cpu.close = lambda: closed.append(cpu)  # type: ignore[method-assign]
    gpu.close = lambda: closed.append(gpu)  # type: ignore[method-assign]
    engines.release(cpu)
    engines.release(gpu)
    assert closed == [gpu]
    engines.close()
    assert closed == [gpu, cpu]


def lease(directory: Path, holder: str = "ragputin-live",
          priority: str = "live", pid: Optional[int] = None,
          expires_in: float = 10.0) -> None:
    now = datetime.now(timezone.utc)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "lease.json").write_text(json.dumps({
        "schema": "gpu-lease.v1", "holder": holder, "priority": priority,
        "pid": pid if pid is not None else os.getpid(),
        "acquired_at": now.isoformat(),
        "expires_at": (now + timedelta(seconds=expires_in)).isoformat()
        .replace("+00:00", "Z"),
        "starts_at": None, "need_mib": 2500}), encoding="utf-8")


def test_the_gate_allows_the_gpu_only_under_a_live_lease(
    tmp_path: Path,
) -> None:
    free = [6000]
    gate = LeaseGpuGate(tmp_path, free_mib=lambda: free[0],
                        sleep=lambda _: None)
    assert gate.check(4400) == "no live GPU lease"
    lease(tmp_path)
    assert gate.check(4400) is None
    free[0] = 3000
    assert gate.check(4400) == "3000 MiB free on the GPU, 4400 needed"
    assert gate.check(0) is None, "a loaded engine needs nothing more"
    lease(tmp_path, holder="transcriber-batch", priority="batch")
    assert gate.check(0) == "the GPU lease is held by transcriber-batch"
    lease(tmp_path, expires_in=-1)
    assert gate.check(0) == "no live GPU lease"
    lease(tmp_path, pid=0)
    assert gate.check(0) == "no live GPU lease", "its holder is gone"
    for bad in ("{nope", "[1]", '{"expires_at": "x", "pid": 1}',
                '{"expires_at": "2026-11-21T10:00:00", "pid": 1}'):
        (tmp_path / "lease.json").write_text(bad, encoding="utf-8")
        assert gate.check(0) == "no live GPU lease"
    nogpu = LeaseGpuGate(tmp_path, free_mib=lambda: None)
    lease(tmp_path)
    assert nogpu.check(800) == "no GPU found"


def test_a_sample_takes_the_lease_only_when_nobody_holds_it(
    tmp_path: Path,
) -> None:
    free = [1000, 1000, 6000]
    gate = LeaseGpuGate(tmp_path / "gpu-lease",
                        free_mib=lambda: free.pop(0) if len(free) > 1
                        else free[0], sleep=lambda _: None)
    assert gate.take(4400) is None, "the memory was freed meanwhile"
    written = json.loads((tmp_path / "gpu-lease" / "lease.json")
                         .read_text(encoding="utf-8"))
    assert written["holder"] == HOLDER and written["priority"] == "live"
    assert written["pid"] == os.getpid() and written["need_mib"] == 4400
    assert gate.check(0) is None
    gate.drop()
    assert not (tmp_path / "gpu-lease" / "lease.json").exists()
    gate.drop()
    lease(tmp_path / "gpu-lease", pid=os.getppid())
    assert gate.take(100) is None
    gate.drop()
    assert (tmp_path / "gpu-lease" / "lease.json").exists(), \
        "someone else's lease is never removed"
    lease(tmp_path / "gpu-lease", holder="batch-job", priority="batch",
          pid=os.getppid())
    assert gate.take(100) == "the GPU lease is held by batch-job"
    short = LeaseGpuGate(tmp_path / "other", free_mib=lambda: 10,
                         sleep=lambda _: None, wait_seconds=1.0)
    assert short.take(4400) == "10 MiB free on the GPU, 4400 needed"
    short.drop()


def test_the_lease_is_renewed_while_held(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from orchestrator.adapters.outbound import gpu_gate

    monkeypatch.setattr(gpu_gate, "HEARTBEAT_SECONDS", 0.05)
    gate = LeaseGpuGate(tmp_path, free_mib=lambda: 9000)
    assert gate.take(100) is None
    first = json.loads((tmp_path / "lease.json").read_text())["expires_at"]
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        now = json.loads((tmp_path / "lease.json").read_text())["expires_at"]
        if now != first:
            break
        time.sleep(0.02)
    assert now != first
    gate.drop()


def test_the_lease_folder_and_the_probes() -> None:
    assert lease_directory({"XDG_RUNTIME_DIR": "/run/user/7"}) == Path(
        "/run/user/7/gpu-lease")
    assert lease_directory({}).name.startswith("gpu-lease-")
    assert process_alive(os.getpid()) and not process_alive(0)
    free = nvidia_free_mib()
    assert free is None or free >= 0


def test_the_sample_route_says_which_voice_spoke() -> None:
    live = bot_sessions([], tts=FakeTts())
    web = TestClient(create_app(live, None))
    answer = web.post("/voicebot/sample", json={
        "choices": {"voicebot.voice": "pocket:lola"}})
    assert answer.status_code == 200
    assert answer.headers["x-voice-engine"] == "piper"
    assert answer.headers["x-voice-note"] == "natural voices off"


def test_serve_wires_the_natural_voices_when_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = argparse.Namespace(natural_tts="on", tts="piper",
                              natural_dir=tmp_path, gpu_lease_dir=None,
                              natural_budget_ms=700.0)
    assert cli.build_natural(args) is None, "nothing installed"
    install(tmp_path, ["qwen3"])
    wiring = cli.build_natural(args)
    assert isinstance(wiring, NaturalWiring)
    assert wiring.budget_ms == 700.0 and wiring.reference is None
    warmed: List[EngineVoice] = []
    monkeypatch.setattr(WorkerSpeech, "warm",
                        lambda self, voice: warmed.append(voice))
    install(tmp_path, ["pocket"])
    args.gpu_lease_dir = tmp_path / "lease"
    wiring = cli.build_natural(args)
    assert wiring is not None and wiring.gate is not None
    deadline = time.monotonic() + 3
    while not warmed and time.monotonic() < deadline:
        time.sleep(0.01)
    assert warmed == [POCKET], "Pocket on the CPU is warmed at start"

    def failing(self: WorkerSpeech, voice: EngineVoice) -> None:
        warmed.append(voice)
        raise TtsUnavailable("no model")

    monkeypatch.setattr(WorkerSpeech, "warm", failing)
    assert cli.build_natural(args) is not None
    deadline = time.monotonic() + 3
    while len(warmed) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    args.natural_tts = "off"
    assert cli.build_natural(args) is None
    parsed = cli.build_parser().parse_args(["serve"])
    assert parsed.natural_tts == "on" and parsed.natural_budget_ms == 800.0


def test_two_samples_at_once_share_one_lease_and_the_last_drop_deletes(
    tmp_path: Path,
) -> None:
    gate = LeaseGpuGate(tmp_path / "gpu-lease", free_mib=lambda: 6000,
                        sleep=lambda _: None)
    path = tmp_path / "gpu-lease" / "lease.json"
    assert gate.take(4400) is None and path.exists()
    assert gate.take(800) is None, "a second sample: the lease is ours"
    gate.drop()
    assert path.exists(), "the other sample is still speaking"
    gate.drop()
    assert not path.exists()
    # a take over the live session's lease is balanced by its drop too
    lease(tmp_path / "gpu-lease", pid=os.getppid())
    assert gate.take(100) is None
    assert gate.take(100) is None
    gate.drop()
    gate.drop()
    assert path.exists(), "the live session's lease is never removed"


def test_serve_gives_the_voices_and_the_cloner_one_gate(
    tmp_path: Path,
) -> None:
    args = argparse.Namespace(gpu_lease_dir=tmp_path / "lease")
    gate = cli.build_gate(args)
    assert gate.path == tmp_path / "lease" / "lease.json"
    install(tmp_path / "n", ["qwen3"])
    args.natural_tts, args.tts = "on", "piper"
    args.natural_dir, args.natural_budget_ms = tmp_path / "n", 800.0
    wiring = cli.build_natural(args, gate)
    assert wiring is not None and wiring.gate is gate
