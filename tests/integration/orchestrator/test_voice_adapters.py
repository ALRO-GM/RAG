"""The voice bot's voice adapters (#207): profiles on disk, the cloning
worker's line protocol (a stand-in worker, no model), Piper's control
flags, the sample route and its proxy."""

import json
import sys
import textwrap
import wave
from io import BytesIO
from pathlib import Path
from typing import Any, Iterator, List

import httpx
import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from orchestrator.adapters.inbound.http_server import create_app
from orchestrator.adapters.outbound.fakes import FakeTts
from orchestrator.adapters.outbound.fs_voice_profiles import DirVoiceProfiles
from orchestrator.adapters.outbound.piper_tts import PiperTts
from orchestrator.adapters.outbound.subprocess_cloner import (
    SubprocessCloner,
)
from orchestrator.application.ports import CancelToken, TtsUnavailable
from orchestrator.domain.speech import tts_request
from tests.integration.orchestrator.test_piper_tts import FAKE_PIPER
from tests.unit.orchestrator.test_voicebot_sessions import bot_sessions
from transcriber.adapters.inbound.live_assist_proxy import (
    add_live_assist_proxy,
)

FAKE_WORKER = textwrap.dedent('''
    import json, sys, wave
    print("loading", file=sys.stderr, flush=True)
    print("not json", flush=True)
    print(json.dumps({"ready": True, "engine": "fake",
                      "sample_rate": 24000}), flush=True)
    for line in sys.stdin:
        req = json.loads(line)
        if req["text"] == "DIE.":
            sys.exit(2)
        if req["text"] == "FAIL.":
            print(json.dumps({"id": req["id"], "error": "boom"}), flush=True)
            continue
        print(json.dumps({"id": "other"}), flush=True)
        if req["text"] != "NOWAV.":
            with wave.open(req["out"], "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
                w.writeframes(b"\\x03\\x00" * 50 * len(req["text"])
                              + (b"\\x09\\x00"
                                 if req.get("length_scale") == 1.4
                                 else b""))
        print(json.dumps({"id": req["id"], "path": req["out"],
                          "sample_rate": 24000, "synth_ms": 1.0}),
              flush=True)
''')


def write_profile(folder: Path, speaker: str, clone: bool,
                  rate: object = 3.5, clip: bool = True) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{speaker}.json").write_text(json.dumps({
        "voiceprint": {"speaker_id": speaker, "words_per_second": rate},
        "consent": {"speaker_id": speaker, "voice_clone": clone},
    }), encoding="utf-8")
    if clip:
        (folder / f"{speaker}.clone.wav").write_bytes(b"RIFF")


def test_profiles_are_read_from_the_transcriber_s_files(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "voiceprints"
    write_profile(folder, "ana", clone=True)
    write_profile(folder, "bea", clone=False, rate=True)
    write_profile(folder, "cid", clone=True, clip=False)
    (folder / "bad.json").write_text("{nope", encoding="utf-8")
    (folder / "odd.json").write_text("[1]", encoding="utf-8")
    (folder / "half.json").write_text('{"voiceprint": 1}', encoding="utf-8")
    profiles = DirVoiceProfiles(folder)
    ana = profiles.get("ana")
    assert ana is not None and ana.clonable
    assert ana.words_per_second == 3.5
    assert ana.reference == str(folder / "ana.clone.wav")
    bea = profiles.get("bea")
    assert bea is not None and not bea.clone_consent
    assert bea.reference is None, "no clip is used without consent"
    assert bea.words_per_second is None
    cid = profiles.get("cid")
    assert cid is not None and not cid.clonable
    for missing in ("nobody", "bad", "odd", "half", "../ana", "a/b", ""):
        assert profiles.get(missing) is None, missing


@pytest.fixture
def cloner(tmp_path: Path) -> Iterator[SubprocessCloner]:
    script = tmp_path / "worker.py"
    script.write_text(FAKE_WORKER, encoding="utf-8")
    worker = SubprocessCloner(Path("unused"), "fake", timeout=5.0,
                              load_timeout=5.0,
                              command=[sys.executable, str(script)])
    yield worker
    worker.close()


def test_the_worker_speaks_one_sentence_per_line(
    cloner: SubprocessCloner, tmp_path: Path
) -> None:
    reference = tmp_path / "ref.wav"
    reference.write_bytes(b"RIFF")
    cloner.warm(str(reference), "es")
    pcm, rate = cloner.speak("Hola a todos.", str(reference), "es",
                             CancelToken())
    assert rate == 24000 and len(pcm) == 2 * 50 * len("Hola a todos.")
    assert not list(cloner.workdir.glob("*.wav")), "the WAV is deleted"
    with pytest.raises(TtsUnavailable, match="boom"):
        cloner.speak("FAIL.", str(reference), "es", CancelToken())
    with pytest.raises(TtsUnavailable, match="missing"):
        cloner.speak("NOWAV.", str(reference), "es", CancelToken())
    with pytest.raises(TtsUnavailable, match="stopped"):
        cloner.speak("DIE.", str(reference), "es", CancelToken())
    pcm, _ = cloner.speak("Otra vez.", str(reference), "es", CancelToken())
    assert pcm, "a dead worker is started again"
    with pytest.raises(TtsUnavailable, match="reference"):
        cloner.speak("Hola.", str(tmp_path / "none.wav"), "es",
                     CancelToken())


def test_controls_reach_the_worker_and_unload_frees_it(
    cloner: SubprocessCloner, tmp_path: Path
) -> None:
    reference = tmp_path / "ref.wav"
    reference.write_bytes(b"RIFF")
    assert cloner.gpu is True and cloner.loaded is False
    cloner.warm(str(reference), "es", {"length_scale": 1.4})
    assert cloner.loaded is True
    plain, _ = cloner.speak("Hola.", str(reference), "es", CancelToken())
    slow, _ = cloner.speak("Hola.", str(reference), "es", CancelToken(),
                           {"length_scale": 1.4, "noise_w": 0.5})
    assert len(slow) == len(plain) + 2, "the control was in the request"
    cloner.unload()
    assert cloner.loaded is False, "the worker is stopped: memory free"
    cloner.unload()
    cpu = SubprocessCloner(Path("unused"), "fake", device="cpu",
                           command=[sys.executable, "-c", "pass"])
    assert cpu.gpu is False
    cpu.close()


def test_unload_kills_a_load_in_progress_without_waiting_for_it(
    tmp_path: Path,
) -> None:
    import threading
    import time

    reference = tmp_path / "ref.wav"
    reference.write_bytes(b"RIFF")
    slow = SubprocessCloner(
        Path("unused"), "fake", load_timeout=30.0,
        command=[sys.executable, "-c", "import time; time.sleep(60)"])
    outcome: List[str] = []

    def load() -> None:
        try:
            slow.warm(str(reference), "es")
        except TtsUnavailable as exc:
            outcome.append(str(exc))

    thread = threading.Thread(target=load)
    thread.start()
    end = time.monotonic() + 5
    while not slow.loaded and time.monotonic() < end:
        time.sleep(0.01)
    assert slow.loaded, "the process is up while the model would load"
    started = time.monotonic()
    slow.unload()
    thread.join(5)
    assert not thread.is_alive() and "stopped" in outcome[0]
    assert time.monotonic() - started < 3, "killed, not waited for"
    assert slow.loaded is False
    slow.release()  # a GPU worker with no holder: nothing left to stop
    slow.close()


TMP_WORKER = textwrap.dedent('''
    import json, sys, tempfile
    d = tempfile.TemporaryDirectory(prefix="voice-clone-")
    print(json.dumps({"ready": True, "sample_rate": 24000}), flush=True)
    for line in sys.stdin:
        pass
    d.cleanup()
''')


def test_a_ready_worker_exits_on_eof_and_leaves_no_temp_dir(
    tmp_path: Path,
) -> None:
    script = tmp_path / "worker.py"
    script.write_text(TMP_WORKER, encoding="utf-8")
    reference = tmp_path / "ref.wav"
    reference.write_bytes(b"RIFF")
    worker = SubprocessCloner(Path("unused"), "fake", load_timeout=5.0,
                              command=[sys.executable, str(script)])
    worker._ensure()
    assert any(worker.worker_tmp.glob("voice-clone-*")), "dir exists"
    process = worker._process
    assert process is not None
    worker.unload()
    assert process.returncode == 0, "it left by itself, not killed"
    assert not worker.worker_tmp.exists() or not any(
        worker.worker_tmp.iterdir())
    worker.close()


def test_a_loading_worker_is_killed_but_its_temp_dir_is_cleaned(
    tmp_path: Path,
) -> None:
    import threading
    import time

    script = tmp_path / "loader.py"
    script.write_text(
        "import tempfile, time\n"
        "d = tempfile.TemporaryDirectory(prefix='voice-clone-')\n"
        "time.sleep(60)\n", encoding="utf-8")
    worker = SubprocessCloner(Path("unused"), "fake", load_timeout=30.0,
                              command=[sys.executable, str(script)])
    thread = threading.Thread(
        target=lambda: pytest.raises(TtsUnavailable, worker._ensure))
    thread.start()
    end = time.monotonic() + 5
    while (not any(worker.worker_tmp.glob("voice-clone-*"))
           and time.monotonic() < end):
        time.sleep(0.01)
    started = time.monotonic()
    worker.unload()
    thread.join(5)
    assert time.monotonic() - started < 1.5, "no grace while loading"
    assert not thread.is_alive()
    assert not worker.worker_tmp.exists()
    worker.close()


def test_an_unload_during_the_start_aborts_that_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deterministic race: unload() lands while Popen is in flight."""
    import subprocess

    worker = SubprocessCloner(
        Path("unused"), "fake", load_timeout=30.0,
        command=[sys.executable, "-c", "import time; time.sleep(60)"])
    real = subprocess.Popen

    def popen(*args: Any, **kwargs: Any) -> Any:
        process = real(*args, **kwargs)
        worker.unload()  # the watcher's unload sees no process yet
        return process

    monkeypatch.setattr(subprocess, "Popen", popen)
    with pytest.raises(TtsUnavailable, match="stopped"):
        worker._ensure()
    assert worker.loaded is False
    monkeypatch.setattr(subprocess, "Popen", real)
    worker.close()


def test_a_worker_that_cannot_start_or_load_is_unavailable(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "ref.wav"
    reference.write_bytes(b"RIFF")
    missing = SubprocessCloner(tmp_path / "no-python", "chatterbox")
    assert missing.command[2:] == ["--engine", "chatterbox", "--device",
                                   "cuda"]
    with pytest.raises(TtsUnavailable, match="cannot run"):
        missing.speak("Hola.", str(reference), "es", CancelToken())
    missing.close()
    silent = SubprocessCloner(
        Path("unused"), "fake", load_timeout=0.3,
        command=[sys.executable, "-c", "import time; time.sleep(3)"])
    with pytest.raises(TtsUnavailable, match="in time"):
        silent.speak("Hola.", str(reference), "es", CancelToken())
    silent.close()


def test_piper_gets_the_control_flags(tmp_path: Path) -> None:
    script = tmp_path / "fake_piper.py"
    script.write_text(FAKE_PIPER, encoding="utf-8")
    voices = tmp_path / "voices"
    voices.mkdir()
    (voices / "es_ES-test-medium.onnx").write_bytes(b"onnx")
    (tmp_path / "out").mkdir()
    piper = PiperTts(voices_dir=voices, timeout=2.0,
                     command=[sys.executable, str(script)],
                     workdir=tmp_path / "out")
    try:
        doc = tts_request("r1", "Hola a todos.", "es_ES-test-medium",
                          length_scale=1.2, noise_scale=0.3, noise_w=0.6,
                          sentence_silence=0.25)
        assert list(piper.synthesize(doc, CancelToken()))
        [process] = piper._processes.values()
        args: List[str] = list(process.process.args)  # type: ignore
        assert args[args.index("--length-scale") + 1] == "1.2"
        assert args[args.index("--noise-scale") + 1] == "0.3"
        assert args[args.index("--noise-w-scale") + 1] == "0.6"
        assert args[args.index("--sentence-silence") + 1] == "0.25"
    finally:
        piper.close()


def test_the_sample_route_answers_a_wav() -> None:
    web = TestClient(create_app(bot_sessions([], tts=FakeTts()), "tok"))
    auth = {"Authorization": "Bearer tok"}
    answer = web.post("/voicebot/sample", headers=auth, json={
        "choices": {"voicebot.length_scale": "1.3"}})
    assert answer.status_code == 200
    assert answer.headers["content-type"] == "audio/wav"
    assert answer.headers["x-voice-length-scale"] == "1.3"
    assert answer.headers["x-voice-clone"] == "0"
    with wave.open(BytesIO(answer.content), "rb") as reader:
        assert reader.getnframes() > 0
    for bad in ({"choices": {"live.asr": "gpu"}}, {"choices": [1]},
                {"choices": {"voicebot.voice": 3}}):
        assert web.post("/voicebot/sample", headers=auth,
                        json=bad).status_code == 422
    assert web.post("/voicebot/sample", json={}).status_code == 401
    quiet = TestClient(create_app(bot_sessions([]), None))
    assert quiet.post("/voicebot/sample", json={}).status_code == 503


def test_the_proxy_forwards_the_sample_and_its_headers() -> None:
    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=b"RIFFwav", headers={
            "content-type": "audio/wav", "x-voice-clone": "1",
            "x-other": "no"})

    api = APIRouter(prefix="/api")
    add_live_assist_proxy(api, None, "http://orch:8772",
                          transport=httpx.MockTransport(handler))
    app = FastAPI()
    app.include_router(api)
    answer = TestClient(app).post("/api/assist/voicebot/sample",
                                  json={"choices": {}})
    assert answer.status_code == 200 and answer.content == b"RIFFwav"
    assert answer.headers["content-type"] == "audio/wav"
    assert answer.headers["x-voice-clone"] == "1"
    assert "x-other" not in answer.headers
    assert str(seen[0].url) == "http://orch:8772/voicebot/sample"


def test_the_cloner_is_on_only_with_its_venv(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from orchestrator.adapters.inbound import cli

    parse = cli.build_parser().parse_args
    assert cli.build_cloner(parse(["serve", "--clone-engine", "none"])) \
        is None
    assert cli.build_cloner(parse(["serve", "--tts", "fake"])) is None
    args = parse(["serve", "--clone-python", str(tmp_path / "none")])
    assert cli.build_cloner(args) is None
    assert "voice cloning off" in capsys.readouterr().err
    python = tmp_path / "python"
    python.write_text("", encoding="utf-8")
    args = parse(["serve", "--clone-python", str(python), "--voices-dir",
                  str(tmp_path), "--clone-device", "cpu"])
    cloner = cli.build_cloner(args)
    assert cloner is not None
    assert cloner.command[2:6] == ["--engine", "openvoice", "--device",
                                   "cpu"]
    assert cloner.command[-1] == str(tmp_path / "es_ES-davefx-medium.onnx")
    assert args.clone_budget_ms == 800.0
    cloner.close()


def test_a_stale_unload_cannot_kill_a_newer_worker(
    cloner: SubprocessCloner,
) -> None:
    """Deterministic: the decision is taken about start 1, start 2 runs."""
    cloner._ensure()
    stale = cloner.generation
    cloner.unload(stale)  # the decided stop of the old worker
    assert not cloner.loaded
    cloner._ensure()  # another session starts a new worker
    assert cloner.generation == stale + 1 and cloner.loaded
    cloner.unload(stale)  # the late, stale decision arrives
    assert cloner.loaded, "a stale unload must not kill the new worker"
    cloner.unload(cloner.generation)
    assert not cloner.loaded
    cloner.unload()  # no generation: unconditional, as before


def test_a_stale_unload_during_a_start_does_not_abort_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stale unload lands while Popen is in flight: ignored."""
    import subprocess

    script = tmp_path / "worker.py"
    script.write_text(FAKE_WORKER, encoding="utf-8")
    worker = SubprocessCloner(Path("unused"), "fake", load_timeout=5.0,
                              command=[sys.executable, str(script)])
    stale = worker.generation
    real = subprocess.Popen

    def popen(*args: Any, **kwargs: Any) -> Any:
        process = real(*args, **kwargs)
        worker.unload(stale)
        return process

    monkeypatch.setattr(subprocess, "Popen", popen)
    worker._ensure()
    assert worker.loaded
    monkeypatch.setattr(subprocess, "Popen", real)
    worker.close()


def test_piper_children_die_with_the_worker(tmp_path: Path) -> None:
    """Unload and stop kill the worker's whole process group."""
    import time

    pidfile = tmp_path / "child.pid"
    script = tmp_path / "parent.py"
    script.write_text(
        "import json, subprocess, sys\n"
        "c = subprocess.Popen([sys.executable, '-c',"
        " 'import time; time.sleep(60)'])\n"
        f"open({str(pidfile)!r}, 'w').write(str(c.pid))\n"
        "print(json.dumps({'ready': True}), flush=True)\n"
        "sys.stdin.read()\n"
        "import time; time.sleep(60)\n", encoding="utf-8")

    def alive(pid: int) -> bool:
        try:
            state = Path(f"/proc/{pid}/stat").read_text().split()[2]
        except OSError:
            return False
        return state != "Z"

    for graceful in (True, False):
        pidfile.unlink(missing_ok=True)
        worker = SubprocessCloner(Path("unused"), "fake", load_timeout=5.0,
                                  command=[sys.executable, str(script)])
        worker._ensure()
        child = int(pidfile.read_text())
        assert alive(child)
        if graceful:
            worker.unload()  # EOF, grace, then the group is killed
        else:
            worker._stop()
        end = time.monotonic() + 5
        while alive(child) and time.monotonic() < end:
            time.sleep(0.01)
        assert not alive(child), "Piper outlived the worker"
        worker.close()


EXITS_AFTER_EOF = textwrap.dedent('''
    import json, subprocess, sys
    c = subprocess.Popen([sys.executable, "-c",
                          "import time; time.sleep(60)"])
    open(sys.argv[1], "w").write(str(c.pid))
    print(json.dumps({"ready": True}), flush=True)
    sys.stdin.read()
''')


def pid_alive(pid: int) -> bool:
    """Whether ``pid`` runs (a zombie is not running)."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except OSError:
        return False


def test_a_worker_that_exits_after_eof_still_loses_its_piper_child(
    tmp_path: Path,
) -> None:
    import time

    pidfile = tmp_path / "child.pid"
    script = tmp_path / "parent.py"
    script.write_text(EXITS_AFTER_EOF, encoding="utf-8")
    worker = SubprocessCloner(
        Path("unused"), "fake", load_timeout=5.0,
        command=[sys.executable, str(script), str(pidfile)])
    worker._ensure()
    child = int(pidfile.read_text())
    worker.unload()  # EOF: the worker exits by itself, the child stays
    end = time.monotonic() + 5
    while pid_alive(child) and time.monotonic() < end:
        time.sleep(0.01)
    assert not pid_alive(child)
    worker.close()


def test_no_signal_is_ever_sent_to_a_reaped_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    script = tmp_path / "worker.py"
    script.write_text(FAKE_WORKER, encoding="utf-8")
    worker = SubprocessCloner(Path("unused"), "fake", load_timeout=5.0,
                              command=[sys.executable, str(script)])
    sent: List[int] = []
    real = os.killpg

    def killpg(pgid: int, sig: int) -> None:
        sent.append(pgid)
        real(pgid, sig)

    monkeypatch.setattr(os, "killpg", killpg)
    worker._ensure()
    pid = worker._process.pid  # type: ignore[union-attr]
    worker.unload()  # graceful: exits on EOF, then killed once and reaped
    assert sent == [pid]
    assert worker.loaded is False
    worker.unload()  # again, the stale handle: nothing is signalled
    worker._stop()
    worker.close()
    assert sent == [pid], "a reaped pid may belong to somebody else"


def test_warming_a_running_worker_makes_older_decisions_stale(
    cloner: SubprocessCloner, tmp_path: Path
) -> None:
    clip = tmp_path / "ref.wav"
    clip.write_bytes(b"x")
    cloner.warm(str(clip), "es")
    first = cloner.generation
    cloner.warm(str(clip), "es")  # a new session reuses the live worker
    assert cloner.generation == first + 1
    cloner.unload(first)
    assert cloner.loaded, "the decision was about the previous hold"
