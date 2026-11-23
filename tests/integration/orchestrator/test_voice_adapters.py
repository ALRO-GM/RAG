"""The voice bot's voice adapters (#207): profiles on disk, the cloning
worker's line protocol (a stand-in worker, no model), Piper's control
flags, the sample route and its proxy."""

import json
import sys
import textwrap
import wave
from io import BytesIO
from pathlib import Path
from typing import Iterator, List

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
    assert not any(cloner.workdir.iterdir()), "the WAV is deleted"
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
