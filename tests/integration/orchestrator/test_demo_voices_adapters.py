"""The examples' voices on disk, on the command line and over HTTP (#254)."""

import json
import os
import shutil
import stat
import subprocess
from pathlib import Path
from typing import AbstractSet, Any, Dict, Iterator, List, Optional, Tuple

import pytest
from fastapi.testclient import TestClient

from orchestrator.adapters.inbound import cli
from orchestrator.adapters.inbound.demo_voices_cli import parse_pairs
from orchestrator.adapters.inbound.http_server import create_app
from orchestrator.adapters.outbound.demo_speaker import (
    EngineDemoSpeaker,
    ToneDemoSpeaker,
)
from orchestrator.adapters.outbound.fakes import (
    FakeKnowledge,
    FakeReasoner,
    FakeTts,
)
from orchestrator.adapters.outbound.fs_demo_voices import (
    FsDemoVoices,
    check_outside_repository,
)
from orchestrator.adapters.outbound.fs_demos import DirDemos
from orchestrator.adapters.outbound.fs_voice_profiles import DirVoiceProfiles
from orchestrator.adapters.outbound.runtime import InlineExecutor
from orchestrator.application.demo_voices import (
    DemoVoiceLibrary,
    RenderDemoVoices,
)
from orchestrator.application.ports import CancelToken, TtsUnavailable
from orchestrator.application.sessions import LiveSessions
from orchestrator.domain.demo_voices import DemoVoiceManifest, parse_voice
from orchestrator.domain.voices import EngineVoice
from tests.unit.orchestrator.fixtures import REPO, product
from tests.unit.orchestrator.test_demo_voices import DEMOS, STOCK
from tests.unit.orchestrator.test_sessions import (
    Clock,
    InlineRunner,
    Transcripts,
)

CALL = "llamada-lumbre"


def mode(path: Path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


# ---- the store ----

def test_clips_and_manifest_are_private_and_checked(tmp_path: Path) -> None:
    store = FsDemoVoices(tmp_path / "dv")
    assert store.manifest(CALL) is None
    store.write_clip(CALL, "t0.s0-abc.wav", b"RIFFxxxx")
    store.save_manifest(DemoVoiceManifest(demo=CALL, voices=STOCK))
    folder = tmp_path / "dv" / CALL
    assert mode(tmp_path / "dv") == mode(folder) == 0o700
    assert mode(folder / "t0.s0-abc.wav") == 0o600
    assert mode(folder / "manifest.json") == 0o600
    assert json.loads((folder / "manifest.json").read_text())["schema"] \
        == "demo.voices.v1"
    assert store.manifest(CALL) == DemoVoiceManifest(demo=CALL,
                                                     voices=STOCK)
    assert store.read_clip(CALL, "t0.s0-abc.wav") == b"RIFFxxxx"
    assert store.has_clip(CALL, "t0.s0-abc.wav")
    for bad in ("../x.wav", "x.mp3", "", ".hidden.wav"):
        assert not store.has_clip(CALL, bad)
        with pytest.raises(KeyError):
            store.read_clip(CALL, bad)
        store.delete_clip(CALL, bad)
    with pytest.raises(KeyError):
        store.write_clip("../up", "a.wav", b"")
    with pytest.raises(KeyError):
        store.read_clip(CALL, "missing.wav")
    (folder / "manifest.json").write_text("{", encoding="utf-8")
    assert store.manifest(CALL) is None
    store.save_manifest(DemoVoiceManifest(demo="other"))
    shutil.copy(tmp_path / "dv" / "other" / "manifest.json",
                folder / "manifest.json")
    assert store.manifest(CALL) is None, "a manifest of another demo"
    store.delete_clip(CALL, "t0.s0-abc.wav")
    assert not store.has_clip(CALL, "t0.s0-abc.wav")
    store.write_clip(CALL, "t1.s0-abc.wav", b"RIFF")
    assert store.clear(CALL) == 1 and not folder.exists()
    assert store.clear(CALL) == 0


def test_the_clips_never_go_where_git_sees_them(tmp_path: Path) -> None:
    check_outside_repository(tmp_path / "dv", REPO)
    check_outside_repository(REPO / "data" / "transcriber" / "demo-voices",
                             REPO)
    for inside in (REPO, REPO / "examples" / "demo", REPO / "datax"):
        with pytest.raises(ValueError, match="inside the repository"):
            check_outside_repository(inside, REPO)


def test_no_audio_file_is_committed() -> None:
    """The repository is public: no recording or demo clip is tracked."""
    if shutil.which("git") is None or not (REPO / ".git").exists():
        pytest.skip("not a git checkout")
    listed = subprocess.run(
        ["git", "ls-files", "-z"], cwd=REPO, capture_output=True,
        check=True).stdout.decode("utf-8", "replace").split("\0")
    audio = (".wav", ".mp3", ".ogg", ".opus", ".flac", ".m4a", ".webm",
             ".aac")
    # The engine study's synthetic samples are the one audio kept.
    known = "experiments/live/tts_samples/"
    assert [f for f in listed if f.lower().endswith(audio)
            and not f.startswith(known)] == []
    ignored = subprocess.run(
        ["git", "check-ignore", "-q",
         "data/transcriber/demo-voices/x/t0.s0-a.wav"], cwd=REPO)
    assert ignored.returncode == 0, "the default clip folder is ignored"


# ---- the speaker over the voice bot's engines ----

class Stream:
    """A SpeechStream that yields two pieces per sentence."""

    def __init__(self) -> None:
        self.said: List[Tuple[str, EngineVoice]] = []
        self.loaded = True

    def stream(self, text: str, voice: EngineVoice, language: str,
               cancel: CancelToken) -> Iterator[Tuple[bytes, int]]:
        self.said.append((text, voice))
        yield b"\x01\x00" * 10, 24000
        yield b"\x02\x00" * 10, 24000

    def warm(self, voice: EngineVoice) -> None:
        return None


class Engines:
    """SpeechEngines with one stream per worker voice."""

    def __init__(self, installed: AbstractSet[str] = frozenset(
            {"pocket", "qwen3"})):
        self._installed = installed
        self.streams: Dict[EngineVoice, Stream] = {}
        self.released: List[Stream] = []

    def installed(self) -> AbstractSet[str]:
        return self._installed

    def get(self, voice: EngineVoice) -> Stream:
        return self.streams.setdefault(voice, Stream())

    def release(self, engine: Any) -> None:
        self.released.append(engine)


class Cloner:
    """A VoiceCloner on the CPU."""

    gpu = False
    loaded = True
    generation = 1

    def __init__(self) -> None:
        self.released = 0

    def speak(self, text: str, reference: str, language: str,
              cancel: CancelToken,
              controls: Optional[Dict[str, float]] = None
              ) -> Tuple[bytes, int]:
        return b"\x03\x00" * 5, 22050

    def warm(self, reference: str, language: str,
             controls: Optional[Dict[str, float]] = None) -> None:
        return None

    def unload(self, generation: Optional[int] = None) -> None:
        return None

    def release(self) -> None:
        self.released += 1


def test_each_engine_says_the_whole_line() -> None:
    engines = Engines()
    tts = FakeTts(rate=22050)
    cloner = Cloner()
    speaker = EngineDemoSpeaker(tts, engines, cloner)  # type: ignore[arg-type]
    text = "Hola, buenas tardes. ¿En qué puedo ayudarle hoy mismo?"
    pcm, rate = speaker.speak(text, parse_voice("piper:sharvard-f"), None,
                              "es")
    assert rate == 22050 and pcm
    assert tts.requests[0]["speaker"] == 1
    assert tts.requests[0]["voice"] == "es_ES-sharvard-medium"
    pcm, rate = speaker.speak(text, parse_voice("pocket:lola"), None, "es")
    lola = engines.streams[EngineVoice("pocket", "cpu", voice="lola")]
    assert rate == 24000 and len(lola.said) == 2, "one call per sentence"
    assert len(pcm) == 2 * 40 + 2 * int(24000 * 0.18)
    clone = parse_voice("clone:ana")
    speaker.speak(text, clone, "/x/ana.clone.wav", "es")
    speaker.speak(text, parse_voice("clone:luis"), "/x/luis.clone.wav", "es")
    assert len(engines.released) == 1, "one GPU engine at a time"
    pcm, rate = speaker.speak("Hola.", parse_voice("clone:ana", "openvoice"),
                              "/x/ana.clone.wav", "es")
    assert rate == 22050
    assert speaker.engine(parse_voice("pocket:lola")) == "pocket-cpu"
    assert speaker.engine(clone) == "qwen3-cuda"
    assert speaker.engine(parse_voice("piper:davefx")) == "piper"
    assert speaker.engine(parse_voice("clone:a", "openvoice")) \
        == "openvoice-cpu"
    assert speaker.need_mib(parse_voice("pocket:lola")) == 0
    assert speaker.need_mib(clone) == 4400
    assert speaker.need_mib(parse_voice("piper:davefx")) == 0
    assert speaker.need_mib(parse_voice("clone:a", "openvoice")) == 0
    assert EngineDemoSpeaker(pocket_device="cuda").need_mib(
        parse_voice("pocket:lola")) == 800
    assert EngineDemoSpeaker().need_mib(
        parse_voice("clone:a", "openvoice")) == 800
    speaker.close()
    assert len(engines.released) == 2 and cloner.released == 1


def test_a_missing_engine_is_unavailable() -> None:
    speaker = EngineDemoSpeaker(engines=Engines(frozenset()))
    for voice in ("piper:davefx", "pocket:lola"):
        with pytest.raises(TtsUnavailable):
            speaker.speak("Hola.", parse_voice(voice), None, "es")
    with pytest.raises(TtsUnavailable):
        speaker.speak("Hola.", parse_voice("clone:a", "openvoice"), None,
                      "es")


# ---- the command line ----

def run(*argv: str) -> int:
    return cli.main(list(argv))


def test_render_show_clear_from_the_command_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    out = str(tmp_path / "dv")
    common = ["--out", out, "--engines", "tone",
              "--voiceprints-dir", str(tmp_path / "vp"),
              "--gpu-lease-dir", str(tmp_path / "lease")]
    voices = [f"--voice={p}={v}" for p, v in STOCK.items()]
    assert run("demo-voices", "render", CALL, *common, *voices) == 0
    done = json.loads(capsys.readouterr().out)
    assert len(done["rendered"]) == 27 and done["voices"] == STOCK
    assert run("demo-voices", "render", CALL, *common) == 0
    assert json.loads(capsys.readouterr().out)["rendered"] == []
    assert run("demo-voices", "render", CALL, *common,
               "--voice", "agente=piper:davefx") == 1
    assert "female" in capsys.readouterr().err
    assert run("demo-voices", "render", CALL, *common,
               "--voice", "agente=clone:ana") == 1
    assert "no enrolled person" in capsys.readouterr().err
    assert run("demo-voices", "show", CALL, "--out", out) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["clips"] == 27 and shown["engines"] == ["tone"]
    assert run("demo-voices", "clear", CALL, "--out", out) == 0
    assert json.loads(capsys.readouterr().out)["deleted"] == 27
    assert run("demo-voices", "show", CALL, "--out", out) == 0
    assert json.loads(capsys.readouterr().out)["clips"] == 0
    assert run("demo-voices", "render", CALL, "--out",
               str(REPO / "examples"), "--engines", "tone") == 2


def test_recast_renames_a_person_in_the_demo_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    folder = tmp_path / "demos"
    shutil.copytree(DEMOS, folder)
    before = (folder / f"{CALL}.demo.json").read_text("utf-8")
    assert run("demo-voices", "recast", CALL, "agente=Mario",
               "--gender", "male", "--demos-dir", str(folder)) == 0
    assert json.loads(capsys.readouterr().out)["was"] == "Marta"
    demo = DirDemos(folder).load(CALL)
    assert demo.transcript.speakers[0].name == "Mario"
    assert run("demo-voices", "recast", CALL, "agente=Marta",
               "--gender", "female", "--demos-dir", str(folder)) == 0
    assert (folder / f"{CALL}.demo.json").read_text("utf-8") == before
    assert run("demo-voices", "recast", CALL, "cliente=Ana Gil",
               "--gender", "female", "--demos-dir", str(folder)) == 2
    assert run("demo-voices", "recast", "nada", "a=b",
               "--demos-dir", str(folder)) == 2


def test_pairs_are_read_once_each() -> None:
    assert parse_pairs(["a=x", " b = y "]) == {"a": "x", "b": "y"}
    for bad in (["a"], ["=x"], ["a=x", "a=y"]):
        with pytest.raises(ValueError):
            parse_pairs(bad)


# ---- over HTTP ----

def rendered(tmp_path: Path) -> DemoVoiceLibrary:
    store = FsDemoVoices(tmp_path / "dv")
    profiles = DirVoiceProfiles(tmp_path / "vp")
    RenderDemoVoices(DirDemos(DEMOS), store, ToneDemoSpeaker(),
                     profiles).render(CALL, STOCK)
    return DemoVoiceLibrary(DirDemos(DEMOS), store, profiles)


def app(library: Optional[DemoVoiceLibrary]) -> TestClient:
    sessions = LiveSessions(
        {"live_assist.support": product("live_assist.support")},
        FakeKnowledge([]), FakeReasoner(), Transcripts({}), Clock(),
        InlineRunner(), InlineExecutor,
        demos=library or DirDemos(DEMOS),
    )
    return TestClient(create_app(sessions, demo_voices=library))


def test_the_page_gets_the_view_and_the_clips(tmp_path: Path) -> None:
    web = app(rendered(tmp_path))
    view = web.get(f"/demos/{CALL}/voices").json()
    assert view["examples"]["call"]["available"] is True
    assert set(view["examples"]["voicebot"]["clips"]) == {
        f"vb{n}.{side}" for n in range(4) for side in ("user", "bot")}
    clip = web.get(f"/demos/{CALL}/voices/t0.s0.wav")
    assert clip.status_code == 200
    assert clip.headers["content-type"] == "audio/wav"
    assert "no-store" in clip.headers["cache-control"]
    assert clip.content[:4] == b"RIFF"
    for bad in ("t0.s0", "t99.s0.wav", "..%2Fmanifest.json"):
        assert web.get(f"/demos/{CALL}/voices/{bad}").status_code == 404
    assert web.get("/demos/nada/voices").status_code == 404
    # The report still answers, and the call lists at the clips' pace.
    assert web.get(f"/demos/{CALL}/report").status_code == 200
    listed = web.get("/transcripts").json()["demos"]
    assert listed[0]["seconds"] != 173.2


def test_without_voices_the_routes_say_not_found(tmp_path: Path) -> None:
    web = app(None)
    assert web.get(f"/demos/{CALL}/voices").status_code == 404
    assert web.get(f"/demos/{CALL}/voices/t0.s0.wav").status_code == 404
    assert web.get(f"/demos/{CALL}/report").status_code == 200


def test_serve_wires_the_library_from_its_folder(tmp_path: Path) -> None:
    args = cli.build_parser().parse_args([
        "serve", "--demo-voices-dir", str(tmp_path / "dv"),
        "--voiceprints-dir", str(tmp_path / "vp")])
    library = cli.build_demo_voices(args)
    assert isinstance(library, DemoVoiceLibrary)
    assert library.view(CALL)["kinds"] == []
    args.demo_voices_dir = REPO / "examples"
    with pytest.raises(ValueError):
        cli.build_demo_voices(args)
