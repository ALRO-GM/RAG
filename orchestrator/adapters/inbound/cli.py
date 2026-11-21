"""Command line: ``python -m orchestrator check-config | replay | serve``.

``replay`` talks to knowledge and the reasoner through ports. Each is
``fake`` (in process, no service), ``http`` (the RAG service) or, for
the reasoner, ``module:factory`` naming a callable that returns a
reasoner port. The orchestrator may not import ``src``, so an
in-process reasoner over the real model lives outside this package
(``experiments/live/orchestrator_reasoner.py``) and is named here.

``serve`` runs the live page's API (:mod:`.http_server`) on a loopback
address: live sessions through the transcriber, and replays of recorded
transcripts for a demo without a microphone. With ``--tts piper`` (the
default) it also serves the voice bot (``voicebot`` product): Piper
runs from its own venv (``--piper-python``, voices in
``--voices-dir``) and is warmed in the background at start; ``--tts
fake`` answers with short tones, ``--tts none`` leaves the voice bot
out. The voice bot can also speak with an enrolled person's cloned
voice (``voicebot.voice = clone:<person>`` on «Ajustes»): the cloning
model runs in its own venv (``--clone-python``, ``--clone-engine``;
off when that venv is missing or with ``--clone-engine none``), and
the people's profiles are read from the transcriber's
``--voiceprints-dir`` (``docs/live/VOICE_CLONING.md``).
"""

import argparse
import importlib
import ipaddress
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from orchestrator.adapters.outbound.fakes import (
    FakeKnowledge,
    FakeReasoner,
    FakeTts,
)
from orchestrator.adapters.outbound.files import (
    ConfigError,
    check_dir,
    load_product,
    load_transcript,
)
from orchestrator.adapters.outbound.fs_interview_plans import JsonPlanStore
from orchestrator.adapters.outbound.http_interview import (
    HttpHandoff,
    HttpOffers,
)
from orchestrator.adapters.outbound.http_knowledge import HttpKnowledge
from orchestrator.adapters.outbound.http_reasoner import HttpReasoner
from orchestrator.adapters.outbound.http_settings import HttpSettings
from orchestrator.adapters.outbound.http_mandatory import HttpMandatory
from orchestrator.adapters.outbound.fs_mandatory import FileMandatory
from orchestrator.adapters.outbound.jsonl_sink import (
    JsonHandoff,
    JsonlSink,
    JsonlTurns,
)
from orchestrator.adapters.outbound.piper_tts import (
    DEFAULT_PYTHON,
    DEFAULT_VOICES,
    PiperTts,
)
from orchestrator.adapters.outbound.fs_transcripts import DirTranscripts
from orchestrator.adapters.outbound.fs_voice_profiles import (
    DirVoiceProfiles,
)
from orchestrator.adapters.outbound.subprocess_cloner import (
    SubprocessCloner,
)
from orchestrator.adapters.outbound.http_transcriber import (
    HttpTranscriber,
    read_code,
)
from orchestrator.adapters.outbound.runtime import (
    InlineExecutor,
    SystemClock,
    ThreadExecutor,
    ThreadRunner,
)
from orchestrator.application.interview import (
    InterviewPlanner,
    InterviewWiring,
)
from orchestrator.application.live_assist import LiveAssist
from orchestrator.application.ports import (
    HandoffSink,
    KnowledgePort,
    MandatoryPort,
    ReasonerPort,
    TextToSpeech,
    TtsUnavailable,
    TurnSink,
    SuggestionSink,
)
from orchestrator.application.sessions import LiveSessions
from orchestrator.application.voice import DEFAULT_BUDGET_MS
from orchestrator.application.replay import (
    Replay,
    default_roles,
    transcript_to_events,
)
from orchestrator.application.summary import summarise
from orchestrator.domain.config import InterviewSection, ProductConfig

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_DIR = ROOT / "config" / "products"
DEFAULT_PORT = 8772
RAG_TOKEN_FILE = Path("~/.config/ragputin/rag-token").expanduser()
"""The token ``make rag-token`` writes; the RAG wants it as bearer."""
TRANSCRIBER_CODE_FILE = ROOT / "data" / "transcriber" / "access_code"
"""Where ``transcriber serve --keep-code`` keeps its access code."""
CLONE_ENGINES = ("chatterbox", "openvoice")
CLONE_ENGINE = "openvoice"
"""The cloning model used by default (#207: the only one measured
within the latency budget; ``docs/live/VOICE_CLONING.md``)."""
CLONE_PYTHON = Path(
    "~/.local/share/ragputin/clone-openvoice-venv/bin/python"
)
CLONE_BASE_VOICE = "es_ES-davefx-medium"
"""The Piper voice OpenVoice converts (the one it was measured with)."""
TRANSCRIPT_DIRS = (ROOT / "data" / "transcripts",
                   ROOT / "examples" / "platform")


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser of the two commands."""
    parser = argparse.ArgumentParser(prog="python -m orchestrator")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check-config", help="validate the product YAMLs")
    check.add_argument("--dir", type=Path, default=DEFAULT_CONFIG_DIR)
    replay = sub.add_parser(
        "replay", help="replay a transcript.v1 as live events"
    )
    replay.add_argument("transcript", type=Path)
    replay.add_argument("--product", required=True)
    replay.add_argument("--speed", default="max",
                        help="1 for real time, a number, or max")
    replay.add_argument("--out", type=Path, default=None,
                        help="suggestions JSONL (default: data/live/...)")
    replay.add_argument("--events-out", type=Path, default=None,
                        help="also write the live.event.v1 stream here")
    replay.add_argument("--config-dir", type=Path,
                        default=DEFAULT_CONFIG_DIR)
    replay.add_argument("--roles", default="",
                        help="speaker=role,... (default: by order)")
    replay.add_argument("--knowledge", default="fake",
                        help="fake | http")
    replay.add_argument("--knowledge-url", default="http://127.0.0.1:8000")
    replay.add_argument("--reasoner", default="fake",
                        help="fake | http | module:factory")
    replay.add_argument("--reasoner-url", default="http://127.0.0.1:8000")
    replay.add_argument("--token", default=None)
    replay.add_argument("--fake-delay-ms", type=float, default=0.0,
                        help="latency of the fakes")
    add_serve_parser(sub)
    return parser


def add_serve_parser(
    sub: "argparse._SubParsersAction[argparse.ArgumentParser]",
) -> None:
    """Add ``serve``: the live page's API."""
    serve = sub.add_parser("serve", help="the live page's API (SSE)")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=DEFAULT_PORT)
    serve.add_argument("--token-file", type=Path, default=None,
                       help="bearer token the callers must send")
    serve.add_argument("--config-dir", type=Path,
                       default=DEFAULT_CONFIG_DIR)
    serve.add_argument("--knowledge", default="http", help="fake | http")
    serve.add_argument("--knowledge-url", default="http://127.0.0.1:8000")
    serve.add_argument("--reasoner", default="http",
                       help="fake | http | module:factory")
    serve.add_argument("--reasoner-url", default="http://127.0.0.1:8000")
    serve.add_argument("--rag-token-file", type=Path,
                       default=RAG_TOKEN_FILE)
    serve.add_argument("--transcriber-url", default="http://127.0.0.1:8770")
    serve.add_argument("--transcriber-code-file", type=Path,
                       default=TRANSCRIBER_CODE_FILE)
    serve.add_argument("--transcripts", type=Path, action="append",
                       default=None,
                       help="folder of *.transcript.v1.json (repeatable)")
    serve.add_argument("--log-dir", type=Path,
                       default=ROOT / "data" / "live" / "serve")
    serve.add_argument("--fake-delay-ms", type=float, default=0.0)
    serve.add_argument("--tts", default="piper",
                       help="piper | fake | none (the voice bot's voice)")
    serve.add_argument("--piper-python", type=Path, default=DEFAULT_PYTHON,
                       help="python of Piper's own venv (GPL-3.0)")
    serve.add_argument("--voices-dir", type=Path, default=DEFAULT_VOICES)
    serve.add_argument("--clone-engine", default=CLONE_ENGINE,
                       choices=("none",) + CLONE_ENGINES,
                       help="voice-cloning model (its own venv)")
    serve.add_argument("--clone-python", type=Path, default=CLONE_PYTHON,
                       help="python of the cloning model's own venv")
    serve.add_argument("--clone-device", default="cuda",
                       choices=("cuda", "cpu"))
    serve.add_argument("--clone-budget-ms", type=float,
                       default=DEFAULT_BUDGET_MS,
                       help="longest wait for a cloned sentence (Piper "
                            "says it after that)")
    serve.add_argument("--voiceprints-dir", type=Path,
                       default=ROOT / "data" / "transcriber" / "voiceprints",
                       help="the transcriber's voiceprints (profiles)")
    serve.add_argument("--interview-dir", type=Path,
                       default=ROOT / "data" / "interview",
                       help="interview plans and sheets (#208)")


def parse_speed(text: str) -> Optional[float]:
    """Return the speed as a number, or ``None`` for ``max``."""
    if text == "max":
        return None
    value = float(text)
    if value <= 0:
        raise ValueError("speed must be positive")
    return value


def parse_roles(text: str) -> Dict[str, str]:
    """Read ``a=customer,b=agent`` into a dict."""
    roles: Dict[str, str] = {}
    for part in filter(None, text.split(",")):
        speaker, _, role = part.partition("=")
        if not role:
            raise ValueError(f"bad role {part!r}, expected speaker=role")
        roles[speaker.strip()] = role.strip()
    return roles


def build_knowledge(args: argparse.Namespace) -> KnowledgePort:
    """Return the knowledge adapter the arguments name."""
    if args.knowledge == "fake":
        return FakeKnowledge(delay=args.fake_delay_ms / 1000.0)
    if args.knowledge == "http":
        return HttpKnowledge(args.knowledge_url, args.token)
    raise ValueError(f"unknown knowledge adapter {args.knowledge!r}")


def build_reasoner(args: argparse.Namespace) -> ReasonerPort:
    """Return the reasoner adapter the arguments name."""
    if args.reasoner == "fake":
        return FakeReasoner(delay=args.fake_delay_ms / 1000.0)
    if args.reasoner == "http":
        return HttpReasoner(args.reasoner_url, args.token)
    module, sep, name = args.reasoner.partition(":")
    if not sep or not name:
        raise ValueError(f"unknown reasoner adapter {args.reasoner!r}")
    factory = getattr(importlib.import_module(module), name)
    port: ReasonerPort = factory()
    return port


def run_replay(args: argparse.Namespace) -> int:
    """Run ``replay``; print the summary as JSON."""
    config = load_product(args.config_dir / f"{args.product}.yaml")
    if not config.is_live:
        raise ValueError(f"{config.product} is not a live product")
    transcript = load_transcript(args.transcript)
    roles = {**default_roles(config, transcript),
             **parse_roles(args.roles)}
    speed = parse_speed(args.speed)
    out = args.out or ROOT / "data" / "live" / (
        f"{args.transcript.name.split('.')[0]}.{config.product}"
        ".suggestions.jsonl"
    )
    sink = JsonlSink(out)
    assist = LiveAssist(
        config, build_knowledge(args), build_reasoner(args), sink,
        SystemClock(),
        InlineExecutor() if speed is None else ThreadExecutor(), roles,
        {"type": transcript.session.type, "goal": transcript.session.goal},
    )
    sent = Replay(assist, SystemClock(), speed).run(
        transcript_to_events(transcript, roles)
    )
    if args.events_out:
        args.events_out.parent.mkdir(parents=True, exist_ok=True)
        args.events_out.write_text(
            "".join(
                json.dumps(e.to_contract(), ensure_ascii=False) + "\n"
                for e in sent
            ), encoding="utf-8",
        )
    summary = summarise(sink.records, assist.skips)
    summary.update(events=len(sent), suggestions=str(out), roles=roles)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def is_loopback(host: str) -> bool:
    """Tell whether ``host`` is an address of this machine only."""
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def load_products(directory: Path) -> Dict[str, ProductConfig]:
    """Read every valid product of ``directory``."""
    products: Dict[str, ProductConfig] = {}
    for path in sorted(directory.glob("*.yaml")):
        config = load_product(path)
        products[config.product] = config
    return products


def build_tts(
    args: argparse.Namespace, products: Dict[str, ProductConfig]
) -> Optional[TextToSpeech]:
    """Return the voice bot's text to speech, warmed in the background."""
    kind = getattr(args, "tts", "none")
    if kind == "fake":
        return FakeTts()
    if kind == "none":
        return None
    if kind != "piper":
        raise ValueError(f"unknown tts {kind!r}")
    python = Path(args.piper_python).expanduser()
    if not python.exists():
        print(f"voice bot off: no Piper venv at {python}", file=sys.stderr)
        return None
    tts = PiperTts(python=python, voices_dir=args.voices_dir)
    voices = {c.tts.voice for c in products.values() if c.tts is not None}

    def warm() -> None:
        for voice in sorted(voices):
            try:
                tts.warm(voice)
            except TtsUnavailable as exc:
                print(f"Piper warm-up failed: {exc}", file=sys.stderr)

    ThreadRunner().start(warm)
    return tts


def build_cloner(args: argparse.Namespace) -> Optional[SubprocessCloner]:
    """Return the voice cloner, or ``None`` when it is off or missing."""
    engine = getattr(args, "clone_engine", "none")
    if engine == "none" or getattr(args, "tts", "none") != "piper":
        return None
    python = Path(args.clone_python).expanduser()
    if not python.exists():
        print(f"voice cloning off: no venv at {python}", file=sys.stderr)
        return None
    extra = [
        "--piper-python", str(Path(args.piper_python).expanduser()),
        "--piper-model",
        str(Path(args.voices_dir).expanduser() / f"{CLONE_BASE_VOICE}.onnx"),
    ] if engine == "openvoice" else []
    return SubprocessCloner(python, engine, device=args.clone_device,
                            extra=extra)


def build_sessions(args: argparse.Namespace) -> LiveSessions:
    """Wire the live sessions of ``serve`` from the arguments."""
    args.token = read_code(args.rag_token_file)
    code = read_code(args.transcriber_code_file)
    transcriber = events = None
    if code:
        from orchestrator.adapters.outbound.ws_events import WsEvents

        transcriber = HttpTranscriber(args.transcriber_url, code)
        events = WsEvents(args.transcriber_url, code)
    log_dir: Path = args.log_dir

    def log(session_id: str, product: str) -> SuggestionSink:
        return JsonlSink(
            log_dir / f"{session_id}.{product}.suggestions.jsonl"
        )

    def turns(session_id: str, product: str) -> TurnSink:
        return JsonlTurns(log_dir / f"{session_id}.{product}.turns.jsonl")

    def handoff(session_id: str, product: str) -> HandoffSink:
        return JsonHandoff(log_dir / f"{session_id}.{product}.handoff.json")

    sessions_ref: List[LiveSessions] = []
    new_id: Callable[[], str] = lambda: datetime.now().strftime(
        "live-%Y%m%d-%H%M%S"
    )
    products = load_products(args.config_dir)
    sessions = LiveSessions(
        products, build_knowledge(args), build_reasoner(args),
        DirTranscripts(args.transcripts or list(TRANSCRIPT_DIRS)),
        SystemClock(), ThreadRunner(), ThreadExecutor, transcriber,
        events, log, new_id, tts=build_tts(args, products),
        turn_log=turns,
        settings=HttpSettings(args.knowledge_url, args.token)
        if args.knowledge == "http" else None,
        mandatory=build_mandatory(args), handoff=handoff,
        profiles=DirVoiceProfiles(args.voiceprints_dir),
        cloner=build_cloner(args), clone_budget_ms=args.clone_budget_ms,
        interview=build_interview(args, products, sessions_ref),
        ticker=ThreadRunner(),
    )
    sessions_ref.append(sessions)
    return sessions


def build_interview(
    args: argparse.Namespace, products: Dict[str, ProductConfig],
    sessions_ref: "list[LiveSessions]",
) -> Optional[InterviewWiring]:
    """Wire the AI interviewer (plans, handoff), if a product has one."""
    product = products.get("voicebot.interview")
    if product is None or product.interview is None:
        return None
    store = JsonPlanStore(args.interview_dir)
    http = args.knowledge == "http"
    reasoner = build_reasoner(args)
    planner_reasoner = (
        HttpReasoner(args.reasoner_url, args.token, timeout=150.0)
        if args.reasoner == "http" else reasoner
    )
    fixed = product.interview

    def section() -> InterviewSection:
        if not sessions_ref:
            return fixed
        return sessions_ref[0].product(product.product).interview or fixed

    planner = InterviewPlanner(
        store, build_knowledge(args), planner_reasoner,
        HttpOffers(args.knowledge_url, args.token) if http else None,
        section, product.language,
        model=product.reasoner.model if product.reasoner else "haiku",
    )
    report = (HttpReasoner(args.reasoner_url, args.token, timeout=100.0)
              if args.reasoner == "http" else reasoner)
    return InterviewWiring(
        planner, store, report,
        HttpHandoff(args.knowledge_url, args.token) if http else None,
    )


def build_mandatory(args: argparse.Namespace) -> MandatoryPort:
    """The mandatory items: the RAG's, or the manifests' with ``fake``."""
    if args.knowledge == "http":
        return HttpMandatory(args.knowledge_url, args.token)
    return FileMandatory(ROOT / "knowledge" / "domains")


def run_serve(args: argparse.Namespace) -> int:
    """Run ``serve`` until interrupted."""
    if not is_loopback(args.host):
        raise ValueError(
            f"serve listens on this machine only: use 127.0.0.1, "
            f"not {args.host}"
        )
    token = None
    if args.token_file is not None:
        token = read_code(args.token_file)
        if not token:
            raise ValueError(f"no token in {args.token_file}")
    sessions = build_sessions(args)
    import uvicorn

    from orchestrator.adapters.inbound.http_server import create_app

    live = "on" if sessions.transcriber is not None else (
        f"off (no access code in {args.transcriber_code_file})"
    )
    print(f"orchestrator on http://{args.host}:{args.port} "
          f"(products: {', '.join(sorted(sessions.products))}; "
          f"live: {live}; token: {'yes' if token else 'no'})", flush=True)
    uvicorn.run(create_app(sessions, token), host=args.host,
                port=args.port, log_level="warning")
    return 0


def run_check(args: argparse.Namespace) -> int:
    """Run ``check-config``; exit 1 when a file is invalid."""
    names, errors = check_dir(args.dir)
    for name in names:
        print(f"ok  {name}")
    for error in errors:
        print(f"ERR {error}", file=sys.stderr)
    return 1 if errors else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the command line; return the exit status."""
    args = build_parser().parse_args(argv)
    try:
        if args.command == "check-config":
            return run_check(args)
        if args.command == "serve":
            return run_serve(args)
        return run_replay(args)
    except (ConfigError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
