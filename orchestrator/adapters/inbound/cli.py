"""Command line: ``python -m orchestrator check-config | replay | serve |
demo-voices``.

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
``--voiceprints-dir`` (``docs/live/VOICE_CLONING.md``). The natural
voices (#214: Pocket TTS, Qwen3-TTS; ``voicebot.voice = pocket:lola`` or
``qwen3``, and ``voicebot.clone_engine = qwen3``) run from their own
venvs under ``--natural-dir`` (off with ``--natural-tts off`` or when
none is installed; ``docs/live/NATURAL_TTS.md`` §6); Pocket TTS on the
CPU is warmed at start, a GPU engine only under the live session's GPU
lease (``--gpu-lease-dir``).

The scripted examples are heard with the clips ``demo-voices render``
keeps in ``--demo-voices-dir`` (:mod:`.demo_voices_cli`, #254); a call
whose clips are all there is replayed at their pace.
"""

import argparse
import logging
import importlib
import ipaddress
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from orchestrator.application.event_follow import ResumingEvents
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
from orchestrator.adapters.outbound.fs_candidates import JsonCandidateStore
from orchestrator.adapters.outbound.fs_core_questions import FileCoreQuestions
from orchestrator.adapters.outbound.fs_interview_plans import JsonPlanStore
from orchestrator.adapters.outbound.http_interview import (
    HttpHandoff,
    HttpOffers,
)
from orchestrator.adapters.outbound.http_knowledge import HttpKnowledge
from orchestrator.adapters.outbound.http_reasoner import HttpReasoner
from orchestrator.adapters.outbound.http_settings import HttpSettings
from orchestrator.adapters.outbound.http_mandatory import HttpMandatory
from orchestrator.adapters.outbound.http_avoid import HttpAvoid
from orchestrator.adapters.outbound.fs_demos import DirDemos
from orchestrator.adapters.outbound.fs_demo_voices import (
    FsDemoVoices,
    check_outside_repository,
)
from orchestrator.adapters.inbound.interview_cli import (
    add_interview_parser,
    run_interview,
)
from orchestrator.adapters.inbound.round_cli import (
    add_round_parser,
    run_round,
)
from orchestrator.adapters.inbound.demo_voices_cli import (
    DEFAULT_OUT as DEMO_VOICES_DIR,
    add_demo_voices_parser,
    run_demo_voices,
)
from orchestrator.adapters.outbound.fs_mandatory import FileMandatory
from orchestrator.adapters.outbound.fs_mandatory_lists import (
    JsonMandatoryLists,
)
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
from orchestrator.adapters.outbound.gpu_gate import (
    LeaseGpuGate,
    lease_directory,
)
from orchestrator.adapters.outbound.subprocess_speech import (
    DEFAULT_DIR as NATURAL_DIR,
    WorkerEngines,
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
from orchestrator.application.candidates import CandidateBook
from orchestrator.application.demo_voices import DemoVoiceLibrary
from orchestrator.application.interview import (
    PLANNER_TIMEOUT_S,
    REPORT_TIMEOUT_S,
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
from orchestrator.application.mandatory_lists import MandatoryLists
from orchestrator.application.sessions import LiveSessions, list_kind
from orchestrator.application.voice import DEFAULT_BUDGET_MS
from orchestrator.application.natural_voice import NaturalWiring
from orchestrator.domain.voices import EngineVoice
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
    add_demo_voices_parser(sub)
    add_interview_parser(sub)
    add_round_parser(sub, DEFAULT_CONFIG_DIR)
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
    serve.add_argument("--natural-tts", default="on", choices=("on", "off"),
                       help="the natural voices (Pocket TTS, Qwen3-TTS)")
    serve.add_argument("--natural-dir", type=Path, default=NATURAL_DIR,
                       help="their venvs, weights and reference voice")
    serve.add_argument("--natural-budget-ms", type=float,
                       default=DEFAULT_BUDGET_MS,
                       help="longest wait for a sentence's first audio "
                            "before the next voice says it")
    serve.add_argument("--gpu-lease-dir", type=Path, default=None,
                       help="folder of the GPU lease (default "
                            "$XDG_RUNTIME_DIR/gpu-lease)")
    serve.add_argument("--voiceprints-dir", type=Path,
                       default=ROOT / "data" / "transcriber" / "voiceprints",
                       help="the transcriber's voiceprints (profiles)")
    serve.add_argument("--interview-dir", type=Path,
                       default=ROOT / "data" / "interview",
                       help="interview plans and sheets (#208)")
    serve.add_argument("--mandatory-dir", type=Path,
                       default=ROOT / "data" / "mandatory",
                       help="the owner's lists of «Lo que hay que decir» "
                            "(#294)")
    serve.add_argument("--demo-voices-dir", type=Path,
                       default=DEMO_VOICES_DIR,
                       help="the scripted examples' rendered voices (#254)")


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


def build_gate(args: argparse.Namespace) -> LeaseGpuGate:
    """Return the one GPU gate of this orchestrator.

    The natural voices and the cloner share it: its ``take``/``drop``
    are counted, so two samples at once never delete each other's lease
    (``docs/live/VOICE_CLONING.md`` §4).
    """
    directory = args.gpu_lease_dir or lease_directory(os.environ)
    return LeaseGpuGate(Path(directory))


def build_natural(
    args: argparse.Namespace, gate: Optional[LeaseGpuGate] = None,
) -> Optional[NaturalWiring]:
    """Return the natural voices, or ``None`` when off or not installed.

    Pocket TTS on the CPU (the shared one) is warmed in the background.
    """
    if (getattr(args, "natural_tts", "off") != "on"
            or getattr(args, "tts", "none") != "piper"):
        return None
    engines = WorkerEngines(Path(args.natural_dir))
    installed = engines.installed()
    if not installed:
        print(f"natural voices off: nothing installed in {args.natural_dir}",
              file=sys.stderr)
        return None
    if "pocket" in installed:
        pocket = engines.get(EngineVoice("pocket", "cpu", voice="lola"))

        def warm() -> None:
            try:
                pocket.warm(EngineVoice("pocket", "cpu", voice="lola"))
            except TtsUnavailable as exc:
                print(f"Pocket TTS warm-up failed: {exc}", file=sys.stderr)

        ThreadRunner().start(warm)
    return NaturalWiring(engines, gate or build_gate(args),
                         engines.reference(), args.natural_budget_ms)


def build_sessions(args: argparse.Namespace) -> LiveSessions:
    """Wire the live sessions of ``serve`` from the arguments."""
    args.token = read_code(args.rag_token_file)
    code = read_code(args.transcriber_code_file)
    transcriber = events = None
    if code:
        from orchestrator.adapters.outbound.ws_events import WsEvents

        transcriber = HttpTranscriber(args.transcriber_url, code)
        # #264: a dropped event stream is reopened where it was.
        events = ResumingEvents(WsEvents(args.transcriber_url, code),
                                time.sleep, time.monotonic)
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
    gate = build_gate(args)
    mandatory = build_mandatory(args)
    sessions = LiveSessions(
        products, build_knowledge(args), build_reasoner(args),
        DirTranscripts(args.transcripts or list(TRANSCRIPT_DIRS)),
        SystemClock(), ThreadRunner(), ThreadExecutor, transcriber,
        events, log, new_id, tts=build_tts(args, products),
        turn_log=turns,
        settings=HttpSettings(args.knowledge_url, args.token)
        if args.knowledge == "http" else None,
        mandatory=mandatory, handoff=handoff,
        profiles=DirVoiceProfiles(args.voiceprints_dir),
        cloner=build_cloner(args), clone_budget_ms=args.clone_budget_ms,
        natural=build_natural(args, gate), clone_gate=gate,
        interview=build_interview(args, products, sessions_ref),
        ticker=ThreadRunner(),
        avoid=HttpAvoid(args.knowledge_url, args.token)
        if args.knowledge == "http" else None,
        demos=build_demo_voices(args),
        lists=build_lists(args, products, mandatory),
    )
    sessions_ref.append(sessions)
    return sessions


def build_demo_voices(args: argparse.Namespace) -> DemoVoiceLibrary:
    """The demos, heard with their rendered clips when there are some."""
    folder = Path(getattr(args, "demo_voices_dir", DEMO_VOICES_DIR))
    check_outside_repository(folder, ROOT)
    return DemoVoiceLibrary(
        DirDemos(ROOT / "examples" / "demo"), FsDemoVoices(folder),
        DirVoiceProfiles(args.voiceprints_dir),
    )


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
        HttpReasoner(args.reasoner_url, args.token,
                     timeout=PLANNER_TIMEOUT_S)
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
    report = (HttpReasoner(args.reasoner_url, args.token,
                           timeout=REPORT_TIMEOUT_S)
              if args.reasoner == "http" else reasoner)

    def plan_exists(plan_id: str) -> bool:
        try:
            store.load(plan_id)
        except KeyError:
            return False
        return True

    candidates = CandidateBook(
        JsonCandidateStore(args.interview_dir),
        FileCoreQuestions(ROOT / "knowledge" / "interview"
                          / "core-questions.json"),
        plan_exists, retention_days(products), plans=store,
    )
    candidates.purge()  # the retention applies from the start
    return InterviewWiring(
        planner, store, report,
        HttpHandoff(args.knowledge_url, args.token) if http else None,
        candidates,
    )


def retention_days(products: Dict[str, ProductConfig]) -> Optional[int]:
    """Days a candidate profile is kept: the shortest text retention of
    the interview products (``privacy.store_text_days``), if any."""
    days = [
        c.privacy.store_text_days for c in products.values()
        if (c.interview is not None or any(
            ch.role == "interviewer" for ch in c.source.channels.values()))
        and c.privacy.store_text_days is not None
    ]
    return min(days) if days else None


def build_lists(
    args: argparse.Namespace, products: Dict[str, ProductConfig],
    mandatory: MandatoryPort,
) -> MandatoryLists:
    """The owner's lists of «Lo que hay que decir» (#294).

    The general list of each kind is the knowledge domain of the first
    live product of that kind.
    """
    domains: Dict[str, str] = {}
    for name in sorted(products):
        config = products[name]
        kind = list_kind(config) if config.tts is None else None
        if kind and config.knowledge is not None and \
                config.knowledge.domains:
            domains.setdefault(kind, config.knowledge.domains[0])
    folder: Path = getattr(args, "mandatory_dir",
                           ROOT / "data" / "mandatory")
    plans = JsonPlanStore(getattr(args, "interview_dir",
                                  ROOT / "data" / "interview"))

    def plan_exists(plan_id: str) -> bool:
        try:
            plans.load(plan_id)
        except KeyError:
            return False
        return True

    return MandatoryLists(JsonMandatoryLists(folder), mandatory, domains,
                          lambda: datetime.now(timezone.utc), plan_exists)


def build_mandatory(args: argparse.Namespace) -> MandatoryPort:
    """The mandatory items: the RAG's, or the manifests' with ``fake``."""
    if args.knowledge == "http":
        return HttpMandatory(args.knowledge_url, args.token)
    return FileMandatory(ROOT / "knowledge" / "domains")


def log_to_stderr() -> logging.Logger:
    """Send the orchestrator's own log lines (INFO and up) to stderr.

    Under ``ragputin-servicios`` stderr is ``orchestrator.log``; only the
    ``orchestrator`` loggers are routed (no library's request lines; the
    root logger has no handler under ``serve``, so nothing is doubled).
    The lines never carry conversation text or audio (#253).
    """
    logger = logging.getLogger("orchestrator")
    if not any(getattr(h, "_orchestrator", False) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s"))
        setattr(handler, "_orchestrator", True)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


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
    log_to_stderr()
    sessions = build_sessions(args)
    import uvicorn

    from orchestrator.adapters.inbound.http_server import create_app

    live = "on" if sessions.transcriber is not None else (
        f"off (no access code in {args.transcriber_code_file})"
    )
    print(f"orchestrator on http://{args.host}:{args.port} "
          f"(products: {', '.join(sorted(sessions.products))}; "
          f"live: {live}; token: {'yes' if token else 'no'})", flush=True)
    voices = sessions.demos if isinstance(
        sessions.demos, DemoVoiceLibrary) else None
    uvicorn.run(create_app(sessions, token, demo_voices=voices,
                           round_executor=InlineExecutor()),
                host=args.host,
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


def run_rebuild(args: argparse.Namespace) -> int:
    """Wire and run ``rebuild-round`` (#264)."""
    args.token = read_code(args.rag_token_file)
    reasoner = (HttpReasoner(args.reasoner_url, args.token,
                             timeout=REPORT_TIMEOUT_S)
                if args.reasoner == "http" else build_reasoner(args))
    product = load_product(args.config_dir / f"{args.product}.yaml")
    http = args.knowledge == "http"
    return run_round(
        args, product, reasoner, build_knowledge(args),
        HttpMandatory(args.knowledge_url, args.token) if http else None,
        HttpAvoid(args.knowledge_url, args.token) if http else None,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the command line; return the exit status."""
    args = build_parser().parse_args(argv)
    try:
        if args.command == "check-config":
            return run_check(args)
        if args.command == "serve":
            return run_serve(args)
        if args.command == "demo-voices":
            return run_demo_voices(args)
        if args.command == "interview":
            args.token = read_code(args.rag_token_file)
            reasoner = (HttpReasoner(args.reasoner_url, args.token,
                                     timeout=REPORT_TIMEOUT_S)
                        if args.reasoner == "http" else build_reasoner(args))
            return run_interview(args, reasoner, build_knowledge(args))
        if args.command == "rebuild-round":
            return run_rebuild(args)
        return run_replay(args)
    except (ConfigError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
