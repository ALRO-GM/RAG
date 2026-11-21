"""Command line: ``python -m orchestrator check-config | replay``.

``replay`` talks to knowledge and the reasoner through ports. Each is
``fake`` (in process, no service), ``http`` (the RAG service) or, for
the reasoner, ``module:factory`` naming a callable that returns a
reasoner port. The orchestrator may not import ``src``, so an
in-process reasoner over the real model lives outside this package
(``experiments/live/orchestrator_reasoner.py``) and is named here.
"""

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

from orchestrator.adapters.outbound.fakes import FakeKnowledge, FakeReasoner
from orchestrator.adapters.outbound.files import (
    ConfigError,
    check_dir,
    load_product,
    load_transcript,
)
from orchestrator.adapters.outbound.http_knowledge import HttpKnowledge
from orchestrator.adapters.outbound.http_reasoner import HttpReasoner
from orchestrator.adapters.outbound.jsonl_sink import JsonlSink
from orchestrator.adapters.outbound.runtime import (
    InlineExecutor,
    SystemClock,
    ThreadExecutor,
)
from orchestrator.application.live_assist import LiveAssist
from orchestrator.application.ports import KnowledgePort, ReasonerPort
from orchestrator.application.replay import (
    Replay,
    default_roles,
    transcript_to_events,
)
from orchestrator.application.summary import summarise

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_DIR = ROOT / "config" / "products"


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
    return parser


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
        return run_replay(args)
    except (ConfigError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
