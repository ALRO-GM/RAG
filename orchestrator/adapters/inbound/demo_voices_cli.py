"""``python -m orchestrator demo-voices``: the examples' voices (#254).

- ``render <demo> --voice <person>=<voice> ...`` says every line of the
  demo with the voice the operator chose for each scripted person
  (``pocket:lola``, ``piper:davefx``, ``piper:sharvard-f``,
  ``piper:sharvard-m``, ``clone:<enrolled person>[:female|:male]``) and
  keeps the clips outside the repository in
  ``data/transcriber/demo-voices``, the folder the transcriber cleans when
  a clone consent is withdrawn; another ``--out`` needs
  ``--outside-consent-hook`` (its clips are the operator's to delete).
  A person left out keeps the voice of the last render. Only lines whose
  text, voice or engine changed are rendered again. A clone needs the
  person's clone consent and is rendered by ``--clone-engine``
  (Qwen3-TTS by default, on the GPU under the lease, taken only when
  nobody holds it: never while a live session runs).
- ``show <demo>``: the manifest (who reads whom, how many clips).
- ``clear <demo>``: delete the demo's clips and manifest.
- ``recast <demo> <person>=<new name> [--gender ...]``: rename a scripted
  person everywhere in the demo file (``examples/demo``), so a name fits
  the voice that reads it.
- ``--engines tone`` renders tones instead of voices (a dry run).

On the workstation run it through ``systemd-run`` (``docs/live/
SUPPORT_DEMO.md``), never beside a running live session.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

from orchestrator.adapters.outbound.demo_speaker import (
    EngineDemoSpeaker,
    ToneDemoSpeaker,
)
from orchestrator.adapters.outbound.fs_demo_voices import (
    FsDemoVoices,
    check_outside_repository,
)
from orchestrator.adapters.outbound.fs_demos import SUFFIX, DirDemos
from orchestrator.adapters.outbound.fs_voice_profiles import (
    DirVoiceProfiles,
)
from orchestrator.adapters.outbound.gpu_gate import (
    LeaseGpuGate,
    lease_directory,
)
from orchestrator.adapters.outbound.piper_tts import (
    DEFAULT_PYTHON,
    DEFAULT_VOICES,
    PiperTts,
)
from orchestrator.adapters.outbound.subprocess_cloner import (
    SubprocessCloner,
)
from orchestrator.adapters.outbound.subprocess_speech import (
    DEFAULT_DIR as NATURAL_DIR,
    WorkerEngines,
)
from orchestrator.application.demo_voices import (
    DemoVoiceError,
    RenderDemoVoices,
)
from orchestrator.application.ports import DemoSpeaker
from orchestrator.domain.demo import DemoScript, recast
from orchestrator.domain.demo_voices import CLONE_ENGINES, STOCK_VOICES

ROOT = Path(__file__).resolve().parents[3]
DEMOS_DIR = ROOT / "examples" / "demo"
DEFAULT_OUT = ROOT / "data" / "transcriber" / "demo-voices"
"""Beside the voiceprints, where the transcriber's consent hook looks."""
CLONE_PYTHON = Path("~/.local/share/ragputin/clone-openvoice-venv/bin/python")
CLONE_BASE_VOICE = "es_ES-davefx-medium"


def add_demo_voices_parser(
    sub: "argparse._SubParsersAction[argparse.ArgumentParser]",
) -> None:
    """Add ``demo-voices`` and its four actions."""
    parser = sub.add_parser(
        "demo-voices", help="render the scripted examples' voices (#254)")
    actions = parser.add_subparsers(dest="action", required=True)
    render = actions.add_parser("render", help="render the clips")
    render.add_argument("demo")
    render.add_argument(
        "--voice", action="append", default=[], metavar="PERSON=VOICE",
        help=f"one of {', '.join(STOCK_VOICES)} or "
             "clone:<person>[:female|:male] (repeat per person)")
    render.add_argument(
        "--outside-consent-hook", action="store_true",
        help="allow --out other than the default: a withdrawn clone consent"
             " does not reach clips there, deleting them is yours")
    render.add_argument("--engines", default="real", choices=("real", "tone"))
    render.add_argument("--clone-engine", default="qwen3",
                        choices=CLONE_ENGINES)
    render.add_argument("--pocket-device", default="cpu",
                        choices=("cpu", "cuda"))
    render.add_argument("--piper-python", type=Path, default=DEFAULT_PYTHON)
    render.add_argument("--voices-dir", type=Path, default=DEFAULT_VOICES)
    render.add_argument("--natural-dir", type=Path, default=NATURAL_DIR)
    render.add_argument("--clone-python", type=Path, default=CLONE_PYTHON)
    render.add_argument("--gpu-lease-dir", type=Path, default=None)
    render.add_argument(
        "--voiceprints-dir", type=Path,
        default=ROOT / "data" / "transcriber" / "voiceprints")
    for action in (render,
                   actions.add_parser("show", help="print the manifest"),
                   actions.add_parser("clear", help="delete the clips")):
        if action is not render:
            action.add_argument("demo")
        action.add_argument("--out", type=Path, default=DEFAULT_OUT,
                            help="where the clips are kept")
        action.add_argument("--demos-dir", type=Path, default=DEMOS_DIR)
    rename = actions.add_parser("recast", help="rename a scripted person")
    rename.add_argument("demo")
    rename.add_argument("name", metavar="PERSON=NEW NAME")
    rename.add_argument("--gender", default=None, choices=("female", "male"))
    rename.add_argument("--demos-dir", type=Path, default=DEMOS_DIR)


def parse_pairs(items: List[str]) -> Dict[str, str]:
    """Read ``person=value`` items; a person given twice is an error."""
    pairs: Dict[str, str] = {}
    for item in items:
        person, _, value = item.partition("=")
        person, value = person.strip(), value.strip()
        if not person or not value:
            raise ValueError(f"bad {item!r}: expected person=value")
        if person in pairs:
            raise ValueError(f"{person!r} given twice")
        pairs[person] = value
    return pairs


def build_speaker(args: argparse.Namespace) -> DemoSpeaker:
    """The engines the voices need (only the installed ones)."""
    if args.engines == "tone":
        return ToneDemoSpeaker()
    python = Path(args.piper_python).expanduser()
    tts = PiperTts(python=python, voices_dir=args.voices_dir) \
        if python.exists() else None
    engines = WorkerEngines(Path(args.natural_dir))
    cloner = None
    clone_python = Path(args.clone_python).expanduser()
    if args.clone_engine == "openvoice" and clone_python.exists():
        cloner = SubprocessCloner(clone_python, "openvoice", extra=[
            "--piper-python", str(python), "--piper-model",
            str(Path(args.voices_dir).expanduser()
                / f"{CLONE_BASE_VOICE}.onnx"),
        ])
    return EngineDemoSpeaker(tts, engines, cloner, args.pocket_device)


def run_render(args: argparse.Namespace) -> int:
    """Render; print what was done as JSON."""
    check_outside_repository(args.out, ROOT)
    if (Path(args.out).expanduser().resolve() != DEFAULT_OUT.resolve()
            and not args.outside_consent_hook):
        raise ValueError(
            f"{args.out} is not {DEFAULT_OUT}, where withdrawing a clone"
            " consent deletes the clips made with it; add"
            " --outside-consent-hook to render there and delete them"
            " yourself")
    speaker = build_speaker(args)
    directory = args.gpu_lease_dir or lease_directory(os.environ)
    render = RenderDemoVoices(
        DirDemos(args.demos_dir), FsDemoVoices(args.out), speaker,
        DirVoiceProfiles(args.voiceprints_dir), LeaseGpuGate(Path(directory)),
    )
    try:
        result = render.render(args.demo, parse_pairs(args.voice),
                               args.clone_engine)
    except DemoVoiceError as exc:
        print(f"error: {exc} (the clips kept before are untouched)",
              file=sys.stderr)
        return 1
    finally:
        speaker.close()
    print(json.dumps(result.as_document(), ensure_ascii=False, indent=2))
    return 0


def run_show(args: argparse.Namespace) -> int:
    """Print who reads whom and the clips' total length."""
    manifest = FsDemoVoices(args.out).manifest(args.demo)
    if manifest is None:
        print(json.dumps({"demo": args.demo, "clips": 0}))
        return 0
    clips = manifest.clips.values()
    print(json.dumps({
        "demo": manifest.demo, "voices": manifest.voices,
        "clips": len(manifest.clips),
        "seconds": round(sum(c.seconds for c in clips), 1),
        "engines": sorted({c.engine for c in clips}),
        "updated_at": manifest.updated_at,
    }, ensure_ascii=False, indent=2))
    return 0


def run_clear(args: argparse.Namespace) -> int:
    """Delete the demo's clips."""
    count = FsDemoVoices(args.out).clear(args.demo)
    print(json.dumps({"demo": args.demo, "deleted": count}))
    return 0


def recast_file(path: Path, person: str, name: str,
                gender: Optional[str]) -> str:
    """Rename ``person`` in the demo file at ``path``; return the old name.

    The renamed demo is checked before the file is written.
    """
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    old = next((str(c.get("name")) for c in data.get("cast") or []
                if c.get("id") == person), "")
    renamed = recast(data, person, name, gender)
    DemoScript.model_validate(renamed)
    text = json.dumps(renamed, ensure_ascii=False, indent=1)
    path.write_text(text + ("\n" if raw.endswith("\n") else ""),
                    encoding="utf-8")
    return old


def run_recast(args: argparse.Namespace) -> int:
    """Rename a person in a demo file; say what changed."""
    pairs = parse_pairs([args.name])
    (person, name), = pairs.items()
    path = Path(args.demos_dir) / f"{args.demo}{SUFFIX}"
    if not path.is_file():
        raise ValueError(f"no demo file {path}")
    old = recast_file(path, person, name, args.gender)
    print(json.dumps({"demo": args.demo, "person": person, "was": old,
                      "now": " ".join(name.split())}, ensure_ascii=False))
    return 0


def run_demo_voices(args: argparse.Namespace) -> int:
    """Run one ``demo-voices`` action."""
    if args.action == "render":
        return run_render(args)
    if args.action == "show":
        return run_show(args)
    if args.action == "clear":
        return run_clear(args)
    return run_recast(args)
