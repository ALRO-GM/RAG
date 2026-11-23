"""Checks of a cloned voice on a spare orchestrator (#211).

Two sub-commands, both writing JSON::

    clone_checks.py speed --orchestrator URL --token-file T --out R.json \\
        [--choice voicebot.voice=clone:synth ...] [--speeds 1 1.4 0.8 1]
    clone_checks.py watch --seconds N --out R.json [--lease-dir D]

``speed`` asks «Escuchar ejemplo» (``POST /voicebot/sample``) for the
fixed sentence at each speed (``voicebot.length_scale``) with the given
unsaved choices and records the WAV's seconds and the answer's
``X-Voice-*`` headers (who spoke, why not the clone). ``watch`` samples
every 2 s, for ``--seconds``, who holds the GPU lease and which
processes use GPU memory (``nvidia-smi``), to see the clone's worker
appear under the live lease and go away afterwards. Nothing here
starts or stops a service.
"""

import argparse
import io
import json
import os
import subprocess
import time
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import httpx

from orchestrator.adapters.outbound.gpu_gate import lease_directory


def parse(pairs: Sequence[str]) -> Dict[str, str]:
    """Read ``step=value`` pairs."""
    out: Dict[str, str] = {}
    for pair in pairs:
        step, sep, value = pair.partition("=")
        if not sep:
            raise SystemExit(f"bad choice {pair!r}")
        out[step] = value
    return out


def speed(args: argparse.Namespace) -> Dict[str, Any]:
    """Say the sample at each speed; return durations and headers."""
    token = args.token_file.expanduser().read_text(encoding="utf-8").strip()
    rows: List[Dict[str, Any]] = []
    with httpx.Client(timeout=180) as client:
        for value in args.speeds:
            choices = {**parse(args.choice), "voicebot.length_scale": value}
            started = time.monotonic()
            answer = client.post(
                f"{args.orchestrator}/voicebot/sample",
                headers={"Authorization": f"Bearer {token}"},
                json={"choices": choices})
            answer.raise_for_status()
            with wave.open(io.BytesIO(answer.content), "rb") as reader:
                seconds = reader.getnframes() / reader.getframerate()
                rate = reader.getframerate()
            rows.append({
                "length_scale": value, "audio_s": round(seconds, 2),
                "sample_rate": rate,
                "wall_s": round(time.monotonic() - started, 2),
                "engine": answer.headers.get("x-voice-engine"),
                "clone": answer.headers.get("x-voice-clone"),
                "note": answer.headers.get("x-voice-note"),
            })
    return {"choices": parse(args.choice), "rows": rows}


def gpu_apps() -> List[Dict[str, Any]]:
    """The processes using GPU memory: pid and MiB."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    apps = []
    for line in out.splitlines():
        pid, _, mib = line.partition(",")
        if mib.strip().isdigit():
            apps.append({"pid": int(pid), "mib": int(mib)})
    return apps


def holder(directory: Path) -> Optional[str]:
    """The lease file's holder, or ``None``."""
    try:
        data = json.loads((directory / "lease.json").read_text("utf-8"))
        return str(data["holder"])
    except (OSError, ValueError, KeyError):
        return None


def watch(args: argparse.Namespace) -> Dict[str, Any]:
    """Sample the lease holder and the GPU processes for a while."""
    directory = args.lease_dir or lease_directory(os.environ)
    rows = []
    end = time.monotonic() + args.seconds
    while time.monotonic() < end:
        rows.append({"at": time.strftime("%H:%M:%S"),
                     "lease": holder(directory), "gpu": gpu_apps()})
        time.sleep(2)
    return {"rows": rows}


def main() -> None:
    """Run one sub-command and write its JSON."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("speed")
    one.add_argument("--orchestrator", required=True)
    one.add_argument("--token-file", type=Path, required=True)
    one.add_argument("--choice", action="append", default=[])
    one.add_argument("--speeds", nargs="+", default=["1", "1.4", "0.8", "1"])
    two = sub.add_parser("watch")
    two.add_argument("--seconds", type=float, required=True)
    two.add_argument("--lease-dir", type=Path, default=None)
    for each in (one, two):
        each.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = speed(args) if args.command == "speed" else watch(args)
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
