#!/usr/bin/env python3
"""Tell whether a live interview is running on this machine (#264).

``ragputin-servicios stop|restart`` asks this before it stops anything:
a restart under a running interview loses everything the live session
holds in memory. Two signals, either is enough:

* the GPU lease (``$XDG_RUNTIME_DIR/gpu-lease/lease.json``,
  ``gpu-lease.v1``) held by ``ragputin-live``, not expired, its process
  alive;
* the transcriber's live marker (``$XDG_RUNTIME_DIR/ragputin/
  live-active.json``), which also covers a session waiting for its audio
  source to come back after a lost connection (#263), and one that has
  ended but is still handing its transcript over to the RAG: such a
  session may not hold the lease. A marker whose process is gone is
  stale.

Exit status 0: nothing is running; 3: an interview is running; 2: the
files could not be read (the script then refuses too). It prints one
line in Spanish for the owner. Standard library only: it runs with the
system's ``python3``, without the project's environment.
"""

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

LIVE_HOLDER = "ragputin-live"
FREE, BUSY, UNKNOWN = 0, 3, 2


def runtime_dir(environ: Dict[str, str]) -> Tuple[Path, Path]:
    """Return the lease's and the marker's directories."""
    runtime = environ.get("XDG_RUNTIME_DIR")
    if runtime:
        return Path(runtime) / "gpu-lease", Path(runtime) / "ragputin"
    temp = Path(tempfile.gettempdir())
    uid = os.getuid()
    return temp / f"gpu-lease-{uid}", temp / f"ragputin-{uid}"


def pid_alive(pid: Any) -> bool:
    """Tell whether process ``pid`` exists (read from ``/proc``)."""
    return isinstance(pid, int) and pid > 0 \
        and Path("/proc", str(pid)).exists()


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    """The JSON object in ``path``, or ``None`` when there is no file.

    Raises ``ValueError`` when it cannot be read as one.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def parse_time(text: Any) -> datetime:
    """An ISO time with its zone (``Z`` accepted)."""
    if not isinstance(text, str):
        raise ValueError("no time")
    when = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if when.tzinfo is None:
        raise ValueError("a time without its zone")
    return when


def lease_busy(
    lease: Optional[Dict[str, Any]], now: datetime,
    alive: Callable[[Any], bool],
) -> bool:
    """Tell whether the live mode holds a valid GPU lease."""
    if lease is None or lease.get("holder") != LIVE_HOLDER:
        return False
    return parse_time(lease.get("expires_at")) > now \
        and alive(lease.get("pid"))


def marked_sessions(
    marker: Optional[Dict[str, Any]], alive: Callable[[Any], bool],
) -> List[Dict[str, Any]]:
    """The live sessions the transcriber says are on (none if stale)."""
    if marker is None or not alive(marker.get("pid")):
        return []
    sessions = marker.get("sessions")
    return [s for s in sessions if isinstance(s, dict)] \
        if isinstance(sessions, list) else []


def describe(sessions: Sequence[Dict[str, Any]], leased: bool) -> str:
    """The owner's line for a running interview."""
    if not sessions:
        return ("Entrevista en directo en marcha (la tarjeta gráfica está"
                " reservada para el directo).")
    parts = []
    for session in sessions:
        note = ""
        if session.get("state") == "handing_over":
            note = " (terminada; enviando su transcripción)"
        elif session.get("source") == "lost":
            note = " (esperando a que vuelva el audio)"
        parts.append(f"{session.get('id', '?')}{note}")
    return "Entrevista en directo en marcha: " + ", ".join(parts) + "."


def check(
    environ: Dict[str, str], now: datetime,
    alive: Callable[[Any], bool] = pid_alive,
) -> Tuple[int, str]:
    """Return the exit status and the owner's line."""
    lease_dir, marker_dir = runtime_dir(environ)
    try:
        leased = lease_busy(read_json(lease_dir / "lease.json"), now, alive)
        sessions = marked_sessions(
            read_json(marker_dir / "live-active.json"), alive)
    except (OSError, ValueError) as exc:
        return UNKNOWN, ("No se pudo comprobar si hay una entrevista en"
                         f" directo ({exc}).")
    if leased or sessions:
        return BUSY, describe(sessions, leased)
    return FREE, "No hay ninguna entrevista en directo en marcha."


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Print the line; return 0 (free), 3 (running) or 2 (unknown)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args(argv)
    status, line = check(dict(os.environ), datetime.now(timezone.utc))
    print(line)
    return status


if __name__ == "__main__":
    sys.exit(main())
