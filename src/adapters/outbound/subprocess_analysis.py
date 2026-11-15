"""Analysis launcher: ``skills analyze`` in a child process.

The RAG's API starts an analysis and answers at once; the analysis runs
as ``python -m src skills ... analyze --session <id>`` from the
repository root, with the same transcripts and analysis directories as
the server, its output appended to ``<analysis dir>/<id>/analyze.log``.
The child's models are freed when it ends, and a crash of the analysis
never takes the API down. The run's record (``run.json``) is what the
API reads back; this launcher only knows which children are alive.

Each child runs in its own process group, with a reduced environment
(no variable whose name holds TOKEN, SECRET, PASSWORD or KEY: the
RAG's bearer token never reaches it), is stopped after ``timeout``
seconds, can be stopped on request (cancel, deleted session) and is
stopped when the server process exits. A stop sends SIGTERM to the
child's group (the child turns it into a clean exit that kills its
judge) and, once the child is gone, kills the groups of every
descendant it had started in a session of its own (``claude -p``), so
no judge outlives a cancelled analysis. At most ``max_running``
children run at once. Session ids are checked by the caller and again
here, passed as one quoted argument so the command line never turns an
id like ``1e5`` into a number, and never go through a shell.
"""

import atexit
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

from src.domain.errors import BusyError, LimitError, RagError
from src.domain.skills.transcript_contract import is_session_id

SECRET = re.compile(r"TOKEN|SECRET|PASSWORD|KEY", re.IGNORECASE)
"""Names of environment variables never handed to a child."""


def descendants(pid: int, proc: Path = Path("/proc")) -> List[Tuple[int, int]]:
    """Return ``(pid, process group)`` of every descendant of ``pid``.

    Read from ``/proc``; where it does not exist the answer is empty.
    """
    parents: Dict[int, List[Tuple[int, int]]] = {}
    try:
        entries = list(proc.iterdir())
    except OSError:
        return []
    for entry in entries:
        if not entry.name.isdigit():
            continue
        stat = _stat(entry)
        if stat is not None:
            parent, group = stat
            parents.setdefault(parent, []).append((int(entry.name), group))
    found: List[Tuple[int, int]] = []
    todo = [pid]
    while todo:
        for child in parents.get(todo.pop(), []):
            found.append(child)
            todo.append(child[0])
    return found


def _stat(entry: Path) -> Optional[Tuple[int, int]]:
    """Return ``(parent pid, process group)`` of a ``/proc/<pid>`` entry."""
    try:
        text = (entry / "stat").read_text()
    except OSError:
        return None
    fields = text.rpartition(")")[2].split()
    if len(fields) < 3 or fields[0] == "Z":
        return None
    return int(fields[1]), int(fields[2])


def kill_groups(
    pid: int, family: List[Tuple[int, int]], proc: Path = Path("/proc")
) -> None:
    """SIGKILL the groups of ``family`` that are not ``pid``'s own.

    A process is killed only when it still lives in the same group, so
    a reused pid is left alone; this server's group is never touched.
    """
    own = os.getpgrp()
    for member, group in family:
        if group in (pid, own):
            continue
        stat = _stat(proc / str(member))
        if stat is None or stat[1] != group:
            continue
        try:
            os.killpg(group, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def child_environment(environ: Mapping[str, str]) -> Dict[str, str]:
    """Return ``environ`` without the variables that may hold secrets."""
    return {k: v for k, v in environ.items() if not SECRET.search(k)}


class SubprocessAnalysisLauncher:
    """Start, watch and stop ``skills analyze`` children.

    Args:
        root: The checkout to run from (``python -m src``).
        transcripts_dir: Where the published transcripts are.
        analysis_dir: Where each session's analysis is kept.
        python: The interpreter (default: this one).
        timeout: Seconds a child may run before it is stopped.
        max_running: Most children running at once.
        grace: Seconds a stopped child gets before it is killed.
    """

    def __init__(
        self,
        root: Path,
        transcripts_dir: str,
        analysis_dir: str,
        python: Optional[str] = None,
        timeout: float = 3600.0,
        max_running: int = 1,
        grace: float = 10.0,
    ) -> None:
        """Keep the paths; nothing is started yet."""
        self.root = root
        self.transcripts_dir = str(Path(transcripts_dir).resolve())
        self.analysis_dir = Path(analysis_dir).resolve()
        self.python = python or sys.executable
        self.timeout = timeout
        self.max_running = max(max_running, 1)
        self.grace = grace
        self._children: Dict[str, "subprocess.Popen[bytes]"] = {}
        self._timers: Dict[str, threading.Timer] = {}
        self._lock = threading.Lock()
        atexit.register(self.stop_all)

    def command(
        self, session_id: str, restart: bool, synthetic: bool
    ) -> List[str]:
        """Return the child's argument list."""
        args = [
            self.python, "-m", "src", "skills",
            "--transcripts_dir", self.transcripts_dir,
            "--analysis_dir", str(self.analysis_dir),
            "analyze", f'--session="{session_id}"',
        ]
        if restart:
            args.append("--restart")
        if synthetic:
            args.append("--synthetic")
        return args

    def start(self, session_id: str, restart: bool, synthetic: bool) -> None:
        """Start the analysis of ``session_id`` unless it is running.

        Raises :class:`~src.domain.errors.BusyError` when this session is
        running and :class:`~src.domain.errors.LimitError` when
        ``max_running`` analyses already run.
        """
        if not is_session_id(session_id):
            raise RagError(f"Invalid session id {session_id!r}.")
        with self._lock:
            if self._alive(session_id):
                raise BusyError(
                    f"The analysis of {session_id!r} is already running."
                )
            running = [s for s in list(self._children) if self._alive(s)]
            if len(running) >= self.max_running:
                raise LimitError(
                    f"{len(running)} analysis already running (at most "
                    f"{self.max_running} at a time); try again when it ends."
                )
            folder = self.analysis_dir / session_id
            try:
                folder.mkdir(parents=True, exist_ok=True)
                with open(folder / "analyze.log", "ab") as log:
                    child = subprocess.Popen(
                        self.command(session_id, restart, synthetic),
                        cwd=self.root, stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=True,
                        env=child_environment(os.environ),
                    )
            except OSError as exc:
                raise RagError(f"Cannot start the analysis: {exc}")
            self._children[session_id] = child
            timer = threading.Timer(
                self.timeout, self._expire, (session_id, child)
            )
            timer.daemon = True
            timer.start()
            self._timers[session_id] = timer

    def running(self, session_id: str) -> bool:
        """Tell whether a child of this server analyses ``session_id``."""
        with self._lock:
            return self._alive(session_id)

    def stop(self, session_id: str) -> bool:
        """Stop the child of ``session_id``; tell whether one ran."""
        with self._lock:
            child = self._children.pop(session_id, None)
            timer = self._timers.pop(session_id, None)
        if timer is not None:
            timer.cancel()
        if child is None or child.poll() is not None:
            return False
        self._terminate(child)
        return True

    def stop_all(self) -> None:
        """Stop every child (the server is shutting down)."""
        for session_id in list(self._children):
            self.stop(session_id)

    def _expire(
        self, session_id: str, child: "subprocess.Popen[bytes]"
    ) -> None:
        """Stop a child that ran past the time-out."""
        with self._lock:
            if self._children.get(session_id) is not child:
                return
        self.stop(session_id)

    def _terminate(self, child: "subprocess.Popen[bytes]") -> None:
        """Stop the child's group, then its descendants' own groups."""
        family = descendants(child.pid)
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            kill_groups(child.pid, family)
            return
        deadline = time.monotonic() + self.grace
        while child.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()
        kill_groups(child.pid, family)

    def _alive(self, session_id: str) -> bool:
        """Tell whether the child is alive; forget it once it ended."""
        child = self._children.get(session_id)
        if child is None:
            return False
        if child.poll() is None:
            return True
        del self._children[session_id]
        timer = self._timers.pop(session_id, None)
        if timer is not None:
            timer.cancel()
        return False
