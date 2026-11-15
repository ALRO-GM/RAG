"""Analysis launcher: ``skills analyze`` in a child process.

The RAG's API starts an analysis and answers at once; the analysis runs
as ``python -m src skills ... analyze --session <id>`` from the
repository root, with the same transcripts and analysis directories as
the server, its output appended to ``<analysis dir>/<id>/analyze.log``.
The child's models are freed when it ends, and a crash of the analysis
never takes the API down. The run's record (``run.json``) is what the
API reads back; this launcher only knows which children are alive.
Session ids are checked by the caller (the use case) and are passed as
one argument each, never through a shell.
"""

import subprocess
import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional

from src.domain.errors import BusyError, RagError
from src.domain.skills.transcript_contract import is_session_id


class SubprocessAnalysisLauncher:
    """Start and watch ``skills analyze`` children, one per session.

    Args:
        root: The checkout to run from (``python -m src``).
        transcripts_dir: Where the published transcripts are.
        analysis_dir: Where each session's analysis is kept.
        python: The interpreter (default: this one).
    """

    def __init__(
        self,
        root: Path,
        transcripts_dir: str,
        analysis_dir: str,
        python: Optional[str] = None,
    ) -> None:
        """Keep the paths; nothing is started yet."""
        self.root = root
        self.transcripts_dir = str(Path(transcripts_dir).resolve())
        self.analysis_dir = Path(analysis_dir).resolve()
        self.python = python or sys.executable
        self._children: Dict[str, "subprocess.Popen[bytes]"] = {}
        self._lock = threading.Lock()

    def command(
        self, session_id: str, restart: bool, synthetic: bool
    ) -> List[str]:
        """Return the child's argument list."""
        args = [
            self.python, "-m", "src", "skills",
            "--transcripts_dir", self.transcripts_dir,
            "--analysis_dir", str(self.analysis_dir),
            "analyze", "--session", session_id,
        ]
        if restart:
            args.append("--restart")
        if synthetic:
            args.append("--synthetic")
        return args

    def start(self, session_id: str, restart: bool, synthetic: bool) -> None:
        """Start the analysis of ``session_id`` unless it is running."""
        if not is_session_id(session_id):
            raise RagError(f"Invalid session id {session_id!r}.")
        with self._lock:
            if self._alive(session_id):
                raise BusyError(
                    f"The analysis of {session_id!r} is already running."
                )
            folder = self.analysis_dir / session_id
            try:
                folder.mkdir(parents=True, exist_ok=True)
                with open(folder / "analyze.log", "ab") as log:
                    self._children[session_id] = subprocess.Popen(
                        self.command(session_id, restart, synthetic),
                        cwd=self.root, stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
            except OSError as exc:
                raise RagError(f"Cannot start the analysis: {exc}")

    def running(self, session_id: str) -> bool:
        """Tell whether a child of this server analyses ``session_id``."""
        with self._lock:
            return self._alive(session_id)

    def _alive(self, session_id: str) -> bool:
        """Tell whether the child is alive; forget it once it ended."""
        child = self._children.get(session_id)
        if child is None:
            return False
        if child.poll() is None:
            return True
        del self._children[session_id]
        return False
