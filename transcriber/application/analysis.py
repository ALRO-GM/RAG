"""The RAG's soft-skills analysis, seen from the web.

:class:`AnalysisJudgeSetting`: which judge the analysis uses.
:class:`SessionAnalysisAccess`: start the whole analysis of a session
and read its steps, results and person reports (#88).

The web chooses the judge (``claude_code``, ``local``, ``both`` or
``opencode``) and
sees whether Claude's quota is available. The RAG keeps the setting;
the transcriber only forwards the request on loopback with the RAG's
token, so the RAG itself stays unreachable from the tunnel
(``docs/transcriber/ARCHITECTURE.md`` §5).
"""

import re
from typing import Any, Dict, Optional, Sequence

from transcriber.application.ports import AnalysisJudge, SessionAnalysisPort
from transcriber.domain.errors import (
    AnalysisUnavailableError,
    InvalidInputError,
)
from transcriber.domain.models import SESSION_ID_PATTERN

JUDGE_MODES = ("claude_code", "local", "both", "opencode")
"""The judges the web may choose (the RAG checks them again)."""

ANALYSIS_STEPS = (
    "sentences", "interaction", "candidates", "acts", "events",
    "confirmation", "assessment", "reports", "synthesis",
)
"""The steps whose results the web may read (the RAG checks them too)."""

MAX_JOB = 4000
"""Longest job description forwarded (the RAG's own limit)."""

MAX_REPORTS = 99
"""Highest report number accepted (one per person of a session)."""


class AnalysisJudgeSetting:
    """Read and change the judge of the RAG's analysis.

    ``judge`` is ``None`` when transcripts do not go to a RAG over
    HTTP (``publisher: "directory"``): then there is nothing to set.
    """

    def __init__(self, judge: Optional[AnalysisJudge]) -> None:
        """Wire the port (or its absence)."""
        self._judge = judge

    def _port(self) -> AnalysisJudge:
        """Return the port, or say the RAG is not connected."""
        if self._judge is None:
            raise AnalysisUnavailableError(
                "Transcripts are not sent to a RAG over HTTP, so there is "
                "no analysis to configure."
            )
        return self._judge

    def status(self) -> Dict[str, Any]:
        """Return the judge in use and Claude's quota status."""
        return self._port().status()

    def choose(self, mode: str) -> Dict[str, Any]:
        """Set the judge of the next analyses."""
        if mode not in JUDGE_MODES:
            raise InvalidInputError(
                f"Unknown judge {mode!r}; choose one of "
                f"{', '.join(JUDGE_MODES)}."
            )
        return self._port().choose(mode)


class SessionAnalysisAccess:
    """Start a session's analysis on the RAG and read what it stored.

    Ids, step names and report numbers are checked here before anything
    is forwarded, so nothing else reaches the RAG's paths. ``port`` is
    ``None`` without an HTTP link to the RAG.
    """

    def __init__(self, port: Optional[SessionAnalysisPort]) -> None:
        """Wire the port (or its absence)."""
        self._port_or_none = port

    def _port(self, session_id: str) -> SessionAnalysisPort:
        """Check the id; return the port or say the RAG is not linked."""
        if not re.fullmatch(SESSION_ID_PATTERN, session_id):
            raise InvalidInputError(f"Invalid session id {session_id!r}.")
        if self._port_or_none is None:
            raise AnalysisUnavailableError(
                "Transcripts are not sent to a RAG over HTTP, so there is "
                "no analysis."
            )
        return self._port_or_none

    def start(
        self,
        session_id: str,
        restart: bool = False,
        synthetic: bool = False,
        steps: Sequence[str] = (),
        job: Optional[str] = None,
        claude_synthesis: bool = False,
    ) -> Dict[str, Any]:
        """Start the analysis of a published session.

        ``steps`` runs only those steps again (``synthesis`` after the
        job sought changed); ``job`` is the job sought, kept with the
        analysis, and ``claude_synthesis`` the leave to use Claude for
        the profiles when the RAG's judge is the local model (#113).
        """
        port = self._port(session_id)
        unknown = [s for s in steps if s not in ANALYSIS_STEPS]
        if unknown:
            raise InvalidInputError(f"Unknown step {unknown[0]!r}.")
        if job is not None and len(job) > MAX_JOB:
            raise InvalidInputError(
                f"The job description is longer than {MAX_JOB} characters."
            )
        return port.start(
            session_id, restart, synthetic, list(steps), job,
            claude_synthesis,
        )

    def status(self, session_id: str) -> Dict[str, Any]:
        """Return the steps of the session's analysis."""
        return self._port(session_id).status(session_id)

    def cancel(self, session_id: str) -> None:
        """Stop the session's running analysis."""
        self._port(session_id).cancel(session_id)

    def result(self, session_id: str, step: str) -> Any:
        """Return one step's result."""
        port = self._port(session_id)
        if step not in ANALYSIS_STEPS:
            raise InvalidInputError(f"Unknown step {step!r}.")
        return port.result(session_id, step)

    def report(self, session_id: str, position: int) -> str:
        """Return one person's report (HTML)."""
        port = self._port(session_id)
        if not 1 <= position <= MAX_REPORTS:
            raise InvalidInputError(
                f"Reports are numbered 1 to {MAX_REPORTS}."
            )
        return port.report(session_id, position)
