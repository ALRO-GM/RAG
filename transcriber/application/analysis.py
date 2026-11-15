"""Analysis settings: which judge the RAG's soft-skills analysis uses.

The web chooses the judge (``claude_code``, ``local`` or ``both``) and
sees whether Claude's quota is available. The RAG keeps the setting;
the transcriber only forwards the request on loopback with the RAG's
token, so the RAG itself stays unreachable from the tunnel
(``docs/transcriber/ARCHITECTURE.md`` §5).
"""

from typing import Any, Dict, Optional

from transcriber.application.ports import AnalysisJudge
from transcriber.domain.errors import (
    AnalysisUnavailableError,
    InvalidInputError,
)

JUDGE_MODES = ("claude_code", "local", "both")
"""The judges the web may choose (the RAG checks them again)."""


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
