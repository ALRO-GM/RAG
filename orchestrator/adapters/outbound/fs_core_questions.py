"""The core questions of KN-4 read from the knowledge folder (#235).

``knowledge/interview/core-questions.json`` (``interview.core.v1``) is
plain JSON; the RAG validates it fully with its own module, and the
orchestrator reads only the fields it needs
(:func:`orchestrator.domain.candidate.core_questions`), never the RAG's
code. The file is read again when it changes; an unreadable file gives
no questions (and the reason in :attr:`FileCoreQuestions.error`).
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple

from orchestrator.domain.candidate import CoreQuestion, core_questions


class FileCoreQuestions:
    """The :class:`~orchestrator.application.ports.CoreQuestionSource`.

    Args:
        path: The ``interview.core.v1`` file.
    """

    def __init__(self, path: Path):
        """Keep the path; read on first use."""
        self.path = Path(path)
        self.error: Optional[str] = None
        self._cached: Optional[Tuple[float, List[CoreQuestion]]] = None

    def questions(self) -> List[CoreQuestion]:
        """The questions in order (empty when the file is unreadable)."""
        try:
            mtime = self.path.stat().st_mtime
            if self._cached is not None and self._cached[0] == mtime:
                return list(self._cached[1])
            found = core_questions(json.loads(
                self.path.read_text(encoding="utf-8")))
        except (OSError, ValueError) as exc:
            self.error = str(exc)[:200]
            return []
        self.error = None
        self._cached = (mtime, found)
        return list(found)
