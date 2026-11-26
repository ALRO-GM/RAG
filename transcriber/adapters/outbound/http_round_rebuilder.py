"""Round rebuilder adapter: ask the orchestrator to rebuild a round (#265).

Implements :class:`~transcriber.application.ports.RoundRebuilder` with
``POST <orchestrator>/sessions/{id}/recovered`` and the session's
merged ``transcript.v1`` as the body; the orchestrator replays it
through the round's trackers, judges it again and updates the
candidate's file (``RoundRebuild``). The bearer token is the one the
live page's proxy adds (the RAG token file, read per call). A refusal
or no answer is a :class:`PublishError` (the recovery tries again).
"""

import re
from pathlib import Path
from typing import Any, Mapping, Optional

import httpx

from transcriber.domain.errors import PublishError
from transcriber.domain.models import SESSION_ID_PATTERN

TIMEOUT = 600.0
"""Seconds for a rebuild: the judgement asks the reasoner per answer."""


class HttpRoundRebuilder:
    """The orchestrator's ``/sessions/{id}/recovered``."""

    def __init__(
        self, base_url: str, token_file: Optional[Path],
        transport: Optional[httpx.BaseTransport] = None,
    ) -> None:
        """Keep the address, the token's file and a test transport."""
        self._base = base_url.rstrip("/")
        self._token_file = token_file
        self._transport = transport

    def _token(self) -> Optional[str]:
        """The bearer token, read now (``None`` without one)."""
        if self._token_file is None:
            return None
        try:
            text = self._token_file.expanduser().read_text(encoding="utf-8")
        except OSError:
            return None
        return text.strip() or None

    def rebuild(self, session_id: str, transcript: Mapping[str, Any]) -> None:
        """Ask for the rebuild and wait for it."""
        if not re.fullmatch(SESSION_ID_PATTERN, session_id):
            raise PublishError("invalid session id")
        token = self._token()
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        try:
            with httpx.Client(transport=self._transport,
                              timeout=TIMEOUT) as client:
                answer = client.post(
                    f"{self._base}/sessions/{session_id}/recovered",
                    json={"transcript": dict(transcript)}, headers=headers)
        except httpx.HTTPError as exc:
            raise PublishError(
                f"the orchestrator did not answer ({type(exc).__name__})"
            ) from exc
        if answer.status_code >= 400:
            raise PublishError(
                f"the orchestrator refused the rebuild ({answer.status_code})")
