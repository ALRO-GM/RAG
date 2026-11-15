"""Session analysis: forward ``/analysis/sessions/...`` to the RAG.

As :mod:`~transcriber.adapters.outbound.http_analysis_judge`: the RAG
listens on loopback only, so the web asks the transcriber (behind the
access code) and this adapter asks the RAG with its bearer token,
never following a redirect. Start, step records, step results and the
person reports (HTML, returned as text for the API to serve sandboxed).
The token never appears in a message; the RAG's own error text does
not either, only its status.
"""

import json
import socket
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, Optional
from urllib.parse import quote

from transcriber.adapters.outbound.no_redirect import OPENER
from transcriber.domain.errors import (
    AnalysisUnavailableError,
    InvalidInputError,
    LimitReachedError,
    NotFoundError,
    PublishError,
    StateError,
)

ROUTE = "/analysis/sessions/"
MAX_BYTES = 16 * 1024 * 1024
"""Largest answer read from the RAG (a long session's step result)."""


class HttpSessionAnalysis:
    """Start a session's analysis on the RAG and read its results.

    Args:
        base_url: The RAG's base URL (the publisher's).
        token: Returns the RAG's bearer token when a call is made.
        timeout: Seconds to wait for the RAG.
    """

    def __init__(
        self,
        base_url: str,
        token: Callable[[], str],
        timeout: float = 10.0,
    ) -> None:
        """Keep the target; nothing is sent yet."""
        self.url = base_url.rstrip("/") + ROUTE
        self._token = token
        self.timeout = timeout

    def start(
        self, session_id: str, restart: bool, synthetic: bool
    ) -> Dict[str, Any]:
        """Ask the RAG to start the analysis (``POST``)."""
        answer = self._call("POST", session_id, {
            "restart": restart, "synthetic": synthetic,
        })
        return self._object(json.loads(answer))

    def status(self, session_id: str) -> Dict[str, Any]:
        """Return the RAG's record of the session's analysis."""
        return self._object(json.loads(self._call("GET", session_id)))

    def cancel(self, session_id: str) -> None:
        """Ask the RAG to stop the analysis (``DELETE …/run``)."""
        self._call("DELETE", f"{session_id}/run")

    def result(self, session_id: str, step: str) -> Any:
        """Return one step's stored result (JSON data)."""
        return json.loads(
            self._call("GET", f"{session_id}/steps/{quote(step)}")
        )

    def report(self, session_id: str, position: int) -> str:
        """Return the HTML report of the ``position``-th person."""
        return self._call(
            "GET", f"{session_id}/reports/{int(position)}", accept="text/html"
        )

    @staticmethod
    def _object(answer: Any) -> Dict[str, Any]:
        """Return ``answer`` if it is a JSON object."""
        if not isinstance(answer, dict):
            raise AnalysisUnavailableError("The RAG did not answer JSON.")
        return answer

    def _call(
        self,
        method: str,
        path: str,
        body: Optional[Dict[str, Any]] = None,
        accept: str = "application/json",
    ) -> str:
        """Send one request; return the text answered."""
        try:
            token = self._token()
        except PublishError as exc:
            raise AnalysisUnavailableError(str(exc)) from None
        headers = {"Accept": accept, "Authorization": f"Bearer {token}"}
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        url = self.url + quote(path, safe="/")
        request = urllib.request.Request(
            url, data=data, method=method, headers=headers
        )
        try:
            with OPENER.open(request, timeout=self.timeout) as response:
                raw: bytes = response.read(MAX_BYTES + 1)
        except urllib.error.HTTPError as exc:
            status = exc.code
            exc.close()
            raise refusal(status, method, url) from None
        except (socket.timeout, TimeoutError, urllib.error.URLError,
                OSError):
            raise AnalysisUnavailableError(
                f"Cannot reach the RAG at {self.url}."
            ) from None
        if len(raw) > MAX_BYTES:
            raise AnalysisUnavailableError("The RAG's answer is too large.")
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            raise AnalysisUnavailableError(
                "The RAG did not answer text."
            ) from None


def refusal(status: int, method: str, url: str) -> Exception:
    """Return the domain error for a RAG answer ``status``."""
    if status == 404:
        return NotFoundError("No such analysis, step or report.")
    if status == 409:
        return StateError(
            "The analysis of this session is running (or runs from the "
            "command line)."
        )
    if status in (400, 422):
        return InvalidInputError("The RAG refused the request.")
    if status == 429:
        return LimitReachedError(
            "A limit on analyses refuses this start now (one at a time, "
            "a pause between two starts of a session, a daily cap)."
        )
    return AnalysisUnavailableError(
        f"The RAG answered {status} to {method} {url}."
    )
