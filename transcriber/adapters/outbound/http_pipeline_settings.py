"""Pipeline settings: forward ``GET/PUT /settings`` to the RAG.

The RAG owns the ``settings.v1`` document (``data/settings.json``) and
listens on loopback only, so the web asks the transcriber (behind the
access code) and this adapter asks the RAG with the publisher's base
URL and bearer token, like :mod:`.http_analysis_judge`. A job or live
session that starts reads the choices with :meth:`current`, which never
fails: without an answer the service keeps its configuration.
"""

import json
import logging
import socket
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, Optional

from transcriber.adapters.outbound.no_redirect import OPENER
from transcriber.domain.errors import (
    AnalysisUnavailableError,
    InvalidInputError,
    PublishError,
    StateError,
)

ROUTE = "/settings"
LOG = logging.getLogger(__name__)


class HttpPipelineSettings:
    """Read and change the RAG's pipeline settings over HTTP.

    Args:
        base_url: The RAG's base URL (the publisher's).
        token: Returns the RAG's bearer token when a call is made.
        timeout: Seconds to wait for the RAG from the page.
        read_timeout: Seconds :meth:`current` waits when a job starts.
    """

    def __init__(
        self,
        base_url: str,
        token: Callable[[], str],
        timeout: float = 5.0,
        read_timeout: float = 2.0,
    ) -> None:
        """Keep the target; nothing is sent yet."""
        self.url = base_url.rstrip("/") + ROUTE
        self._token = token
        self.timeout = timeout
        self.read_timeout = read_timeout

    def show(self) -> Dict[str, Any]:
        """Return the RAG's answer to ``GET /settings``."""
        return self._call("GET", None, self.timeout)

    def change(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Return the RAG's answer to ``PUT /settings``.

        Its 422 (an unknown step or option) is an
        :class:`InvalidInputError` and its 409 (saved meanwhile) a
        :class:`StateError`, both with the RAG's reason.
        """
        return self._call("PUT", body, self.timeout)

    def current(self) -> Optional[Dict[str, str]]:
        """Return the saved choices, or ``None`` (none saved, no answer)."""
        try:
            answer = self._call("GET", None, self.read_timeout)
        except AnalysisUnavailableError as exc:
            LOG.warning("Pipeline settings not read: %s", exc)
            return None
        choices = answer.get("choices")
        if not answer.get("stored") or not isinstance(choices, dict):
            return None
        return {
            str(k): v for k, v in choices.items() if isinstance(v, str)
        }

    def _call(
        self, method: str, body: Optional[Dict[str, Any]], timeout: float
    ) -> Dict[str, Any]:
        """Send one request; return the JSON object answered."""
        try:
            token = self._token()
        except PublishError as exc:
            raise AnalysisUnavailableError(str(exc)) from None
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
        }
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self.url, data=data, method=method, headers=headers
        )
        try:
            with OPENER.open(request, timeout=timeout) as response:
                answer = json.loads(response.read(262144).decode("utf-8"))
        except urllib.error.HTTPError as exc:
            status, reason = exc.code, _detail(exc)
            exc.close()
            if status == 422:
                raise InvalidInputError(reason) from None
            if status == 409:
                raise StateError(reason) from None
            raise AnalysisUnavailableError(
                f"The RAG answered {status} to {method} {self.url}."
            ) from None
        except (socket.timeout, TimeoutError, urllib.error.URLError,
                OSError):
            raise AnalysisUnavailableError(
                f"Cannot reach the RAG at {self.url}."
            ) from None
        except ValueError:
            raise AnalysisUnavailableError(
                "The RAG did not answer JSON."
            ) from None
        if not isinstance(answer, dict):
            raise AnalysisUnavailableError("The RAG did not answer JSON.")
        return answer


def _detail(exc: urllib.error.HTTPError) -> str:
    """Return the ``detail`` of an error answer (short), or a default."""
    try:
        detail = json.loads(exc.read(4096).decode("utf-8")).get("detail")
    except (ValueError, AttributeError, OSError):
        detail = None
    if isinstance(detail, str) and detail:
        return detail[:300]
    return "The RAG refused the settings."
