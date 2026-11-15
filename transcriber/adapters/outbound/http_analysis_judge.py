"""Analysis judge: forward ``GET/PUT /analysis/judge`` to the RAG.

The RAG listens on loopback only; the tunnel forwards the transcriber's
port, never the RAG's (``docs/DEPLOYMENT.md``). So the web asks the
transcriber (behind the access code) and this adapter asks the RAG,
with the same base URL and bearer token the transcript publisher uses.
One attempt with a short time-out: a person is waiting on the page.
The token never appears in a message.
"""

import json
import socket
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, Optional

from transcriber.adapters.outbound.no_redirect import OPENER
from transcriber.domain.errors import AnalysisUnavailableError, PublishError

ROUTE = "/analysis/judge"


class HttpAnalysisJudge:
    """Read and change the RAG's judge setting over HTTP.

    Args:
        base_url: The RAG's base URL (the publisher's).
        token: Returns the RAG's bearer token when a call is made.
        timeout: Seconds to wait for the RAG.
    """

    def __init__(
        self,
        base_url: str,
        token: Callable[[], str],
        timeout: float = 5.0,
    ) -> None:
        """Keep the target; nothing is sent yet."""
        self.url = base_url.rstrip("/") + ROUTE
        self._token = token
        self.timeout = timeout

    def status(self) -> Dict[str, Any]:
        """Return the RAG's answer to ``GET /analysis/judge``."""
        return self._call("GET", None)

    def choose(self, mode: str) -> Dict[str, Any]:
        """Return the RAG's answer to ``PUT /analysis/judge``."""
        return self._call("PUT", {"mode": mode})

    def _call(
        self, method: str, body: Optional[Dict[str, Any]]
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
            with OPENER.open(
                request, timeout=self.timeout
            ) as response:
                answer = json.loads(response.read(65536).decode("utf-8"))
        except urllib.error.HTTPError as exc:
            status = exc.code
            exc.close()
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
