"""A service's health route as a :class:`ServiceProbe` (#247).

The orchestrator (``GET /health``) and the RAG (``GET /health``) listen
on loopback; the transcriber asks them for the live page's connection
check. Only the status matters: a ``200`` within the timeout is
available, anything else (a refusal, a redirect, a timeout, a closed
port) is not. The answer's body is never read beyond a few bytes nor
passed on, so the RAG's index statistics stay internal. No token is
sent: both health routes are open on loopback.
"""

import urllib.error
import urllib.request

from transcriber.adapters.outbound.no_redirect import OPENER

TIMEOUT = 1.5
"""Seconds to wait: a loopback health route answers in milliseconds,
and the browser's whole check should take a couple of seconds."""


class HttpHealthProbe:
    """Tell whether ``GET url`` answers ``200`` in time.

    Args:
        url: The health route (``http://127.0.0.1:8772/health``).
        timeout: Seconds to wait for the answer.
    """

    def __init__(self, url: str, timeout: float = TIMEOUT) -> None:
        """Keep the address; nothing is sent yet."""
        self.url = url
        self.timeout = timeout

    def available(self) -> bool:
        """Return ``True`` when the route answers ``200`` in time."""
        try:
            request = urllib.request.Request(self.url, method="GET")
            with OPENER.open(request, timeout=self.timeout) as answer:
                answer.read(256)
                return int(answer.status) == 200
        except (urllib.error.URLError, OSError, ValueError):
            return False
