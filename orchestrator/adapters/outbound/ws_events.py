"""The ``live.event.v1`` stream of the transcriber, over WebSocket.

``WS /api/live/{id}/events`` (``docs/live/LIVE_SESSION.md``): the
server sends every event as one JSON text message, past ones first, and
closes with 1000 after the last. The access code goes in the
``X-Access-Code`` header. The socket is read with a short timeout so a
stop asked by the orchestrator is noticed within ``poll`` seconds. A
stream can start at any event (``?from_seq=``): the orchestrator resumes
a dropped one there (:mod:`orchestrator.application.event_follow`).
"""

import json
from typing import Any, Callable, Iterator, Optional

from websockets.exceptions import ConnectionClosed, WebSocketException
from websockets.sync.client import connect

from orchestrator.adapters.outbound.http_transcriber import ACCESS_HEADER
from orchestrator.application.ports import (
    CancelToken,
    Document,
    TranscriberUnavailable,
)

Connect = Callable[..., Any]


class WsEvents:
    """Follow a live session's events.

    Args:
        base_url: Root of the transcriber (``http://127.0.0.1:8770``);
            ``http`` becomes ``ws``.
        code: The access code.
        poll: Seconds between looks at the stop flag.
        opener: ``websockets.sync.client.connect`` (replaced in tests).
    """

    def __init__(
        self,
        base_url: str,
        code: str,
        poll: float = 0.5,
        opener: Optional[Connect] = None,
    ):
        """Keep the address and the code."""
        root = base_url.rstrip("/")
        if root.startswith("http"):
            root = "ws" + root[len("http"):]
        self.root = root
        self.code = code
        self.poll = poll
        self.opener: Connect = opener or connect

    def url(self, session_id: str) -> str:
        """Address of one session's stream."""
        return f"{self.root}/api/live/{session_id}/events"

    def follow(
        self, session_id: str, stop: CancelToken, from_seq: int = 0,
    ) -> Iterator[Document]:
        """Yield the events from ``from_seq`` until the stream closes or
        ``stop`` is set."""
        url = self.url(session_id)
        if from_seq > 0:
            url += f"?from_seq={int(from_seq)}"
        try:
            socket = self.opener(
                url,
                additional_headers={ACCESS_HEADER: self.code},
                open_timeout=10,
            )
        except (OSError, WebSocketException, TimeoutError) as exc:
            raise TranscriberUnavailable(type(exc).__name__) from exc
        with socket:
            while not stop.cancelled:
                try:
                    text = socket.recv(timeout=self.poll)
                except TimeoutError:
                    continue
                except ConnectionClosed as exc:
                    if exc.rcvd is None:
                        # No close frame: a reset, a ping timeout (1006).
                        # The session goes on; the stream is resumed
                        # (#264 review), not taken for its end.
                        raise TranscriberUnavailable(
                            "closed abnormally (no close frame)"
                        ) from exc
                    if exc.rcvd.code != 1000:
                        raise TranscriberUnavailable(
                            f"closed {exc.rcvd.code}: {exc.rcvd.reason}"
                        ) from exc
                    return
                try:
                    message = json.loads(text)
                except ValueError:
                    continue
                if isinstance(message, dict):
                    if message.get("type") == "error":
                        raise TranscriberUnavailable(
                            str(message.get("code") or "error")
                        )
                    yield message
