"""Inbound adapter: live sessions over HTTP and WebSocket (LV-2, LV-7).

Routes (all need the access code, as the rest of the API):

``POST /api/live/sessions``
    Start a session. Body: :class:`~transcriber.domain.live_session.
    LiveSessionRequest` as JSON. Answers 201 with the session (its
    ``state``, ``device`` and ``speakers``) and the paths of its two
    sockets.
``GET /api/live/sessions`` and ``GET /api/live/sessions/{id}``
    The known sessions, or one.
``POST /api/live/sessions/{id}/stop``
    Stop the session and wait for its end; answers the session.
``POST /api/live/sessions/{id}/speakers/{speaker}``
    ``{"name"}``: name a speaker of a running session; the hand-off's
    ``transcript.v1`` lists the name (LV-16).
``POST /api/live/sessions/{id}/voice``
    ``{"action": "skip"}`` skips the employee's voice check of a
    two-party session; ``{"action": "mine" | "not_mine", "utterance":
    "t3.s0"}`` says that a sentence's voice is (not) the employee's,
    which corrects that voice from then on and its earlier sentences
    (#249). Answers the session.
``WS /api/live/{id}/audio``
    The browser's microphone: binary messages of signed 16-bit
    little-endian PCM at 16 kHz, channels interleaved (at most one
    second each). The text message ``{"type": "stop"}`` ends the
    session. The server first sends ``{"type": "ready", ...}`` with
    ``received_samples``, the samples (per channel) the session has
    taken so far: a client that reconnects resends from there (#263).
    About every :data:`ACK_SECONDS` it sends ``{"type": "ack",
    "samples": N}`` (everything before N is taken; the client forgets
    it). ``{"type": "gap", "samples": k}`` says the client lost ``k``
    samples (they become silence); ``{"type": "align", "held": n}``,
    from a page that reattached after a reload, fills the time no audio
    came for with silence, leaving room for the ``n`` samples it is
    about to send. A full audio queue stops the server reading
    (backpressure); a connected source is never refused for it. A newer
    socket takes the source over; the older one is closed with
    ``4409`` and the code ``replaced``.

A browser cannot set a header on a WebSocket, so a socket without the
``X-Access-Code`` header must send ``{"type": "auth", "access_code":
"..."}`` as its first message; wrong codes are throttled like every
other route. Refusals close the socket with ``4400`` (invalid input),
``4401`` (no valid code), ``4404`` (unknown session), ``4409`` (wrong
state) or ``4429`` (too many wrong codes), after an ``{"type":
"error", "code", "message"}`` message.
"""

import asyncio
import json
import logging
import re
import time
from typing import Any, Callable, Dict, Mapping, Optional

from fastapi import APIRouter, FastAPI, Request, WebSocket
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from starlette.websockets import WebSocketDisconnect, WebSocketState

from transcriber.adapters.inbound.access import ACCESS_HEADER
from transcriber.adapters.inbound.http_api import (
    ApiError,
    parse_json,
    read_body,
)
from transcriber.application.live_session import (
    LIFECYCLE,
    LiveSession,
    LiveSessions,
)
from transcriber.domain.errors import (
    InvalidInputError,
    NotFoundError,
    SourceReplacedError,
    StateError,
    TranscriberError,
)
from transcriber.domain.live import SAMPLE_RATE
from transcriber.domain.live_session import (
    MAX_MESSAGE_SECONDS,
    PCM_BYTES,
    SPEAKER_ID_PATTERN,
    LiveSessionRequest,
)

LOG = logging.getLogger(__name__)

AUTH_TIMEOUT = 5.0
"""Seconds a socket without the header has to send its code."""

Admit = Callable[[Optional[str], Optional[str], Mapping[str, str]], None]
"""Checks an access code: raises an ``ApiError`` 401 or 429 if wrong."""

CLOSE_CODES = {
    "invalid_input": 4400,
    "unauthorized": 4401,
    "not_found": 4404,
    "wrong_state": 4409,
    "busy": 4409,
    "replaced": 4409,
    "too_many_attempts": 4429,
}

ACK_SECONDS = 1.0
"""How often the audio socket acknowledges what it took (#263)."""


READY_TIMEOUT = 120.0
"""Seconds a socket waits for the recogniser before it says ready."""


def session_json(session: LiveSession) -> Dict[str, Any]:
    """Return a session as the API shows it, with its socket paths."""
    data = session.info()
    data["audio"] = f"/api/live/{session.id}/audio"
    data["events"] = f"/api/live/{session.id}/events"
    return data


def error_code(exc: TranscriberError) -> str:
    """Return the stable code of a domain error on a socket."""
    if isinstance(exc, NotFoundError):
        return "not_found"
    if isinstance(exc, SourceReplacedError):
        return "replaced"
    if isinstance(exc, InvalidInputError):
        return "invalid_input"
    if isinstance(exc, StateError):
        return "wrong_state"
    return "internal_error"


async def refuse(socket: WebSocket, code: str, message: str) -> None:
    """Tell the client why, then close the socket with ``code``'s number."""
    if socket.application_state == WebSocketState.CONNECTED:
        try:
            await socket.send_json(
                {"type": "error", "code": code, "message": message}
            )
            await socket.close(CLOSE_CODES.get(code, 1011))
        except (WebSocketDisconnect, RuntimeError):
            pass


async def authenticate(socket: WebSocket, admit: Admit) -> bool:
    """Accept the socket and check its access code; tell if it passed.

    The code comes from the ``X-Access-Code`` header or, without it,
    from a first text message ``{"type": "auth", "access_code": ...}``
    sent within :data:`AUTH_TIMEOUT` seconds.
    """
    await socket.accept()
    given = socket.headers.get(ACCESS_HEADER)
    if given is None:
        try:
            first = await asyncio.wait_for(
                socket.receive_text(), AUTH_TIMEOUT
            )
            message = json.loads(first)
            if isinstance(message, dict) and message.get("type") == "auth":
                code = message.get("access_code")
                given = code if isinstance(code, str) else None
        except (asyncio.TimeoutError, ValueError, KeyError):
            given = None
        except WebSocketDisconnect:
            return False
    peer = socket.client.host if socket.client is not None else None
    try:
        admit(given, peer, socket.headers)
    except ApiError as exc:
        await refuse(socket, exc.code, exc.message)
        return False
    return True


def add_live_routes(
    app: FastAPI, api: APIRouter, live: LiveSessions, admit: Admit,
    json_limit: int,
) -> None:
    """Add the live routes: HTTP ones to ``api``, sockets to ``app``."""

    @api.post("/live/sessions", status_code=201)
    async def start_live(request: Request) -> Dict[str, Any]:
        """Start a live session."""
        body = parse_json(
            LiveSessionRequest, await read_body(request, json_limit)
        )
        session = await run_in_threadpool(live.start, body)
        return session_json(session)

    @api.get("/live/sessions")
    async def list_live() -> Dict[str, Any]:
        """List the known live sessions."""
        return {"sessions": [session_json(s) for s in live.list()]}

    @api.get("/live/sessions/{session_id}")
    async def show_live(session_id: str) -> Dict[str, Any]:
        """Show one live session."""
        return session_json(live.get(session_id))

    @api.post("/live/sessions/{session_id}/stop")
    async def stop_live(session_id: str) -> Any:
        """Stop a live session and wait for its end."""
        session = await run_in_threadpool(live.stop, session_id)
        return JSONResponse(session_json(session))

    @api.post("/live/sessions/{session_id}/speakers/{speaker}")
    async def rename_live(
        session_id: str, speaker: str, request: Request
    ) -> Dict[str, Any]:
        """Name a speaker of a running live session."""
        try:
            data = json.loads(await read_body(request, json_limit) or b"{}")
        except ValueError as exc:
            raise InvalidInputError("the body is not JSON") from exc
        name = data.get("name") if isinstance(data, dict) else None
        if not isinstance(name, str) or not re.fullmatch(
            SPEAKER_ID_PATTERN, speaker
        ):
            raise InvalidInputError("a speaker id and a name are needed")
        session = live.rename(session_id, speaker, name)
        return session_json(session)

    @api.post("/live/sessions/{session_id}/voice")
    async def voice_live(session_id: str, request: Request) -> Dict[str, Any]:
        """Skip the voice check or correct a sentence's side (#249)."""
        try:
            data = json.loads(await read_body(request, json_limit) or b"{}")
        except ValueError as exc:
            raise InvalidInputError("the body is not JSON") from exc
        if not isinstance(data, dict):
            raise InvalidInputError("the body is a JSON object")
        action, utterance = data.get("action"), data.get("utterance")
        if not isinstance(action, str) or (
            utterance is not None and not isinstance(utterance, str)
        ):
            raise InvalidInputError("an action (and an utterance id)")
        session = live.voice(session_id, action, utterance)
        return session_json(session)

    @app.websocket("/api/live/{session_id}/audio")
    async def live_audio(socket: WebSocket, session_id: str) -> None:
        """Take the session's audio from the browser."""
        if not await authenticate(socket, admit):
            return
        try:
            session = live.get(session_id)
            source = session.claim_source()
            if source is None:
                raise StateError(f"live session {session_id} has ended")
        except TranscriberError as exc:
            await refuse(socket, error_code(exc), str(exc))
            return
        closed = "unknown"
        try:
            # The client starts sending on "ready": hold it back until
            # the recogniser is loaded, so no audio queues meanwhile.
            await run_in_threadpool(session.wait_ready, READY_TIMEOUT)
            await socket.send_json({
                "type": "ready", "session_id": session_id,
                "channels": session.channels, "sample_rate": SAMPLE_RATE,
                "format": "pcm_s16le",
                "max_message_bytes": int(
                    PCM_BYTES * session.channels * SAMPLE_RATE
                    * MAX_MESSAGE_SECONDS
                ),
                "received_samples": session.received,
            })
            closed = await _pump_audio(socket, session, source)
        except (WebSocketDisconnect, RuntimeError) as exc:
            closed = str(getattr(exc, "code", "") or type(exc).__name__)
        finally:
            session.release_source(
                source, int(closed) if closed.isdigit() else None
            )
            LIFECYCLE.info(
                "Live session %s: audio socket %d closed (%s)",
                session_id, source, closed,
            )

    @app.websocket("/api/live/{session_id}/events")
    async def live_events(socket: WebSocket, session_id: str) -> None:
        """Send the session's ``live.event.v1`` events (LV-7)."""
        await follow_events(socket, session_id, live, admit)

    @app.websocket("/live/{session_id}/events")
    async def live_events_alias(socket: WebSocket, session_id: str) -> None:
        """The same, at the path ``docs/platform/CONTRACTS.md`` names."""
        await follow_events(socket, session_id, live, admit)


async def _pump_audio(
    socket: WebSocket, session: LiveSession, source: int
) -> str:
    """Feed the socket's messages to ``session`` until stop or close.

    Each message is queued from the thread pool; while the session's
    queue is full the call blocks and the socket is not read. What was
    taken is acknowledged about every :data:`ACK_SECONDS`, also while
    the client sends nothing (its last audio must be acknowledged for
    it to stop, #263). Returns how the socket ended (its close code or
    the refusal), for the log.
    """
    last = _Acked(session.received, time.monotonic())
    acker = asyncio.create_task(_ack_while_quiet(socket, session, last))
    try:
        return await _read_audio(socket, session, source, last)
    finally:
        acker.cancel()


class _Acked:
    """The samples last acknowledged on a socket, and when."""

    def __init__(self, samples: int, at: float) -> None:
        """Keep both."""
        self.samples = samples
        self.at = at


async def _ack_while_quiet(
    socket: WebSocket, session: LiveSession, last: _Acked
) -> None:
    """Acknowledge new audio every :data:`ACK_SECONDS`, messages or not."""
    try:
        while True:
            await asyncio.sleep(max(ACK_SECONDS, 0.05))
            await _maybe_ack(socket, session, last)
    except (WebSocketDisconnect, RuntimeError):
        return


async def _read_audio(
    socket: WebSocket, session: LiveSession, source: int, last: _Acked
) -> str:
    """The loop of :func:`_pump_audio`."""
    while True:
        try:
            message = await socket.receive()
        except WebSocketDisconnect as exc:
            return str(exc.code)
        if message["type"] == "websocket.disconnect":
            return str(message.get("code", "disconnect"))
        data = message.get("bytes")
        text = message.get("text")
        try:
            if data is not None:
                await run_in_threadpool(session.feed, data, source)
            elif text is not None:
                command = _command(text)
                kind = command.get("type")
                if kind == "stop":
                    await run_in_threadpool(session.stop)
                    await socket.send_json({"type": "stopped"})
                    await socket.close(1000)
                    return "stop"
                if kind == "gap":
                    samples = await run_in_threadpool(
                        session.silence, command.get("samples"), source
                    )
                    LIFECYCLE.info(
                        "Live session %s: the client lost %.1f s of audio;"
                        " silence put in", session.id, samples / SAMPLE_RATE,
                    )
                elif kind == "align":
                    samples = await run_in_threadpool(
                        session.align, command.get("held", 0), source
                    )
                    LIFECYCLE.info(
                        "Live session %s: a reattached page; %.1f s of"
                        " silence put in", session.id, samples / SAMPLE_RATE,
                    )
        except TranscriberError as exc:
            await refuse(socket, error_code(exc), str(exc))
            return error_code(exc)
        await _maybe_ack(socket, session, last)


async def _maybe_ack(
    socket: WebSocket, session: LiveSession, last: _Acked
) -> None:
    """Send ``{"type": "ack"}`` when new audio was taken and it is time.

    ``last`` is the samples last acknowledged and when; it is updated.
    """
    now = time.monotonic()
    taken = session.received
    if taken == last.samples or now - last.at < ACK_SECONDS:
        return
    last.samples, last.at = taken, now
    await socket.send_json({"type": "ack", "samples": taken})


def _command(text: str) -> Dict[str, Any]:
    """Return a text message as a JSON object (empty when it is not)."""
    try:
        message = json.loads(text)
    except ValueError:
        return {}
    return message if isinstance(message, dict) else {}


EVENT_POLL_SECONDS = 1.0
"""How long one wait for new events lasts (the socket is checked between)."""


async def follow_events(
    socket: WebSocket, session_id: str, live: LiveSessions, admit: Admit,
) -> None:
    """Send every event of the session from ``?from_seq=`` on, then close.

    Events already emitted are sent first (a reader may join late); the
    socket closes with 1000 once the session's last event is sent.
    """
    if not await authenticate(socket, admit):
        return
    try:
        session = live.get(session_id)
        start = int(socket.query_params.get("from_seq", "0"))
        if start < 0:
            raise InvalidInputError("from_seq must be 0 or more")
    except ValueError:
        await refuse(socket, "invalid_input", "from_seq must be an integer")
        return
    except TranscriberError as exc:
        await refuse(socket, error_code(exc), str(exc))
        return
    position = start
    try:
        while True:
            events, closed = await run_in_threadpool(
                session.hub.read, position, EVENT_POLL_SECONDS
            )
            for event in events:
                await socket.send_text(json.dumps(event, ensure_ascii=False))
            position += len(events)
            if closed and not events:
                await socket.close(1000)
                return
    except (WebSocketDisconnect, RuntimeError):
        return
