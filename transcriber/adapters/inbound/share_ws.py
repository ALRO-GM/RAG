"""Inbound adapter: the relay of «Compartir pantalla» over a WebSocket (#278).

``WS /api/share`` — behind the access code like every other route (a
device must already be inside the application): a browser sends
``{"type": "auth", "access_code"}`` first, as on the live sockets. The
next message, within :data:`HELLO_TIMEOUT` seconds, is one of

``{"type": "host", "name", "pin", "label"}``
    share this device's page; ``pin`` is four digits.
``{"type": "join", "name", "pin", "label"}``
    join a shared session.
``{"type": "resume", "token"}``
    come back after a dropped link, without the password (within
    :data:`~transcriber.domain.screen_share.RESUME_SECONDS`).

The server answers ``{"type": "welcome", "device", "host", "token",
"limits"}`` and then routes messages as
:class:`~transcriber.application.screen_share.ShareRelay` says; the
message types are in ``docs/web/SCREEN_SHARE.md``. Refusals send
``{"type": "error", "code", "message", "retry_after"}`` and close with
``4400`` (invalid input), ``4401`` (no valid access code), ``4403``
(full or too many sessions), ``4404`` (no session with that name and
password: the same for both), ``4408`` (no hello in time), ``4410``
(the session ended, or this device was removed), ``4413`` (message
too large) or ``4429`` (too many wrong passwords). ``4409`` closes a
connection that a newer one of the same device replaced.

Nothing that passes through is logged or stored.
"""

import asyncio
import json
import logging
from typing import Any, Callable, Dict, Optional, Tuple

from fastapi import FastAPI, WebSocket
from starlette.websockets import WebSocketDisconnect, WebSocketState

from transcriber.adapters.inbound.live_ws import Admit, authenticate
from transcriber.adapters.inbound.login_throttle import LoginThrottle
from transcriber.application.screen_share import (
    Device,
    Outbox,
    ShareRefused,
    ShareRelay,
)
from transcriber.domain.errors import InvalidInputError
from transcriber.domain.screen_share import (
    HOST_MESSAGE_BYTES,
    MAX_VIEWERS,
    PIN_DIGITS,
    RESUME_SECONDS,
    VIEWER_MESSAGE_BYTES,
)

LOG = logging.getLogger(__name__)

HELLO_TIMEOUT = 10.0
"""Seconds a connection has, after its code, to say what it wants."""

SWEEP_SECONDS = 5.0
"""How often expired sessions and devices are looked for."""

CLOSE_CODES = {
    "invalid_input": 4400,
    "unauthorized": 4401,
    "full": 4403,
    "busy": 4403,
    "no_match": 4404,
    "timeout": 4408,
    "ended": 4410,
    "too_large": 4413,
    "too_many_attempts": 4429,
}

ClientKey = Callable[[WebSocket], str]
"""The key wrong passwords are counted under for one connection."""

LIMITS = {
    "pin_digits": PIN_DIGITS, "max_viewers": MAX_VIEWERS,
    "resume_seconds": RESUME_SECONDS,
    "host_message_bytes": HOST_MESSAGE_BYTES,
    "viewer_message_bytes": VIEWER_MESSAGE_BYTES,
}


def share_relay(now: Callable[[], float]) -> ShareRelay:
    """A relay whose wrong passwords are limited per client and per name.

    Per client and per name alike: 5 wrong passwords within 15 minutes
    lock the key for 1 minute, doubled at every further lock up to an
    hour. All clients together: 30 wrong passwords within an hour lock
    every join for 15 minutes.
    """
    clients = LoginThrottle(
        per_client_limit=5, window=900.0, base_cooldown=60.0,
        max_cooldown=3600.0, global_limit=30, global_window=3600.0,
        global_cooldown=900.0, now=now,
    )
    names = LoginThrottle(
        per_client_limit=5, window=900.0, base_cooldown=60.0,
        max_cooldown=3600.0, global_limit=10**9, now=now,
    )
    return ShareRelay(clients, names, now=now)


async def refuse_share(
    socket: WebSocket, code: str, message: str, retry_after: float = 0.0
) -> None:
    """Say why, then close with ``code``'s number."""
    if socket.application_state != WebSocketState.CONNECTED:
        return
    body: Dict[str, Any] = {"type": "error", "code": code, "message": message}
    if retry_after > 0:
        body["retry_after"] = int(retry_after + 0.999)
    try:
        await socket.send_json(body)
        await socket.close(CLOSE_CODES.get(code, 1011))
    except (WebSocketDisconnect, RuntimeError):
        pass


async def _hello(
    socket: WebSocket, relay: ShareRelay, client: str
) -> Optional[Tuple[Device, Optional[str]]]:
    """Read the first message; return the device (and a new token)."""
    try:
        text = await asyncio.wait_for(socket.receive_text(), HELLO_TIMEOUT)
    except asyncio.TimeoutError:
        await refuse_share(socket, "timeout", "say host, join or resume")
        return None
    except (WebSocketDisconnect, RuntimeError, KeyError):
        return None
    try:
        message = json.loads(text) if len(text) <= VIEWER_MESSAGE_BYTES else 0
    except ValueError:
        message = None
    if not isinstance(message, dict):
        await refuse_share(socket, "invalid_input", "a JSON object")
        return None
    kind = message.get("type")
    try:
        if kind == "host":
            device, token = relay.host(
                message.get("name"), message.get("pin"), message.get("label")
            )
            return device, token
        if kind == "join":
            device, token = relay.join(
                message.get("name"), message.get("pin"),
                message.get("label"), client,
            )
            return device, token
        if kind == "resume":
            return relay.resume(message.get("token")), None
        raise InvalidInputError("say host, join or resume")
    except ShareRefused as exc:
        await refuse_share(socket, exc.code, str(exc), exc.retry_after)
    except InvalidInputError as exc:
        await refuse_share(socket, "invalid_input", str(exc))
    return None


def add_share_routes(
    app: FastAPI, relay: ShareRelay, admit: Admit, client_of: ClientKey
) -> None:
    """Add ``WS /api/share`` to ``app``."""
    sweeper: Dict[str, "asyncio.Task[None]"] = {}

    async def sweep_while_open() -> None:
        """End expired sessions and devices while any session is open."""
        while relay.sweep() > 0:
            await asyncio.sleep(SWEEP_SECONDS)

    def keep_sweeping() -> None:
        """Start the sweeper on this event loop unless it runs already."""
        task = sweeper.get("task")
        loop = asyncio.get_running_loop()
        if task is None or task.done() or task.get_loop() is not loop:
            sweeper["task"] = loop.create_task(sweep_while_open())

    @app.websocket("/api/share")
    async def share(socket: WebSocket) -> None:
        """Share this device's page, or join one (#278)."""
        if not await authenticate(socket, admit):
            return
        hello = await _hello(socket, relay, client_of(socket))
        if hello is None:
            return
        device, token = hello
        await _serve(socket, relay, device, token, keep_sweeping)


async def _serve(
    socket: WebSocket, relay: ShareRelay, device: Device,
    token: Optional[str], keep_sweeping: Callable[[], None],
) -> None:
    """Write the device's outbox and route what it sends, until closed."""
    loop = asyncio.get_running_loop()
    ready = asyncio.Event()

    def notify() -> None:
        """Wake the writer (from any thread: tests run several loops)."""
        loop.call_soon_threadsafe(ready.set)

    outbox = Outbox(notify=notify)
    welcome: Dict[str, Any] = {
        "type": "welcome", "device": device.id, "host": device.host,
        "limits": LIMITS,
    }
    if token is not None:
        welcome["token"] = token
    try:
        await socket.send_json(welcome)
    except (WebSocketDisconnect, RuntimeError):
        return
    relay.attach(device, outbox)
    keep_sweeping()
    writer = asyncio.create_task(_write(socket, outbox, ready))
    try:
        await _read(socket, relay, device)
    finally:
        writer.cancel()
        relay.detach(device, outbox)


async def _write(
    socket: WebSocket, outbox: Outbox, ready: asyncio.Event
) -> None:
    """Send what the outbox holds; close when it says so."""
    try:
        while True:
            await ready.wait()
            ready.clear()
            for text in outbox.take():
                await socket.send_text(text)
            if outbox.closed is not None:
                for text in outbox.take():
                    await socket.send_text(text)
                await socket.close(outbox.closed[0])
                return
    except (WebSocketDisconnect, RuntimeError):
        return


async def _read(socket: WebSocket, relay: ShareRelay, device: Device) -> None:
    """Route every message of the device to the relay."""
    limit = relay.limit(device)
    while True:
        try:
            message = await socket.receive()
        except (WebSocketDisconnect, RuntimeError):
            return
        if message["type"] == "websocket.disconnect":
            return
        text = message.get("text")
        if text is None:
            await refuse_share(socket, "invalid_input", "text messages only")
            return
        if len(text) > limit or (
            len(text) * 4 > limit and len(text.encode("utf-8")) > limit
        ):
            await refuse_share(socket, "too_large", f"at most {limit} bytes")
            return
        relay.receive(device, text)
