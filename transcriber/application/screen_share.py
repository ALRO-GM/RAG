"""The relay of «Compartir pantalla» (#278), kept in memory only.

A shared session has one *sharing* device (the host: the only page that
runs the application, the source of truth) and a few *joined* devices.
The relay never looks inside what it passes on and never keeps it:

* ``{"type": "m", ...}`` from the host (its page, its changes, its
  pointer, its sounds) goes to every joined device, or to the one named
  in ``to``. Each device has an :class:`Outbox`; a device that cannot
  keep up loses the changes waiting for it (never the host's time) and
  the host is asked for its whole page again (``need_snapshot``), so
  the latest state always arrives and nothing drifts.
* ``{"type": "in", "body"}`` from a joined device (a click, a key, a
  value, a scroll, its pointer) goes to the host *only if that device
  may send actions*; otherwise it is dropped here, whatever the page
  did. The host applies actions in the order they reach it.
* The host sets who sends actions, where the sound plays, removes a
  device or ends the session; any device can set itself to only watch
  or leave.

Every operation runs on one thread (the server's event loop); the clock,
the random source and the two attempt limiters are injected. What is
logged is counts and reasons, never a name, a password or a message.
"""

import hashlib
import json
import logging
import secrets
import time
from collections import deque
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
)

from transcriber.domain.errors import TranscriberError
from transcriber.domain.screen_share import (
    HOST_BURST,
    HOST_MESSAGE_BYTES,
    HOST_RATE,
    IDLE_SECONDS,
    MAX_LIFETIME_SECONDS,
    MAX_SESSIONS,
    MAX_VIEWERS,
    MAX_WRONG_PINS,
    OUTBOX_BYTES,
    PIN_HASH_ITERATIONS,
    RESUME_SECONDS,
    VIEWER_BURST,
    VIEWER_MESSAGE_BYTES,
    VIEWER_RATE,
    TokenBucket,
    check_pin,
    clean_label,
    hash_pin,
    normalise_name,
    pin_matches,
)

LOG = logging.getLogger(__name__)

CLOSE_REPLACED = 4409
"""A newer connection of the same device took its place."""

CLOSE_ENDED = 4410
"""The shared session ended, or this device was removed or left."""


class AttemptLimiter(Protocol):
    """Counts wrong passwords per key and says how long a key must wait.

    :class:`~transcriber.adapters.inbound.login_throttle.LoginThrottle`
    is one.
    """

    def retry_after(self, key: str) -> Tuple[float, bool]:
        """Return ``(seconds to wait, whether every key waits)``."""

    def failed(self, key: str) -> bool:
        """Count one wrong password of ``key``."""

    def succeeded(self, key: str) -> None:
        """Forget the failures of ``key``."""


class ShareRefused(TranscriberError):
    """A share, join or resume was refused; ``code`` says why.

    Codes: ``invalid_input``, ``no_match`` (no such name or a wrong
    password: the same answer on purpose), ``too_many_attempts`` (with
    ``retry_after`` seconds), ``full``, ``busy`` (too many sessions on
    the server), ``name_taken`` (a session open with that name),
    ``ended`` (nothing to resume).
    """

    def __init__(
        self, code: str, message: str, retry_after: float = 0.0
    ) -> None:
        """Keep the code and, for a lock, the wait."""
        super().__init__(message)
        self.code = code
        self.retry_after = retry_after


class Outbox:
    """Messages waiting to be written to one device's connection.

    Droppable messages (changes of the page) are bounded by ``limit``
    bytes; when one more would not fit they are all forgotten and
    :meth:`put` says so, so the relay asks for the whole page again.
    Control messages (who is connected, the end) are always kept.
    ``notify`` is called when there is something to write.
    """

    def __init__(
        self, limit: int = OUTBOX_BYTES,
        notify: Callable[[], None] = lambda: None,
    ) -> None:
        """An empty outbox."""
        self.limit = limit
        self.notify = notify
        self.items: Deque[Tuple[str, bool]] = deque()
        self.size = 0
        self.closed: Optional[Tuple[int, str]] = None

    def put(
        self, text: str, droppable: bool = False, fresh: bool = False
    ) -> bool:
        """Queue ``text``; ``fresh`` forgets the droppable ones first.

        Returns False when droppable messages had to be forgotten.
        """
        if self.closed is not None:
            return True
        kept = True
        if fresh:
            self._forget()
        elif droppable and self.size + len(text) > self.limit:
            self._forget()
            kept = False
        if not kept:
            self.notify()
            return False
        self.items.append((text, droppable))
        if droppable:
            self.size += len(text)
        self.notify()
        return True

    def _forget(self) -> None:
        """Drop every droppable message still waiting."""
        self.items = deque(item for item in self.items if not item[1])
        self.size = 0

    def take(self) -> List[str]:
        """Return everything waiting and empty the outbox."""
        texts = [text for text, _ in self.items]
        self.items.clear()
        self.size = 0
        return texts

    def close(self, code: int, reason: str) -> None:
        """Ask the writer to close the connection after what is queued."""
        if self.closed is None:
            self.closed = (code, reason)
        self.notify()


@dataclass(eq=False)
class Device:
    """One connected (or briefly away) device of a shared session."""

    id: str
    share: "Share"
    label: str
    host: bool
    sends: bool
    token_hash: bytes
    bucket: TokenBucket
    outbox: Optional[Outbox] = None
    away_since: Optional[float] = None
    stale: bool = True
    asked: bool = False
    dropped: int = 0


@dataclass(eq=False)
class Share:
    """One shared session: its name, the password's hash, its devices."""

    id: str
    key: str
    name: str
    salt: bytes
    digest: bytes
    created: float
    active: float
    devices: Dict[str, Device] = field(default_factory=dict)
    host_id: str = ""
    sound: str = ""
    wrong: int = 0
    ended: bool = False

    @property
    def host(self) -> Device:
        """The sharing device."""
        return self.devices[self.host_id]

    def viewers(self) -> List[Device]:
        """The joined devices, in the order they joined."""
        return [d for d in self.devices.values() if not d.host]


def _digest(token: str) -> bytes:
    """The key a resume token is kept under (the token is not kept)."""
    return hashlib.sha256(token.encode("utf-8")).digest()


def _dumps(message: Dict[str, Any]) -> str:
    """One message as compact JSON."""
    return json.dumps(message, separators=(",", ":"), ensure_ascii=False)


class ShareRelay:
    """The shared sessions of this server and the routing between devices.

    ``clients`` limits wrong passwords per client address, ``names`` per
    session name (counted for names that do not exist too).
    """

    def __init__(
        self,
        clients: AttemptLimiter,
        names: AttemptLimiter,
        now: Callable[[], float] = time.monotonic,
        new_token: Callable[[], str] = lambda: secrets.token_urlsafe(32),
        new_id: Callable[[], str] = lambda: secrets.token_hex(6),
        new_salt: Callable[[], bytes] = lambda: secrets.token_bytes(16),
        iterations: int = PIN_HASH_ITERATIONS,
    ) -> None:
        """No shared session yet."""
        self._clients = clients
        self._names = names
        self._now = now
        self._new_token = new_token
        self._new_id = new_id
        self._new_salt = new_salt
        self._iterations = iterations
        self._shares: Dict[str, Share] = {}
        self._tokens: Dict[bytes, Device] = {}
        self.dropped_inputs = 0

    # ---- opening ----------------------------------------------------

    def secret(self, pin: object) -> Tuple[bytes, bytes]:
        """A new salt and the password's hash (slow: run it off the loop)."""
        code = check_pin(pin)
        salt = self._new_salt()
        return salt, hash_pin(code, salt, self._iterations)

    def host(
        self, name: object, pin: object, label: object,
        secret: Optional[Tuple[bytes, bytes]] = None,
    ) -> Tuple[Device, str]:
        """Open a shared session; return the host device and its token.

        Names are unique among open sessions (``name_taken``): a second
        session with the same name could not be told apart by a join.
        ``secret`` is :meth:`secret`'s answer when it was computed off
        the event loop.
        """
        key = normalise_name(name)
        check_pin(pin)
        self.sweep()
        if any(s.key == key for s in self._shares.values()):
            raise ShareRefused("name_taken", "a shared session has that name")
        if len(self._shares) >= MAX_SESSIONS:
            raise ShareRefused("busy", "too many shared sessions open")
        now = self._now()
        salt, digest = secret if secret is not None else self.secret(pin)
        share = Share(
            id=self._new_id(), key=key, name=" ".join(str(name).split()),
            salt=salt, digest=digest, created=now, active=now,
        )
        device, token = self._add(share, label, host=True)
        share.host_id = share.sound = device.id
        self._shares[share.id] = share
        LOG.info("Shared session opened (%d open)", len(self._shares))
        return device, token

    def locked(self, name: object, client: str) -> None:
        """Raise ``too_many_attempts`` while the client or the name waits."""
        key = normalise_name(name)
        for limiter, which in ((self._clients, client), (self._names, key)):
            wait, _ = limiter.retry_after(which)
            if wait > 0:
                raise ShareRefused(
                    "too_many_attempts", "too many wrong passwords", wait
                )

    def check(self, name: object, pin: object) -> Dict[str, bool]:
        """Which open sessions called ``name`` ``pin`` opens (slow: hashes).

        Only reads; meant to run off the event loop before :meth:`join`.
        For a name no session has, a hash is computed all the same.
        """
        key = normalise_name(name)
        code = check_pin(pin)
        shares = [s for s in list(self._shares.values()) if s.key == key]
        if not shares:
            hash_pin(code, b"\0" * 16, self._iterations)
        return {
            s.id: pin_matches(code, s.salt, s.digest, self._iterations)
            for s in shares
        }

    def join(
        self, name: object, pin: object, label: object, client: str,
        checked: Optional[Dict[str, bool]] = None,
    ) -> Tuple[Device, str]:
        """Join the session called ``name`` with ``pin``.

        Raises :class:`ShareRefused`: ``no_match`` alike for an unknown
        name and a wrong password, ``too_many_attempts`` while the
        client or the name is locked. ``checked`` is :meth:`check`'s
        answer when the hashing ran off the event loop. A right
        password clears the client's own failures, never the name's
        (one's own session must not reset the lock of another's name).
        """
        key = normalise_name(name)
        check_pin(pin)
        self.sweep()
        self.locked(name, client)
        if checked is None:
            checked = self.check(name, pin)
        candidates = [s for s in self._shares.values() if s.key == key]
        found: Optional[Share] = None
        for share in candidates:
            if checked.get(share.id):
                found = share
        if found is None:
            self._clients.failed(client)
            self._names.failed(key)
            for share in candidates:
                share.wrong += 1
                if share.wrong >= MAX_WRONG_PINS:
                    self._end(share, "too_many_attempts")
            LOG.info("Shared session: a wrong name or password")
            raise ShareRefused("no_match", "no shared session matches")
        self._clients.succeeded(client)
        if len(found.viewers()) >= MAX_VIEWERS:
            raise ShareRefused("full", "the shared session is full")
        device, token = self._add(found, label, host=False)
        LOG.info(
            "Shared session: a device joined (%d devices)",
            len(found.devices),
        )
        return device, token

    def resume(self, token: object) -> Device:
        """The device of ``token``, back after a dropped link."""
        self.sweep()
        device = (
            self._tokens.get(_digest(token))
            if isinstance(token, str) else None
        )
        if device is None or device.share.ended:
            raise ShareRefused("ended", "nothing to resume")
        return device

    def _add(
        self, share: Share, label: object, host: bool
    ) -> Tuple[Device, str]:
        """A new device of ``share`` and its resume token."""
        token = self._new_token()
        rate, burst = (
            (HOST_RATE, HOST_BURST) if host else (VIEWER_RATE, VIEWER_BURST)
        )
        number = len(share.devices) + 1
        device = Device(
            id=self._new_id(), share=share, host=host, sends=host,
            label=clean_label(label, f"#{number}"),
            token_hash=_digest(token), bucket=TokenBucket(rate, burst),
            away_since=self._now(), stale=not host,
        )
        share.devices[device.id] = device
        self._tokens[device.token_hash] = device
        return device, token

    # ---- connections ------------------------------------------------

    def attach(self, device: Device, outbox: Outbox) -> None:
        """Send ``device``'s messages to ``outbox`` from now on.

        An older connection of the device is closed (``replaced``).
        When the host comes back every joined device waits for its
        whole page again; a joined device that comes back asks for it.
        """
        share = device.share
        if share.ended:
            outbox.put(_dumps({"type": "ended", "reason": "ended"}))
            outbox.close(CLOSE_ENDED, "ended")
            return
        if device.outbox is not None:
            device.outbox.close(CLOSE_REPLACED, "replaced")
        device.outbox = outbox
        device.away_since = None
        if device.host:
            for viewer in share.viewers():
                viewer.stale, viewer.asked = True, False
        else:
            device.stale, device.asked = True, False
        self._roster(share)
        if device.host:
            for viewer in share.viewers():
                if viewer.outbox is not None:
                    self._ask(viewer)
        else:
            self._ask(device)

    def detach(self, device: Device, outbox: Outbox) -> None:
        """The connection of ``outbox`` closed; the device may come back."""
        if device.outbox is not outbox:
            return
        device.outbox = None
        device.away_since = self._now()
        if not device.share.ended:
            self._roster(device.share)

    # ---- messages ---------------------------------------------------

    def limit(self, device: Device) -> int:
        """The largest message ``device`` may send, in bytes."""
        return HOST_MESSAGE_BYTES if device.host else VIEWER_MESSAGE_BYTES

    def receive(self, device: Device, text: str) -> None:
        """Route one message of ``device`` (see the module's docstring)."""
        share = device.share
        if share.ended or device.share.devices.get(device.id) is not device:
            return
        if device.host:
            share.active = self._now()  # its ping too: a static slide lives
        if not device.bucket.take(self._now()):
            device.dropped += 1
            if device.host:
                # A change of its page may be lost: the devices catch up.
                for viewer in share.viewers():
                    if not viewer.stale:
                        viewer.stale, viewer.asked = True, False
                        self._ask(viewer)
            return
        try:
            message = json.loads(text)
        except ValueError:
            return
        if not isinstance(message, dict):
            return
        kind = message.get("type")
        if kind == "ping":
            self._send(device, {"type": "pong"})
        elif kind == "m" and device.host:
            self._from_host(share, message, text)
        elif kind == "in" and not device.host:
            self._from_viewer(device, message)
        elif kind == "resync" and not device.host:
            if device.stale and device.asked:
                return  # a snapshot is already on its way
            device.stale, device.asked = True, False
            self._ask(device)
        elif kind == "watch":
            device.sends = False
            self._roster(share)
        elif kind == "leave":
            if device.host:
                self._end(share, "stopped")
            else:
                self._drop(device, "left")
        elif device.host:
            self._host_command(share, kind, message)

    def _from_host(
        self, share: Share, message: Dict[str, Any], text: str
    ) -> None:
        """Pass a message of the host to its joined devices."""
        to = message.get("to")
        snapshot = message.get("snap") is True
        if to is None:
            targets = share.viewers()
        else:
            one = share.devices.get(to) if isinstance(to, str) else None
            targets = [one] if one is not None and not one.host else []
        for viewer in targets:
            if viewer.outbox is None:
                viewer.stale = True
            elif snapshot:
                viewer.outbox.put(text, droppable=True, fresh=True)
                viewer.stale = viewer.asked = False
            elif not viewer.stale and not viewer.outbox.put(
                text, droppable=True
            ):
                viewer.stale = True
                self._ask(viewer)

    def _from_viewer(self, device: Device, message: Dict[str, Any]) -> None:
        """Pass an action of a joined device to the host, if it may send."""
        share = device.share
        host = share.host
        if not device.sends or host.outbox is None:
            self.dropped_inputs += 1
            return
        share.active = self._now()
        host.outbox.put(_dumps({
            "type": "in", "from": device.id, "body": message.get("body"),
        }), droppable=True)

    def _host_command(
        self, share: Share, kind: object, message: Dict[str, Any]
    ) -> None:
        """Apply one of the host's settings."""
        target = message.get("device")
        device = (
            share.devices.get(target) if isinstance(target, str) else None
        )
        if kind == "set" and device is not None:
            device.sends = message.get("sends") is True
            self._roster(share)
        elif kind == "remove" and device is not None and not device.host:
            self._drop(device, "removed")
        elif kind == "sound":
            share.sound = device.id if device is not None else share.host_id
            self._roster(share)

    def _ask(self, viewer: Device) -> None:
        """Ask the host for the whole page for ``viewer`` (once)."""
        host = viewer.share.host
        if viewer.asked or host.outbox is None:
            return
        viewer.asked = True
        host.outbox.put(_dumps({"type": "need_snapshot", "device": viewer.id}))

    def _send(self, device: Device, message: Dict[str, Any]) -> None:
        """Queue a control message for ``device`` if it is connected."""
        if device.outbox is not None:
            device.outbox.put(_dumps(message))

    def _roster(self, share: Share) -> None:
        """Tell every device who is there, who sends, where the sound is."""
        message = {
            "type": "roster", "name": share.name, "host": share.host_id,
            "sound": share.sound,
            "devices": [
                {
                    "id": d.id, "label": d.label, "host": d.host,
                    "sends": d.sends, "online": d.outbox is not None,
                }
                for d in share.devices.values()
            ],
        }
        for device in share.devices.values():
            self._send(device, message)

    # ---- the end ----------------------------------------------------

    def _drop(self, device: Device, reason: str) -> None:
        """Take ``device`` out of its session (it must join again)."""
        share = device.share
        share.devices.pop(device.id, None)
        self._tokens.pop(device.token_hash, None)
        if device.outbox is not None:
            device.outbox.put(_dumps({"type": "ended", "reason": reason}))
            device.outbox.close(CLOSE_ENDED, reason)
            device.outbox = None
        if share.sound == device.id:
            share.sound = share.host_id
        LOG.info("Shared session: a device went (%s)", reason)
        self._roster(share)

    def _end(self, share: Share, reason: str) -> None:
        """End ``share``: every device is told and disconnected."""
        if share.ended:
            return
        share.ended = True
        self._shares.pop(share.id, None)
        for device in share.devices.values():
            self._tokens.pop(device.token_hash, None)
            if device.outbox is not None:
                device.outbox.put(_dumps({"type": "ended", "reason": reason}))
                device.outbox.close(CLOSE_ENDED, reason)
                device.outbox = None
        LOG.info(
            "Shared session ended (%s); %d open", reason, len(self._shares)
        )

    def sweep(self) -> int:
        """End what expired; return how many sessions are still open.

        A session ends after :data:`MAX_LIFETIME_SECONDS`, after
        :data:`IDLE_SECONDS` without any message of its host, or when its
        host has been away for :data:`RESUME_SECONDS`; a joined device
        away that long loses its place.
        """
        now = self._now()
        for share in list(self._shares.values()):
            host = share.host
            if now - share.created > MAX_LIFETIME_SECONDS or (
                now - share.active > IDLE_SECONDS
            ):
                self._end(share, "expired")
            elif host.away_since is not None and (
                now - host.away_since > RESUME_SECONDS
            ):
                self._end(share, "host_left")
            else:
                for viewer in share.viewers():
                    if viewer.away_since is not None and (
                        now - viewer.away_since > RESUME_SECONDS
                    ):
                        self._drop(viewer, "gone")
        return len(self._shares)

    def open_sessions(self) -> int:
        """How many shared sessions are open (for the logs and tests)."""
        return len(self._shares)
