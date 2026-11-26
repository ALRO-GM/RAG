"""«Compartir pantalla» (#278): the rules and the in-memory relay.

The relay runs with a fake clock, fake attempt limiters and a cheap
password hash; every routing rule is checked on the outboxes it fills.
Invented names and passwords only.
"""

import json
import logging
from typing import Any, Dict, List, Tuple

import pytest

from transcriber.application.screen_share import (
    CLOSE_ENDED,
    CLOSE_REPLACED,
    Device,
    Outbox,
    ShareRefused,
    ShareRelay,
)
from transcriber.domain import screen_share as rules
from transcriber.domain.errors import InvalidInputError
from transcriber.domain.screen_share import (
    TokenBucket,
    check_pin,
    clean_label,
    hash_pin,
    normalise_name,
    pin_matches,
)


class Clock:
    """A clock the test moves."""

    def __init__(self) -> None:
        """Start at 1000 s."""
        self.now = 1000.0

    def __call__(self) -> float:
        """The current time."""
        return self.now


class Limiter:
    """Locks a key after ``limit`` failures until it succeeds."""

    def __init__(self, limit: int = 5) -> None:
        """No failures yet."""
        self.limit = limit
        self.failures: Dict[str, int] = {}

    def retry_after(self, key: str) -> Tuple[float, bool]:
        """60 s while locked."""
        return (60.0 if self.failures.get(key, 0) >= self.limit else 0.0,
                False)

    def failed(self, key: str) -> bool:
        """Count one."""
        self.failures[key] = self.failures.get(key, 0) + 1
        return False

    def succeeded(self, key: str) -> None:
        """Forget."""
        self.failures.pop(key, None)


Relay = Tuple[ShareRelay, Limiter, Limiter]


def relay_with(clock: Clock, limit: int = 5) -> Relay:
    """A relay with fake limiters, a fake clock and a one-round hash."""
    clients, names = Limiter(limit), Limiter(limit)
    counter = iter(range(10**6))
    relay = ShareRelay(
        clients, names, now=clock,
        new_token=lambda: f"token-{next(counter)}",
        new_id=lambda: f"d{next(counter)}",
        new_salt=lambda: b"salt-salt-salt-1", iterations=1,
    )
    return relay, clients, names


def messages(outbox: Outbox) -> List[Dict[str, Any]]:
    """What waits in ``outbox``, decoded, emptied."""
    return [json.loads(text) for text in outbox.take()]


def types(outbox: Outbox) -> List[str]:
    """The types of what waits in ``outbox``, emptied."""
    return [m["type"] for m in messages(outbox)]


def connect(relay: ShareRelay, device: Device) -> Outbox:
    """Attach a fresh outbox to ``device``."""
    outbox = Outbox()
    relay.attach(device, outbox)
    return outbox


def send(relay: ShareRelay, sender: Device, **message: Any) -> None:
    """Send one message of ``sender``."""
    relay.receive(sender, json.dumps(message))


@pytest.fixture()
def clock() -> Clock:
    """A fresh clock."""
    return Clock()


# ---- the rules ------------------------------------------------------------


def test_names_are_compared_trimmed_and_without_case() -> None:
    assert normalise_name("  Sala   Uno ") == "sala uno"
    assert normalise_name("Presentación (1)") == "presentación (1)"
    for bad in ["", "   ", "x" * 41, "<script>", "a/b", 3]:
        with pytest.raises(InvalidInputError):
            normalise_name(bad)


def test_the_password_is_four_digits() -> None:
    assert check_pin("0420") == "0420"
    for bad in ["123", "12345", "12a4", " 1234", "١٢٣٤", 1234, None]:
        with pytest.raises(InvalidInputError):
            check_pin(bad)


def test_a_device_label_is_made_safe_and_short() -> None:
    assert clean_label("Tablet\x00 de\n prueba", "#1") == "Tablet de prueba"
    assert clean_label("x" * 99, "#1") == "x" * rules.LABEL_MAX
    assert clean_label("   ", "#2") == "#2"
    assert clean_label(None, "#3") == "#3"


def test_the_password_hash_is_salted_and_checked() -> None:
    first = hash_pin("1234", b"a" * 16, 10)
    assert first != hash_pin("1234", b"b" * 16, 10)
    assert b"1234" not in first
    assert pin_matches("1234", b"a" * 16, first, 10)
    assert not pin_matches("1235", b"a" * 16, first, 10)


def test_the_token_bucket_allows_a_burst_then_the_rate() -> None:
    bucket = TokenBucket(rate=2.0, burst=3.0)
    assert [bucket.take(0.0) for _ in range(4)] == [True, True, True, False]
    assert bucket.take(0.5)
    assert not bucket.take(0.5)
    assert bucket.take(10.0)


def test_the_outbox_forgets_changes_that_do_not_fit_but_never_control() -> (
    None
):
    woken: List[int] = []
    outbox = Outbox(limit=10, notify=lambda: woken.append(1))
    assert outbox.put("roster")
    assert outbox.put("123456", droppable=True)
    assert not outbox.put("1234567", droppable=True)
    assert outbox.take() == ["roster"]
    assert outbox.put("abc", droppable=True)
    assert outbox.put("whole", droppable=True, fresh=True)
    assert outbox.take() == ["whole"]
    outbox.close(4410, "ended")
    outbox.close(4409, "later")
    assert outbox.closed == (4410, "ended")
    assert outbox.put("after")
    assert outbox.take() == []
    assert woken


# ---- opening and joining --------------------------------------------------


def test_a_device_shares_and_another_joins_with_name_and_password(
    clock: Clock,
) -> None:
    relay, _, _ = relay_with(clock)
    host, host_token = relay.host("Sala Uno", "1234", "Tablet")
    assert host.host and host.sends
    host_box = connect(relay, host)
    viewer, token = relay.join("sala  uno", "1234", "Proyector", "ip-a")
    assert not viewer.host and not viewer.sends
    assert token != host_token
    viewer_box = connect(relay, viewer)
    roster = [m for m in messages(viewer_box) if m["type"] == "roster"][-1]
    assert roster["name"] == "Sala Uno"
    assert [(d["label"], d["host"], d["sends"], d["online"])
            for d in roster["devices"]] == [
        ("Tablet", True, True, True), ("Proyector", False, False, True)]
    assert roster["sound"] == host.id
    assert {"type": "need_snapshot", "device": viewer.id} in messages(
        host_box)


def test_unknown_name_and_wrong_password_get_the_same_answer(
    clock: Clock,
) -> None:
    relay, clients, names = relay_with(clock)
    relay.host("Sala", "1234", "Tablet")
    answers = []
    for name, pin in [("Otra", "1234"), ("Sala", "9999")]:
        with pytest.raises(ShareRefused) as caught:
            relay.join(name, pin, "x", "ip-a")
        answers.append((caught.value.code, str(caught.value)))
    assert answers[0] == answers[1] == (
        "no_match", "no shared session matches")
    assert clients.failures == {"ip-a": 2}
    assert names.failures == {"otra": 1, "sala": 1}


def test_wrong_passwords_lock_the_client_and_the_name(clock: Clock) -> None:
    relay, _, _ = relay_with(clock, limit=3)
    relay.host("Sala", "1234", "Tablet")
    for pin in ["0000", "0001", "0002"]:
        with pytest.raises(ShareRefused):
            relay.join("Sala", pin, "x", "ip-a")
    with pytest.raises(ShareRefused) as locked:
        relay.join("Sala", "1234", "x", "ip-a")
    assert locked.value.code == "too_many_attempts"
    assert locked.value.retry_after == 60.0
    # Another client is locked out of that name too.
    with pytest.raises(ShareRefused) as by_name:
        relay.join("Sala", "1234", "x", "ip-b")
    assert by_name.value.code == "too_many_attempts"


def test_a_session_that_takes_too_many_wrong_passwords_ends(
    clock: Clock,
) -> None:
    relay, _, _ = relay_with(clock, limit=10**6)
    host, _ = relay.host("Sala", "1234", "Tablet")
    box = connect(relay, host)
    messages(box)
    for i in range(rules.MAX_WRONG_PINS):
        with pytest.raises(ShareRefused):
            relay.join("Sala", f"{5000 + i:04d}", "x", f"ip-{i}")
    assert relay.open_sessions() == 0
    assert messages(box)[-1] == {
        "type": "ended", "reason": "too_many_attempts"}
    assert box.closed == (CLOSE_ENDED, "too_many_attempts")
    with pytest.raises(ShareRefused) as caught:
        relay.join("Sala", "1234", "x", "ip-z")
    assert caught.value.code == "no_match"


def test_a_session_is_full_after_the_maximum_of_devices(clock: Clock) -> None:
    relay, _, _ = relay_with(clock)
    relay.host("Sala", "1234", "Tablet")
    for i in range(rules.MAX_VIEWERS):
        relay.join("Sala", "1234", f"d{i}", "ip-a")
    with pytest.raises(ShareRefused) as caught:
        relay.join("Sala", "1234", "one more", "ip-a")
    assert caught.value.code == "full"


def test_the_server_holds_a_bounded_number_of_sessions(clock: Clock) -> None:
    relay, _, _ = relay_with(clock)
    for i in range(rules.MAX_SESSIONS):
        relay.host(f"Sala {i}", "1234", "x")
    with pytest.raises(ShareRefused) as caught:
        relay.host("Sala extra", "1234", "x")
    assert caught.value.code == "busy"


def test_nothing_of_the_password_or_the_name_is_kept_in_clear_or_logged(
    clock: Clock, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    relay, _, _ = relay_with(clock)
    host, token = relay.host("Sala Secreta", "4321", "Tablet")
    connect(relay, host)
    viewer, _ = relay.join("Sala Secreta", "4321", "Proyector", "ip-a")
    connect(relay, viewer)
    send(relay, host, type="m", body={"k": "d", "text": "Ana Ruiz"})
    with pytest.raises(ShareRefused):
        relay.join("Sala Secreta", "1111", "x", "ip-a")
    share = host.share
    kept = repr(vars(share)) + repr(relay._tokens)
    assert "4321" not in kept and token not in kept
    logged = caplog.text
    for secret in ["Sala Secreta", "sala secreta", "4321", "1111",
                   "Ana Ruiz", "Tablet", "Proyector", token]:
        assert secret not in logged


# ---- routing --------------------------------------------------------------


def joined(relay: ShareRelay) -> Tuple[Device, Outbox, Device, Outbox]:
    """A host and one joined device, both connected, outboxes emptied."""
    host, _ = relay.host("Sala", "1234", "Tablet")
    host_box = connect(relay, host)
    viewer, _ = relay.join("Sala", "1234", "Proyector", "ip-a")
    viewer_box = connect(relay, viewer)
    messages(host_box)
    messages(viewer_box)
    return host, host_box, viewer, viewer_box


def test_a_late_joiner_gets_nothing_before_its_whole_page(
    clock: Clock,
) -> None:
    relay, _, _ = relay_with(clock)
    host, host_box, viewer, viewer_box = joined(relay)
    send(relay, host, type="m", body={"k": "d"})
    assert messages(viewer_box) == []
    send(relay, host, type="m", snap=True, to=viewer.id, body={"k": "s"})
    send(relay, host, type="m", body={"k": "d", "n": 2})
    got = messages(viewer_box)
    assert [m["body"]["k"] for m in got] == ["s", "d"]
    assert not viewer.stale


def test_a_snapshot_for_one_device_goes_to_that_device_only(
    clock: Clock,
) -> None:
    relay, _, _ = relay_with(clock)
    host, host_box, first, first_box = joined(relay)
    send(relay, host, type="m", snap=True, body={"k": "s"})
    messages(first_box)
    second, _ = relay.join("Sala", "1234", "Móvil", "ip-b")
    second_box = connect(relay, second)
    messages(second_box)
    send(relay, host, type="m", snap=True, to=second.id, body={"k": "s2"})
    assert "m" not in types(first_box)
    assert [m["body"]["k"] for m in messages(second_box)] == ["s2"]


def test_a_slow_device_loses_changes_and_gets_the_whole_page_again(
    clock: Clock,
) -> None:
    relay, _, _ = relay_with(clock)
    host, host_box, viewer, viewer_box = joined(relay)
    send(relay, host, type="m", snap=True, body={"k": "s"})
    viewer_box.take()
    viewer_box.limit = 200
    for i in range(10):
        send(relay, host, type="m", body={"k": "d", "pad": "x" * 30, "i": i})
    assert viewer.stale
    asked = [m for m in messages(host_box) if m["type"] == "need_snapshot"]
    assert asked == [{"type": "need_snapshot", "device": viewer.id}]
    send(relay, host, type="m", body={"k": "d", "i": 99})
    send(relay, host, type="m", snap=True, to=viewer.id, body={"k": "s"})
    got = messages(viewer_box)
    assert got[-1]["body"] == {"k": "s"}
    assert all(m["body"].get("i") != 99 for m in got)


def test_a_watch_only_device_cannot_act_even_if_its_page_sends(
    clock: Clock,
) -> None:
    relay, _, _ = relay_with(clock)
    host, host_box, viewer, viewer_box = joined(relay)
    send(relay, viewer, type="in", body={"k": "click", "node": 7})
    assert messages(host_box) == []
    assert relay.dropped_inputs == 1
    send(relay, host, type="set", device=viewer.id, sends=True)
    assert viewer.sends
    host_box.take()
    send(relay, viewer, type="in", body={"k": "click", "node": 7})
    assert messages(host_box) == [{
        "type": "in", "from": viewer.id,
        "body": {"k": "click", "node": 7}}]
    send(relay, viewer, type="watch")
    assert not viewer.sends
    send(relay, viewer, type="in", body={"k": "click", "node": 8})
    assert "in" not in types(host_box)


def test_only_the_host_changes_settings_or_sends_the_page(
    clock: Clock,
) -> None:
    relay, _, _ = relay_with(clock)
    host, host_box, viewer, viewer_box = joined(relay)
    send(relay, viewer, type="set", device=viewer.id, sends=True)
    send(relay, viewer, type="sound", device=viewer.id)
    send(relay, viewer, type="remove", device=host.id)
    send(relay, viewer, type="m", snap=True, body={})
    assert not viewer.sends and host.share.sound == host.id
    assert host.id in host.share.devices
    assert messages(host_box) == []


def test_the_host_chooses_the_sound_device_and_removes_a_device(
    clock: Clock,
) -> None:
    relay, _, _ = relay_with(clock)
    host, _ = relay.host("Sala", "1234", "Tablet")
    host_box = connect(relay, host)
    viewer, token = relay.join("Sala", "1234", "Proyector", "ip-a")
    viewer_box = connect(relay, viewer)
    send(relay, host, type="sound", device=viewer.id)
    assert host.share.sound == viewer.id
    send(relay, host, type="remove", device=viewer.id)
    assert messages(viewer_box)[-1] == {"type": "ended", "reason": "removed"}
    assert viewer_box.closed == (CLOSE_ENDED, "removed")
    assert host.share.sound == host.id
    roster = messages(host_box)[-1]
    assert [d["id"] for d in roster["devices"]] == [host.id]
    with pytest.raises(ShareRefused):
        relay.resume(token)


def test_a_device_reconnects_with_its_token_and_resyncs(clock: Clock) -> None:
    relay, _, _ = relay_with(clock)
    host, _ = relay.host("Sala", "1234", "Tablet")
    host_box = connect(relay, host)
    viewer, token = relay.join("Sala", "1234", "Proyector", "ip-a")
    first = connect(relay, viewer)
    send(relay, host, type="m", snap=True, body={"k": "s"})
    relay.detach(viewer, first)
    roster = [m for m in messages(host_box) if m["type"] == "roster"][-1]
    assert [d["online"] for d in roster["devices"]] == [True, False]
    clock.now += rules.RESUME_SECONDS - 1
    assert relay.resume(token) is viewer
    again = connect(relay, viewer)
    assert viewer.stale
    assert {"type": "need_snapshot", "device": viewer.id} in messages(
        host_box)
    # A third connection of the same device replaces the second.
    third = connect(relay, viewer)
    assert again.closed == (CLOSE_REPLACED, "replaced")
    relay.detach(viewer, again)
    assert viewer.outbox is third


def test_when_the_host_comes_back_every_device_waits_for_its_page(
    clock: Clock,
) -> None:
    relay, _, _ = relay_with(clock)
    host, host_box, viewer, viewer_box = joined(relay)
    send(relay, host, type="m", snap=True, body={"k": "s"})
    relay.detach(host, host_box)
    messages(viewer_box)
    send(relay, viewer, type="in", body={"k": "click"})
    again = connect(relay, host)
    assert viewer.stale
    assert {"type": "need_snapshot", "device": viewer.id} in messages(again)


def test_the_host_leaving_ends_the_session_for_everyone(clock: Clock) -> None:
    relay, _, _ = relay_with(clock)
    host, host_box, viewer, viewer_box = joined(relay)
    send(relay, host, type="leave")
    assert messages(viewer_box)[-1] == {"type": "ended", "reason": "stopped"}
    assert relay.open_sessions() == 0


def test_a_viewer_leaving_keeps_the_session(clock: Clock) -> None:
    relay, _, _ = relay_with(clock)
    host, host_box, viewer, viewer_box = joined(relay)
    send(relay, viewer, type="leave")
    assert viewer_box.closed == (CLOSE_ENDED, "left")
    assert relay.open_sessions() == 1


def test_sessions_expire_when_idle_too_long_or_the_host_stays_away(
    clock: Clock,
) -> None:
    relay, _, _ = relay_with(clock)
    host, host_box, viewer, viewer_box = joined(relay)
    clock.now += rules.IDLE_SECONDS + 1
    assert relay.sweep() == 0
    assert messages(viewer_box)[-1]["reason"] == "expired"

    host, host_box, viewer, viewer_box = joined(relay)
    relay.detach(host, host_box)
    clock.now += rules.RESUME_SECONDS + 1
    relay.sweep()
    assert messages(viewer_box)[-1]["reason"] == "host_left"


def test_no_session_outlives_its_maximum_life(clock: Clock) -> None:
    relay, _, _ = relay_with(clock)
    host, host_box, viewer, viewer_box = joined(relay)
    for _ in range(int(rules.MAX_LIFETIME_SECONDS // 600) + 1):
        clock.now += 600
        send(relay, host, type="m", body={})
        relay.sweep()
    assert relay.open_sessions() == 0


def test_a_device_away_too_long_loses_its_place(clock: Clock) -> None:
    relay, _, _ = relay_with(clock)
    host, host_box, viewer, viewer_box = joined(relay)
    relay.detach(viewer, viewer_box)
    clock.now += rules.RESUME_SECONDS + 1
    relay.sweep()
    assert viewer.id not in host.share.devices
    assert relay.open_sessions() == 1


def test_messages_over_the_rate_are_dropped(clock: Clock) -> None:
    relay, _, _ = relay_with(clock)
    host, host_box, viewer, viewer_box = joined(relay)
    send(relay, host, type="set", device=viewer.id, sends=True)
    host_box.take()
    for _ in range(int(rules.VIEWER_BURST) + 20):
        send(relay, viewer, type="in", body={})
    assert len(messages(host_box)) == int(rules.VIEWER_BURST)
    assert viewer.dropped == 20
    assert relay.limit(viewer) == rules.VIEWER_MESSAGE_BYTES
    assert relay.limit(host) == rules.HOST_MESSAGE_BYTES


def test_odd_messages_are_ignored(clock: Clock) -> None:
    relay, _, _ = relay_with(clock)
    host, host_box, viewer, viewer_box = joined(relay)
    for text in ["not json", "[1, 2]", '{"type": "nope"}']:
        relay.receive(viewer, text)
        relay.receive(host, text)
    send(relay, viewer, type="ping")
    assert messages(viewer_box) == [{"type": "pong"}]
    assert messages(host_box) == []
    send(relay, viewer, type="resync")
    assert types(host_box) == ["need_snapshot"]


def test_resuming_an_unknown_or_ended_token_is_refused(clock: Clock) -> None:
    relay, _, _ = relay_with(clock)
    for token in ["nope", None, 3]:
        with pytest.raises(ShareRefused) as caught:
            relay.resume(token)
        assert caught.value.code == "ended"
    host, token = relay.host("Sala", "1234", "Tablet")
    late = Outbox()
    relay._end(host.share, "stopped")
    relay.attach(host, late)
    assert late.closed == (CLOSE_ENDED, "ended")
