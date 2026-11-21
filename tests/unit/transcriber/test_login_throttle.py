"""The login throttle: per-client back-off and the global lock."""

from typing import List

import pytest

import transcriber.adapters.inbound.login_throttle as module
from transcriber.adapters.inbound.login_throttle import LoginThrottle


class Clock:
    """A clock moved by hand."""

    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def make(clock: Clock, global_limit: int = 8) -> LoginThrottle:
    """A throttle: 3 failures per client, ``global_limit`` overall."""
    return LoginThrottle(
        per_client_limit=3, window=100.0, base_cooldown=10.0,
        max_cooldown=50.0, global_limit=global_limit, global_window=1000.0,
        global_cooldown=200.0, now=clock,
    )


def test_a_client_is_locked_after_the_limit_with_growing_cooldowns() -> None:
    clock = Clock()
    throttle = make(clock, 1000)
    for _ in range(2):
        throttle.failed("a")
    assert throttle.retry_after("a") == (0.0, False)
    throttle.failed("a")
    assert throttle.retry_after("a") == (10.0, False)
    assert throttle.retry_after("b") == (0.0, False)
    clock.now += 10
    assert throttle.retry_after("a")[0] == 0.0
    waits: List[float] = []
    for _ in range(4):
        for _ in range(3):
            throttle.failed("a")
        wait = throttle.retry_after("a")[0]
        waits.append(wait)
        clock.now += wait
    assert waits == [20.0, 40.0, 50.0, 50.0]


def test_failures_age_out_of_the_window() -> None:
    clock = Clock()
    throttle = make(clock)
    throttle.failed("a")
    throttle.failed("a")
    clock.now += 101
    throttle.failed("a")
    assert throttle.retry_after("a")[0] == 0.0


def test_many_clients_start_the_global_lock_once() -> None:
    clock = Clock()
    throttle = make(clock)
    started = [throttle.failed(f"ip{i}") for i in range(8)]
    assert started == [False] * 7 + [True]
    wait, everyone = throttle.retry_after("someone-new")
    assert (wait, everyone) == (200.0, True)
    clock.now += 200
    assert throttle.retry_after("someone-new") == (0.0, False)


def test_success_forgets_the_client() -> None:
    clock = Clock()
    throttle = make(clock)
    throttle.failed("a")
    throttle.failed("a")
    throttle.succeeded("a")
    throttle.failed("a")
    assert throttle.retry_after("a")[0] == 0.0


def test_the_tables_stay_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "MAX_CLIENTS", 3)
    clock = Clock()
    throttle = LoginThrottle(
        per_client_limit=2, window=10.0, global_limit=10_000, now=clock,
    )
    for i in range(6):
        throttle.failed(f"ip{i}")
        clock.now += 11
    assert len(throttle._failures) <= 3
