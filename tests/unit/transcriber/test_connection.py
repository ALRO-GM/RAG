"""The live page's connection check of the services (#247)."""

import threading
import time

from transcriber.application.connection import (
    AVAILABLE,
    NOT_SET_UP,
    SERVICES,
    UNAVAILABLE,
    CheckServices,
    FixedProbe,
)


class FakeProbe:
    """A probe with a scripted answer, optionally slow or broken."""

    def __init__(self, answer: bool, wait: float = 0.0,
                 broken: bool = False) -> None:
        """Keep the script."""
        self.answer = answer
        self.wait = wait
        self.broken = broken
        self.calls = 0
        self.thread = ""

    def available(self) -> bool:
        """Answer as scripted."""
        self.calls += 1
        self.thread = threading.current_thread().name
        time.sleep(self.wait)
        if self.broken:
            raise RuntimeError("boom")
        return self.answer


def test_every_service_gets_one_word_in_order() -> None:
    check = CheckServices({
        "transcription": FixedProbe(True),
        "assistant": FakeProbe(False),
        "knowledge": None,
    })
    answer = check()
    assert list(answer) == list(SERVICES)
    assert answer == {
        "transcription": AVAILABLE,
        "assistant": UNAVAILABLE,
        "knowledge": NOT_SET_UP,
    }


def test_a_broken_probe_counts_as_unavailable() -> None:
    broken = FakeProbe(True, broken=True)
    answer = CheckServices({"assistant": broken})()
    assert answer["assistant"] == UNAVAILABLE
    assert answer["transcription"] == NOT_SET_UP
    assert broken.calls == 1


def test_unknown_names_are_ignored_and_nothing_is_asked_without_probes(
) -> None:
    other = FakeProbe(True)
    answer = CheckServices({"other": other})()
    assert set(answer.values()) == {NOT_SET_UP}
    assert other.calls == 0


def test_the_probes_are_asked_at_the_same_time() -> None:
    slow = [FakeProbe(True, wait=0.3) for _ in SERVICES]
    started = time.monotonic()
    answer = CheckServices(dict(zip(SERVICES, slow)))()
    took = time.monotonic() - started
    assert set(answer.values()) == {AVAILABLE}
    # Three probes of 0.3 s each in turn would take 0.9 s.
    assert took < 0.8
    assert len({p.thread for p in slow}) == len(SERVICES)
