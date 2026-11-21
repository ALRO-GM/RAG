"""The session feed: numbered, resumable, bounded, closable."""

import threading

import pytest

from orchestrator.application.feed import Feed


def test_messages_are_numbered_and_read_after_a_number() -> None:
    feed = Feed()
    assert feed.last == 0
    assert feed.publish({"a": 1}) == 1
    assert feed.publish({"a": 2}) == 2
    found, finished = feed.read(0, 0.01)
    assert found == [(1, {"a": 1}), (2, {"a": 2})]
    assert not finished
    assert feed.read(1, 0.01)[0] == [(2, {"a": 2})]
    assert feed.read(2, 0.01) == ([], False)


def test_a_closed_feed_finishes_readers_and_refuses_more() -> None:
    feed = Feed()
    feed.publish({"a": 1})
    feed.close()
    assert feed.closed
    assert feed.publish({"a": 2}) == 0
    assert feed.read(0, 5.0) == ([(1, {"a": 1})], True)
    assert feed.read(1, 5.0) == ([], True)


def test_a_waiting_reader_wakes_on_a_new_message() -> None:
    feed = Feed()
    got = []

    def reader() -> None:
        got.append(feed.read(0, 5.0))

    thread = threading.Thread(target=reader)
    thread.start()
    feed.publish({"x": 1})
    thread.join(5.0)
    assert got == [([(1, {"x": 1})], False)]


def test_old_messages_are_forgotten_past_the_limit() -> None:
    feed = Feed(limit=2)
    for n in range(5):
        feed.publish({"n": n})
    assert feed.last == 5
    assert feed.read(0, 0.01)[0] == [(4, {"n": 3}), (5, {"n": 4})]
    with pytest.raises(ValueError):
        Feed(limit=0)
