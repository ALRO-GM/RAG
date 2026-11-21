"""The message feed of one session: what the live page receives.

Every message (a ``live.event.v1`` event, a suggestion record, a status
change) is appended in order and numbered from 1. Any number of readers
follow it: a reader asks for the messages after the last number it saw,
waiting up to a timeout, so a page that reconnects gets what it missed
(the SSE ``Last-Event-ID``). The feed is bounded: past ``limit``
messages the oldest are forgotten and a late reader starts after them.
"""

import threading
from typing import Any, Dict, List, Tuple

Message = Dict[str, Any]


class Feed:
    """An append-only, numbered list of messages with waiting readers.

    Args:
        limit: Most messages kept.
    """

    def __init__(self, limit: int = 5000) -> None:
        """Start empty and open."""
        if limit < 1:
            raise ValueError("limit must be positive")
        self.limit = limit
        self._items: List[Message] = []
        self._first = 1
        self._closed = False
        self._cond = threading.Condition()

    @property
    def last(self) -> int:
        """Number of the newest message (0 when there is none)."""
        with self._cond:
            return self._first + len(self._items) - 1

    @property
    def closed(self) -> bool:
        """Whether no more messages will come."""
        with self._cond:
            return self._closed

    def publish(self, message: Message) -> int:
        """Append ``message``; return its number (0 once closed)."""
        with self._cond:
            if self._closed:
                return 0
            self._items.append(dict(message))
            if len(self._items) > self.limit:
                drop = len(self._items) - self.limit
                del self._items[:drop]
                self._first += drop
            self._cond.notify_all()
            return self._first + len(self._items) - 1

    def close(self) -> None:
        """Say no more messages will come; wake every reader."""
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def read(
        self, after: int, timeout: float
    ) -> Tuple[List[Tuple[int, Message]], bool]:
        """Return the messages numbered above ``after``.

        Waits up to ``timeout`` seconds when there is none yet.

        Returns:
            ``([(number, message)], finished)``; ``finished`` is true
            when the feed is closed and the reader has everything.
        """
        with self._cond:
            last = self._first + len(self._items) - 1
            if after >= last and not self._closed:
                self._cond.wait(timeout)
                last = self._first + len(self._items) - 1
            start = max(after + 1, self._first)
            found = [
                (n, self._items[n - self._first])
                for n in range(start, last + 1)
            ]
            return found, self._closed
