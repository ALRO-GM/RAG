"""Use case: follow a live session's events through a dropped stream (#264).

:class:`ResumingEvents` wraps an :class:`~orchestrator.application.ports.
EventSource` (the transcriber's ``live.event.v1`` socket). Before #264 any
close other than a normal one (1000) ended the following and failed the
session in the orchestrator, although the transcriber keeps every event
and serves them again from any ``from_seq``. Now a dropped stream is
opened again from the next sequence number, with a short backoff, for as
long as ``give_up_s`` passes without a single event; events already seen
are never handed on twice. A refusal that will not change (the session
is unknown, the code is refused) ends the following at once, as before.

No text is logged here: only ids, counts and reasons.
"""

import logging
from typing import Any, Callable, Iterator

from orchestrator.application.ports import (
    CancelToken,
    Document,
    EventSource,
    TranscriberUnavailable,
)

LOG = logging.getLogger(__name__)

PERMANENT = ("4404", "not_found", "4401", "unauthorized", "4429",
             "too_many_attempts", "4400", "invalid_input")
"""Reasons of a refusal that a new attempt would get again."""


def permanent(exc: TranscriberUnavailable) -> bool:
    """Tell whether opening the stream again cannot help."""
    reason = str(exc)
    return any(code in reason for code in PERMANENT)


class ResumingEvents:
    """An event source that reopens a dropped stream where it was.

    Args:
        inner: The source that opens the stream once (``follow`` with a
            ``from_seq``).
        sleep: Waits that many seconds (``time.sleep``; tests pass a fake).
        clock: Monotonic seconds.
        give_up_s: How long without any event the stream is retried.
        first_wait_s: The first wait before reopening; doubles up to
            ``max_wait_s``.
    """

    def __init__(
        self,
        inner: EventSource,
        sleep: Callable[[float], None],
        clock: Callable[[], float],
        give_up_s: float = 120.0,
        first_wait_s: float = 0.5,
        max_wait_s: float = 5.0,
    ) -> None:
        """Keep the source and the retry policy."""
        self.inner = inner
        self.sleep = sleep
        self.clock = clock
        self.give_up_s = give_up_s
        self.first_wait_s = first_wait_s
        self.max_wait_s = max_wait_s

    def follow(
        self, session_id: str, stop: CancelToken, from_seq: int = 0,
    ) -> Iterator[Document]:
        """Yield every event once, from ``from_seq``, until the end.

        Ends when the stream closes normally (the session is over) or
        ``stop`` is set. Raises :class:`TranscriberUnavailable` for a
        refusal that cannot change, or when ``give_up_s`` passed since
        the last event with the stream still down.
        """
        position = from_seq
        failing_since: Any = None
        attempt = 0
        while not stop.cancelled:
            try:
                for document in self.inner.follow(
                        session_id, stop, from_seq=position):
                    seq = document.get("seq")
                    if isinstance(seq, int):
                        if seq < position:
                            continue  # already handed on before the drop
                        position = seq + 1
                    else:
                        position += 1
                    failing_since, attempt = None, 0
                    yield document
                return
            except TranscriberUnavailable as exc:
                if permanent(exc) or stop.cancelled:
                    raise
                now = self.clock()
                failing_since = now if failing_since is None \
                    else failing_since
                if now - failing_since >= self.give_up_s:
                    raise
                wait = min(self.first_wait_s * 2 ** attempt,
                           self.max_wait_s)
                attempt += 1
                LOG.warning(
                    "Live session %s: event stream lost (%s); reopening"
                    " from event %d in %.1f s", session_id, exc, position,
                    wait,
                )
                self.sleep(wait)
