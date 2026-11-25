"""Use case: follow a live session's events through a dropped stream (#264).

:class:`ResumingEvents` wraps an :class:`~orchestrator.application.ports.
EventSource` (the transcriber's ``live.event.v1`` socket). Before #264 any
close other than a normal one (1000) ended the following and failed the
session in the orchestrator, although the transcriber keeps every event
and serves them again from any ``from_seq``. Now a dropped stream is
opened again from the next sequence number, with a short backoff; events
already seen are never handed on twice. It gives up only when the stream
has been impossible to keep for ``give_up_s`` in a row: a stream that
stayed open ``healthy_s`` (an event or not: a session waiting for its
audio source sends none for up to 10 min) counts as working again. A
refusal that will not change (the session is unknown, the code is
refused) ends the following at once, as before.

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
        give_up_s: How long the stream may stay impossible to keep before
            the following ends: the transcriber's grace for a lost source
            (600 s) plus a margin, as the session cannot outlive it.
        healthy_s: How long an opened stream must stay up to count as
            working (no event needed).
        first_wait_s: The first wait before reopening; doubles up to
            ``max_wait_s``.
    """

    def __init__(
        self,
        inner: EventSource,
        sleep: Callable[[float], None],
        clock: Callable[[], float],
        give_up_s: float = 660.0,
        first_wait_s: float = 0.5,
        max_wait_s: float = 5.0,
        healthy_s: float = 5.0,
    ) -> None:
        """Keep the source and the retry policy."""
        self.inner = inner
        self.sleep = sleep
        self.clock = clock
        self.give_up_s = give_up_s
        self.first_wait_s = first_wait_s
        self.max_wait_s = max_wait_s
        self.healthy_s = healthy_s

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
            opened = self.clock()
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
                if now - opened >= self.healthy_s:
                    failing_since, attempt = None, 0  # it was working
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
