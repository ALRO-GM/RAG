"""Use case: rebuild a round whose lost audio was recovered (#265).

When a live interview lost its connection for longer than the page
could hold, the page kept the audio and sent it after the end; the
transcriber transcribed it, merged it into the round's transcript and
asks here (``POST /sessions/{id}/recovered``) for every indicator to be
computed again from the merged transcript. :class:`RecoveredRound`
replays it through :class:`~orchestrator.application.rebuild_round.
RoundRebuild`: the topics and concepts of the role's plan, the
product's mandatory items, the lines to avoid, and the judgement of the
answers, whose report replaces the one stored at the end of the round
(in the plan's results and the candidate's file: the round now has more
of the conversation). The candidate and the plan are found from the
candidate's rounds; the product from the session when this process
still knows it, else the interview product. A session still in memory
gets the new judgement in its view and on its stream, and its hand-off
record is written again.

:func:`forget_recovery` asks the transcriber to delete the recovered
audio still kept for sessions whose file is deleted (#265: deleting a
candidate's file deletes it too). Nothing here logs words.
"""

import logging
from typing import Any, Callable, Iterable, Optional, Protocol, Tuple
from typing import runtime_checkable

from orchestrator.application.ports import (
    Document,
    Executor,
    TranscriberUnavailable,
)
from orchestrator.application.rebuild_round import (
    RoundRebuild,
    events_of_transcript,
)
from orchestrator.domain.events import Transcript
from orchestrator.domain.interview_coverage import earlier_coverage

LOG = logging.getLogger(__name__)
INTERVIEW = "live_assist.interview"


@runtime_checkable
class RecoveryDiscard(Protocol):
    """The transcriber's deletion of a session's recovered audio."""

    def discard_recovery(self, session_id: str) -> None:
        """Delete what is kept for the session (absent: nothing)."""
        ...


class RecoveredRound:
    """Recompute a round's indicators from its merged transcript.

    Args:
        sessions: The live sessions (products, ports, wiring and the
            sessions this process still knows).
        executor: Runs the trackers' checks; inline, so every check is
            done when the replay ends (as ``rebuild-round`` does).
    """

    def __init__(self, sessions: Any, executor: Executor) -> None:
        """Keep the live sessions and the executor."""
        self.sessions = sessions
        self.executor = executor

    def _owner(self, session_id: str) -> Tuple[Any, Any]:
        """The plan and the candidate's profile of a session, if any."""
        wiring = self.sessions.interview
        book = wiring.candidates if wiring is not None else None
        if book is None:
            return None, None
        try:
            found = book.by_session(session_id)
            plan = wiring.store.load(found["plan_id"])
            profile = book.store.load(found["plan_id"], found["id"])
        except Exception:  # noqa: BLE001 - no candidate: no judgement
            return None, None
        return plan, profile

    def rebuild(self, session_id: str, document: Document) -> Document:
        """Replay ``document`` (``transcript.v1``); return the summary.

        Raises ``ValueError`` for a transcript of another session.
        """
        transcript = Transcript.model_validate(document)
        if transcript.session.id != session_id:
            raise ValueError("the transcript is of another session")
        events, roles = events_of_transcript(transcript)
        live = self.sessions.find(session_id)
        plan, profile = self._owner(session_id)
        product = live.product if live is not None else \
            self.sessions.products.get(INTERVIEW) if plan is not None \
            else None
        publish: Callable[[Document], object] = (
            live.feed.publish if live is not None else (lambda m: None))
        wiring = self.sessions.interview if plan is not None else None
        rebuilt = RoundRebuild(
            self.executor, self.sessions.mandatory,
            self.sessions.avoid, wiring,
        ).rebuild(
            session_id, events, roles, product=product, plan=plan,
            rounds=profile.rounds if profile is not None else (),
            candidate=profile.name if profile is not None else None,
            candidate_id=profile.id if profile is not None else None,
            judge=plan is not None and profile is not None,
            publish=publish,
            earlier=earlier_coverage(live.handoff or {})
            if live is not None else None,
        )
        if live is not None:
            if rebuilt.accuracy is not None:
                live.accuracy = rebuilt.accuracy
            if live.handoff is not None:
                record = dict(live.handoff, recovered=True)
                if rebuilt.mandatory is not None:
                    record["mandatory"] = rebuilt.mandatory
                if rebuilt.accuracy is not None:
                    record["accuracy"] = rebuilt.accuracy
                if rebuilt.coverage is not None:
                    # #286: the same coverage as the stored report.
                    record["coverage"] = rebuilt.coverage
                live.handoff = record
                self.sessions.write_handoff(live)
        summary = rebuilt.summary()
        LOG.info("Round %s rebuilt after a recovery: %s", session_id,
                 {k: v for k, v in summary.items() if k != "session_id"})
        return summary


def forget_recovery(transcriber: Optional[object],
                    session_ids: Iterable[str]) -> int:
    """Ask the transcriber to delete the recovered audio of sessions.

    Best effort (the transcriber's own expiry deletes it anyway);
    returns how many requests it accepted, logged as a count.
    """
    if not isinstance(transcriber, RecoveryDiscard):
        return 0
    done = 0
    for session_id in session_ids:
        try:
            transcriber.discard_recovery(session_id)
            done += 1
        except TranscriberUnavailable:
            continue
    if done:
        LOG.info("Recovered audio of %d session(s) deleted", done)
    return done
