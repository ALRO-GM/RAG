"""Use case: hand a live session's transcript over, also after a failure
or a restart (LV-11, #264).

:class:`HandOver` turns the events of a live session into its
``transcript.v1`` document, keeps it next to the live log, and then
publishes it to the RAG and starts the post-call analysis when the
start request asked for it. It is used three ways:

* at the normal end of a session;
* when the session's worker died with an exception (CUDA out of memory,
  or anything else): what was heard until then is still handed over,
  marked ``partial`` in the session's summary;
* at startup (:meth:`HandOver.recover`): a session a stopped process left
  with a ``session.json`` record (it started after #264) and either no
  transcript (rebuilt from ``events.jsonl``, marked ``recovered``) or a
  transcript it was asked to publish and had not (published again; the
  RAG replaces a transcript of the same session). Sessions logged before
  the record existed are never touched. At most :data:`RECOVER_LIMIT`
  per start, newest first; a session that cannot be read or rebuilt is
  logged (its id and the error's type) and left as it is: recovery never
  stops a start.

Publishing happens after the session has ended (#264), so a stop never
waits for the RAG. The session record holds the start request, the
speakers and names, the model and the state: no audio, and no words
beyond what ``events.jsonl`` already holds.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence

from transcriber.application.ports import (
    Clock,
    LiveLog,
    SessionAnalysisPort,
    TranscriptPublisher,
)
from transcriber.domain.errors import TranscriberError
from transcriber.domain.live import LiveSpeaker
from transcriber.domain.live_session import LiveSessionRequest
from transcriber.domain.live_transcript import live_transcript
from transcriber.domain.models import Pipeline

LOG = logging.getLogger(__name__)
LIFECYCLE = logging.getLogger("transcriber.live.lifecycle")

RECOVER_LIMIT = 20
"""Most sessions handed over again per start (the rest wait, counted)."""
SESSION_SCHEMA = "live.session.record"
"""The internal record kept as ``session.json`` (not a public contract)."""

Document = Dict[str, Any]


@dataclass
class Handed:
    """A session's transcript, kept and maybe about to be published.

    ``summary`` is what the API shows as the session's ``transcript``:
    ``utterances``, ``published``, ``analysis``, ``error``, and
    ``partial`` (the worker failed) or ``recovered`` (rebuilt at start).
    """

    session_id: str
    document: Document
    summary: Document
    handoff_v1: bool = False
    post_call: bool = False
    utterances: int = 0
    facts: Optional["SessionFacts"] = None
    state: str = "ended"
    """The record's state to write once published."""
    extra: Dict[str, Any] = field(default_factory=dict)
    """More of the record to keep when it is written again."""


@dataclass
class SessionFacts:
    """What a transcript is assembled from, for a running or a dead one."""

    session_id: str
    request: LiveSessionRequest
    speakers: Sequence[LiveSpeaker]
    channels: int
    model: Optional[str] = None
    compute: Optional[str] = None
    diarizer: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class HandOver:
    """Assemble, keep, publish and analyse a live session's transcript."""

    def __init__(
        self,
        log: Optional[LiveLog],
        publisher: Optional[TranscriptPublisher],
        analysis: Optional[SessionAnalysisPort],
        clock: Clock,
        version: str,
    ) -> None:
        """Keep the ports (each may be absent) and the version."""
        self._log = log
        self._publisher = publisher
        self._analysis = analysis
        self._clock = clock
        self._version = version

    def keep(
        self, facts: SessionFacts, events: Sequence[Mapping[str, Any]],
        mark: Optional[str] = None,
    ) -> Handed:
        """Build the transcript of ``events``; keep it next to the log.

        ``mark`` is ``partial`` (the worker failed) or ``recovered``
        (rebuilt at start); it goes into the summary, not the contract.
        """
        started = events[0].get("at") if events else None
        model = facts.model or "unknown"
        compute = facts.compute or "unknown"
        transcript = live_transcript(
            facts.session_id, events, facts.speakers,
            Pipeline(
                recogniser=f"faster-whisper {model} {compute}",
                voice_model="none", transcriber=self._version,
                diarizer=("channels" if facts.channels == 2
                          else facts.diarizer),
            ),
            language=facts.request.language,
            started_at=parse_time(started if isinstance(started, str)
                                  else None),
        )
        document = transcript.to_contract()
        summary: Document = {
            "utterances": len(transcript.utterances),
            "published": False, "analysis": None, "error": None,
        }
        if mark is not None:
            summary[mark] = True
        if self._log is not None:
            try:
                self._log.save_transcript(facts.session_id, document)
            except OSError:
                LOG.exception("The transcript of %s was not kept",
                              facts.session_id)
        handoff = facts.request.handoff
        return Handed(
            facts.session_id, document, summary,
            handoff_v1=handoff.transcript_v1,
            post_call=handoff.run_post_call,
            utterances=len(transcript.utterances), facts=facts,
        )

    def publish(self, handed: Handed) -> None:
        """Publish the transcript and start the analysis, if asked.

        A failure is recorded in the summary, not raised: the live part
        went well (or, for a failed session, as well as it could).
        """
        summary = handed.summary
        if not handed.handoff_v1 or not handed.utterances:
            return
        if self._publisher is None:
            summary["error"] = "no RAG to publish to"
            return
        try:
            self._publisher.publish(handed.document)
            summary["published"] = True
            if handed.post_call:
                if self._analysis is None:
                    summary["error"] = "no RAG analysis configured"
                    return
                self._analysis.start(handed.session_id, False, False)
                summary["analysis"] = "started"
        except TranscriberError as exc:
            summary["error"] = str(exc)

    def record(self, facts: SessionFacts, state: str, **more: Any) -> None:
        """Write the session's ``session.json`` (start, rename, end).

        The start request, speakers and names, the model and the state:
        enough to rebuild the transcript from ``events.jsonl`` after a
        restart. A failure to write is logged, never raised.
        """
        if self._log is None:
            return
        document: Document = {
            "schema": SESSION_SCHEMA, "session_id": facts.session_id,
            "state": state,
            "request": facts.request.model_dump(mode="json"),
            "speakers": [
                {"id": s.id, "channel": s.channel, "role": s.role,
                 "name": s.name, "enrolled": s.enrolled}
                for s in facts.speakers
            ],
            "channels": facts.channels, "model": facts.model,
            "compute": facts.compute, "diarizer": facts.diarizer,
            "updated_at": self._clock.now().isoformat(),
            **facts.extra, **more,
        }
        try:
            self._log.save_session(facts.session_id, document)
        except OSError:
            LOG.exception("The record of live session %s was not kept",
                          facts.session_id)

    def recover(
        self, skip: Sequence[str] = (), limit: int = RECOVER_LIMIT,
    ) -> List[Handed]:
        """Find what a stopped process left without its hand-over.

        Only sessions with a record (started after #264), newest first,
        at most ``limit`` (the rest are counted in the log and wait for
        the next start). Without a transcript, it is rebuilt from the
        events and kept (``recovered``); with one that was to be
        published and was not, it is published again. ``skip``: ids
        still running here. Never raises: a session that cannot be read
        or rebuilt is logged and left as it is. Returns what is to be
        published (:meth:`publish`, then :meth:`record`).
        """
        if self._log is None:
            return []
        try:
            ids = [i for i in self._log.recorded() if i not in skip]
        except Exception as exc:  # noqa: BLE001 - never stop a start
            LOG.warning("Live sessions cannot be listed for recovery (%s)",
                        type(exc).__name__)
            return []
        done: List[Handed] = []
        waiting = 0
        for session_id in ids:
            try:
                handed = self._recover_one(session_id, len(done) < limit)
            except Exception as exc:  # noqa: BLE001 - one bad log only
                LOG.warning("Live session %s cannot be recovered (%s)",
                            session_id, type(exc).__name__)
                continue
            if handed is True:
                waiting += 1
            elif isinstance(handed, Handed):
                done.append(handed)
        if waiting:
            LOG.warning("%d more live session(s) to recover wait for the"
                        " next start", waiting)
        return done

    def _recover_one(self, session_id: str, room: bool) -> Any:
        """One session: a :class:`Handed`, ``True`` (to do, no room
        left), or ``None`` (nothing to do)."""
        assert self._log is not None
        record = self._log.read_session(session_id)
        if record is None:
            return None
        document = self._log.read_transcript(session_id)
        summary = record.get("transcript") or {}
        handoff = (record.get("request") or {}).get("handoff") or {}
        if document is None:
            if not room:
                return True
            events = self._log.read_events(session_id)
            if not events:
                return None
            facts = facts_of(session_id, record)
            handed = self.keep(facts, events, mark="recovered")
            handed.state = "recovered"
            handed.extra = {"end_reason": "restart"}
            self.record(facts, "recovered", end_reason="restart",
                        transcript=handed.summary)
            LIFECYCLE.info(
                "Live session %s recovered after a restart: %d events,"
                " %d utterances", session_id, len(events),
                handed.utterances,
            )
            return handed
        utterances = len(document.get("utterances") or [])
        if not handoff.get("transcript_v1") or summary.get("published") \
                or not utterances:
            return None
        if not room:
            return True
        facts = facts_of(session_id, record)
        state = str(record.get("state") or "ended")
        LIFECYCLE.info("Live session %s: its transcript was not published;"
                       " publishing it again", session_id)
        return Handed(
            session_id, document,
            {"utterances": utterances, "published": False,
             "analysis": None, "error": None,
             **{k: v for k, v in summary.items()
                if k in ("partial", "recovered")}},
            handoff_v1=True, post_call=bool(handoff.get("run_post_call")),
            utterances=utterances, facts=facts,
            state="recovered" if state in ("starting", "running")
            else state,
            extra={k: record[k] for k in ("end_reason", "error")
                   if k in record},
        )

    def finish(self, handed: Handed) -> None:
        """Publish ``handed`` and write its record with the result."""
        self.publish(handed)
        if handed.facts is not None:
            self.record(handed.facts, handed.state,
                        transcript=handed.summary, **handed.extra)


def facts_of(session_id: str, record: Mapping[str, Any]) -> SessionFacts:
    """What a dead session's transcript is assembled from: its record.

    Raises ``ValueError`` for a record that does not hold a valid
    request.
    """
    request = LiveSessionRequest.model_validate(record.get("request"))
    if request.session_id != session_id:
        raise ValueError("the record is of another session")
    speakers = [
        LiveSpeaker(id=str(s["id"]), channel=s.get("channel"),
                    role=str(s.get("role") or "unknown"),
                    name=s.get("name"), enrolled=bool(s.get("enrolled")))
        for s in record.get("speakers") or []
    ]
    return SessionFacts(
        session_id, request, speakers,
        int(record.get("channels") or request.channels),
        record.get("model"), record.get("compute"), record.get("diarizer"),
    )


def parse_time(text: Optional[str]) -> Optional[datetime]:
    """Return the ISO time ``text`` of an event as a datetime, if any."""
    if not text:
        return None
    return datetime.fromisoformat(text.replace("Z", "+00:00"))
