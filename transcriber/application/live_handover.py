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
  with events but no transcript is rebuilt from ``events.jsonl`` and the
  small ``session.json`` record, handed over, and marked ``recovered``.

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
            utterances=len(transcript.utterances),
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

    def recover(self, skip: Sequence[str] = ()) -> List[Handed]:
        """Rebuild the sessions a stopped process left unfinished.

        Each one with events but no transcript gets its transcript from
        ``events.jsonl`` and ``session.json`` (or, without the record,
        from the events alone), kept and published as at a normal end,
        and its record says ``recovered``. ``skip``: ids that are still
        running here. Returns what was handed over (to publish).
        """
        if self._log is None:
            return []
        done: List[Handed] = []
        for session_id in self._log.unfinished():
            if session_id in skip:
                continue
            try:
                events = self._log.read_events(session_id)
                record = self._log.read_session(session_id)
                facts = facts_of(session_id, events, record)
            except (OSError, ValueError) as exc:
                LOG.warning("Live session %s cannot be recovered: %s",
                            session_id, exc)
                continue
            if not events:
                continue
            handed = self.keep(facts, events, mark="recovered")
            self.record(facts, "recovered", end_reason="restart",
                        transcript=handed.summary)
            LIFECYCLE.info(
                "Live session %s recovered after a restart: %d events,"
                " %d utterances", session_id, len(events),
                handed.utterances,
            )
            done.append(handed)
        return done


def facts_of(
    session_id: str, events: Sequence[Mapping[str, Any]],
    record: Optional[Mapping[str, Any]],
) -> SessionFacts:
    """What a dead session's transcript is assembled from.

    From its ``session.json`` record when there is one; else from the
    events alone (the speakers of ``session_started``), as for a session
    kept before the record existed. Raises ``ValueError`` for a record
    that does not hold a valid request.
    """
    if record is not None:
        request = LiveSessionRequest.model_validate(record.get("request"))
        speakers = [
            LiveSpeaker(id=str(s["id"]), channel=s.get("channel"),
                        role=str(s.get("role") or "unknown"),
                        name=s.get("name"),
                        enrolled=bool(s.get("enrolled")))
            for s in record.get("speakers") or []
        ]
        return SessionFacts(
            session_id, request, speakers,
            int(record.get("channels") or request.channels),
            record.get("model"), record.get("compute"),
            record.get("diarizer"),
        )
    started = next((e for e in events
                    if e.get("type") == "session_started"), None)
    listed = (started or {}).get("speakers") or []
    speakers = [
        LiveSpeaker(id=str(s["id"]), channel=s.get("channel"),
                    role=str(s.get("role") or "unknown"), name=s.get("name"))
        for s in listed
    ]
    channels = int(((started or {}).get("source") or {}).get("channels")
                   or 1)
    request = LiveSessionRequest(session_id=session_id, channels=channels)
    return SessionFacts(session_id, request, speakers, channels)


def parse_time(text: Optional[str]) -> Optional[datetime]:
    """Return the ISO time ``text`` of an event as a datetime, if any."""
    if not text:
        return None
    return datetime.fromisoformat(text.replace("Z", "+00:00"))
