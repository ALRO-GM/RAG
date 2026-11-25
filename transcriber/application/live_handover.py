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
MAX_PUBLISH_TRIES = 3
"""Publishes tried for one transcript (at its end, then at later starts)
before it is left unpublished for good (said in its summary)."""
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
    tries: int = 0
    """Publishes tried so far (kept in the record)."""
    analysis_done: bool = False
    """The RAG already has this session's analysis: do not start it."""


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
            if handed.post_call and not handed.analysis_done:
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

        Only sessions whose hand-over is not done (the log keeps that
        mark, so the work does not grow with history), with a record
        (started after #264), latest end first, at most ``limit`` (the
        rest are counted in the log and wait for the next start).
        Without a transcript, it is rebuilt from the events and kept
        (``recovered``); with one that was to be published and was not,
        it is published again, at most :data:`MAX_PUBLISH_TRIES` times
        in all. ``skip``: ids still running here. Never raises: a
        session that cannot be read or rebuilt is logged and left as it
        is. Returns what is to be published (:meth:`finish`).
        """
        if self._log is None:
            return []
        try:
            ids = [i for i in self._log.pending() if i not in skip]
        except Exception as exc:  # noqa: BLE001 - never stop a start
            LOG.warning("Live sessions cannot be listed for recovery (%s)",
                        type(exc).__name__)
            return []
        records: List[Any] = []
        for session_id in ids:
            try:
                record = self._log.read_session(session_id)
            except Exception as exc:  # noqa: BLE001 - one bad record only
                LOG.warning("Live session %s cannot be recovered (%s)",
                            session_id, type(exc).__name__)
                continue
            if record is not None:
                records.append((ended_at(record), session_id, record))
        records.sort(key=lambda found: (found[0], found[1]), reverse=True)
        done: List[Handed] = []
        waiting = given_up = 0
        for _, session_id, record in records:
            try:
                handed = self._recover_one(session_id, record,
                                           len(done) < limit)
            except Exception as exc:  # noqa: BLE001 - one bad log only
                LOG.warning("Live session %s cannot be recovered (%s)",
                            session_id, type(exc).__name__)
                continue
            if handed == "wait":
                waiting += 1
            elif handed == "gave_up":
                given_up += 1
            elif isinstance(handed, Handed):
                done.append(handed)
        if waiting:
            LOG.warning("%d more live session(s) to recover wait for the"
                        " next start", waiting)
        if given_up:
            LOG.warning("%d live transcript(s) left unpublished after %d"
                        " tries", given_up, MAX_PUBLISH_TRIES)
        return done

    def _recover_one(
        self, session_id: str, record: Mapping[str, Any], room: bool,
    ) -> Any:
        """One session: a :class:`Handed`, ``"wait"`` (no room left),
        ``"gave_up"``, or ``None`` (nothing to do). Its transcript is
        read only when it is to be published now."""
        assert self._log is not None
        summary = dict(record.get("transcript") or {})
        handoff = (record.get("request") or {}).get("handoff") or {}
        publishing = record.get("publish_state")
        tries = int(record.get("publish_tries") or 0)
        if not self._log.has_transcript(session_id):
            if not room:
                return "wait"
            events = self._log.read_events(session_id)
            if not events:
                self._log.mark_handed(session_id)
                return None
            facts = facts_of(session_id, record)
            handed = self.keep(facts, events, mark="recovered")
            handed.state = "recovered"
            handed.extra = {"end_reason": "restart",
                            "ended_at": ended_at(record) or None}
            self.record(facts, "recovered", end_reason="restart",
                        transcript=handed.summary)
            LIFECYCLE.info(
                "Live session %s recovered after a restart: %d events,"
                " %d utterances", session_id, len(events),
                handed.utterances,
            )
            return handed
        if not handoff.get("transcript_v1") or summary.get("published") \
                or publishing == "published" or not summary.get(
                    "utterances", 1):
            self._log.mark_handed(session_id)
            return None
        facts = facts_of(session_id, record)
        extra = {k: record[k] for k in ("end_reason", "error", "ended_at")
                 if k in record}
        state = str(record.get("state") or "ended")
        state = "recovered" if state in ("starting", "running") else state
        if tries >= MAX_PUBLISH_TRIES:
            summary.update(published=False,
                           error=f"not published after {tries} tries")
            self.record(facts, state, transcript=summary,
                        publish_state="given_up", publish_tries=tries,
                        **extra)
            self._log.mark_handed(session_id)
            return "gave_up"
        if not room:
            return "wait"
        document = self._log.read_transcript(session_id)
        if document is None:
            return None
        utterances = len(document.get("utterances") or [])
        if publishing == "publishing":
            # The last publish may have reached the RAG (the result was
            # never written): once more at most, and no second analysis
            # when the RAG already has one for the session.
            tries = max(tries, MAX_PUBLISH_TRIES - 1)
        LIFECYCLE.info("Live session %s: its transcript was not published;"
                       " publishing it again", session_id)
        return Handed(
            session_id, document,
            {"utterances": utterances, "published": False,
             "analysis": None, "error": None,
             **{k: v for k, v in summary.items()
                if k in ("partial", "recovered")}},
            handoff_v1=True, post_call=bool(handoff.get("run_post_call")),
            utterances=utterances, facts=facts, state=state, extra=extra,
            tries=tries,
            analysis_done=publishing == "publishing"
            and self._has_analysis(session_id),
        )

    def _has_analysis(self, session_id: str) -> bool:
        """Tell whether the RAG already has an analysis of the session."""
        if self._analysis is None:
            return False
        try:
            status = self._analysis.status(session_id)
        except TranscriberError:
            return False
        steps = status.get("steps") if isinstance(status, dict) else None
        return bool(steps) or bool(status.get("running"))

    def finish(self, handed: Handed) -> None:
        """Publish ``handed`` and keep the result in its record.

        The intent (``publish_state: publishing``) is written first: a
        process that dies before writing the result is found in that
        state by the next start, which publishes once more at most. A
        session whose hand-over is done (published, not asked, or tried
        :data:`MAX_PUBLISH_TRIES` times) is marked so in the log and
        never looked at again.
        """
        asked = handed.handoff_v1 and handed.utterances > 0
        if asked:
            handed.tries += 1
            self._write(handed, "publishing")
            self.publish(handed)
        published = bool(handed.summary.get("published"))
        gave_up = asked and not published \
            and handed.tries >= MAX_PUBLISH_TRIES
        if gave_up:
            handed.summary["error"] = (
                f"not published after {handed.tries} tries"
                + (f": {handed.summary['error']}"
                   if handed.summary.get("error") else ""))
        state = ("published" if published else "given_up" if gave_up
                 else "failed" if asked else "not_asked")
        self._write(handed, state)
        if not asked or published or gave_up:
            if self._log is not None:
                try:
                    self._log.mark_handed(handed.session_id)
                except OSError:
                    LOG.exception("Live session %s: hand-over not marked",
                                  handed.session_id)

    def _write(self, handed: Handed, publish_state: str) -> None:
        """Write the record with the publish's state and tries."""
        if handed.facts is None:
            return
        self.record(handed.facts, handed.state, transcript=handed.summary,
                    publish_state=publish_state, publish_tries=handed.tries,
                    **handed.extra)


def ended_at(record: Mapping[str, Any]) -> str:
    """When a recorded session ended (its last write before that)."""
    return str(record.get("ended_at") or record.get("updated_at") or "")


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
