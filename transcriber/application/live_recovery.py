"""Use case: recover the audio a live session lost, asynchronously (#265).

The page keeps the audio the server never acknowledged (IndexedDB, in
the browser) and, once the session has ended, uploads it here in
numbered chunks (:meth:`LiveRecovery.put_chunk`, idempotent: a chunk
sent again replaces itself) and closes the upload
(:meth:`LiveRecovery.close`). Only for a session that is over, that
this service ran, whose product allows it (the start request's
``buffer_audio``), and only inside the time the session lasted. From
then on a job moves through ``queued → transcribing → merging →
publishing → analysing → rejudging → done`` (or ``failed``), one step
per :meth:`LiveRecovery.tick`.

**Crash safety.** The job's state is written after every step, and
always *before* anything is deleted: the recovered parts' transcripts
are kept and the job is written as ``merging`` (audio marked deleted)
before the audio goes, so a crash at any point either redoes a step
from what is still on disk or finds the audio already gone and goes on
from the texts; a job whose audio is gone and whose texts are not fails
honestly (``audio_lost``), it never "completes" with nothing. A batch
job a restart lost is submitted again without counting a try (at most
:data:`MAX_RESTARTS` times); only real failures count, three at most
(:data:`~transcriber.domain.live_recovery.MAX_TRIES`, with a wait). Any
unexpected exception (``MemoryError`` included) is a failed try. A
failed job never blocks anything else.

* ``queued`` waits while the live session is still on (or handing its
  transcript over). Then the parts of the uploaded audio the live
  transcript lacks (:func:`~transcriber.domain.live_recovery.
  recovered_spans`, grouped by :func:`~transcriber.domain.live_recovery.
  group_spans`: one bounded WAV per group and channel, never one array
  across a long gap) go to the batch pipeline of the recovery's own
  repository (:class:`RecoveryTranscriber`: the same job runner, so the
  batch GPU lease is honoured and the job waits while a live session
  holds the GPU). Nothing is put in «Sesiones».
* ``transcribing`` follows those jobs; while one of them runs (holds
  the GPU) the job is announced as busy (``busy``: the services guard
  then refuses a restart). Once every part is transcribed, the texts
  and the next state are written, then the audio is deleted.
* ``merging`` puts the recovered sentences into the live transcript in
  place of the gaps (always from the live transcript as it was before
  the first merge, so a merge run twice gives the same result).
* ``publishing`` / ``analysing`` hand it to the RAG as the start request
  asked; ``rejudging`` asks the orchestrator to rebuild the round.

External work (the batch submission, the RAG, the orchestrator) runs
outside the service's lock, with the job marked in progress: an upload,
a «Descartar» or a sweep never waits for it, and a discard during a step
wins (the step's result is dropped and its parts deleted).

Audio is kept only until its transcription succeeds, and never past the
job's expiry: :meth:`LiveRecovery.sweep` deletes what is left, at start
and every hour, and the batch parts of no live job. Deletions are logged
as counts. Nothing here logs words.
"""

import hashlib
import logging
import re
import threading
import uuid
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Mapping, Optional, Set

from transcriber.application.ports import (
    Clock,
    LiveLog,
    RecoveryStore,
    RecoveryTranscriber,
    RoundRebuilder,
    SessionAnalysisPort,
    TranscriptPublisher,
)
from transcriber.domain.errors import (
    InvalidInputError,
    NotBufferedError,
    NotFoundError,
    SessionRunningError,
    StateError,
    TooLargeError,
)
from transcriber.domain.live import LiveSpeaker
from transcriber.domain.live_recovery import (
    RecoveredPart,
    RecoveryJob,
    RecoverySettings,
    SAMPLE_RATE,
    group_spans,
    merge_recovered,
    recovered_spans,
    speaker_map,
)
from transcriber.domain.models import SESSION_ID_PATTERN, Transcript

LOG = logging.getLogger(__name__)

MAX_CHUNK_NUMBER = 999_999
"""Largest chunk number accepted (the page numbers them from 0)."""
LIVE_TEXT = "live"
"""The live transcript as it was before the first merge."""
MAX_RESTARTS = 20
"""Restarts that may lose a recovery's batch job before it fails."""
MAX_PARTS = 200
"""Parts one recovery may be split into."""

Busy = Callable[[str, bool], None]
"""Told when a recovery's batch job starts or stops holding the GPU."""


def check_session(session_id: str) -> str:
    """Return ``session_id`` if it is a valid session id, else raise."""
    if not isinstance(session_id, str) \
            or not re.fullmatch(SESSION_ID_PATTERN, session_id):
        raise InvalidInputError(f"Invalid session id {session_id!r}.")
    return session_id


def part_id(
    session_id: str, channel: Optional[int], group: int = 0,
    generation: str = "",
) -> str:
    """The id of one part in the recovery's repository (short, safe).

    ``generation`` tells apart the parts of a job discarded and the
    parts of a new job of the same session.
    """
    digest = hashlib.sha256(
        f"{session_id}/{generation}".encode("utf-8")).hexdigest()[:24]
    side = "m" if channel is None else f"c{channel}"
    return f"rc-{digest}-{group}-{side}"


def _time(text: Any) -> Optional[datetime]:
    """An ISO time of a record, or ``None``."""
    if not isinstance(text, str) or not text:
        return None
    try:
        when = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return when if when.tzinfo is not None else None


class _Discarded(Exception):
    """The job was discarded while one of its steps ran."""


class LiveRecovery:
    """Receive, keep, transcribe and merge a live session's lost audio.

    Args:
        store: The temporary store (jobs, chunks, working documents).
        log: The live logs (each session's record and transcript).
        transcriber: The batch pipeline over the recovery's repository.
        clock: The wall clock.
        settings: Limits and the hard expiry.
        running: Tells whether a live session is still on here (its
            transcript not handed over yet).
        publisher: The RAG's transcripts, if the session asked for it.
        analysis: The RAG's post-call analysis, if asked.
        rebuilder: The orchestrator's rebuild of the round, if any.
        busy: Told while a recovery's transcription holds the GPU.
    """

    def __init__(
        self,
        store: RecoveryStore,
        log: LiveLog,
        transcriber: RecoveryTranscriber,
        clock: Clock,
        settings: RecoverySettings = RecoverySettings(),
        running: Callable[[str], bool] = lambda session_id: False,
        publisher: Optional[TranscriptPublisher] = None,
        analysis: Optional[SessionAnalysisPort] = None,
        rebuilder: Optional[RoundRebuilder] = None,
        busy: Busy = lambda session_id, on: None,
    ) -> None:
        """Keep the ports; nothing is read until the first call."""
        self._store = store
        self._log = log
        self._transcriber = transcriber
        self._clock = clock
        self._settings = settings
        self._running = running
        self._publisher = publisher
        self._analysis = analysis
        self._rebuilder = rebuilder
        self._busy = busy
        self._lock = threading.RLock()
        self._stepping: Set[str] = set()
        self._discarded: Set[str] = set()

    # ---- the upload ----------------------------------------------------

    def _new_job(self, session_id: str, channels: int) -> RecoveryJob:
        """A job for an ended session that allows it, with its bounds.

        Audio may lie anywhere from the session's start to the product's
        limit (``buffer_max_hours``, else ``ttl_hours``): the page goes
        on recording offline long after the server ended the session
        (a lost source is waited for only 10 minutes), so the server's
        own end is no bound. Memory stays bounded by the byte limits and
        the parts' length.
        """
        record = self._log.read_session(session_id)
        if record is None:
            raise NotFoundError(f"No live session {session_id} to recover.")
        request = record.get("request") or {}
        if not request.get("buffer_audio"):
            raise NotBufferedError(
                "This session's product keeps no audio during an outage.")
        now = self._clock.now()
        expires = now + timedelta(hours=self._settings.ttl_hours)
        started = _time(record.get("started_at"))
        hours = request.get("buffer_max_hours")
        if not isinstance(hours, int) or hours <= 0:
            hours = None
        if started is not None and hours is not None:
            expires = min(expires, started + timedelta(hours=hours))
        span = float(hours) if hours is not None \
            else self._settings.ttl_hours
        job = RecoveryJob(
            session_id, "uploading", now.isoformat(), expires.isoformat(),
            channels=channels, updated_at=now.isoformat(),
        )
        job.meta["max_end"] = int(span * 3600 * SAMPLE_RATE)
        job.meta["generation"] = uuid.uuid4().hex
        return job

    def put_chunk(
        self, session_id: str, number: int, offset: int, data: bytes,
        channels: int = 1,
    ) -> Dict[str, Any]:
        """Keep chunk ``number`` (PCM s16le, ``channels`` interleaved).

        ``offset``: the stream sample it starts at. Only for an ended
        live session this service kept a record of, whose start request
        allowed buffering (else :class:`StateError`, 409), and only
        inside the time it lasted plus a slack (else
        :class:`InvalidInputError`); resent chunks replace themselves
        (resumable, idempotent). Refused once the upload is closed and
        past the limits (:class:`TooLargeError`, which the page shows).
        Returns the job's view.
        """
        check_session(session_id)
        if not self._settings.enabled:
            raise StateError("Recovering live audio is turned off.")
        if not 0 <= number <= MAX_CHUNK_NUMBER:
            raise InvalidInputError("Invalid chunk number.")
        if channels not in (1, 2):
            raise InvalidInputError("channels is 1 or 2.")
        frame = 2 * channels
        if not data or len(data) % frame:
            raise InvalidInputError(
                "A chunk is whole frames of 16-bit PCM.")
        if len(data) > self._settings.chunk_max_bytes:
            raise TooLargeError(
                f"A chunk is at most {self._settings.chunk_max_bytes}"
                " bytes.")
        samples = len(data) // frame
        if offset < 0:
            raise InvalidInputError("Invalid chunk offset.")
        if self._running(session_id):
            raise SessionRunningError(
                f"Live session {session_id} is still on: go back to it.")
        with self._lock:
            job = self._store.load(session_id)
            if job is None:
                job = self._new_job(session_id, channels)
            if job.state != "uploading":
                raise StateError(
                    f"The recovery of {session_id} is {job.state}.")
            if channels != job.channels:
                raise InvalidInputError("The channels changed.")
            limit = job.meta.get("max_end")
            if isinstance(limit, int) and offset + samples > limit:
                raise InvalidInputError(
                    "The chunk lies past the time this session's audio may"
                    " be kept.")
            key = str(number)
            old = job.chunks.get(key)
            old_bytes = old[1] * frame if old else 0
            if len(job.chunks) - (1 if old else 0) \
                    >= self._settings.max_chunks:
                raise TooLargeError("Too many chunks for one session.")
            if job.bytes - old_bytes + len(data) \
                    > self._settings.session_max_bytes:
                raise TooLargeError(
                    "The audio of this session is over the limit.")
            if self._store.audio_bytes() - old_bytes + len(data) \
                    > self._settings.total_max_bytes:
                raise TooLargeError(
                    "There is no room left for recovered audio.")
            self._store.put_chunk(session_id, number, data)
            job.chunks[key] = [offset, samples]
            job.bytes += len(data) - old_bytes
            job.updated_at = self._clock.now().isoformat()
            self._store.save(job)
            return job.view()

    def close(
        self, session_id: str, count: int,
        expires_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        """End the upload: ``count`` chunks were sent; queue the job.

        A missing chunk is a :class:`StateError` (the page sends what
        is missing and closes again), and so is a session still on.
        ``expires_at`` (ISO, from the page: the session's start plus
        the product's limit) may only bring the expiry closer. Closing
        a closed job changes nothing.
        """
        check_session(session_id)
        if self._running(session_id):
            raise SessionRunningError(
                f"Live session {session_id} is still on: go back to it.")
        with self._lock:
            job = self._require(session_id)
            if job.state != "uploading":
                return job.view()
            if count != len(job.chunks) or count <= 0:
                raise StateError(
                    f"{len(job.chunks)} of {count} chunks arrived.")
            if expires_at:
                asked = _time(expires_at)
                if asked is None:
                    raise InvalidInputError("Invalid expires_at.")
                if asked < datetime.fromisoformat(job.expires_at):
                    job.expires_at = asked.isoformat()
            job.move("queued", self._clock.now())
            self._store.save(job)
            LOG.info("Live session %s: recovery queued (%d chunks, %.0f s)",
                     session_id, count,
                     sum(n for _, n in job.chunks.values()) / SAMPLE_RATE)
            return job.view()

    def status(self, session_id: str) -> Dict[str, Any]:
        """Return the job's view (:class:`NotFoundError` without one)."""
        return self._require(check_session(session_id)).view()

    def list(self) -> List[Dict[str, Any]]:
        """Return every job's view (newest first)."""
        views = []
        for session_id in self._store.ids():
            try:
                job = self._store.load(session_id)
            except Exception:  # noqa: BLE001 - one bad job only
                continue
            if job is not None:
                views.append(job.view())
        return sorted(views, key=lambda v: str(v.get("updated_at") or ""),
                      reverse=True)

    def discard(self, session_id: str) -> bool:
        """Delete everything kept for the session now; ``True`` if any.

        The page's «Descartar», and the deletion of a session or of a
        candidate's file. A step in progress finds the job gone and
        keeps nothing; a batch transcription in progress finds its part
        gone and saves nothing.
        """
        check_session(session_id)
        with self._lock:
            job = self._store.load(session_id)
            found = job is not None or session_id in self._store.ids()
            if job is not None and session_id in self._stepping:
                self._discarded.add(self._generation(job))
            self._forget_parts(session_id, job)
            self._store.delete(session_id)
        self._busy(session_id, False)
        if found:
            LOG.info("Live session %s: recovered audio discarded",
                     session_id)
        return found

    def _require(self, session_id: str) -> RecoveryJob:
        """Return the session's job or raise :class:`NotFoundError`."""
        job = self._store.load(session_id)
        if job is None:
            raise NotFoundError(f"No recovery for live session {session_id}.")
        return job

    # ---- the job -------------------------------------------------------

    def sweep(self) -> int:
        """Delete the audio past its expiry or of ended jobs, and batch
        parts of no live job; return how many sessions lost audio
        (logged as a count)."""
        now = self._clock.now()
        swept = files = 0
        with self._lock:
            live_parts: Set[str] = set()
            for session_id in self._store.ids():
                if session_id in self._stepping:
                    continue
                try:
                    job = self._store.load(session_id)
                except Exception:  # noqa: BLE001 - unreadable: delete
                    job = None
                if job is None:
                    files += self._store.delete_audio(session_id)
                    self._store.delete(session_id)
                    swept += 1
                    continue
                if not job.final and not job.audio_deleted \
                        and not job.expired(now):
                    live_parts.update(self._part_ids(job))
                    continue
                if not job.final and job.expired(now):
                    job.failed_step = job.state
                    job.state = "failed"
                    job.error = "expired"
                    job.updated_at = now.isoformat()
                    self._store.save(job)
                gone = self._store.delete_audio(session_id)
                if gone or not job.audio_deleted:
                    files += gone
                    swept += 1
                if not job.audio_deleted:
                    job.audio_deleted = True
                    self._store.save(job)
                self._forget_parts(session_id, job)
                if job.final:
                    # Working copies a process left when it stopped
                    # between the end and their deletion.
                    self._store.delete_texts(session_id)
            stray = [p for p in self._transcriber.ids()
                     if p.startswith("rc-") and p not in live_parts
                     and not any(p in self._part_ids_of(s)
                                 for s in self._stepping)]
            for part in stray:
                self._transcriber.forget(part)
        if swept or stray:
            LOG.info("Recovered live audio deleted: %d session(s), %d"
                     " file(s), %d batch part(s)", swept, files, len(stray))
        return swept

    def _part_ids_of(self, session_id: str) -> List[str]:
        """The parts of a job in progress (empty when unreadable)."""
        try:
            job = self._store.load(session_id)
        except Exception:  # noqa: BLE001 - nothing to protect
            return []
        return self._part_ids(job) if job is not None else []

    def tick(self) -> int:
        """Move every due job one step on; return how many moved.

        Never raises: a job that breaks is logged (its id and the
        error's type) and the others go on.
        """
        moved = 0
        for session_id in self._store.ids():
            try:
                if self._tick_one(session_id):
                    moved += 1
            except Exception as exc:  # noqa: BLE001 - the others go on
                LOG.warning("Live session %s: recovery step failed (%s)",
                            session_id, type(exc).__name__)
        return moved

    @staticmethod
    def _generation(job: RecoveryJob) -> str:
        """The job's own token (a job discarded and made again for the
        same session is another generation)."""
        return str(job.meta.get("generation") or "")

    def _tick_one(self, session_id: str) -> bool:
        """One step of one job, the external work outside the lock."""
        with self._lock:
            if session_id in self._stepping:
                return False
            job = self._store.load(session_id)
            if job is None or not job.due(self._clock.now()):
                return False
            self._stepping.add(session_id)
        before = (job.state, job.tries, job.progress, job.restarts)
        try:
            self._step(job)
            self._persist(job)
        except _Discarded:
            return False
        finally:
            with self._lock:
                self._stepping.discard(session_id)
                generation = self._generation(job)
                if generation in self._discarded:
                    # Discarded while it ran: only this generation's own
                    # parts go (a newer job of the session is left alone).
                    self._discarded.discard(generation)
                    self._forget_parts(session_id, job)
        return (job.state, job.tries, job.progress, job.restarts) != before

    def _persist(self, job: RecoveryJob) -> None:
        """Write the job, unless it was discarded (or replaced by a new
        job of the same session) meanwhile."""
        with self._lock:
            current = self._store.load(job.session_id)
            if self._generation(job) in self._discarded or current is None \
                    or self._generation(current) != self._generation(job):
                raise _Discarded()
            self._store.save(job)

    def _step(self, job: RecoveryJob) -> None:
        """Run the job's current step once; any error is a failed try."""
        now = self._clock.now()
        if job.expired(now):
            job.failed_step, job.state, job.error = job.state, "failed", \
                "expired"
            self._busy(job.session_id, False)
            self._persist(job)
            self._drop_audio(job)
            return
        step = getattr(self, f"_{job.state}")
        try:
            step(job)
        except _Discarded:
            raise
        except Exception as exc:  # noqa: BLE001 - MemoryError too
            self._failed(job, type(exc).__name__)

    def _failed(self, job: RecoveryJob, error: str) -> None:
        """Count a failed try; a job that ends deletes its audio."""
        step = job.state
        ended = job.failed_try(error, self._clock.now())
        LOG.warning("Live session %s: recovery step %s failed (%s)%s",
                    job.session_id, step, error,
                    "; given up" if ended else "")
        if step == "transcribing":
            self._busy(job.session_id, False)
            if not ended:
                job.state = "queued"  # transcribed again from the chunks
        if ended:
            self._persist(job)
            self._drop_audio(job)
            self._store.delete_texts(job.session_id)

    def _record(self, job: RecoveryJob) -> Dict[str, Any]:
        """The live session's record (``ValueError`` without one)."""
        record = self._log.read_session(job.session_id)
        if record is None:
            raise ValueError("no live record")
        return record

    def _parts(self, job: RecoveryJob) -> List[Dict[str, Any]]:
        """The job's parts: ``{"id", "channel", "start", "end"}``."""
        parts = job.meta.get("parts") or []
        return [p for p in parts if isinstance(p, dict)]

    def _part_ids(self, job: RecoveryJob) -> List[str]:
        """The batch ids of the job's parts."""
        return [str(p["id"]) for p in self._parts(job)]

    def _texts_kept(self, job: RecoveryJob) -> bool:
        """Every part's transcript is kept (a crash after transcribing)."""
        parts = self._parts(job)
        return bool(parts) and all(
            self._store.read_text(job.session_id, f"part-{n}") is not None
            for n in range(len(parts)))

    def _queued(self, job: RecoveryJob) -> None:
        """Send the parts the live transcript lacks to the batch pipeline."""
        if self._running(job.session_id):
            return  # the live session is still on: wait
        if job.audio_deleted or self._store.missing(
                job.session_id, list(job.chunks)):
            # A crash after the transcription: go on from its texts, or
            # say honestly that the audio is lost.
            if self._texts_kept(job):
                job.audio_deleted = True
                job.move("merging", self._clock.now())
                return
            job.failed_step, job.state, job.error = "queued", "failed", \
                "audio_lost"
            job.updated_at = self._clock.now().isoformat()
            self._persist(job)
            self._drop_audio(job)
            return
        record = self._record(job)
        if not self._log.has_transcript(job.session_id):
            raise ValueError("no live transcript")
        received = record.get("received_samples")
        if not isinstance(received, int):
            received = self._stream_end(job.session_id)
        silences = [(int(a), int(b)) for a, b in record.get("silences") or []]
        spans = recovered_spans(job.uploaded(), silences, received)
        job.meta["spans"] = [list(s) for s in spans]
        if not spans:
            job.meta["parts"] = []
            job.audio_deleted = True
            job.move("merging", self._clock.now())
            self._persist(job)
            self._drop_audio(job)
            return
        groups = group_spans(spans)
        if len(groups) * (1 if job.channels == 1 else 2) > MAX_PARTS:
            raise ValueError("too many parts")
        request = record.get("request") or {}
        language = request.get("language") or "auto"
        diarized = bool(request.get("diarize"))
        channels: List[Optional[int]] = \
            [None] if job.channels == 1 else [0, 1]
        parts: List[Dict[str, Any]] = [
            {"id": part_id(job.session_id, channel, n,
                           self._generation(job)), "channel": channel,
             "start": start, "end": end}
            for n, (start, end) in enumerate(groups) for channel in channels
        ]
        job.meta["parts"] = parts
        self._persist(job)  # the parts are known before any is submitted
        for part in parts:
            channel = part["channel"]
            wav = self._store.wav(job.session_id, job.chunks,
                                  (part["start"], part["end"]), spans,
                                  job.channels, channel)
            speakers: Optional[int]
            expected: Optional[List[str]]
            if channel is not None or not diarized:
                speakers, expected = 1, []
            else:
                enrolled = request.get("enrolled")
                speakers = 2 if request.get("parties") else None
                expected = list(enrolled) if enrolled is not None else None
            self._transcriber.submit(part["id"], wav, language, speakers,
                                     expected)
        job.state = "transcribing"  # the tries go on counting
        job.next_try = None
        job.progress = 0.0
        job.updated_at = self._clock.now().isoformat()

    def _stream_end(self, session_id: str) -> int:
        """Where a session without the count ended (its last event)."""
        times = [float(e.get("stream_time") or 0.0)
                 for e in self._log.read_events(session_id)]
        return int(max(times + [0.0]) * SAMPLE_RATE)

    def _transcribing(self, job: RecoveryJob) -> None:
        """Follow the parts; once all are done, keep their texts, write
        the next state, and only then delete the audio."""
        parts = self._parts(job)
        states = [self._transcriber.status(str(p["id"])) for p in parts]
        self._busy(job.session_id,
                   any(state == "running" for state, _ in states))
        if any(state == "failed" for state, _ in states):
            raise ValueError("transcription failed")
        if any(state == "lost" for state, _ in states):
            # A restart lost the batch job: submitted again, no try spent.
            self._busy(job.session_id, False)
            job.restarts += 1
            if job.restarts > MAX_RESTARTS:
                raise ValueError("lost too many times")
            LOG.warning("Live session %s: the recovery's transcription was"
                        " lost (a restart); it is submitted again",
                        job.session_id)
            job.state = "queued"
            job.next_try = None
            return
        job.progress = sum(f if s != "done" else 1.0 for s, f in states) \
            / max(len(states), 1)
        if any(state != "done" for state, _ in states):
            return
        for n, part in enumerate(parts):
            document = self._transcriber.transcript(str(part["id"]))
            if document is None:
                raise ValueError("no transcript")
            self._store.save_text(job.session_id, f"part-{n}", document)
        self._busy(job.session_id, False)
        job.progress = 1.0
        job.audio_deleted = True
        job.move("merging", self._clock.now())
        self._persist(job)  # before any audio goes
        self._drop_audio(job)

    def _merging(self, job: RecoveryJob) -> None:
        """Put the recovered sentences into the live transcript."""
        live = self._store.read_text(job.session_id, LIVE_TEXT)
        if live is None:
            live = self._log.read_transcript(job.session_id)
            if live is None:
                raise ValueError("no live transcript")
            self._store.save_text(job.session_id, LIVE_TEXT, live)
        record = self._record(job)
        request = record.get("request") or {}
        speakers = [
            LiveSpeaker(id=str(s["id"]), channel=s.get("channel"),
                        role=str(s.get("role") or "unknown"),
                        name=s.get("name"), enrolled=bool(s.get("enrolled")))
            for s in record.get("speakers") or []
        ]
        parties = request.get("parties")
        pair = ((str(parties["employee"]), str(parties["other"]))
                if isinstance(parties, dict) else None)
        spans = [(int(a), int(b)) for a, b in job.meta.get("spans") or []]
        parts: List[RecoveredPart] = []
        for n, part in enumerate(self._parts(job)):
            document = self._store.read_text(job.session_id, f"part-{n}")
            if document is None:
                raise ValueError("a recovered part is missing")
            batch = Transcript.from_contract(document)
            names = speaker_map(
                batch.utterances, batch.speakers, speakers, job.channels,
                bool(request.get("diarize")), part["channel"], pair)
            parts.append(RecoveredPart(batch, int(part["start"]), names))
        merged = merge_recovered(
            Transcript.from_contract(live), parts, spans,
            {s.id: s.role for s in speakers})
        self._log.save_transcript(job.session_id,
                                  merged.transcript.to_contract())
        job.seconds = merged.seconds
        job.added = merged.added
        job.unidentified = merged.unidentified
        LOG.info("Live session %s: %d recovered sentence(s) merged, %d"
                 " dropped, %.0f s of speech recovered%s", job.session_id,
                 merged.added, merged.dropped, merged.seconds,
                 ", some voices not identified" if merged.unidentified
                 else "")
        job.move("publishing", self._clock.now())

    def _handoff(self, job: RecoveryJob) -> Mapping[str, Any]:
        """What the session's start request asked to do at the end."""
        request = self._record(job).get("request") or {}
        handoff = request.get("handoff")
        return handoff if isinstance(handoff, dict) else {}

    def _publishing(self, job: RecoveryJob) -> None:
        """Hand the merged transcript to the RAG, if the session asked."""
        if self._handoff(job).get("transcript_v1") and job.added \
                and self._publisher is not None:
            document = self._log.read_transcript(job.session_id)
            if document is None:
                raise ValueError("no transcript")
            self._publisher.publish(document)
        job.move("analysing", self._clock.now())

    def _analysing(self, job: RecoveryJob) -> None:
        """Start the post-call analysis again, if the product asks."""
        handoff = self._handoff(job)
        if handoff.get("transcript_v1") and handoff.get("run_post_call") \
                and job.added and self._analysis is not None:
            self._analysis.start(job.session_id, True, False)
        job.move("rejudging", self._clock.now())

    def _rejudging(self, job: RecoveryJob) -> None:
        """Ask the orchestrator to rebuild the round's indicators."""
        if job.added and self._rebuilder is not None:
            document = self._log.read_transcript(job.session_id)
            if document is None:
                raise ValueError("no transcript")
            self._rebuilder.rebuild(job.session_id, document)
        job.move("done", self._clock.now())
        self._persist(job)
        self._store.delete_texts(job.session_id)
        self._drop_audio(job)

    def _forget_parts(
        self, session_id: str, job: Optional[RecoveryJob],
    ) -> None:
        """Delete the batch copies of a job's parts."""
        for part in self._part_ids(job) if job is not None else []:
            self._transcriber.forget(part)

    def _drop_audio(self, job: RecoveryJob) -> int:
        """Delete the job's audio (chunks and the batch copies)."""
        files = self._store.delete_audio(job.session_id)
        self._forget_parts(job.session_id, job)
        if files:
            LOG.info("Live session %s: recovered audio deleted (%d"
                     " file(s))", job.session_id, files)
        job.audio_deleted = True
        return files
