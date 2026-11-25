"""Use case: recover the audio a live session lost, asynchronously (#265).

The page keeps the audio the server never acknowledged (IndexedDB, in
the browser) and, once the session has ended, uploads it here in
numbered chunks (:meth:`LiveRecovery.put_chunk`, idempotent: a chunk
sent again replaces itself) and closes the upload
(:meth:`LiveRecovery.close`). From then on a job moves through
``queued → transcribing → merging → publishing → analysing → rejudging
→ done`` (or ``failed``), one step per :meth:`LiveRecovery.tick`; its
state is written after every step (:class:`RecoveryStore`), so a
restart goes on where it stopped. A step that fails is tried again
after a wait, at most three times (:data:`~transcriber.domain.
live_recovery.MAX_TRIES`); a failed job never blocks anything else.

* ``queued`` waits while the live session is still on (or handing its
  transcript over). Then the parts of the uploaded audio that the live
  transcript lacks (:func:`~transcriber.domain.live_recovery.
  recovered_spans`) go, as one WAV per channel, to the batch pipeline
  of the recovery's own repository (:class:`RecoveryTranscriber`: the
  same job runner, so the batch GPU lease is honoured and the job waits
  while a live session holds the GPU). Nothing is put in «Sesiones».
* ``transcribing`` follows those jobs; once every part is transcribed
  the audio is deleted at once (chunks and the batch copy).
* ``merging`` puts the recovered sentences into the live transcript in
  place of the gaps (always from the live transcript as it was before
  the first merge, so a merge run twice gives the same result) and
  keeps it as the session's transcript.
* ``publishing`` and ``analysing`` hand it to the RAG as the start
  request asked (the RAG replaces a transcript of the same session;
  the post-call analysis is started again);
* ``rejudging`` asks the orchestrator to rebuild the round's
  indicators from it (topics and concepts, mandatory items, lines to
  avoid, the judgement and the candidate's file).

Audio is kept only until its transcription succeeds, and never past the
job's expiry (the page's own limit, at most ``ttl_hours`` after the
upload began): :meth:`LiveRecovery.sweep` deletes what is left, at
start and every hour; a job that ends (done or failed) deletes its
audio too. Deletions are logged as counts. :meth:`LiveRecovery.discard`
deletes everything of a session at once (the page's «Descartar», a
deleted session or candidate file). Nothing here logs words.
"""

import hashlib
import logging
import re
import threading
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Mapping, Optional

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
    NotFoundError,
    StateError,
    TooLargeError,
    TranscriberError,
)
from transcriber.domain.live import LiveSpeaker
from transcriber.domain.live_recovery import (
    RecoveredPart,
    RecoveryJob,
    RecoverySettings,
    SAMPLE_RATE,
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


def check_session(session_id: str) -> str:
    """Return ``session_id`` if it is a valid session id, else raise."""
    if not isinstance(session_id, str) \
            or not re.fullmatch(SESSION_ID_PATTERN, session_id):
        raise InvalidInputError(f"Invalid session id {session_id!r}.")
    return session_id


def part_id(session_id: str, channel: Optional[int]) -> str:
    """The id of one part in the recovery's repository (short, safe)."""
    digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:24]
    return f"rc-{digest}-{'m' if channel is None else f'c{channel}'}"


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
        self._lock = threading.RLock()

    # ---- the upload ----------------------------------------------------

    def put_chunk(
        self, session_id: str, number: int, offset: int, data: bytes,
        channels: int = 1,
    ) -> Dict[str, Any]:
        """Keep chunk ``number`` (PCM s16le, ``channels`` interleaved).

        ``offset``: the stream sample it starts at. Only for a live
        session this service kept a record of; resent chunks replace
        themselves (resumable, idempotent). Refused once the upload is
        closed (:class:`StateError`) and past the limits
        (:class:`TooLargeError`, which the page shows). Returns the
        job's view.
        """
        check_session(session_id)
        if not self._settings.enabled:
            raise StateError("Recovering live audio is turned off.")
        if not 0 <= number <= MAX_CHUNK_NUMBER:
            raise InvalidInputError("Invalid chunk number.")
        if offset < 0 or offset > 2 ** 40:
            raise InvalidInputError("Invalid chunk offset.")
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
        with self._lock:
            job = self._store.load(session_id)
            if job is None:
                if self._log.read_session(session_id) is None:
                    raise NotFoundError(
                        f"No live session {session_id} to recover.")
                now = self._clock.now()
                job = RecoveryJob(
                    session_id, "uploading", now.isoformat(),
                    (now + timedelta(hours=self._settings.ttl_hours)
                     ).isoformat(),
                    channels=channels, updated_at=now.isoformat(),
                )
            if job.state != "uploading":
                raise StateError(
                    f"The recovery of {session_id} is {job.state}.")
            if channels != job.channels:
                raise InvalidInputError("The channels changed.")
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
            job.chunks[key] = [offset, len(data) // frame]
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
        is missing and closes again). ``expires_at`` (ISO, from the
        page: the session's start plus the product's limit) may only
        bring the expiry closer. Closing a closed job changes nothing.
        """
        check_session(session_id)
        with self._lock:
            job = self._require(session_id)
            if job.state != "uploading":
                return job.view()
            if count != len(job.chunks) or count <= 0:
                raise StateError(
                    f"{len(job.chunks)} of {count} chunks arrived.")
            if expires_at:
                try:
                    asked = datetime.fromisoformat(
                        expires_at.replace("Z", "+00:00"))
                except ValueError as exc:
                    raise InvalidInputError("Invalid expires_at.") from exc
                if asked.tzinfo is not None and \
                        asked < datetime.fromisoformat(job.expires_at):
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
        candidate's file. A batch transcription in progress finds its
        part gone and saves nothing.
        """
        check_session(session_id)
        with self._lock:
            job = self._store.load(session_id)
            found = job is not None or session_id in self._store.ids()
            for channel in (None, 0, 1):
                self._transcriber.forget(part_id(session_id, channel))
            self._store.delete(session_id)
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
        """Delete the audio past its expiry or of ended jobs; return how
        many sessions lost audio (logged as a count)."""
        now = self._clock.now()
        swept = files = 0
        with self._lock:
            for session_id in self._store.ids():
                try:
                    job = self._store.load(session_id)
                except Exception:  # noqa: BLE001 - unreadable: delete
                    job = None
                if job is None:
                    files += self._store.delete_audio(session_id)
                    self._store.delete(session_id)
                    swept += 1
                    continue
                if job.final and job.audio_deleted:
                    continue
                if not job.final and not job.expired(now):
                    continue
                if not job.final:
                    job.failed_step = job.state
                    job.state = "failed"
                    job.error = "expired"
                    job.updated_at = now.isoformat()
                files += self._drop_audio(job)
                self._store.save(job)
                swept += 1
        if swept:
            LOG.info("Recovered live audio deleted: %d session(s), %d"
                     " file(s)", swept, files)
        return swept

    def tick(self) -> int:
        """Move every due job one step on; return how many moved.

        Never raises: a job that breaks is logged (its id and the
        error's type) and the others go on.
        """
        moved = 0
        for session_id in self._store.ids():
            try:
                with self._lock:
                    job = self._store.load(session_id)
                    if job is None or not job.due(self._clock.now()):
                        continue
                    before = (job.state, job.tries, job.progress)
                    self._step(job)
                    self._store.save(job)
                    if (job.state, job.tries, job.progress) != before:
                        moved += 1
            except Exception as exc:  # noqa: BLE001 - the others go on
                LOG.warning("Live session %s: recovery step failed (%s)",
                            session_id, type(exc).__name__)
        return moved

    def _step(self, job: RecoveryJob) -> None:
        """Run the job's current step once."""
        now = self._clock.now()
        if job.expired(now):
            job.failed_step, job.state, job.error = job.state, "failed", \
                "expired"
            self._drop_audio(job)
            return
        step = getattr(self, f"_{job.state}")
        try:
            step(job)
        except TranscriberError as exc:
            self._failed(job, type(exc).__name__)
        except (OSError, ValueError, KeyError) as exc:
            self._failed(job, type(exc).__name__)

    def _failed(self, job: RecoveryJob, error: str) -> None:
        """Count a failed try; a job that ends deletes its audio."""
        step = job.state
        ended = job.failed_try(error, self._clock.now())
        LOG.warning("Live session %s: recovery step %s failed (%s)%s",
                    job.session_id, step, error,
                    "; given up" if ended else "")
        if ended:
            self._drop_audio(job)
            self._forget_texts(job)
        elif step == "transcribing":
            job.state = "queued"  # transcribed again from the chunks

    def _record(self, job: RecoveryJob) -> Dict[str, Any]:
        """The live session's record (``ValueError`` without one)."""
        record = self._log.read_session(job.session_id)
        if record is None:
            raise ValueError("no live record")
        return record

    def _queued(self, job: RecoveryJob) -> None:
        """Send the parts the live transcript lacks to the batch pipeline."""
        if self._running(job.session_id):
            return  # the live session is still on: wait
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
            self._drop_audio(job)
            job.move("merging", self._clock.now())
            return
        request = record.get("request") or {}
        language = request.get("language") or "auto"
        diarized = bool(request.get("diarize"))
        parts: List[Optional[int]] = [None] if job.channels == 1 else [0, 1]
        for channel in parts:
            speakers: Optional[int]
            expected: Optional[List[str]]
            wav = self._store.wav(job.session_id, job.chunks, spans,
                                  job.channels, channel)
            if channel is not None or not diarized:
                speakers, expected = 1, []
            else:
                enrolled = request.get("enrolled")
                speakers = 2 if request.get("parties") else None
                expected = list(enrolled) if enrolled is not None else None
            self._transcriber.submit(
                part_id(job.session_id, channel), wav, language, speakers,
                expected)
        job.meta["parts"] = parts
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
        """Follow the parts; once all are done, delete the audio."""
        parts: List[Optional[int]] = list(job.meta.get("parts") or [None])
        states = [self._transcriber.status(part_id(job.session_id, c))
                  for c in parts]
        if any(state in ("failed", "lost") for state, _ in states):
            raise ValueError("transcription failed")
        job.progress = sum(f if s != "done" else 1.0 for s, f in states) \
            / len(states)
        if any(state != "done" for state, _ in states):
            return
        for n, channel in enumerate(parts):
            document = self._transcriber.transcript(
                part_id(job.session_id, channel))
            if document is None:
                raise ValueError("no transcript")
            self._store.save_text(job.session_id, f"part-{n}", document)
        self._drop_audio(job)
        job.progress = 1.0
        job.move("merging", self._clock.now())

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
        start = spans[0][0] if spans else 0
        parts: List[RecoveredPart] = []
        for n, channel in enumerate(job.meta.get("parts") or []):
            document = self._store.read_text(job.session_id, f"part-{n}")
            if document is None:
                raise ValueError("a recovered part is missing")
            batch = Transcript.from_contract(document)
            names = speaker_map(
                batch.utterances, batch.speakers, speakers, job.channels,
                bool(request.get("diarize")), channel, pair)
            parts.append(RecoveredPart(batch, start, names))
        merged = merge_recovered(
            Transcript.from_contract(live), parts, spans,
            {s.id: s.role for s in speakers})
        self._log.save_transcript(job.session_id,
                                  merged.transcript.to_contract())
        job.seconds = merged.seconds
        job.added = merged.added
        LOG.info("Live session %s: %d recovered sentence(s) merged, %d"
                 " dropped, %.0f s recovered", job.session_id, merged.added,
                 merged.dropped, merged.seconds)
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
        self._forget_texts(job)
        self._drop_audio(job)

    def _drop_audio(self, job: RecoveryJob) -> int:
        """Delete the job's audio (chunks and the batch copies)."""
        files = self._store.delete_audio(job.session_id)
        for channel in (None, 0, 1):
            self._transcriber.forget(part_id(job.session_id, channel))
        if not job.audio_deleted:
            LOG.info("Live session %s: recovered audio deleted (%d"
                     " file(s))", job.session_id, files)
        job.audio_deleted = True
        return files

    def _forget_texts(self, job: RecoveryJob) -> None:
        """Delete the job's working documents (the state stays)."""
        self._store.delete_texts(job.session_id)
