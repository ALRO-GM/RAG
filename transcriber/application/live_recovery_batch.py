"""The batch pipeline over the recovery's own repository (#265).

:class:`BatchRecoveryTranscriber` implements
:class:`~transcriber.application.ports.RecoveryTranscriber` the way the
knowledge library's transcriptions do (KN-3a): a session repository of
its own (``<data_dir>/live-recovery-batch``), never the one «Sesiones»,
the player, publishing and the analysis read; the service's job runner
and models, so the batch GPU lease is honoured (the job waits while a
live session holds the GPU). A part's audio there is deleted with
:meth:`forget`, which the recovery calls as soon as the transcription
succeeded, when the job ends and when it is discarded.
"""

from typing import Any, Dict, List, Optional, Tuple

from transcriber.application.ports import (
    JobRunner,
    RecordedAudio,
    SessionLocks,
    SessionRepository,
)
from transcriber.application.record import ImportRecording
from transcriber.application.transcribe import (
    QueueTranscription,
    TranscriptionOptions,
)
from transcriber.domain.models import SessionState


class BatchRecoveryTranscriber:
    """Transcribe recovered audio with the batch pipeline.

    Args:
        sessions: The recovery's own repository.
        locks: Its per-id locks.
        importer: Stores a WAV there as an ``uploaded`` session.
        queue: Queues a session of that repository in the job runner.
        runner: The job runner (its status of a part's job).
    """

    def __init__(
        self,
        sessions: SessionRepository,
        locks: SessionLocks,
        importer: ImportRecording,
        queue: QueueTranscription,
        runner: JobRunner,
    ) -> None:
        """Wire the use cases of the recovery's repository."""
        self._sessions = sessions
        self._locks = locks
        self._import = importer
        self._queue = queue
        self._runner = runner

    def submit(
        self, part_id: str, wav: bytes, language: str,
        num_speakers: Optional[int], expected: Optional[List[str]],
    ) -> None:
        """Store ``wav`` as part ``part_id`` and queue it (replacing)."""
        self.forget(part_id)
        self._import(part_id, language, RecordedAudio(wav, "wav"))
        self._queue(part_id, TranscriptionOptions(
            num_speakers=num_speakers, expected_speakers=expected,
        ))

    def status(self, part_id: str) -> Tuple[str, float]:
        """Where the part is (``lost``: no job here for it)."""
        session = self._sessions.get(part_id)
        if session is None:
            return "lost", 0.0
        if session.state is SessionState.DONE:
            return "done", 1.0
        if session.state is SessionState.FAILED:
            return "failed", 0.0
        job = self._runner.status(part_id)
        if job is None:
            return "lost", 0.0
        state = "running" if session.state is SessionState.RUNNING \
            else "queued"
        return state, job.fraction

    def transcript(self, part_id: str) -> Optional[Dict[str, Any]]:
        """Return the part's ``transcript.v1`` once done."""
        stored = self._sessions.transcript(part_id)
        return stored.to_contract() if stored is not None else None

    def forget(self, part_id: str) -> None:
        """Delete the part and its audio (absent: nothing)."""
        with self._locks.hold(part_id):
            if self._sessions.get(part_id) is not None:
                self._sessions.delete(part_id)
