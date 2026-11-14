"""Session locks in threads: the races the use cases must not lose.

The repositories below make two threads meet right after reading the
session (a barrier), which is where an unlocked read-then-write would
interleave. With the locks, the second thread cannot read until the
first has written, the barrier times out, and the outcome is one of
the serial ones.
"""

import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional

import pytest

from tests.integration.transcriber.audio import sound_file, tone
from transcriber.adapters.outbound.av_audio_decoder import AvAudioDecoder
from transcriber.adapters.outbound.fake_models import (
    FakeSpeechRecogniser,
    FakeVoiceEncoder,
)
from transcriber.adapters.outbound.fs_session_repository import (
    FileSystemSessionRepository,
)
from transcriber.adapters.outbound.fs_voiceprint_repository import (
    FileSystemVoiceprintRepository,
)
from transcriber.adapters.outbound.thread_session_locks import (
    ThreadSessionLocks,
)
from transcriber.application.ports import RecordedAudio
from transcriber.application.record import (
    AppendAudio,
    DeleteSession,
    ImportRecording,
    OpenSession,
)
from transcriber.application.transcribe import TranscribeSession
from transcriber.bootstrap import start_container
from transcriber.config import TranscriberSettings
from transcriber.domain.errors import StateError, TranscriberError
from transcriber.domain.models import Session, SessionState, Transcript

MEET = 0.5  # seconds a thread waits for the other one after reading
WAIT = 10.0


class MeetingRepository(FileSystemSessionRepository):
    """Sessions on disk whose first two reads wait for each other."""

    def __init__(self, root: Path) -> None:
        """Arm the barrier."""
        super().__init__(root)
        self.barrier = threading.Barrier(2, timeout=MEET)
        self.armed = False

    def get(self, session_id: str) -> Optional[Session]:
        """Read the session, then wait for the other thread to read."""
        found = super().get(session_id)
        if self.armed:
            try:
                self.barrier.wait()
            except threading.BrokenBarrierError:
                pass
        return found


def in_threads(*calls: "Call") -> List["Call"]:
    """Run ``calls`` at once in threads; return them when all ended."""
    threads = [threading.Thread(target=call) for call in calls]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(WAIT)
    return list(calls)


class Call:
    """A call run in a thread, keeping its result or its error."""

    def __init__(self, function: Callable[..., object], *args: object) -> None:
        """Remember what to call."""
        self.function = function
        self.args = args
        self.result: object = None
        self.error: Optional[BaseException] = None

    def __call__(self) -> None:
        """Call it, keeping the outcome."""
        try:
            self.result = self.function(*self.args)
        except Exception as exc:
            self.error = exc


class FixedClock:
    """A clock for opening sessions."""

    def now(self) -> datetime:
        """Return a fixed aware time."""
        return datetime(2026, 11, 14, tzinfo=timezone.utc)


class TestThreadSessionLocks:
    def test_one_thread_at_a_time_per_session(self) -> None:
        locks = ThreadSessionLocks()
        inside = 0
        most = 0
        guard = threading.Lock()

        def work() -> None:
            nonlocal inside, most
            with locks.hold("s1"):
                with guard:
                    inside += 1
                    most = max(most, inside)
                threading.Event().wait(0.01)
                with guard:
                    inside -= 1

        threads = [threading.Thread(target=work) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(WAIT)
        assert most == 1
        assert locks.active() == 0

    def test_other_sessions_do_not_wait(self) -> None:
        locks = ThreadSessionLocks()
        entered = threading.Event()
        with locks.hold("a"):
            def other() -> None:
                with locks.hold("b"):
                    entered.set()

            thread = threading.Thread(target=other)
            thread.start()
            assert entered.wait(WAIT)
            thread.join(WAIT)
            assert locks.active() == 1

    def test_the_holder_may_take_it_again(self) -> None:
        locks = ThreadSessionLocks()
        with locks.hold("s1"):
            with locks.hold("s1"):
                assert locks.active() == 1
        assert locks.active() == 0

    def test_the_lock_is_released_on_error(self) -> None:
        locks = ThreadSessionLocks()
        with pytest.raises(RuntimeError):
            with locks.hold("s1"):
                raise RuntimeError("boom")
        assert locks.active() == 0


class TestRaces:
    def test_two_uploads_of_one_chunk_with_other_bytes(
        self, tmp_path: Path
    ) -> None:
        sessions = MeetingRepository(tmp_path)
        locks = ThreadSessionLocks()
        OpenSession(sessions, FixedClock(), locks)("s1", "es")
        append = AppendAudio(sessions, locks)
        sessions.armed = True
        first, second = in_threads(
            Call(append, "s1", 0, b"\x01\x00"),
            Call(append, "s1", 0, b"\x02\x00"),
        )
        outcomes = sorted([
            "ok" if c.error is None else type(c.error).__name__
            for c in (first, second)
        ])
        assert outcomes == ["InvalidInputError", "ok"]
        winner = first if first.error is None else second
        session = sessions.get("s1")
        assert session is not None and session.chunks == 1
        assert sessions.recording("s1").data == winner.args[2]

    def test_a_delete_and_the_job_start_do_not_interleave(
        self, tmp_path: Path
    ) -> None:
        sessions = MeetingRepository(tmp_path)
        locks = ThreadSessionLocks()
        wav = sound_file(tone(3.0), 16000, "wav")
        ImportRecording(sessions, locks)(
            "s1", "es", RecordedAudio(wav, "wav")
        )
        stored = sessions.get("s1")
        assert stored is not None
        sessions.save(stored.model_copy(update={
            "state": SessionState.QUEUED, "run_id": "a" * 32,
        }))
        transcribe = TranscribeSession(
            sessions, FileSystemVoiceprintRepository(tmp_path, "fake"),
            AvAudioDecoder(), FakeSpeechRecogniser(), FakeVoiceEncoder(),
            locks,
        )
        sessions.armed = True
        delete, job = in_threads(
            Call(DeleteSession(sessions, locks), "s1"),
            Call(transcribe, "s1", None, None, "a" * 32),
        )
        assert job.error is None, job.error
        if delete.error is None:  # deleted first: the job did nothing
            assert job.result is None
            assert sessions.get("s1") is None
            assert not (tmp_path / "sessions" / "s1").exists()
        else:  # the job started first: the delete was refused
            assert isinstance(delete.error, StateError)
            assert isinstance(job.result, Transcript)
            done = sessions.get("s1")
            assert done is not None and done.state is SessionState.DONE


def test_start_container_fails_interrupted_sessions(tmp_path: Path) -> None:
    settings = TranscriberSettings(data_dir=tmp_path, runtime="fake")
    sessions = FileSystemSessionRepository(tmp_path)
    for session_id, state in (
        ("queued", SessionState.QUEUED), ("running", SessionState.RUNNING),
        ("uploaded", SessionState.UPLOADED),
    ):
        sessions.create(Session(id=session_id, language="es", state=state))
    container = start_container(settings)
    states = {
        sid: container.sessions.get(sid) for sid in container.sessions.ids()
    }
    assert {sid: (s.state, s.failure) for sid, s in states.items()
            if s is not None} == {
        "queued": (SessionState.FAILED, "interrupted"),
        "running": (SessionState.FAILED, "interrupted"),
        "uploaded": (SessionState.UPLOADED, None),
    }
    container.delete_session("running")
    assert container.sessions.get("running") is None
    with pytest.raises(TranscriberError):
        container.delete_session("running")
