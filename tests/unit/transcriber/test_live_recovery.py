"""The recovery job of a live session's lost audio, with fakes (#265).

The upload (resumable, idempotent, bounded in size and in time, only for
an ended session whose product allows it), the job's steps from
``queued`` to ``done`` (waiting while the live session is on, retries
with backoff and a cap, restarts that lose the batch job without
spending a try, a crash at every step boundary), the audio deleted as
soon as its transcription succeeds and only after the next state is
written, expiry, discard (also during a step) and the services guard.
"""

import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytest

from tests.unit.transcriber.live_fakes import (
    MemoryLiveLog,
    RecordingAnalysis,
    RecordingPublisher,
)
from tests.unit.transcriber.recovery_fakes import (
    Clock,
    MemoryRecoveryStore,
    RecordingRebuilder,
    ScriptedTranscriber,
    transcript,
)
from transcriber.application.live_recovery import (
    LiveRecovery,
    check_session,
    part_id,
)
from transcriber.domain.errors import (
    InvalidInputError,
    NotFoundError,
    PublishError,
    StateError,
    TooLargeError,
)
from transcriber.domain.live_recovery import RecoverySettings
from transcriber.domain.models import Speaker

S = 16000
SECOND = b"\x01\x00" * S  # one second of invented mono PCM
PART = part_id("iv-1", None, 0)


def record(**more: Any) -> Dict[str, Any]:
    """The live session's record as #264 writes it, with #265's facts."""
    return {
        "schema": "live.session.record", "session_id": "iv-1",
        "state": "ended",
        "request": {"session_id": "iv-1", "channels": 1, "language": "es",
                    "buffer_audio": True, "buffer_max_hours": 48,
                    "handoff": {"transcript_v1": True,
                                "run_post_call": True}},
        "speakers": [{"id": "S1", "channel": 0, "role": "unknown"}],
        "channels": 1, "received_samples": 10 * S,
        "started_at": "2026-11-26T09:59:50+00:00",
        "ended_at": "2026-11-26T10:00:00+00:00",
        "silences": [[4 * S, 8 * S]], **more,
    }


class World:
    """The service over fakes, with a live session ``iv-1`` that ended."""

    def __init__(self, settings: RecoverySettings = RecoverySettings(),
                 **parts: Any) -> None:
        self.log = MemoryLiveLog()
        self.log.sessions["iv-1"] = record()
        self.log.transcripts["iv-1"] = transcript(
            [("S1", 0.5, 3.5, "antes del corte"),
             ("S1", 8.5, 9.5, "después")], duration=10).to_contract()
        self.store = MemoryRecoveryStore()
        self.batch = ScriptedTranscriber({None: transcript(
            [("S1", 0.5, 3.0, "lo que se perdió")], session_id="rc")})
        self.clock = Clock()
        self.publisher = RecordingPublisher()
        self.analysis = RecordingAnalysis()
        self.rebuilder = RecordingRebuilder()
        self.on: List[str] = []
        self.busy: List[Tuple[str, bool]] = []
        self.settings = settings
        self.parts = parts
        self.recovery = self.service()

    def service(self) -> LiveRecovery:
        """A service over the same store, log and batch (a new process)."""
        return LiveRecovery(
            self.store, self.log, self.batch, self.clock, self.settings,
            running=lambda sid: sid in self.on,
            publisher=self.parts.get("publisher", self.publisher),
            analysis=self.analysis,
            rebuilder=self.parts.get("rebuilder", self.rebuilder),
            busy=lambda sid, on: self.busy.append((sid, on)),
        )

    def upload(self, seconds: int = 4, at: int = 4 * S) -> None:
        for n in range(seconds):
            self.recovery.put_chunk("iv-1", n, at + n * S, SECOND)
        self.recovery.close("iv-1", seconds)

    def run(self, rounds: int = 10) -> str:
        for _ in range(rounds):
            self.recovery.tick()
        return str(self.recovery.status("iv-1")["state"])


def test_the_whole_job_fills_the_gap_and_recomputes_everything() -> None:
    world = World()
    world.upload()
    assert world.recovery.status("iv-1")["state"] == "queued"
    assert world.run() == "done"
    merged = world.log.transcripts["iv-1"]
    texts = [u["text"] for u in merged["utterances"]]
    assert texts == ["antes del corte", "lo que se perdió", "después"]
    assert merged["utterances"][1]["start"] == 4.5
    assert world.publisher.documents == [merged]
    assert world.analysis.started == ["iv-1"]
    assert world.rebuilder.asked == [("iv-1", merged)]
    view = world.recovery.status("iv-1")
    assert view["seconds"] == 2.5, "speech merged, not the gap's length"
    assert view["audio_deleted"] and view["progress"] == 1.0
    assert not view["unidentified"]
    assert world.store.chunks == {} and world.store.texts == {}
    assert world.batch.submitted == [(PART, "es", 1, [])]
    assert world.store.wavs == [("iv-1", (4 * S, 8 * S), None)]


def test_the_audio_goes_only_after_the_next_state_is_written() -> None:
    """The texts and ``merging`` are written before any audio goes."""
    world = World()
    world.upload()
    seen: List[Tuple[str, bool]] = []
    drop = world.store.delete_audio

    def delete_audio(session_id: str) -> int:
        job = world.store.jobs.get(session_id) or {}
        seen.append((str(job.get("state")), bool(job.get("audio_deleted"))))
        assert world.store.texts.get(session_id, {}).get("part-0")
        return drop(session_id)

    world.store.delete_audio = delete_audio  # type: ignore[method-assign]
    world.recovery.tick()
    world.recovery.tick()
    assert seen[0] == ("merging", True)


class Crash(BaseException):
    """The process stops here (not an error a step catches)."""


def crashing(target: Any, names: List[str], after: List[int]) -> None:
    """Make every call to ``names`` of ``target`` count down ``after``;
    at zero the call raises :class:`Crash` *after* doing its work."""
    for name in names:
        real = getattr(target, name)

        def wrapped(*args: Any, _real: Callable[..., Any] = real,
                    **kwargs: Any) -> Any:
            result = _real(*args, **kwargs)
            after[0] -= 1
            if after[0] == 0:
                raise Crash()
            return result

        setattr(target, name, wrapped)


def run_with_crash(at: int) -> Tuple[World, bool]:
    """Run the whole job with a crash after the ``at``-th write, then a
    restart (a new service, batch jobs not done lost) to the end."""
    world = World()
    world.upload()
    counter = [at]
    crashing(world.store, ["save", "save_text", "delete_audio",
                           "delete_texts"], counter)
    crashing(world.log, ["save_transcript"], counter)
    crashing(world.batch, ["submit", "forget"], counter)
    crashed = False
    try:
        world.run(12)
    except Crash:
        crashed = True
    counter[0] = -1  # no more crashes
    world.batch.restart()
    world.recovery = world.service()
    world.recovery.sweep()
    for _ in range(12):
        world.recovery.tick()
        world.clock.advance(600)
    return world, crashed


def test_a_crash_at_any_step_boundary_never_loses_the_recovered_speech(
) -> None:
    reference = World()
    reference.upload()
    reference.run()
    expected = reference.log.transcripts["iv-1"]
    crashes = 0
    for at in range(1, 30):
        world, crashed = run_with_crash(at)
        crashes += crashed
        view = world.recovery.status("iv-1")
        assert view["state"] == "done", (at, view)
        assert world.log.transcripts["iv-1"] == expected, at
        assert view["seconds"] == 2.5 and view["added"] == 1, at
        assert world.store.chunks == {} and not world.batch.parts, at
    assert crashes == 18, "every write of the job is a crash point"


def test_audio_gone_without_its_texts_fails_honestly() -> None:
    world = World()
    world.upload()
    world.recovery.tick()  # submitted
    world.store.chunks.clear()  # a crash deleted the audio, not the texts
    world.batch.restart()
    world.recovery.tick()  # lost: queued again, no try spent
    assert world.recovery.status("iv-1")["state"] == "queued"
    assert world.run(2) == "failed"
    view = world.recovery.status("iv-1")
    assert view["error"] == "audio_lost" and view["seconds"] == 0
    assert world.publisher.documents == []


def test_restarts_during_a_transcription_spend_no_try() -> None:
    """Five deploys in the middle of a transcription: it is submitted
    again each time and finally completes; nothing is deleted early."""
    world = World()
    world.upload()
    for _ in range(5):
        world.batch.plan = [("running", 0.3)]
        world.recovery.tick()  # submitted (or submitted again)
        world.recovery.tick()  # running
        world.batch.restart()
        world.recovery = world.service()
        world.recovery.tick()  # lost: back to queued
        view = world.recovery.status("iv-1")
        assert view["state"] == "queued" and view["error"] is None
        assert world.store.chunks["iv-1"]
    assert world.run() == "done"
    assert len(world.batch.submitted) == 6


def test_the_guard_hears_of_a_recovery_only_while_it_holds_the_gpu(
) -> None:
    world = World()
    world.upload()
    world.batch.plan = [("queued", 0.0), ("running", 0.5)]
    world.recovery.tick()  # submitted
    world.recovery.tick()  # queued behind a live session: not busy
    world.recovery.tick()  # running: busy
    world.recovery.tick()  # done: free
    assert world.busy == [("iv-1", False), ("iv-1", True), ("iv-1", False),
                          ("iv-1", False)]


def test_absurd_offsets_are_refused_before_anything_is_kept() -> None:
    world = World()
    with pytest.raises(InvalidInputError):
        world.recovery.put_chunk("iv-1", 0, 2 ** 33, b"\x00" * 20)
    with pytest.raises(InvalidInputError):
        # past the session's ten seconds plus the slack
        world.recovery.put_chunk("iv-1", 0, 700 * S, SECOND)
    with pytest.raises(InvalidInputError):
        world.recovery.put_chunk("iv-1", 0, -1, SECOND)
    world.recovery.put_chunk("iv-1", 0, 600 * S, SECOND)  # inside the slack
    assert world.store.audio_bytes() == len(SECOND)


def test_parts_never_span_a_long_gap() -> None:
    world = World()
    world.log.sessions["iv-1"] = record(
        silences=[[1 * S, 3 * S], [100 * S, 102 * S]],
        received_samples=200 * S, ended_at="2026-11-26T10:03:20+00:00")
    world.recovery.put_chunk("iv-1", 0, 1 * S, SECOND)
    world.recovery.put_chunk("iv-1", 1, 100 * S, SECOND)
    world.recovery.close("iv-1", 2)
    world.recovery.tick()
    assert [w[1] for w in world.store.wavs] == [(1 * S, 2 * S),
                                                (100 * S, 101 * S)]


def test_any_unexpected_error_is_a_counted_try() -> None:
    world = World()
    world.upload()

    def no_memory(*args: Any) -> bytes:
        raise MemoryError()

    world.store.wav = no_memory  # type: ignore[method-assign,assignment]
    for _ in range(6):
        world.recovery.tick()
        world.clock.advance(600)
    view = world.recovery.status("iv-1")
    assert view["state"] == "failed" and view["error"] == "MemoryError"
    assert "iv-1" not in world.store.chunks


def test_the_audio_goes_as_soon_as_its_transcription_succeeds() -> None:
    world = World(rebuilder=RecordingRebuilder(PublishError("down")))
    world.upload()
    world.recovery.tick()  # queued → transcribing
    assert world.store.chunks["iv-1"]
    world.recovery.tick()  # transcribed: the audio is deleted at once
    assert "iv-1" not in world.store.chunks
    assert world.batch.forgotten  # the batch copy too
    assert world.run(4) == "rejudging"
    assert world.recovery.status("iv-1")["error"] == "PublishError"


def test_the_upload_is_resumable_idempotent_and_bounded() -> None:
    world = World(RecoverySettings(chunk_max_bytes=4 * S,
                                   session_max_bytes=6 * S,
                                   total_max_bytes=8 * S, max_chunks=3))
    recovery = world.recovery
    recovery.put_chunk("iv-1", 0, 4 * S, SECOND)
    view = recovery.put_chunk("iv-1", 0, 4 * S, SECOND)  # sent again
    assert view["chunks"] == [0] and view["state"] == "uploading"
    with pytest.raises(StateError):
        recovery.close("iv-1", 2)  # one is missing: send it, close again
    recovery.put_chunk("iv-1", 1, 5 * S, SECOND)
    recovery.put_chunk("iv-1", 2, 6 * S, SECOND)
    with pytest.raises(TooLargeError):
        recovery.put_chunk("iv-1", 3, 7 * S, SECOND)  # over the session
    with pytest.raises(TooLargeError):
        recovery.put_chunk("iv-1", 4, 7 * S, b"\x00" * (4 * S + 2))
    with pytest.raises(InvalidInputError):
        recovery.put_chunk("iv-1", 3, 7 * S, b"\x00\x00\x00")  # odd bytes
    with pytest.raises(InvalidInputError):
        recovery.put_chunk("iv-1", 3, 7 * S, SECOND, channels=2)
    with pytest.raises(InvalidInputError):
        recovery.put_chunk("iv-1", -1, 0, SECOND)
    with pytest.raises(InvalidInputError):
        recovery.put_chunk("../x", 0, 0, SECOND)
    with pytest.raises(NotFoundError):
        recovery.put_chunk("never-ran", 0, 0, SECOND)
    assert recovery.close("iv-1", 3)["state"] == "queued"
    assert recovery.close("iv-1", 3)["state"] == "queued"  # again: same
    with pytest.raises(StateError):
        recovery.put_chunk("iv-1", 0, 4 * S, SECOND)  # closed


def test_a_session_still_on_or_without_the_products_leave_is_refused(
) -> None:
    world = World()
    world.on.append("iv-1")
    with pytest.raises(StateError):
        world.recovery.put_chunk("iv-1", 0, 4 * S, SECOND)
    world.on.clear()
    world.recovery.put_chunk("iv-1", 0, 4 * S, SECOND)
    world.on.append("iv-1")  # reattached meanwhile
    with pytest.raises(StateError):
        world.recovery.close("iv-1", 1)
    off = World()
    off.log.sessions["iv-1"] = record(request={
        "session_id": "iv-1", "handoff": {}})
    with pytest.raises(StateError):
        off.recovery.put_chunk("iv-1", 0, 4 * S, SECOND)
    assert off.store.chunks == {}


def test_another_session_cannot_overwrite_ones_audio_nor_the_total() -> None:
    world = World(RecoverySettings(total_max_bytes=3 * S))
    world.log.sessions["iv-2"] = record(session_id="iv-2")
    world.recovery.put_chunk("iv-1", 0, 0, SECOND)
    with pytest.raises(TooLargeError):
        world.recovery.put_chunk("iv-2", 0, 0, SECOND)  # no room left
    assert world.store.chunks["iv-1"][0] == SECOND
    with pytest.raises(StateError):
        World(RecoverySettings(enabled=False)).recovery.put_chunk(
            "iv-1", 0, 0, SECOND)


def test_the_job_waits_while_the_live_session_is_on() -> None:
    world = World()
    world.upload()
    world.on.append("iv-1")
    assert world.run(3) == "queued" and world.batch.submitted == []
    world.on.clear()
    assert world.run() == "done"


def test_a_failed_transcription_is_tried_again_after_a_wait() -> None:
    world = World()
    world.upload()
    world.batch.fail_next = 1
    world.recovery.tick()  # submitted
    world.recovery.tick()  # failed: back to queued, waits 30 s
    view = world.recovery.status("iv-1")
    assert view["state"] == "queued" and view["error"] == "ValueError"
    assert world.store.chunks["iv-1"]  # kept for the next try
    world.recovery.tick()
    assert len(world.batch.submitted) == 1  # not yet
    world.clock.advance(30)
    assert world.run() == "done"
    assert len(world.batch.submitted) == 2


def test_three_failures_give_up_and_delete_the_audio() -> None:
    world = World()
    world.upload()
    world.batch.fail_next = 99
    for _ in range(12):
        world.recovery.tick()
        world.clock.advance(600)
    view = world.recovery.status("iv-1")
    assert view["state"] == "failed" and view["failed_step"] == "transcribing"
    assert view["audio_deleted"] and "iv-1" not in world.store.chunks


def test_publishing_retries_with_backoff_then_fails() -> None:
    world = World(publisher=RecordingPublisher(PublishError("refused")))
    world.upload()
    for _ in range(20):
        world.recovery.tick()
        world.clock.advance(500)
    view = world.recovery.status("iv-1")
    assert view["state"] == "failed" and view["failed_step"] == "publishing"
    assert world.rebuilder.asked == []


def test_nothing_to_recover_when_the_server_heard_it_all() -> None:
    world = World()
    for n in range(2):
        world.recovery.put_chunk("iv-1", n, n * S, SECOND)  # heard live
    world.recovery.close("iv-1", 2)
    assert world.run() == "done"
    assert world.batch.submitted == [] and world.publisher.documents == []
    assert world.recovery.status("iv-1")["seconds"] == 0.0


def two_party(world: World) -> None:
    world.log.sessions["iv-1"] = record(
        request={"session_id": "iv-1", "channels": 1, "diarize": True,
                 "enrolled": ["emp"], "language": None, "buffer_audio": True,
                 "parties": {"employee": "emp", "other": "cand"},
                 "handoff": {"transcript_v1": True}},
        speakers=[{"id": "emp", "channel": 0, "role": "interviewer",
                   "enrolled": True},
                  {"id": "cand", "channel": 0, "role": "candidate"}])


def test_a_diarized_two_party_session_asks_for_its_people() -> None:
    world = World()
    two_party(world)
    world.batch.results[None] = transcript(
        [("emp", 0.2, 1.0, "pregunta"), ("S2", 1.5, 3.5, "contesta")],
        [Speaker(id="emp", enrolled=True)])
    world.upload()
    assert world.run() == "done"
    assert world.batch.submitted[0][1:] == ("auto", 2, ["emp"])
    speakers = [u["speaker"] for u in world.log.transcripts["iv-1"]
                ["utterances"]]
    assert speakers == ["S1", "emp", "cand", "S1"]
    assert world.analysis.started == []  # not asked by this session
    assert not world.recovery.status("iv-1")["unidentified"]


def test_voices_not_told_apart_are_said_so() -> None:
    world = World()
    two_party(world)
    world.batch.results[None] = transcript(
        [("S1", 0.2, 1.0, "pregunta"), ("S2", 1.5, 3.5, "contesta")])
    world.upload()
    assert world.run() == "done"
    view = world.recovery.status("iv-1")
    assert view["unidentified"] and view["added"] == 2


def test_a_stereo_upload_is_transcribed_per_channel() -> None:
    world = World()
    world.log.sessions["iv-1"] = record(
        channels=2, speakers=[{"id": "ana", "channel": 0,
                               "role": "interviewer"},
                              {"id": "cand", "channel": 1,
                               "role": "candidate"}])
    world.batch.results = {
        0: transcript([("S1", 0.2, 1.0, "pregunta")]),
        1: transcript([("S1", 1.5, 3.5, "respuesta")]),
    }
    for n in range(4):
        world.recovery.put_chunk("iv-1", n, (4 + n) * S, SECOND * 2,
                                 channels=2)
    world.recovery.close("iv-1", 4)
    assert world.run() == "done"
    said = [(u["speaker"], u["text"])
            for u in world.log.transcripts["iv-1"]["utterances"]]
    assert ("ana", "pregunta") in said and ("cand", "respuesta") in said


def test_expired_audio_is_swept_and_discard_deletes_at_once() -> None:
    world = World()
    world.recovery.put_chunk("iv-1", 0, 4 * S, SECOND)
    world.recovery.close("iv-1", 1, expires_at="2026-11-26T12:00:00+00:00")
    assert world.recovery.sweep() == 0
    world.clock.advance(2 * 3600)
    assert world.recovery.sweep() == 1
    view = world.recovery.status("iv-1")
    assert view["state"] == "failed" and view["error"] == "expired"
    assert "iv-1" not in world.store.chunks
    assert world.recovery.sweep() == 0  # nothing more to delete
    assert world.recovery.list()[0]["session_id"] == "iv-1"
    assert world.recovery.discard("iv-1")
    assert world.store.ids() == [] and not world.recovery.discard("iv-1")
    with pytest.raises(NotFoundError):
        world.recovery.status("iv-1")


def test_the_expiry_is_the_sessions_start_plus_the_products_limit(
) -> None:
    world = World()
    world.log.sessions["iv-1"] = record(request={
        **record()["request"], "buffer_max_hours": 1})
    view = world.recovery.put_chunk("iv-1", 0, 4 * S, SECOND)
    assert view["expires_at"] == "2026-11-26T10:59:50+00:00"
    with pytest.raises(InvalidInputError):
        world.recovery.close("iv-1", 1, expires_at="tomorrow")
    view = world.recovery.close("iv-1", 1,
                                expires_at="2030-01-01T00:00:00+00:00")
    assert view["expires_at"] == "2026-11-26T10:59:50+00:00"


def test_sweep_deletes_batch_parts_of_no_job_and_unreadable_jobs() -> None:
    world = World()
    world.batch.parts["rc-stray-0-m"] = None
    world.store.chunks["ghost"] = {0: SECOND}
    assert world.recovery.sweep() == 1 and world.store.ids() == []
    assert world.batch.parts == {}
    with pytest.raises(InvalidInputError):
        check_session("a/b")
    assert part_id("iv-1", 0).endswith("-0-c0")


def test_a_discard_during_a_step_wins() -> None:
    """«Descartar» while the job submits its audio: the step's result is
    dropped, nothing of the session is left, and the discard does not
    wait for the step."""
    world = World()
    world.upload()
    entered, release = threading.Event(), threading.Event()
    submit = world.batch.submit

    def slow_submit(*args: Any) -> None:
        submit(*args)
        entered.set()
        release.wait(5)

    world.batch.submit = slow_submit  # type: ignore[method-assign,assignment]
    stepping = threading.Thread(target=world.recovery.tick)
    stepping.start()
    assert entered.wait(5)
    assert world.recovery.discard("iv-1")  # does not block on the step
    release.set()
    stepping.join(5)
    assert world.store.ids() == [] and world.batch.parts == {}
    with pytest.raises(NotFoundError):
        world.recovery.status("iv-1")


def test_a_job_whose_session_lost_its_transcript_fails_quietly() -> None:
    world = World()
    world.upload()
    del world.log.transcripts["iv-1"]
    for _ in range(10):
        world.recovery.tick()
        world.clock.advance(600)
    assert world.recovery.status("iv-1")["failed_step"] == "queued"


def test_without_counts_the_stream_end_comes_from_the_events() -> None:
    """A record of a session cut by a crash has no end count: what came
    after the last event is the part recovered, and the silences written
    as they happened are recovered too."""
    world = World()
    stored = record(silences=[[1 * S, 2 * S]])
    del stored["received_samples"]
    world.log.sessions["iv-1"] = stored
    world.log.events["iv-1"] = [{"type": "final", "stream_time": 6.0}]
    for n in range(8):
        world.recovery.put_chunk("iv-1", n, n * S, SECOND)
    world.recovery.close("iv-1", 8)
    assert world.run() == "done"
    assert world.store.wavs[0][1] == (1 * S, 8 * S)


def held(log: Optional[MemoryLiveLog]) -> int:
    """How many records a log keeps (absent: none)."""
    return 0 if log is None else len(log.sessions)


def test_the_helpers_count() -> None:
    assert held(None) == 0 and held(MemoryLiveLog()) == 0
