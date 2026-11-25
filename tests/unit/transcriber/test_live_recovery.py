"""The recovery job of a live session's lost audio, with fakes (#265).

The upload (resumable, idempotent, bounded), the job's steps from
``queued`` to ``done`` (waiting while the live session is on, retries
with backoff and a cap, a restart that lost the batch job), the audio
deleted as soon as its transcription succeeds, expiry and discard.
"""

from typing import Any, Dict, List

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


def record(**more: Any) -> Dict[str, Any]:
    """The live session's record as #264 writes it, with #265's counts."""
    return {
        "schema": "live.session.record", "session_id": "iv-1",
        "state": "ended",
        "request": {"session_id": "iv-1", "channels": 1, "language": "es",
                    "handoff": {"transcript_v1": True,
                                "run_post_call": True}},
        "speakers": [{"id": "S1", "channel": 0, "role": "unknown"}],
        "channels": 1, "received_samples": 10 * S,
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
        self.recovery = LiveRecovery(
            self.store, self.log, self.batch, self.clock, settings,
            running=lambda sid: sid in self.on,
            publisher=parts.get("publisher", self.publisher),
            analysis=self.analysis,
            rebuilder=parts.get("rebuilder", self.rebuilder),
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
    assert view["minutes"] == round(4 / 60, 1) and view["seconds"] == 4.0
    assert view["audio_deleted"] and view["progress"] == 1.0
    assert world.store.chunks == {} and world.store.texts == {}
    assert world.batch.submitted == [(part_id("iv-1", None), "es", 1, [])]
    assert world.store.wavs == [("iv-1", [(4 * S, 8 * S)], None)]


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


def test_a_restart_that_lost_the_batch_job_submits_it_again() -> None:
    world = World()
    world.upload()
    world.batch.plan = [("running", 0.4)]
    world.recovery.tick()
    world.recovery.tick()
    assert world.recovery.status("iv-1") == {
        **world.recovery.status("iv-1"), "state": "transcribing",
        "progress": 0.4}
    world.batch.parts.clear()  # the process stopped: no job any more
    world.recovery.tick()
    assert world.recovery.status("iv-1")["state"] == "queued"
    world.clock.advance(30)
    assert world.run() == "done"


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


def test_a_diarized_two_party_session_asks_for_its_people() -> None:
    world = World()
    world.log.sessions["iv-1"] = record(
        request={"session_id": "iv-1", "channels": 1, "diarize": True,
                 "enrolled": ["emp"], "language": None,
                 "parties": {"employee": "emp", "other": "cand"},
                 "handoff": {"transcript_v1": True}},
        speakers=[{"id": "emp", "channel": 0, "role": "interviewer",
                   "enrolled": True},
                  {"id": "cand", "channel": 0, "role": "candidate"}])
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


def test_a_later_expiry_from_the_page_is_ignored_and_bad_ones_refused(
) -> None:
    world = World()
    world.recovery.put_chunk("iv-1", 0, 4 * S, SECOND)
    with pytest.raises(InvalidInputError):
        world.recovery.close("iv-1", 1, expires_at="tomorrow")
    view = world.recovery.close("iv-1", 1,
                                expires_at="2030-01-01T00:00:00+00:00")
    assert view["expires_at"] == "2026-11-27T10:00:00+00:00"


def test_a_job_whose_session_lost_its_transcript_fails_quietly() -> None:
    world = World()
    world.upload()
    del world.log.transcripts["iv-1"]
    for _ in range(10):
        world.recovery.tick()
        world.clock.advance(600)
    assert world.recovery.status("iv-1")["failed_step"] == "queued"


def test_an_unreadable_job_is_swept_and_ids_are_checked() -> None:
    world = World()
    world.store.chunks["ghost"] = {0: SECOND}
    assert world.recovery.sweep() == 1 and world.store.ids() == []
    with pytest.raises(InvalidInputError):
        check_session("a/b")
    assert part_id("iv-1", 0).endswith("-c0")
    assert part_id("x" * 64, None) == part_id("x" * 64, None)


def test_without_counts_the_stream_end_comes_from_the_events() -> None:
    """A record written before #265 has no counts: what came after the
    last event is the part recovered."""
    world = World()
    stored = record()
    del stored["received_samples"]
    stored["silences"] = []
    world.log.sessions["iv-1"] = stored
    world.log.events["iv-1"] = [{"type": "final", "stream_time": 6.0}]
    world.upload()
    assert world.run() == "done"
    assert world.store.wavs[0][1] == [(6 * S, 8 * S)]
