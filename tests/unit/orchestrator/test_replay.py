"""Replay: transcript.v1 to live.event.v1, timing, summary."""

import json
from typing import List

import jsonschema
import pytest

from orchestrator.adapters.outbound.fakes import FakeKnowledge, FakeReasoner
from orchestrator.adapters.outbound.jsonl_sink import MemorySink
from orchestrator.adapters.outbound.runtime import InlineExecutor
from orchestrator.application.live_assist import LiveAssist
from orchestrator.application.ports import CancelToken
from orchestrator.application.replay import (
    STOP_CHECK,
    Replay,
    transcript_to_events,
)
from orchestrator.application.summary import percentile, summarise
from orchestrator.domain.events import Transcript
from tests.unit.orchestrator.fixtures import REPO, product, transcript

ROLES = {"a": "agent", "c": "customer"}
SCHEMA = json.loads(
    (REPO / "contracts" / "live.event.v1.schema.json").read_text("utf-8")
)


class FakeClock:
    def __init__(self) -> None:
        self.t = 100.0
        self.slept: List[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.t += seconds


def test_events_validate_against_the_contract() -> None:
    events = list(transcript_to_events(transcript(), ROLES))
    for event in events:
        jsonschema.validate(event.to_contract(), SCHEMA)
    assert [e.type for e in events][0] == "session_started"
    assert [e.type for e in events][-1] == "session_ended"
    assert [e.seq for e in events] == list(range(len(events)))


def test_turn_end_when_the_speaker_changes() -> None:
    types = [e.type for e in transcript_to_events(transcript(), ROLES)]
    assert types == [
        "session_started", "final", "turn_end", "final", "final",
        "turn_end", "final", "turn_end", "final", "turn_end",
        "session_ended",
    ]


def test_finals_are_stamped_at_sentence_end() -> None:
    finals = [e for e in transcript_to_events(transcript(), ROLES)
              if e.type == "final"]
    assert finals[0].stream_time == 2.0
    assert finals[0].at == "2026-11-21T09:00:02.000Z"
    assert finals[0].utterance is not None
    assert finals[0].utterance.language == "es"


def test_session_id_is_made_valid() -> None:
    first = next(iter(transcript_to_events(transcript(), ROLES)))
    assert first.session_id == "unit-session"
    assert first.source is not None and first.source.kind == "stereo_channels"


def test_more_than_two_speakers_is_mono() -> None:
    data = transcript().model_dump()
    data["speakers"].append({"id": "z", "name": None, "enrolled": None})
    first = next(iter(transcript_to_events(
        Transcript.model_validate(data), ROLES)))
    assert first.source is not None and first.source.kind == "mono_diarized"
    assert first.speakers is not None and first.speakers[0].channel is None


def test_empty_transcript_and_odd_dates() -> None:
    data = transcript().model_dump()
    data["utterances"] = []
    data["session"]["started_at"] = "garbage"
    events = list(transcript_to_events(Transcript.model_validate(data),
                                       ROLES))
    assert [e.type for e in events] == ["session_started", "session_ended"]
    data["session"]["started_at"] = "2026-01-01T00:00:00"
    data["session"]["id"] = "***"
    events = list(transcript_to_events(Transcript.model_validate(data),
                                       ROLES))
    assert events[0].at.startswith("2026-01-01T00:00:00")
    assert events[0].session_id == "replay"
    data["session"]["started_at"] = None
    first = next(iter(transcript_to_events(Transcript.model_validate(data),
                                           ROLES)))
    assert first.at.startswith("1970-01-01")


def make(speed: "float | None", clock: FakeClock) -> Replay:
    assist = LiveAssist(product(), FakeKnowledge(), FakeReasoner(),
                        MemorySink(), clock, InlineExecutor(), ROLES)
    return Replay(assist, clock, speed)


def test_real_time_waits_for_each_event() -> None:
    clock = FakeClock()
    sent = make(1.0, clock).run(transcript_to_events(transcript(), ROLES))
    assert sum(clock.slept) == pytest.approx(sent[-1].stream_time)
    # The first final ends at 2.0 s, waited in steps that check the stop.
    assert sum(clock.slept[:8]) == pytest.approx(2.0)
    assert max(clock.slept) <= STOP_CHECK


def test_a_stopped_replay_ends_early() -> None:
    clock = FakeClock()
    stop = CancelToken()
    replay = make(1.0, clock)
    replay.stop = stop

    def stopping(seconds: float) -> None:
        clock.t += seconds
        if clock.t > 105.0:
            stop.cancel()

    clock.sleep = stopping  # type: ignore[method-assign]
    sent = replay.run(transcript_to_events(transcript(), ROLES))
    assert sent and sent[-1].stream_time < 6.0


def test_speed_scales_the_waits() -> None:
    clock = FakeClock()
    make(4.0, clock).run(transcript_to_events(transcript(), ROLES))
    assert sum(clock.slept) == pytest.approx(13.0 / 4.0)


def test_max_speed_never_waits() -> None:
    clock = FakeClock()
    make(None, clock).run(transcript_to_events(transcript(), ROLES))
    assert clock.slept == []


def test_speed_must_be_positive() -> None:
    with pytest.raises(ValueError):
        make(0, FakeClock())


def test_summary_counts_and_percentiles() -> None:
    clock = FakeClock()
    replay = make(None, clock)
    replay.run(transcript_to_events(transcript(), ROLES))
    assist = replay.assist
    assert isinstance(assist, LiveAssist)
    sink = assist.sink
    assert isinstance(sink, MemorySink)
    summary = summarise(sink.records, assist.skips)
    assert summary["calls"] == 2 and summary["by_status"] == {"ok": 2}
    assert set(summary["stages"]) == {
        "knowledge_ms", "reasoner_ms", "total_ms"}
    empty = summarise([], {})
    assert empty["calls"] == 0


def test_percentile() -> None:
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    assert percentile([5], 0.9) == 5
    with pytest.raises(ValueError):
        percentile([], 0.5)
