"""Domain rules of live sessions: PCM messages, backlog, channels, GPU,
and the transcript assembled from the finals (LV-2, LV-5, LV-8, LV-11)."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import jsonschema
import numpy as np
import pytest
from pydantic import ValidationError

from tests.unit.transcriber.live_fakes import LIVE_SCHEMA, pcm
from transcriber.domain.errors import InvalidInputError
from transcriber.domain.live import LiveEventStream, LiveSpeaker, renamed
from transcriber.domain.live_session import (
    AudioBacklog,
    ChannelSpeaker,
    LiveSessionRequest,
    channel_speakers,
    decode_pcm,
    gpu_choice,
)
from transcriber.domain.live_transcript import last_finals, live_transcript
from transcriber.domain.models import Pipeline

AT = datetime(2026, 11, 21, 10, 0, tzinfo=timezone.utc)
PIPELINE = Pipeline(recogniser="r", voice_model="none", transcriber="t")


def test_mono_pcm_is_decoded_to_floats() -> None:
    samples = np.array([0.0, 0.5, -0.5, 0.25], dtype=np.float32)
    [decoded] = decode_pcm(pcm(samples), 1)
    assert decoded.dtype == np.float32
    np.testing.assert_allclose(decoded, samples, atol=1e-4)


def test_stereo_pcm_is_split_by_channel() -> None:
    left = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    right = np.array([-0.1, -0.2, -0.3], dtype=np.float32)
    a, b = decode_pcm(pcm(left, right), 2)
    np.testing.assert_allclose(a, left, atol=1e-4)
    np.testing.assert_allclose(b, right, atol=1e-4)


@pytest.mark.parametrize("data,channels", [
    (b"\x00\x00\x00", 1), (b"\x00\x00", 2), (b"\x00" * 32002, 1),
])
def test_broken_or_long_pcm_is_refused(data: bytes, channels: int) -> None:
    with pytest.raises(InvalidInputError):
        decode_pcm(data, channels)


def test_backlog_fills_and_warns_once_per_episode() -> None:
    backlog = AudioBacklog(limit=2.0, behind=1.0)
    backlog.add(8000)
    assert not backlog.full() and not backlog.warn()
    backlog.add(8000)
    assert backlog.warn() and not backlog.warn()
    backlog.add(16000)
    assert backlog.full()
    backlog.take(16000 * 2)
    assert backlog.seconds == 0.0 and not backlog.warn()
    backlog.add(16000)
    assert backlog.warn()


def test_channels_get_the_speakers_and_roles_of_the_request() -> None:
    request = LiveSessionRequest(session_id="call-1", channels=2, speakers=[
        ChannelSpeaker(channel=1, id="S2", role="candidate"),
        ChannelSpeaker(channel=0, id="S1", role="interviewer", name="Ana"),
    ])
    assert channel_speakers(request) == [
        LiveSpeaker("S1", 0, "interviewer", "Ana"),
        LiveSpeaker("S2", 1, "candidate", None),
    ]


def test_defaults_are_anonymous_and_mono_names_no_channel() -> None:
    assert channel_speakers(LiveSessionRequest(session_id="a")) == [
        LiveSpeaker("S1", None, "unknown")
    ]
    mono = LiveSessionRequest(session_id="a", speakers=[
        ChannelSpeaker(channel=0, id="user", role="user")
    ])
    assert channel_speakers(mono) == [LiveSpeaker("user", None, "user")]
    assert [s.id for s in channel_speakers(
        LiveSessionRequest(session_id="a", channels=2)
    )] == ["S1", "S2"]


@pytest.mark.parametrize("body", [
    {"session_id": "a", "channels": 2,
     "speakers": [{"channel": 0, "id": "S1"}]},
    {"session_id": "a", "channels": 2, "speakers": [
        {"channel": 0, "id": "S1"}, {"channel": 1, "id": "S1"}]},
    {"session_id": "a", "speakers": [{"channel": 0, "id": "S1",
                                      "role": "oracle"}]},
    {"session_id": "../x"},
    {"session_id": "a", "channels": 3},
    {"session_id": "a", "hang_ms": 100},
    {"session_id": "a", "unknown": 1},
])
def test_bad_requests_are_refused(body: Dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        LiveSessionRequest.model_validate(body)


@pytest.mark.parametrize("waited,acked,free,expected", [
    (0.0, True, 3000, "gpu"),
    (0.0, False, 3000, "wait"),
    (4.0, False, 3000, "gpu"),
    (0.0, True, 1000, "wait"),
    (20.0, True, 1000, "cpu"),
    (5.0, False, None, "cpu"),
    (0.0, False, None, "cpu"),
])
def test_gpu_choice(
    waited: float, acked: bool, free: Optional[int], expected: str
) -> None:
    assert gpu_choice(waited, acked, free, 2200, 4.0, 20.0) == expected


def test_warning_events_follow_the_contract() -> None:
    stream = LiveEventStream("s-1")
    event = stream.warning(AT, 1.5, "gpu_degraded", "CPU mode")
    jsonschema.validate(event, LIVE_SCHEMA)
    assert event["warning"] == {"code": "gpu_degraded", "message": "CPU mode"}


def final(
    ident: str, speaker: str, start: float, end: float, text: str,
    revision: int = 0, language: Optional[str] = "es",
) -> Dict[str, Any]:
    turn, index = (int(x[1:]) for x in ident.split("."))
    return {"type": "final", "stream_time": end, "utterance": {
        "id": ident, "turn": turn, "index": index, "speaker": speaker,
        "start": start, "end": end, "text": text, "revision": revision,
        "confidence": 0.8, "language": language,
    }}


def test_the_last_revision_of_each_final_is_kept_in_time_order() -> None:
    events: List[Dict[str, Any]] = [
        {"type": "partial", "utterance": {"id": "t0.s0"}},
        final("t1.s0", "S2", 3.0, 4.0, "Dos."),
        final("t0.s0", "S1", 0.0, 2.0, "Uno."),
        final("t0.s0", "S1", 0.0, 2.0, "Uno bis.", revision=2),
        final("t0.s0", "S1", 0.0, 2.0, "Uno viejo.", revision=1),
    ]
    assert [u["text"] for u in last_finals(events)] == ["Uno bis.", "Dos."]


def test_the_transcript_regroups_turns_and_follows_its_contract() -> None:
    events = [
        {"type": "session_started", "stream_time": 0.0},
        final("t0.s0", "S1", 0.0, 1.0, "Hola."),
        final("t0.s2", "S1", 1.2, 2.0, "¿Qué tal?"),
        final("t1.s0", "S2", 2.5, 3.0, "  "),
        final("t2.s0", "S2", 3.5, 4.0, "Bien."),
        {"type": "session_ended", "stream_time": 5.0},
    ]
    speakers = [
        LiveSpeaker("S1", 0, "interviewer", "Ana"),
        LiveSpeaker("S2", 1, "candidate"), LiveSpeaker("S3", None),
    ]
    transcript = live_transcript(
        "call-1", events, speakers, PIPELINE, None, AT
    )
    document = transcript.to_contract()
    assert [u["id"] for u in document["utterances"]] == [
        "t0.s0", "t0.s1", "t1.s0"
    ]
    assert document["utterances"][2]["previous_utterance"] == "t0.s1"
    assert document["speakers"] == [
        {"id": "S1", "name": "Ana", "enrolled": False},
        {"id": "S2", "name": None, "enrolled": False},
    ]
    assert document["session"]["duration"] == 5.0
    assert document["session"]["language"] == "es"
    assert "role" not in str(document)


def test_unknown_speakers_and_languages_are_still_a_transcript() -> None:
    events = [final("t0.s0", "X", 0.0, 1.0, "Hi.", language=None)]
    transcript = live_transcript("a", events, [], PIPELINE)
    assert [s.id for s in transcript.speakers] == ["X"]
    assert transcript.session.language == "und"
    empty = live_transcript("a", [], [], PIPELINE, "es")
    assert empty.utterances == [] and empty.session.duration == 0.0


def test_a_speaker_is_renamed_or_added_with_its_name() -> None:
    given = [LiveSpeaker(id="S1", channel=0, role="agent"),
             LiveSpeaker(id="S2", channel=1, role="customer")]
    out = renamed(given, "S2", "  Carla \n Ruiz ")
    assert out[1] == LiveSpeaker(id="S2", channel=1, role="customer",
                                 name="Carla Ruiz")
    assert given[1].name is None
    added = renamed(given, "ana", "Ana")
    assert added[-1] == LiveSpeaker(id="ana", name="Ana")
    for bad in ("", "  ", "x" * 81):
        with pytest.raises(ValueError):
            renamed(given, "S1", bad)
