"""Transcriber domain model and its ``transcript.v1`` translation."""

import copy
import math
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, Tuple

import pytest
from pydantic import ValidationError

from transcriber.domain.errors import ContractError, TranscriberError
from transcriber.domain.models import (
    Pipeline,
    Session,
    SessionState,
    Speaker,
    Transcript,
    Turn,
    Utterance,
    Voiceprint,
    Word,
)
from transcriber.domain.sentences import build_utterances
from transcriber.domain.speakers import SpeakerSpan


def document() -> Dict[str, Any]:
    """A small valid ``transcript.v1`` document."""
    return {
        "schema": "transcript.v1",
        "session": {"id": "s-1", "title": None, "language": "es",
                    "started_at": "2026-11-14T10:02:00+01:00",
                    "duration": 3.0, "audio_sha256": "a" * 64},
        "speakers": [{"id": "ana", "name": "Ana", "enrolled": True},
                     {"id": "S1", "name": None, "enrolled": False}],
        "utterances": [
            {"id": "t0.s0", "turn": 0, "index": 0, "speaker": "ana",
             "start": 0.0, "end": 1.0, "text": "Hola.",
             "previous_speaker": None, "previous_utterance": None,
             "confidence": 0.9, "speaker_confidence": 0.7},
            {"id": "t1.s0", "turn": 1, "index": 0, "speaker": "S1",
             "start": 1.5, "end": 3.0, "text": "Buenas.",
             "previous_speaker": "ana", "previous_utterance": "t0.s0",
             "confidence": None, "speaker_confidence": None},
        ],
        "pipeline": {"recogniser": "r", "voice_model": "v",
                     "transcriber": "0.1.0"},
    }


class TestContract:
    def test_round_trip(self) -> None:
        source = document()
        transcript = Transcript.from_contract(source)
        assert transcript.session.state is SessionState.TRANSCRIBED
        assert transcript.to_contract() == source

    def test_words_are_not_in_the_contract(self) -> None:
        utterances = build_utterances([Turn(
            speaker="ana", text="Hola.", start=0, end=1,
            words=[Word(text="Hola.", start=0, end=1)],
        )])
        transcript = Transcript(
            session=Session(id="s", language="es", duration=1.0),
            speakers=[Speaker(id="ana", enrolled=True)],
            utterances=utterances,
            pipeline=Pipeline(recogniser="r", voice_model="v",
                              transcriber="0.1.0"),
        )
        out = transcript.to_contract()
        assert "words" not in out["utterances"][0]
        assert out["session"]["started_at"] is None
        assert "state" not in out["session"]

    @pytest.mark.parametrize("change, message", [
        (lambda d: d.update(schema="transcript.v2"), "Expected schema"),
        (lambda d: d.update(extra=1), "Unknown transcript field"),
        (lambda d: d.pop("session"), "Missing session"),
        (lambda d: d["session"].pop("duration"), "Missing session field"),
        (lambda d: d.update(speakers={}), "speakers must be a list"),
        (lambda d: d["speakers"].append("ana"), "must be a JSON object"),
        (lambda d: d["speakers"][0].pop("enrolled"), "Missing speaker"),
        (lambda d: d["utterances"][0].update(words=[]), "Unknown utterance"),
        (lambda d: d["utterances"][0].pop("previous_speaker"),
         "Missing utterance"),
        (lambda d: d["pipeline"].pop("transcriber"), "Missing pipeline"),
        (lambda d: d["session"].update(id="-bad"), "session.id"),
        (lambda d: d["session"].update(started_at="2026-11-14T10:02:00"),
         "session.started_at"),
        (lambda d: d["utterances"][0].update(id="t0.s1"), "does not match"),
        (lambda d: d["utterances"][0].update(end=-1), "utterances.0.end"),
        (lambda d: d["utterances"][1].update(speaker="luis"),
         "unlisted speaker luis"),
        (lambda d: d["utterances"][1].update(previous_utterance="t9.s0"),
         "unknown utterance"),
        (lambda d: d["utterances"][1].update(id="t0.s0", turn=0),
         "utterance ids must be unique"),
        (lambda d: d["speakers"].append(dict(d["speakers"][0])),
         "speaker ids must be unique"),
    ])
    def test_invalid_documents(
        self, change: Callable[[Dict[str, Any]], Any], message: str,
    ) -> None:
        broken = copy.deepcopy(document())
        change(broken)
        with pytest.raises(ContractError, match=message):
            Transcript.from_contract(broken)

    @pytest.mark.parametrize("end", [3.0, 3.5])
    def test_end_within_the_session_tolerance(self, end: float) -> None:
        source = document()
        source["utterances"][1]["end"] = end
        assert Transcript.from_contract(source).utterances[1].end == end

    @pytest.mark.parametrize("end", [3.51, 30.0])
    def test_end_after_the_session_is_rejected(self, end: float) -> None:
        broken = document()
        broken["utterances"][1]["end"] = end
        with pytest.raises(ContractError, match="after the session"):
            Transcript.from_contract(broken)

    @pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
    @pytest.mark.parametrize("place", [
        ("session", None, "duration"),
        ("utterances", 0, "start"),
        ("utterances", 1, "end"),
        ("utterances", 0, "confidence"),
        ("utterances", 0, "speaker_confidence"),
    ])
    def test_non_finite_numbers_are_rejected(
        self, place: Tuple[str, Optional[int], str], value: float,
    ) -> None:
        broken = document()
        section, item, key = place
        target = broken[section] if item is None else broken[section][item]
        target[key] = value
        with pytest.raises(ContractError):
            Transcript.from_contract(broken)

    def test_not_json_data_is_rejected(self) -> None:
        broken = document()
        broken["session"]["title"] = object()
        with pytest.raises(ContractError, match="not JSON data"):
            Transcript.from_contract(broken)

    def test_not_an_object(self) -> None:
        with pytest.raises(ContractError):
            Transcript.from_contract([])  # type: ignore[arg-type]

    def test_contract_error_is_a_transcriber_error(self) -> None:
        assert issubclass(ContractError, TranscriberError)


class TestModels:
    def test_word_and_turn_times_are_ordered(self) -> None:
        with pytest.raises(ValidationError):
            Word(text="x", start=2, end=1)
        with pytest.raises(ValidationError):
            Turn(speaker="a", text="x", start=2, end=1)

    def test_utterance_cannot_end_before_it_starts(self) -> None:
        with pytest.raises(ValidationError, match="end before"):
            Utterance(id="t0.s0", turn=0, index=0, speaker="a", start=2,
                      end=1, text="x", previous_speaker=None,
                      previous_utterance=None)

    @pytest.mark.parametrize("value", [math.inf, math.nan])
    def test_times_must_be_finite(self, value: float) -> None:
        with pytest.raises(ValidationError, match="finite"):
            Word(text="x", start=0, end=value)
        with pytest.raises(ValidationError, match="finite"):
            Word(text="x", start=0, end=1, probability=value)
        with pytest.raises(ValidationError, match="finite"):
            Turn(speaker="a", text="x", start=0, end=value)
        with pytest.raises(ValidationError, match="finite"):
            Utterance(id="t0.s0", turn=0, index=0, speaker="a", start=0,
                      end=value, text="x", previous_speaker=None,
                      previous_utterance=None)
        with pytest.raises(ValidationError, match="finite"):
            Session(id="x", language="es", duration=value)
        with pytest.raises(ValidationError, match="finite"):
            SpeakerSpan(speaker="a", start=value, end=value)

    def test_session_defaults_and_rules(self) -> None:
        session = Session(id="2026-11-14-equipo-a", language="es")
        assert session.state is SessionState.RECORDING
        assert session.duration == 0.0
        with pytest.raises(ValidationError):
            Session(id="x", language="es", audio_sha256="not-a-hash")

    def test_voiceprint(self) -> None:
        when = datetime(2026, 11, 14, tzinfo=timezone.utc)
        print_ = Voiceprint(speaker_id="ana", model="m", vector=[0.1, 0.2],
                            consent_version="1", enrolled_at=when)
        assert print_.vector == [0.1, 0.2]
        with pytest.raises(ValidationError):
            Voiceprint(speaker_id="ana", model="m", vector=[],
                       consent_version="1", enrolled_at=when)
        with pytest.raises(ValidationError):
            Voiceprint(speaker_id="ana", model="m", vector=[1.0],
                       consent_version="1",
                       enrolled_at=datetime(2026, 11, 14))
