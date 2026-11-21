"""LocalAgreement-2, sentence assembly and live.event.v1 events (LV-4)."""

from datetime import datetime, timezone

import jsonschema
import pytest

from tests.unit.transcriber.live_fakes import LIVE_SCHEMA
from transcriber.domain.live import (
    Hypothesis,
    LiveEventStream,
    LiveSpeaker,
    LiveUtterance,
    LocalAgreement,
    SentenceAssembler,
    Timings,
    agreement_key,
    iso,
    percentile,
    sentence_spans,
)
from transcriber.domain.models import Word

AT = datetime(2026, 11, 21, 10, 4, 12, 310000, tzinfo=timezone.utc)


def heard(text: str) -> Hypothesis:
    return Hypothesis(text=text, language="es")


# ------------------------------------------------------------ agreement

def test_agreement_commits_the_prefix_of_two_hypotheses() -> None:
    a = LocalAgreement()
    assert a.update("buenos") == []
    assert a.update("buenos días a") == ["buenos"]
    assert a.update("buenos días a todos") == ["buenos", "días", "a"]


def test_agreement_ignores_case_and_punctuation() -> None:
    a = LocalAgreement()
    a.update("Hola, qué tal")
    assert a.update("hola qué tal estás") == ["hola", "qué", "tal"]
    assert agreement_key("¿Qué?") == "qué"


def test_agreement_never_takes_words_back_and_resets() -> None:
    a = LocalAgreement()
    a.update("uno dos tres")
    a.update("uno dos tres cuatro")
    assert a.update("uno otro") == ["uno", "dos", "tres"]
    assert a.committed == ["uno", "dos", "tres"]
    a.reset()
    assert a.committed == [] and a.update("uno") == []


# ------------------------------------------------------------ sentences

def test_sentence_spans_take_the_times_of_their_words() -> None:
    words = [
        Word(text="Hola.", start=0.0, end=0.4),
        Word(text="Qué", start=0.6, end=0.8),
        Word(text="tal?", start=0.8, end=1.2),
    ]
    spans = sentence_spans("Hola. Qué tal?", words, 10.0, 12.0)
    assert spans == [
        ("Hola.", 10.0, 10.4),
        ("Qué tal?", pytest.approx(10.6), pytest.approx(11.2)),
    ]


def test_sentence_spans_fall_back_to_the_segment_and_clamp() -> None:
    assert sentence_spans("Hola. Adiós.", [], 1.0, 2.0) == [
        ("Hola.", 1.0, 2.0), ("Adiós.", 1.0, 2.0),
    ]
    late = [Word(text="Hola", start=0.5, end=9.0)]
    assert sentence_spans("Hola", late, 1.0, 2.0) == [("Hola", 1.5, 2.0)]
    assert sentence_spans("", [], 0.0, 1.0) == []


# ------------------------------------------------------------ assembler

def test_partial_and_final_share_ids_and_revisions_grow() -> None:
    s = SentenceAssembler()
    (p1,) = s.partial("S1", None, 0.0, 1.0, heard("Buenos días"))
    assert (p1.id, p1.revision) == ("t0.s0", 0)
    assert s.partial("S1", None, 0.0, 1.5, heard("Buenos días")) == []
    p2 = s.partial("S1", None, 0.0, 2.0, heard("Buenos días. Hoy"))
    assert [(u.id, u.revision, u.text) for u in p2] == [
        ("t0.s0", 1, "Buenos días."), ("t0.s1", 0, "Hoy"),
    ]
    finals = s.final("S1", None, 0.0, 3.0, heard("Buenos días. Hoy sí."))
    assert [(u.id, u.revision) for u in finals] == [
        ("t0.s0", 2), ("t0.s1", 1),
    ]
    (nxt,) = s.final("S1", None, 4.0, 5.0, heard("Otra."))
    assert (nxt.id, nxt.turn, nxt.index, nxt.revision) == ("t0.s2", 0, 2, 0)


def test_a_new_speaker_opens_a_new_turn() -> None:
    s = SentenceAssembler()
    s.final("S1", 0, 0.0, 1.0, heard("Uno."))
    (b,) = s.final("S2", 1, 1.0, 2.0, heard("Dos."))
    (a,) = s.final("S1", 0, 2.0, 3.0, heard("Tres."))
    assert (b.id, b.channel, a.id) == ("t1.s0", 1, "t2.s0")


def test_overlapping_speakers_keep_their_reserved_ids() -> None:
    s = SentenceAssembler()
    s.partial("S1", 0, 0.0, 1.0, heard("Hola"))
    s.partial("S2", 1, 0.5, 1.0, heard("Sí"))
    (b,) = s.final("S2", 1, 0.5, 1.2, heard("Sí."))
    (a,) = s.final("S1", 0, 0.0, 1.5, heard("Hola."))
    assert (a.id, a.revision, b.id) == ("t0.s0", 1, "t1.s0")


def test_ids_shown_by_a_partial_are_not_reused() -> None:
    s = SentenceAssembler()
    s.partial("S1", None, 0.0, 1.0, heard("Uno. Dos."))
    s.final("S1", None, 0.0, 1.0, heard("Uno dos."))
    (nxt,) = s.final("S1", None, 2.0, 3.0, heard("Tres."))
    assert nxt.id == "t0.s2"


def test_empty_hypotheses_give_nothing() -> None:
    s = SentenceAssembler()
    assert s.partial("S1", None, 0.0, 1.0, heard("")) == []
    assert s.final("S1", None, 0.0, 1.0, heard("")) == []
    s.partial("S1", None, 0.0, 1.0, heard("algo"))
    assert s.final("S1", None, 0.0, 1.0, heard("  ")) == []
    (u,) = s.final("S1", None, 2.0, 3.0, heard("Nada."))
    assert u.id == "t0.s1"


# ------------------------------------------------------------ events

def test_events_follow_the_contract_and_count_up() -> None:
    events = LiveEventStream("int-2026-11-21-a")
    started = events.session_started(
        AT, "stereo_channels", 2,
        [LiveSpeaker("S1", 0), LiveSpeaker("S2", 1, role="candidate")],
    )
    utterance = LiveUtterance(
        id="t7.s1", turn=7, index=1, speaker="S2", channel=1,
        start=247.20000000001, end=251.4, text="Hola.", revision=3,
        confidence=0.912345, language="es",
    )
    final = events.utterance(
        "final", AT, 251.9, utterance,
        Timings(endpoint_ms=480, asr_ms=310.04, attribution_ms=1,
                lag_ms=510),
    )
    ended = events.session_ended(AT, 300.0)
    for event in (started, final, ended):
        jsonschema.validate(event, LIVE_SCHEMA)
    assert [e["seq"] for e in (started, final, ended)] == [0, 1, 2]
    assert final["at"] == "2026-11-21T10:04:12.310Z"
    assert final["utterance"]["start"] == 247.2
    assert final["utterance"]["confidence"] == 0.9123
    assert final["timings"] == {
        "endpoint_ms": 480, "asr_ms": 310.0, "attribution_ms": 1,
        "lag_ms": 510,
    }
    assert started["speakers"][1]["role"] == "candidate"


def test_timings_drop_unknown_values() -> None:
    assert Timings(asr_ms=12.34).to_contract() == {"asr_ms": 12.3}


def test_iso_and_percentile() -> None:
    assert iso(AT) == "2026-11-21T10:04:12.310Z"
    assert percentile([], 0.5) is None
    assert percentile([3.0], 0.9) == 3.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert percentile([0.0, 10.0], 0.9) == pytest.approx(9.0)
    assert percentile([0.0, 10.0], 2.0) == 10.0
