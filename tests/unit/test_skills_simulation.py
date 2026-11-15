"""Simulated discussions: script parsing, timing, transcript.v1."""

import json
from pathlib import Path

import pytest

from src.domain.errors import InvalidInputError
from src.domain.skills.simulation import (
    parse_script,
    script_transcript,
    speaker_id,
    timed_sentences,
)
from src.domain.skills.transcript import from_contract
from src.domain.skills.transcript_contract import read_contract

SCRIPT = """# a comment
id: sim-1
title: Prueba
language: es
type: reunion_decision
goal: Decidir algo

Lucía: Buenos días. Empezamos ya.
Andrés: Yo tengo prisa hoy.
+Lucía: Lo sé, seremos breves.
"""

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "examples" / "skills" / "simulated"


def test_parse_reads_header_turns_and_overlaps() -> None:
    """Header keys, turns in order and the ``+`` mark."""
    script = parse_script(SCRIPT)
    assert script.header["id"] == "sim-1"
    assert script.header["type"] == "reunion_decision"
    assert [t.speaker for t in script.turns] == ["Lucía", "Andrés", "Lucía"]
    assert [t.overlaps for t in script.turns] == [False, False, True]
    assert script.speakers() == ["Lucía", "Andrés"]


@pytest.mark.parametrize("text, message", [
    ("id: s\n+Ana: hola", "first turn cannot overlap"),
    ("id: s\n", "no turns"),
    ("Ana: hola", "valid 'id:'"),
    ("id: ../x\nAna: hola", "valid 'id:'"),
    ("id: s\nAna: hola\nAna: otra", "in a row"),
    ("id: s\nAna: hola\nesto no es un turno", "Line 3"),
])
def test_parse_refuses_bad_scripts(text: str, message: str) -> None:
    """Each mistake is named."""
    with pytest.raises(InvalidInputError, match=message):
        parse_script(text)


def test_timing_follows_words_pauses_and_overlaps() -> None:
    """Sentences last by words; an overlap starts before the end."""
    timed = timed_sentences(parse_script(SCRIPT), words_per_second=2.0)
    assert [(s.turn, s.position) for s in timed] == [
        (0, 0), (0, 1), (1, 0), (2, 0),
    ]
    first, second, third, fourth = timed
    assert first.start == 0.3 and first.end == pytest.approx(1.3)
    assert second.start == pytest.approx(first.end + 0.25)
    assert third.start == pytest.approx(second.end + 0.45)
    assert fourth.overlap and fourth.start == pytest.approx(third.end - 0.6)
    with pytest.raises(InvalidInputError):
        timed_sentences(parse_script(SCRIPT), words_per_second=0)


def test_transcript_follows_the_contract() -> None:
    """The document validates and reads as the analyzer's transcript."""
    document = script_transcript(parse_script(SCRIPT))
    contract = read_contract(document)
    assert contract.session.type == "reunion_decision"
    assert [s.id for s in contract.speakers] == ["lucia", "andres"]
    third = document["utterances"][2]
    assert third["previous_speaker"] == "lucia"
    assert third["previous_utterance"] == "t0.s1"
    transcript = from_contract(contract)
    assert transcript.speakers() == ["Lucía", "Andrés"]
    assert transcript.conversation_goal == "Decidir algo"


def test_speaker_ids_are_plain() -> None:
    """Accents and spaces go; an empty name still gets an id."""
    assert speaker_id("José Ñúñez") == "jose-nunez"
    assert speaker_id("¿?") == "speaker"
    with pytest.raises(InvalidInputError, match="same id"):
        script_transcript(parse_script("id: s\nAna: hola\nAná: adiós"))


def test_the_committed_demo_matches_its_script() -> None:
    """``make demo-analysis``'s transcript is the script's, unchanged."""
    script = parse_script(
        (DEMO / "reunion_verano.txt").read_text(encoding="utf-8")
    )
    stored = json.loads(
        (DEMO / "sim-reunion-verano.transcript.v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert stored == script_transcript(script)
    assert len(script.speakers()) == 4
    assert 13 * 60 < stored["session"]["duration"] < 18 * 60
    assert sum(u["overlap"] for u in stored["utterances"]) >= 5
