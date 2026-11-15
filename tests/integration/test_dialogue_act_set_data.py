"""The committed Spanish dialogue-act set and its datasheet agree.

Checks what the ticket (#64) promises about the set: size, the 13
categories, oversampled rare categories, every conversation type,
reply links, and that the generated JSON and the datasheet's counts are
those of the ``*.txt`` source.
"""

import json
import re
from pathlib import Path
from typing import Dict, List

from src.domain.skills.dialogue_act_set import (
    SetDialogue,
    parse_set,
    set_distribution,
    to_transcript_like,
    weighted_count,
)
from src.domain.skills.dialogue_acts import SECOND_PARTS, SESSION_TYPES

ROOT = Path(__file__).resolve().parents[2]
SET_DIR = ROOT / "examples" / "skills" / "dialogue_acts"


def load() -> List[SetDialogue]:
    """Parse every batch file of the set."""
    return [
        dialogue
        for path in sorted(SET_DIR.glob("*.txt"))
        for dialogue in parse_set(path.read_text(encoding="utf-8"),
                                  path.name)
    ]


DIALOGUES = load()
COUNTS = set_distribution(DIALOGUES)


def test_size_and_rare_categories() -> None:
    total = sum(COUNTS.values())
    assert 1400 <= total <= 1700
    for rare in ("compromiso", "rechazo", "escucha"):
        assert COUNTS[rare] >= 50, rare
    assert all(count > 0 for count in COUNTS.values())


def test_every_conversation_type_and_unique_ids() -> None:
    assert {d.session.type for d in DIALOGUES} == set(SESSION_TYPES)
    ids = [d.session.id for d in DIALOGUES]
    assert len(ids) == len(set(ids))


def test_second_parts_point_at_earlier_sentences() -> None:
    for dialogue in DIALOGUES:
        order = {u.id: n for n, u in enumerate(dialogue.utterances)}
        for n, utterance in enumerate(dialogue.utterances):
            if utterance.act in SECOND_PARTS:
                assert utterance.responds_to is not None
            if utterance.responds_to is not None:
                assert order[utterance.responds_to] < n


def test_generated_json_is_in_sync() -> None:
    committed = json.loads(
        (SET_DIR / "dialogues.json").read_text(encoding="utf-8"))
    assert committed == [to_transcript_like(d) for d in DIALOGUES]


def test_datasheet_counts_match_the_set() -> None:
    sheet = (SET_DIR / "README.md").read_text(encoding="utf-8")
    table: Dict[str, int] = {
        m.group(1): int(m.group(2).replace(",", ""))
        for m in re.finditer(r"^\| (\w+) \| ([\d,]+) \| [\d.]+% \|$", sheet,
                             re.MULTILINE)
    }
    assert table == COUNTS
    total = f"{sum(COUNTS.values()):,}"
    assert f"{len(DIALOGUES)} dialogues, {total} sentences" in sheet


def test_weighted_examples_are_few_and_in_the_datasheet() -> None:
    weighted = [u for d in DIALOGUES for u in d.utterances
                if len(u.labels) > 1]
    assert len(weighted) == weighted_count(DIALOGUES)
    assert 1 <= len(weighted) <= 20
    assert all(u.act == u.labels[0].label for u in weighted)
    sheet = (SET_DIR / "README.md").read_text(encoding="utf-8")
    assert f"{len(weighted)} sentences carry weighted" in sheet


def test_expected_events_point_at_sentences_of_the_set() -> None:
    """Every expected event names a real sentence and speaker (#83)."""
    document = json.loads(
        (SET_DIR / "expected_events.json").read_text(encoding="utf-8"))
    by_id = {d.session.id: d for d in DIALOGUES}
    assert len(document["dialogues"]) == 8
    assert "not human gold" in document["note"]
    for dialogue, events in document["dialogues"].items():
        sentences = {u.id: u for u in by_id[dialogue].utterances}
        for event in events:
            assert event["sentence"] in sentences, (dialogue, event)
            if "owner" in event:
                assert event["owner"] in by_id[dialogue].speakers
