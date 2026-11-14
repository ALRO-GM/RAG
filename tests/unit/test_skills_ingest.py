"""Contract test (RAG side) and the ingest use cases, with a fake store.

``contracts/`` is the only thing the transcriber and the RAG share. The
RAG reads it with its own models (:mod:`src.domain.skills.
transcript_contract`), never with the transcriber's code: the example
must be accepted, and every document the JSON schema rejects must be
rejected too, as must the documents that break the rules of §4 the
schema cannot express.
"""

import copy
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import pytest
from jsonschema import Draft202012Validator, FormatChecker

from src.application.skills.ingest import (
    IngestTranscript,
    ShowIngestedTranscript,
)
from src.domain.errors import InvalidInputError, NotFoundError
from src.domain.skills.transcript_contract import (
    is_session_id,
    read_contract,
)
from tests.fakes import InMemoryTranscriptStore

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "contracts" / "transcript.v1.schema.json"
EXAMPLE_PATH = ROOT / "contracts" / "examples" / "transcript.v1.json"

Change = Callable[[Dict[str, Any]], Any]


def example() -> Dict[str, Any]:
    """A fresh copy of the contract's example document."""
    document: Dict[str, Any] = json.loads(EXAMPLE_PATH.read_text("utf-8"))
    return document


@pytest.fixture(scope="module")
def validator() -> Draft202012Validator:
    """The contract's JSON schema, with format checks."""
    schema = json.loads(SCHEMA_PATH.read_text("utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, format_checker=FormatChecker())


def set_in(path: Sequence[Any], value: Any) -> Change:
    """A change that sets the value at ``path`` of a document."""
    def change(document: Dict[str, Any]) -> None:
        target: Any = document
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
    return change


def drop(path: Sequence[Any]) -> Change:
    """A change that removes the key at ``path``."""
    def change(document: Dict[str, Any]) -> None:
        target: Any = document
        for key in path[:-1]:
            target = target[key]
        del target[path[-1]]
    return change


def test_the_example_is_accepted(validator: Draft202012Validator) -> None:
    document = example()
    validator.validate(document)
    transcript = read_contract(document)
    assert transcript.session.id == "demo-es"
    assert [len(t) for t in transcript.turns()] == [2, 1, 1]
    assert transcript.session.started_at is not None
    assert transcript.session.started_at.utcoffset() is not None


def test_whole_numbers_are_valid_times() -> None:
    document = example()
    document["session"]["duration"] = 15
    document["utterances"][0]["start"] = 0
    assert read_contract(document).session.duration == 15.0


@pytest.mark.parametrize("value", [
    "2026-11-14", "2026-11-14T10:02:00", "14/11/2026 10:02", "",
])
def test_started_at_is_a_date_time_with_an_offset(value: str) -> None:
    """``format: date-time`` (RFC 3339) needs a time and an offset.

    The ``jsonschema`` installed here does not check that format, so
    the RAG's own reading is the check.
    """
    document = example()
    document["session"]["started_at"] = value
    with pytest.raises(InvalidInputError, match="started_at"):
        read_contract(document)
    document["session"]["started_at"] = "2026-11-14T09:02:00Z"
    read_contract(document)


# Documents the JSON schema rejects: the RAG must reject each too.
SCHEMA_BREAKS: List[Tuple[str, Change]] = [
    ("another schema version", set_in(["schema"], "transcript.v2")),
    ("unknown top-level field", set_in(["audio"], "UklGRg==")),
    ("unknown utterance field", set_in(["utterances", 0, "words"], [])),
    ("unknown speaker field", set_in(["speakers", 0, "voiceprint"], [0.1])),
    ("missing pipeline", drop(["pipeline"])),
    ("missing session language", drop(["session", "language"])),
    ("missing previous speaker", drop(["utterances", 0, "previous_speaker"])),
    ("session id with a slash", set_in(["session", "id"], "../etc")),
    ("session id too long", set_in(["session", "id"], "a" * 65)),
    ("duration as a string", set_in(["session", "duration"], "14.2")),
    ("negative duration", set_in(["session", "duration"], -1)),
    ("duration as a boolean", set_in(["session", "duration"], True)),
    ("started_at as a number",
     set_in(["session", "started_at"], 1763110920)),
    ("bad audio hash", set_in(["session", "audio_sha256"], "ABC")),
    ("language too short", set_in(["session", "language"], "e")),
    ("title as a number", set_in(["session", "title"], 5)),
    ("enrolled as a string", set_in(["speakers", 0, "enrolled"], "true")),
    ("enrolled as a number", set_in(["speakers", 0, "enrolled"], 1)),
    ("empty speaker id", set_in(["speakers", 0, "id"], "")),
    ("speaker name as a number", set_in(["speakers", 0, "name"], 3)),
    ("bad utterance id", set_in(["utterances", 0, "id"], "u0")),
    ("turn as a string", set_in(["utterances", 0, "turn"], "0")),
    ("negative index", set_in(["utterances", 0, "index"], -1)),
    ("index as a boolean", set_in(["utterances", 0, "index"], False)),
    ("start as a string", set_in(["utterances", 0, "start"], "0.4")),
    ("end as a boolean", set_in(["utterances", 0, "end"], True)),
    ("empty text", set_in(["utterances", 0, "text"], "")),
    ("confidence above 1", set_in(["utterances", 0, "confidence"], 1.5)),
    ("speaker confidence below -1",
     set_in(["utterances", 0, "speaker_confidence"], -1.5)),
    ("speaker confidence as a string",
     set_in(["utterances", 0, "speaker_confidence"], "0.74")),
    ("previous speaker as a number",
     set_in(["utterances", 2, "previous_speaker"], 7)),
    ("recogniser as a number", set_in(["pipeline", "recogniser"], 1)),
    ("utterances as an object", set_in(["utterances"], {})),
    ("speakers as a string", set_in(["speakers"], "marta")),
]


@pytest.mark.parametrize(
    "change", [c[1] for c in SCHEMA_BREAKS], ids=[c[0] for c in SCHEMA_BREAKS]
)
def test_what_the_schema_rejects_the_rag_rejects(
    validator: Draft202012Validator, change: Change
) -> None:
    broken = copy.deepcopy(example())
    change(broken)
    assert not validator.is_valid(broken)
    with pytest.raises(InvalidInputError):
        read_contract(broken)


def utterance(document: Dict[str, Any], n: int) -> Dict[str, Any]:
    """Utterance ``n`` of ``document``."""
    item: Dict[str, Any] = document["utterances"][n]
    return item


def swap_turns(document: Dict[str, Any]) -> None:
    """Put turn 2 before turn 1."""
    items = document["utterances"]
    items[2], items[3] = items[3], items[2]


# Each breaks a rule of §4 the JSON schema cannot express.
STRUCTURE_BREAKS: List[Tuple[str, Change, str]] = [
    ("id does not match turn and index",
     lambda d: utterance(d, 1).update(id="t0.s5"), "does not match"),
    ("utterance ends before it starts",
     lambda d: utterance(d, 0).update(end=0.1), "ends before"),
    ("duplicated speaker id",
     lambda d: d["speakers"].append(dict(d["speakers"][0])), "unique"),
    ("unlisted speaker",
     lambda d: [utterance(d, 3).update(speaker="nadie")], "unlisted"),
    ("utterances out of order", swap_turns, "out of order"),
    ("gap in turn numbers",
     lambda d: utterance(d, 3).update(id="t3.s0", turn=3), "expected turn 2"),
    ("gap in sentence indices",
     lambda d: (utterance(d, 1).update(id="t0.s2", index=2),
                utterance(d, 2).update(previous_utterance="t0.s2")),
     "expected index 1"),
    ("two speakers inside one turn",
     lambda d: utterance(d, 1).update(speaker="luis"), "two speakers"),
    ("previous differs inside a turn",
     lambda d: utterance(d, 1).update(previous_speaker="luis"),
     "does not share"),
    ("first turn has a previous",
     lambda d: [utterance(d, n).update(previous_speaker="S1")
                for n in (0, 1)],
     "first turn"),
    ("previous utterance is not the last of the turn before",
     lambda d: utterance(d, 2).update(previous_utterance="t0.s0"),
     "must follow t0.s1"),
    ("previous speaker is not the speaker of the turn before",
     lambda d: utterance(d, 3).update(previous_speaker="marta"),
     "previous speaker luis"),
]


@pytest.mark.parametrize(
    "change, message", [c[1:] for c in STRUCTURE_BREAKS],
    ids=[c[0] for c in STRUCTURE_BREAKS],
)
def test_the_turn_structure_is_enforced(
    validator: Draft202012Validator, change: Change, message: str
) -> None:
    broken = copy.deepcopy(example())
    change(broken)
    validator.validate(broken)  # the schema alone lets it through
    with pytest.raises(InvalidInputError, match=message):
        read_contract(broken)


@pytest.mark.parametrize("document, message", [
    ([], "JSON object"),
    ("transcript", "JSON object"),
    ({"schema": "transcript.v1", "session": {"duration": math.nan}},
     "not JSON data"),
    ({}, "Expected schema transcript.v1"),
])
def test_what_is_not_a_document(document: Any, message: str) -> None:
    with pytest.raises(InvalidInputError, match=message):
        read_contract(document)


@pytest.mark.parametrize("value, valid", [
    ("demo-es", True), ("2026-11-14.team_a", True), ("a" * 64, True),
    ("", False), ("-x", False), ("a/b", False), ("..", False),
    ("a" * 65, False), ("a\n", False),
])
def test_session_ids(value: str, valid: bool) -> None:
    assert is_session_id(value) is valid


def test_ingest_stores_and_replaces_by_session() -> None:
    store = InMemoryTranscriptStore()
    ingest = IngestTranscript(store)
    first = ingest(example())
    assert first == ("demo-es", False, 4)
    changed = example()
    changed["session"]["title"] = "Otra vez"
    assert ingest(changed).replaced is True
    assert list(store.documents) == ["demo-es"]
    assert ShowIngestedTranscript(store)("demo-es") == changed


def test_ingest_stores_nothing_invalid() -> None:
    store = InMemoryTranscriptStore()
    broken = example()
    broken["utterances"][0]["words"] = []
    with pytest.raises(InvalidInputError, match="words"):
        IngestTranscript(store)(broken)
    assert store.puts == []


def test_show_checks_the_id_then_the_store() -> None:
    show = ShowIngestedTranscript(InMemoryTranscriptStore())
    with pytest.raises(InvalidInputError, match="Invalid session id"):
        show("../x")
    with pytest.raises(NotFoundError):
        show("demo-es")
