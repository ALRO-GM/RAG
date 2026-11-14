"""Contract test: the model reads and writes ``transcript.v1`` as specified.

The JSON schema in ``contracts/`` is the only thing the transcriber and
the RAG share; this test holds the transcriber side to it.
"""

import copy
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import pytest
from jsonschema import Draft202012Validator, FormatChecker

from transcriber.domain.errors import ContractError
from transcriber.domain.models import (
    Pipeline,
    Session,
    Speaker,
    Transcript,
    Turn,
    Word,
)
from transcriber.domain.sentences import build_utterances

ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = ROOT / "contracts" / "transcript.v1.schema.json"
EXAMPLE_PATH = ROOT / "contracts" / "examples" / "transcript.v1.json"


def load(path: Path) -> Dict[str, Any]:
    """Read a JSON document."""
    data: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


@pytest.fixture(scope="module")
def validator() -> Draft202012Validator:
    """A validator of the contract that also checks formats."""
    schema = load(SCHEMA_PATH)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, format_checker=FormatChecker())


def test_example_is_valid(validator: Draft202012Validator) -> None:
    validator.validate(load(EXAMPLE_PATH))


def test_example_round_trips(validator: Draft202012Validator) -> None:
    example = load(EXAMPLE_PATH)
    out = Transcript.from_contract(example).to_contract()
    assert out == example
    validator.validate(out)
    # Through JSON text too, as a publisher would send it.
    assert json.loads(json.dumps(out)) == example


def test_a_document_without_the_certainty_fields_still_reads(
    validator: Draft202012Validator,
) -> None:
    """The fields added to transcript.v1 are optional both ways."""
    added = ("speaker_certainty", "doubtful", "overlap",
             "alternative_speaker", "overlap_source", "language",
             "language_differs")
    old = load(EXAMPLE_PATH)
    for item in old["utterances"]:
        for key in added:
            item.pop(key, None)
    for item in old["speakers"]:
        item.pop("identification", None)
    validator.validate(old)
    transcript = Transcript.from_contract(old)
    assert all(u.speaker_certainty is None for u in transcript.utterances)
    assert transcript.to_contract() == old


def test_built_transcript_validates(validator: Draft202012Validator) -> None:
    turns = [
        Turn(speaker="marta", text="Creo que sí. Pero no estoy segura.",
             start=0.4, end=6.3, speaker_confidence=0.74,
             words=[Word(text=t, start=0.4 + i, end=0.9 + i,
                         probability=0.9)
                    for i, t in enumerate(
                        "Creo que sí. Pero no estoy segura.".split())]),
        Turn(speaker="S1", text="Espera.", start=6.9, end=7.5),
        Turn(speaker="S1", text="Déjala terminar.", start=7.6, end=9.0),
    ]
    transcript = Transcript(
        session=Session(id="demo-es", language="es", duration=9.0,
                        audio_sha256="0" * 64),
        speakers=[Speaker(id="marta", name="Marta", enrolled=True),
                  Speaker(id="S1", enrolled=False)],
        utterances=build_utterances(turns),
        pipeline=Pipeline(recogniser="openai/whisper-large-v3-turbo",
                          voice_model="microsoft/wavlm-base-plus-sv",
                          transcriber="0.1.0"),
    )
    document = transcript.to_contract()
    validator.validate(document)
    assert [u["id"] for u in document["utterances"]] == [
        "t0.s0", "t0.s1", "t1.s0", "t1.s1",
    ]
    assert Transcript.from_contract(document).to_contract() == document


Change = Callable[[Dict[str, Any]], Any]


def utterance(document: Dict[str, Any], n: int) -> Dict[str, Any]:
    """Utterance ``n`` of ``document``."""
    item: Dict[str, Any] = document["utterances"][n]
    return item


def split_luis_turn(document: Dict[str, Any]) -> None:
    """Give turn 1 a second sentence with a different previous pair."""
    second = dict(utterance(document, 2), id="t1.s1", index=1,
                  start=9.8, end=10.0, text="Seguro.",
                  previous_utterance="t0.s0")
    document["utterances"].insert(3, second)
    utterance(document, 4)["previous_utterance"] = "t1.s1"


def swap_turns(document: Dict[str, Any]) -> None:
    """Put turn 2 before turn 1."""
    items = document["utterances"]
    items[2], items[3] = items[3], items[2]


# Each breaks a rule of §4 that the JSON schema cannot express.
STRUCTURE_BREAKS: List[Tuple[str, Change, str]] = [
    ("an anonymous voice identified",
     lambda d: d["speakers"][2].update(identification="threshold"),
     "not identified"),
    ("previous differs inside a turn", split_luis_turn,
     "does not share the previous"),
    ("first turn has a previous",
     lambda d: [utterance(d, n).update(previous_speaker="S1",
                                       previous_utterance="t2.s0")
                for n in (0, 1)],
     "first turn has no previous"),
    ("previous speaker is not the speaker of previous utterance",
     lambda d: utterance(d, 3).update(previous_speaker="marta"),
     "must have previous speaker luis"),
    ("previous utterance is itself",
     lambda d: utterance(d, 3).update(previous_utterance="t2.s0"),
     "must follow t1.s0"),
    ("previous utterance is not the last of the turn before",
     lambda d: utterance(d, 2).update(previous_utterance="t0.s0"),
     "must follow t0.s1"),
    ("two consecutive turns of one speaker",
     lambda d: utterance(d, 3).update(speaker="luis"),
     "have the same speaker luis"),
    ("two speakers inside one turn",
     lambda d: utterance(d, 1).update(speaker="luis"),
     "turn 0 has two speakers"),
    ("utterances out of order", swap_turns, "out of order"),
    ("gap in turn numbers",
     lambda d: utterance(d, 3).update(id="t3.s0", turn=3),
     "expected turn 2"),
    ("alternative speaker not listed",
     lambda d: utterance(d, 2).update(alternative_speaker="pablo"),
     "unlisted speaker pablo"),
    ("alternative speaker is the speaker",
     lambda d: utterance(d, 2).update(alternative_speaker="luis"),
     "alternative speaker must be someone else"),
    ("gap in sentence indices",
     lambda d: (utterance(d, 1).update(id="t0.s2", index=2),
                utterance(d, 2).update(previous_utterance="t0.s2")),
     "expected index 1"),
]


@pytest.mark.parametrize(
    "change, message", [c[1:] for c in STRUCTURE_BREAKS],
    ids=[c[0] for c in STRUCTURE_BREAKS],
)
def test_turn_structure_is_enforced(
    validator: Draft202012Validator, change: Change, message: str,
) -> None:
    broken = copy.deepcopy(load(EXAMPLE_PATH))
    change(broken)
    validator.validate(broken)  # the schema alone lets it through
    with pytest.raises(ContractError, match=message):
        Transcript.from_contract(broken)


def set_in(path: Sequence[Any], value: Any) -> Change:
    """A change that sets the value at ``path`` of a document."""
    def change(document: Dict[str, Any]) -> None:
        target: Any = document
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
    return change


# Documents the JSON schema rejects for their types or values: the model
# must reject each too (pydantic, unless strict, would coerce many).
SCHEMA_BREAKS: List[Tuple[str, Change]] = [
    ("duration as a string", set_in(["session", "duration"], "14.2")),
    ("negative duration", set_in(["session", "duration"], -1)),
    ("enrolled as a string", set_in(["speakers", 0, "enrolled"], "true")),
    ("enrolled as a number", set_in(["speakers", 0, "enrolled"], 1)),
    ("started_at as a number",
     set_in(["session", "started_at"], 1763110920)),
    ("title as a number", set_in(["session", "title"], 5)),
    ("language too short", set_in(["session", "language"], "e")),
    ("bad audio hash", set_in(["session", "audio_sha256"], "ABC")),
    ("speaker name as a number", set_in(["speakers", 0, "name"], 3)),
    ("turn as a string", set_in(["utterances", 0, "turn"], "0")),
    ("index as a boolean", set_in(["utterances", 0, "index"], False)),
    ("start as a string", set_in(["utterances", 0, "start"], "0.4")),
    ("end as a boolean", set_in(["utterances", 0, "end"], True)),
    ("empty text", set_in(["utterances", 0, "text"], "")),
    ("confidence above 1", set_in(["utterances", 0, "confidence"], 1.5)),
    ("confidence as a string",
     set_in(["utterances", 0, "confidence"], "0.5")),
    ("speaker confidence as a string",
     set_in(["utterances", 0, "speaker_confidence"], "0.74")),
    ("previous speaker as a number",
     set_in(["utterances", 2, "previous_speaker"], 7)),
    ("speaker certainty above 2",
     set_in(["utterances", 0, "speaker_certainty"], 2.5)),
    ("speaker certainty as a string",
     set_in(["utterances", 0, "speaker_certainty"], "0.8")),
    ("doubtful as a string", set_in(["utterances", 0, "doubtful"], "no")),
    ("overlap as a number", set_in(["utterances", 0, "overlap"], 0)),
    ("empty alternative speaker",
     set_in(["utterances", 2, "alternative_speaker"], "")),
    ("alternative speaker as a number",
     set_in(["utterances", 2, "alternative_speaker"], 1)),
    ("unknown identification",
     set_in(["speakers", 0, "identification"], "guess")),
    ("identification as a number",
     set_in(["speakers", 0, "identification"], 1)),
    ("unknown overlap source",
     set_in(["utterances", 0, "overlap_source"], "ears")),
    ("language too short", set_in(["utterances", 0, "language"], "e")),
    ("language differs as a string",
     set_in(["utterances", 0, "language_differs"], "yes")),
    ("recogniser as a number", set_in(["pipeline", "recogniser"], 1)),
    ("schema as another version", set_in(["schema"], "transcript.v2")),
    ("utterances as an object", set_in(["utterances"], {})),
]


@pytest.mark.parametrize(
    "change", [c[1] for c in SCHEMA_BREAKS],
    ids=[c[0] for c in SCHEMA_BREAKS],
)
def test_what_the_schema_rejects_the_model_rejects(
    validator: Draft202012Validator, change: Change,
) -> None:
    broken = copy.deepcopy(load(EXAMPLE_PATH))
    change(broken)
    assert not validator.is_valid(broken)
    with pytest.raises(ContractError):
        Transcript.from_contract(broken)


def test_integral_floats_are_not_turn_numbers() -> None:
    """JSON Schema calls 0.0 an integer; the model, stricter, does not."""
    broken = copy.deepcopy(load(EXAMPLE_PATH))
    utterance(broken, 0)["turn"] = 0.0
    with pytest.raises(ContractError, match="turn"):
        Transcript.from_contract(broken)


def test_strict_reading_still_takes_whole_numbers_as_times() -> None:
    """A JSON number without a fraction is a valid ``number``."""
    example = copy.deepcopy(load(EXAMPLE_PATH))
    example["session"]["duration"] = 15
    utterance(example, 0)["start"] = 0
    transcript = Transcript.from_contract(example)
    assert transcript.session.duration == 15.0
    assert transcript.utterances[0].start == 0.0
