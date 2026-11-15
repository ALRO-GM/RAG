"""Weighted dialogue-act labels: the format and its rules (#73)."""

import json

import pytest
from pydantic import ValidationError

from src.domain.skills.dialogue_acts import (
    ActRecord,
    CorpusInfo,
    SourceDialogue,
    SourceUtterance,
    WeightedLabel,
    check_annotation,
    check_labels,
    format_labels,
    own_categories,
    parse_labels,
    to_records,
)


def test_single_label_carries_one_hundred() -> None:
    assert parse_labels("compromiso") == [
        WeightedLabel(label="compromiso", weight=100)]
    assert parse_labels("compromiso:100") == parse_labels("compromiso")


def test_weighted_labels_keep_their_order() -> None:
    found = parse_labels("compromiso:70 aceptacion:30")
    assert [(x.label, x.weight) for x in found] == [
        ("compromiso", 70), ("aceptacion", 30)]
    assert format_labels(found) == "compromiso:70 aceptacion:30"
    assert format_labels(parse_labels("otro")) == "otro"
    three = "gestion:40 pregunta:30 relacion:30"
    assert format_labels(parse_labels(three)) == three


@pytest.mark.parametrize("text, message", [
    ("", "1 to 3 labels"),
    ("saludo", "unknown label"),
    ("compromiso aceptacion:30", "needs a weight"),
    ("compromiso:70 aceptacion", "needs a weight"),
    ("compromiso:x", "must be 1 to 100"),
    ("compromiso:0 otro:100", "must be 1 to 100"),
    ("compromiso:70 aceptacion:20", "add up to 90"),
    ("compromiso:50 compromiso:50", "repeated"),
    ("aceptacion:30 compromiso:70", "primary label goes first"),
    ("pregunta:40 respuesta:20 otro:20 gestion:20", "1 to 3 labels"),
])
def test_parse_labels_refuses(text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_labels(text)


def test_incomplete_labels_are_allowed_for_a_tagger() -> None:
    top = [WeightedLabel(label="otro", weight=60),
           WeightedLabel(label="pregunta", weight=30)]
    check_labels(top, complete=False)
    with pytest.raises(ValueError, match="not 100"):
        check_labels(top)
    with pytest.raises(ValueError, match="more than 100"):
        check_labels(top + [WeightedLabel(label="gestion", weight=20)],
                     complete=False)


@pytest.mark.parametrize("text, message", [
    ("compromiso:75 aceptacion:25", "steps of 10"),
    ("gestion:30 pregunta:30 relacion:30 otro:10", "1 to 3"),
    ("gestion:34 pregunta:33 relacion:33", "steps of 10"),
])
def test_annotation_rules(text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        check_annotation(parse_labels(text))


def test_annotation_primary_is_at_least_forty_by_construction() -> None:
    # Steps of 10, at most three labels, primary first, sum 100: the
    # smallest possible primary is 40 (40 + 30 + 30).
    check_annotation(parse_labels("gestion:40 pregunta:30 relacion:30"))
    with pytest.raises(ValueError):
        check_annotation(parse_labels("gestion:30 pregunta:30 relacion:30"))


def test_weighted_label_refuses_unknown_category_and_bad_weight() -> None:
    with pytest.raises(ValidationError):
        WeightedLabel(label="saludo", weight=100)
    with pytest.raises(ValidationError):
        WeightedLabel(label="otro", weight=0)


def test_own_set_maps_its_primary_label_and_carries_weights() -> None:
    info = CorpusInfo(name="own", licence="test", non_commercial=False,
                      language="es")
    weighted = parse_labels("compromiso:70 aceptacion:30")
    dialogue = SourceDialogue(id="d", utterances=[
        SourceUtterance(id="a", speaker="A", text="Vale, lo hago.",
                        source_label=format_labels(weighted),
                        labels=weighted),
        SourceUtterance(id="b", speaker="B", text="Bien.",
                        source_label="informacion"),
    ])
    first, second = to_records(info, dialogue, own_categories)
    assert first.categories == ["compromiso"]
    assert first.labels == weighted
    assert second.categories == ["informacion"] and second.labels == []
    assert own_categories(dialogue.utterances[0], None) == ("compromiso",)


def test_records_written_before_weights_still_load() -> None:
    old = {
        "corpus": "own", "licence": "x", "non_commercial": False,
        "language": "es", "dialogue": "d", "id": "a", "order": 0,
        "speaker": "A", "text": "Hola.", "context": [],
        "source_label": "relacion", "categories": ["relacion"],
        "responds_to": None, "session_type": None,
    }
    record = ActRecord.model_validate_json(json.dumps(old))
    assert record.labels == []
    again = ActRecord.model_validate_json(record.model_dump_json())
    assert again == record
