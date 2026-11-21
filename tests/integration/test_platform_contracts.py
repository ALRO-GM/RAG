"""Contract tests of the platform schemas in ``contracts/``.

Each schema is valid JSON Schema, accepts its example and rejects one
document that breaks a rule of the contract.
"""

import copy
import json
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import pytest
from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[2]


def _drop_utterance(doc: Dict[str, Any]) -> None:
    del doc["utterance"]


def _bad_role(doc: Dict[str, Any]) -> None:
    doc["role"] = "oracle"


def _bad_status(doc: Dict[str, Any]) -> None:
    doc["status"] = "great"


def _bad_signal(doc: Dict[str, Any]) -> None:
    doc["suggestions"][0]["signals"] = ["anger_from_voice"]


def _long_text(doc: Dict[str, Any]) -> None:
    doc["text"] = "x" * 1001


def _bad_priority(doc: Dict[str, Any]) -> None:
    doc["priority"] = "urgent"


def _free_text_option(doc: Dict[str, Any]) -> None:
    doc["choices"]["analysis.judge"] = "claude --dangerously-skip"


CASES: Dict[str, Callable[[Dict[str, Any]], None]] = {
    "live.event.v1": _drop_utterance,
    "reasoner.request.v1": _bad_role,
    "reasoner.response.v1": _bad_status,
    "tts.request.v1": _long_text,
    "gpu-lease.v1": _bad_priority,
    "settings.v1": _free_text_option,
}


def _load(name: str) -> Tuple[Draft202012Validator, Dict[str, Any]]:
    path = ROOT / "contracts" / f"{name}.schema.json"
    schema = json.loads(path.read_text("utf-8"))
    Draft202012Validator.check_schema(schema)
    sample = ROOT / "contracts" / "examples" / f"{name}.json"
    example = json.loads(sample.read_text("utf-8"))
    checker = FormatChecker()
    return Draft202012Validator(schema, format_checker=checker), example


@pytest.mark.parametrize("name", sorted(CASES))
def test_example_is_valid(name: str) -> None:
    validator, example = _load(name)
    validator.validate(example)


@pytest.mark.parametrize("name", sorted(CASES))
def test_invalid_document_is_rejected(name: str) -> None:
    validator, example = _load(name)
    bad = copy.deepcopy(example)
    CASES[name](bad)
    assert not validator.is_valid(bad)


def test_response_signals_are_accepted_and_closed() -> None:
    validator, example = _load("reasoner.response.v1")
    good = copy.deepcopy(example)
    good["suggestions"][0]["signals"] = ["complaint", "churn_risk"]
    validator.validate(good)
    bad = copy.deepcopy(example)
    _bad_signal(bad)
    assert not validator.is_valid(bad)
