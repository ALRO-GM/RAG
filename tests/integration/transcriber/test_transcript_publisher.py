"""The directory publisher writes valid ``transcript.v1`` files."""

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from transcriber.adapters.outbound.fs_transcript_publisher import (
    DirectoryTranscriptPublisher,
)
from transcriber.domain.errors import ContractError, InvalidInputError
from transcriber.domain.models import Transcript

EXAMPLES = Path(__file__).resolve().parents[3] / "contracts" / "examples"


def example() -> Dict[str, Any]:
    """The first valid example document shipped with the contract."""
    for path in sorted(EXAMPLES.glob("*.json")):
        document: Dict[str, Any] = json.loads(path.read_text("utf-8"))
        try:
            Transcript.from_contract(document)
        except ContractError:
            continue
        return document
    raise AssertionError("no valid example in contracts/examples")


def test_writes_the_document_under_the_session_id(tmp_path: Path) -> None:
    target = tmp_path / "published"
    publisher = DirectoryTranscriptPublisher(target)
    document = example()
    publisher.publish(document)
    session_id = document["session"]["id"]
    path = target / f"{session_id}.transcript.v1.json"
    assert publisher.path_for(session_id) == path.resolve()
    assert json.loads(path.read_text("utf-8")) == document
    publisher.publish(document)  # publishing again replaces it
    assert [p.name for p in target.iterdir()] == [path.name]


def test_an_invalid_document_is_not_written(tmp_path: Path) -> None:
    publisher = DirectoryTranscriptPublisher(tmp_path)
    broken = example()
    broken["schema"] = "transcript.v0"
    with pytest.raises(ContractError):
        publisher.publish(broken)
    assert list(tmp_path.iterdir()) == []


def test_unsafe_session_ids_are_refused(tmp_path: Path) -> None:
    with pytest.raises(InvalidInputError, match="Invalid session id"):
        DirectoryTranscriptPublisher(tmp_path).path_for("../x")
