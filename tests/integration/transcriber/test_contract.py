"""Contract test: the model reads and writes ``transcript.v1`` as specified.

The JSON schema in ``contracts/`` is the only thing the transcriber and
the RAG share; this test holds the transcriber side to it.
"""

import json
from pathlib import Path
from typing import Any, Dict

import pytest
from jsonschema import Draft202012Validator, FormatChecker

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
