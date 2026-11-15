"""Steps d and e over the existing modules: provisional acts, events."""

import json
import re
from typing import List

import pytest

from src.application.skills.analysis_steps import (
    RULES_NAME,
    ProvisionalActTagger,
    RuleEventExtractor,
)
from src.application.skills.judges import ActClassifier
from src.config import SkillsSettings
from src.domain.errors import InvalidInputError
from src.domain.skills.speech import transcript_sentences
from src.domain.skills.transcript import Transcript
from tests.fakes import FakeChatModel, make_transcript

MEETING = (
    ("Ana", "¿Quién prepara el informe para el viernes?"),
    ("Luis", "Propongo que lo haga Marta, que tiene los datos."),
    ("Marta", "De acuerdo, me parece bien."),
    ("Luis", "Yo me encargo de revisar las cifras antes del jueves."),
    ("Ana", "Perfecto, gracias."),
)


def act_of(system: str, user: str) -> str:
    """Name an act from the target sentence (and fail on one)."""
    target = user.split("Frase objetivo", 1)[1]
    if "¿" in target:
        return json.dumps({"act": "pregunta"})
    if "Propongo" in target:
        return json.dumps({"act": "propuesta"})
    if "De acuerdo" in target:
        return json.dumps({"act": "aceptacion"})
    if re.search(r"me encargo", target):
        return json.dumps({"act": "compromiso"})
    return "no sé"


def meeting() -> Transcript:
    """The short Spanish meeting."""
    return make_transcript(MEETING, session_id="m1", language="es")


def test_the_provisional_tagger_writes_acts_v1() -> None:
    """One call per sentence; an unreadable reply is a doubtful otro."""
    judge = FakeChatModel(responder=act_of, model_name="judge-x")
    released: List[bool] = []
    judge.release = lambda: released.append(True)  # type: ignore[attr-defined]
    tagger = ProvisionalActTagger(lambda: ActClassifier(judge), context=2)
    transcript = meeting()
    sentences = transcript_sentences(transcript)
    out = tagger.tag(transcript, sentences)
    assert len(judge.calls) == len(sentences)
    assert "Ana: ¿Quién" in judge.calls[2][1]  # two sentences of context
    assert out.model == "judge-x (provisional tagger)"
    assert out.counts == {"sentences": len(sentences), "unparsed": 1}
    data = out.data
    assert data["schema"] == "acts.v1" and data["provisional"] is True
    assert data["session"] == {"id": "m1"}
    labels = [u["labels"][0] for u in data["utterances"]]
    assert labels[0] == {"label": "pregunta", "weight": 100}
    assert labels[-1] == {"label": "otro", "weight": 100, "doubtful": True}
    assert released == [True]


def test_events_come_from_the_acts_with_the_rules() -> None:
    """The acts of step d give a commitment and an answered question."""
    judge = FakeChatModel(responder=act_of)
    transcript = meeting()
    sentences = transcript_sentences(transcript)
    acts = ProvisionalActTagger(lambda: ActClassifier(judge)).tag(
        transcript, sentences
    ).data
    rules = SkillsSettings().event_rules()
    out = RuleEventExtractor(rules).extract(transcript, sentences, acts)
    assert out.model == RULES_NAME
    assert out.counts and all(v > 0 for v in out.counts.values())
    assert "events" in out.data and "summary" in out.data
    kinds = {e["type"] for e in out.data["events"]}
    assert "action_item" in kinds
    with pytest.raises(InvalidInputError, match="step d"):
        RuleEventExtractor(rules).extract(transcript, sentences, None)
