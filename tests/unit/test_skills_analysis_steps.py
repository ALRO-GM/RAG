"""Steps d and e over the existing modules: the #82 tagger, #83 events."""

from typing import Dict, List, Sequence

import pytest

from src.application.skills.act_tagger import RuleTagger, TaggerCard
from src.application.skills.analysis_steps import (
    RULES_NAME,
    RuleEventExtractor,
    TaggerActStep,
)
from src.config import SkillsSettings
from src.domain.errors import InvalidInputError
from src.domain.skills.act_calibration import fit_outcomes
from src.domain.skills.act_tagging import ActInput, Probabilities, Reference
from src.domain.skills.dialogue_acts import CATEGORIES
from src.domain.skills.speech import transcript_sentences
from src.domain.skills.transcript import Transcript
from tests.fakes import make_transcript

MEETING = (
    ("Ana", "¿Quién prepara el informe para el viernes?"),
    ("Luis", "Propongo que lo haga Marta, que tiene los datos."),
    ("Marta", "De acuerdo, me parece bien."),
    ("Luis", "Yo me encargo de revisar las cifras antes del jueves."),
    ("Ana", "Perfecto, gracias."),
)

ACTS = {
    "¿Quién": "pregunta", "Propongo": "propuesta", "De acuerdo": "aceptacion",
    "me encargo": "compromiso",
}


class KeywordTagger:
    """A fake encoder: a sure act for known openings, unsure otherwise."""

    name = "encoder"
    version = "test-1"

    def __init__(self) -> None:
        """Not released yet."""
        self.released = False

    def predict(self, items: Sequence[ActInput]) -> List[Probabilities]:
        """Return 0.9 on the matching act, or an even spread."""
        out: List[Dict[str, float]] = []
        for item in items:
            act = next((a for k, a in ACTS.items() if k in item.text), None)
            if act is None:
                out.append({c: 1 / len(CATEGORIES) for c in CATEGORIES})
            else:
                rest = 0.1 / (len(CATEGORIES) - 1)
                out.append({c: 0.9 if c == act else rest for c in CATEGORIES})
        return out

    def release(self) -> None:
        """Remember the release."""
        self.released = True


def card() -> TaggerCard:
    """A small card: two rule rows, even base rates, margin 0.1."""
    table = fit_outcomes(
        ["pregunta:question_mark", "default:statement"],
        [Reference(categories=["pregunta"]),
         Reference(categories=["informacion"])], smoothing=0.5)
    return TaggerCard(
        name="encoder", rules=table,
        base_rates={c: 1 / len(CATEGORIES) for c in CATEGORIES}, margin=0.1,
    )


def meeting() -> Transcript:
    """The short Spanish meeting."""
    return make_transcript(MEETING, session_id="m1", language="es")


def test_the_tagger_step_writes_acts_v1_with_who_answered() -> None:
    """Weighted labels, doubtful counted, the approach and margin noted."""
    tagger = KeywordTagger()
    transcript = meeting()
    sentences = transcript_sentences(transcript)
    out = TaggerActStep(lambda: (tagger, ""), card).tag(transcript, sentences)
    assert out.model == "encoder (test-1)"
    assert out.counts == {
        "sentences": len(sentences), "doubtful": 1, "by_encoder": 5,
    }
    assert out.note == "Doubtful margin 0.1."
    assert out.data["schema"] == "acts.v1"
    assert out.data["session"] == {"id": "m1"}
    first = out.data["utterances"][0]
    assert first["labels"][0]["label"] == "pregunta"
    assert first["turn"] == 0 and first["speaker"] == "Ana"
    assert out.data["utterances"][-1]["labels"][0]["doubtful"] is True
    assert tagger.released


def test_the_rules_answer_when_the_weights_are_missing() -> None:
    """The note of the container's choice reaches the record."""
    transcript = meeting()
    note = "No fine-tuned weights in x: the rules answered instead."
    out = TaggerActStep(
        lambda: (RuleTagger(card().rules), note), card
    ).tag(transcript, transcript_sentences(transcript))
    assert out.model.startswith("rules (")
    assert out.counts["by_rules"] == 5
    assert out.note.endswith(note)


def test_events_come_from_the_acts_with_the_rules() -> None:
    """The acts of step d give an action item."""
    transcript = meeting()
    sentences = transcript_sentences(transcript)
    acts = TaggerActStep(lambda: (KeywordTagger(), ""), card).tag(
        transcript, sentences
    ).data
    rules = SkillsSettings().event_rules()
    out = RuleEventExtractor(rules).extract(transcript, sentences, acts)
    assert out.model == RULES_NAME
    assert out.counts and all(v > 0 for v in out.counts.values())
    kinds = {e["type"] for e in out.data["events"]}
    assert "action_item" in kinds
    with pytest.raises(InvalidInputError, match="step d"):
        RuleEventExtractor(rules).extract(transcript, sentences, None)
