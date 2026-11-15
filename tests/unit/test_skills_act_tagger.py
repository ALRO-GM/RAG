"""Use cases of the dialogue-act tagger with fake parts (#82)."""

import json
from typing import Dict, List, Sequence

import pytest

from src.application.skills.act_tagger import (
    ACTS_FORMAT,
    ActPlace,
    CombinedTagger,
    LlmTagger,
    RuleTagger,
    TaggerCard,
    act_item,
    llm_outcome,
    tag_acts,
)
from src.application.skills.judges import ActClassifier
from src.domain.skills.act_calibration import fit_outcomes
from src.domain.skills.act_rules import RULES_VERSION
from src.domain.skills.act_sequence import session_from_json
from src.domain.skills.act_tagging import (
    ActInput,
    Probabilities,
    Reference,
    Route,
    sentence_inputs,
)
from src.domain.skills.dialogue_acts import CATEGORIES, ContextLine
from tests.fakes import FakeChatModel


def ref(category: str) -> Reference:
    """Return a one-category reference."""
    return Reference(categories=[category])


def items() -> List[ActInput]:
    """Return three sentences: a question, a back-channel, a statement."""
    return sentence_inputs([
        ("t0.s0", "Ana", "¿Cuándo llega el pedido?"),
        ("t1.s0", "Luis", "Ajá."),
        ("t1.s1", "Luis", "Llega el martes por la tarde."),
    ])


def rules() -> RuleTagger:
    """Return a rule tagger calibrated on a tiny table."""
    table = fit_outcomes(
        ["pregunta:question_mark", "escucha:backchannel_after_other",
         "default:statement", "default:statement"],
        [ref("pregunta"), ref("escucha"), ref("informacion"),
         ref("opinion")], smoothing=0.0)
    return RuleTagger(table)


class FixedTagger:
    """A tagger answering one fixed distribution for every sentence."""

    def __init__(self, probabilities: Dict[str, float]):
        """Keep the distribution."""
        self.probabilities = probabilities

    @property
    def name(self) -> str:
        """Return ``fixed``."""
        return "fixed"

    @property
    def version(self) -> str:
        """Return ``1``."""
        return "1"

    def predict(self, batch: Sequence[ActInput]) -> List[Probabilities]:
        """Return the fixed distribution for each item."""
        return [dict(self.probabilities) for _ in batch]


def test_rule_tagger_answers_calibrated_rows() -> None:
    tagger = rules()
    assert (tagger.name, tagger.version) == ("rules", RULES_VERSION)
    out = tagger.predict(items())
    assert out[0]["pregunta"] == 1.0
    assert out[1]["escucha"] == 1.0
    assert out[2]["informacion"] == 0.5 and out[2]["opinion"] == 0.5
    assert set(out[0]) == set(CATEGORIES)


def test_llm_tagger_calibrates_the_judges_label() -> None:
    judge = FakeChatModel(['{"act": "pregunta"}', "nada"], "qwen")
    table = fit_outcomes(["llm:pregunta", "llm:pregunta", "llm:?"],
                         [ref("pregunta"), ref("respuesta"), ref("otro")],
                         smoothing=0.0)
    tagger = LlmTagger(ActClassifier(judge), table)
    assert (tagger.name, tagger.version) == ("llm", "qwen")
    out = tagger.predict(items()[:2])
    assert out[0]["pregunta"] == 0.5 and out[0]["respuesta"] == 0.5
    assert out[1]["otro"] == 1.0
    assert "Frase objetivo, dicha por Ana" in judge.calls[0][1]
    assert llm_outcome(None) == "llm:?"
    assert act_item(items()[1]).context == [
        ("Ana", "¿Cuándo llega el pedido?")]


def test_combined_tagger_routes_and_says_who_answered() -> None:
    encoder = FixedTagger({"opinion": 0.7, "informacion": 0.3})
    combined = CombinedTagger(
        {"rules": rules(), "encoder": encoder},
        [Route(tagger="rules", categories=["pregunta", "escucha"])],
        fallback="encoder")
    out, sources = combined.predict_with_sources(items())
    assert sources == ["rules", "rules", "encoder"]
    assert out[2]["opinion"] == 0.7
    assert combined.predict(items())[0]["pregunta"] == 1.0
    assert combined.name == "combined"
    assert combined.version == f"encoder=1 rules={RULES_VERSION}"
    with pytest.raises(ValueError, match="llm"):
        CombinedTagger({"rules": rules()}, [Route(tagger="llm")], "rules")


def test_tag_acts_writes_top3_rest_and_doubtful() -> None:
    encoder = FixedTagger({"opinion": 0.55, "informacion": 0.3,
                           "propuesta": 0.1, "otro": 0.05})
    rates = {c: 0.05 for c in CATEGORIES}
    card = TaggerCard(name="encoder", rules=rules().table,
                      base_rates=rates, margin=0.2)
    thresholds = card.thresholds()
    assert thresholds["opinion"] == pytest.approx(0.25)
    document = tag_acts("s1", items()[:1], encoder, thresholds, 0.2,
                        note="n")
    sentence = document.utterances[0]
    assert document.schema_name == ACTS_FORMAT
    assert (document.tagger.name, document.tagger.version,
            document.tagger.note) == (
        "fixed", "1", "n")
    assert [(lab.label, lab.weight, lab.doubtful)
            for lab in sentence.labels] == [
        ("opinion", 55, False), ("informacion", 30, False),
        ("propuesta", 10, True)]
    assert sentence.rest == 5 and sentence.by == "fixed"
    data = json.loads(document.to_json())
    assert data["schema"] == "acts.v1" and data["session"] == {"id": "s1"}
    assert data["utterances"][0]["id"] == "t0.s0"
    assert data["utterances"][0]["responds_to"] is None


def test_tag_acts_records_the_route_of_each_sentence() -> None:
    combined = CombinedTagger(
        {"rules": rules(), "encoder": FixedTagger({"opinion": 1.0})},
        [Route(tagger="rules", categories=["pregunta"])], "encoder")
    rates = {c: 0.0 for c in CATEGORIES}
    document = tag_acts("s1", items(), combined, rates, 0.0)
    assert [s.by for s in document.utterances] == [
        "rules", "encoder", "encoder"]
    assert document.utterances[1].labels[0].label == "opinion"
    assert document.utterances[1].rest == 0


def test_sentence_inputs_keep_context_lines() -> None:
    assert items()[2].context == [
        ContextLine(speaker="Ana", text="¿Cuándo llega el pedido?"),
        ContextLine(speaker="Luis", text="Ajá.")]


def test_acts_document_loads_in_the_second_layer() -> None:
    combined = CombinedTagger(
        {"rules": rules(), "encoder": FixedTagger({"opinion": 1.0})},
        [Route(tagger="rules", categories=["pregunta"])], "encoder")
    places = [ActPlace(turn=0, start=1.0, end=2.0),
              ActPlace(turn=1, start=2.5, end=3.0),
              ActPlace(turn=1, start=3.0, end=4.0)]
    document = tag_acts("s1", items(), combined,
                        {c: 0.1 for c in CATEGORIES}, 0.0, places=places)
    session = session_from_json(json.loads(document.to_json()))
    assert session.id == "s1"
    assert [s.id for s in session.sentences] == ["t0.s0", "t1.s0", "t1.s1"]
    first = session.sentences[0]
    assert first.speaker == "Ana" and first.start == 1.0
    assert first.labels[0].label == "pregunta"
    with pytest.raises(ValueError):
        tag_acts("s1", items(), combined, {}, 0.0, places=places[:1])
