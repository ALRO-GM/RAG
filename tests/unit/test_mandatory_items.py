"""Mandatory script items (LV-16): the manifest block, the verdict rule,
the use case over a fake embedder and the ``@2`` suggester modes."""

import json
from typing import Any, Dict, List

import pytest
import yaml
from pydantic import ValidationError

from src.adapters.outbound.serial_embedder import SerialEmbedder
from src.application.knowledge.mandatory import MandatoryItems
from src.application.reasoning import Reasoner
from src.domain.errors import InvalidInputError, NotFoundError
from src.domain.knowledge.domains import DomainManifest, index_version
from src.domain.knowledge.mandatory import (
    MandatoryList,
    check_item_id,
    describe,
    item_scores,
)
from src.domain.reasoning.roles import ReasonRequest
from src.domain.reasoning.templates import (
    PINCELADA_WORDS,
    get_template,
)
from tests.fakes import FakeChatModel, FakeEmbedder

LIST: Dict[str, Any] = {
    "version": 3, "threshold": 0.6, "remind_seconds": 30,
    "items": [
        {"id": "aviso-datos", "text": "Aviso de datos", "phase": "opening",
         "must_say": ["sus datos se tratan según la ley",
                      "le informo del tratamiento de datos"]},
        {"id": "desistimiento", "text": "14 días",
         "must_say": ["tiene catorce días para desistir"]},
    ],
}


def manifest(mandatory: Any = None) -> DomainManifest:
    data: Dict[str, Any] = {"id": "d", "title": "D"}
    if mandatory is not None:
        data["mandatory"] = mandatory
    return DomainManifest.model_validate(data)


def test_the_list_is_validated() -> None:
    listed = MandatoryList.model_validate(LIST)
    assert listed.examples()[2] == "tiene catorce días para desistir"
    assert listed.to_json()["version"] == 3
    bad: List[Dict[str, Any]] = [
        {**LIST, "items": []},
        {**LIST, "version": 0},
        {**LIST, "threshold": 1.0},
        {**LIST, "items": [LIST["items"][0], LIST["items"][0]]},
        {**LIST, "items": [{**LIST["items"][0], "must_say": [" "]}]},
        {**LIST, "items": [{**LIST["items"][0], "id": "Bad Id"}]},
        {**LIST, "extra": 1},
    ]
    for case in bad:
        with pytest.raises(ValidationError):
            MandatoryList.model_validate(case)
    assert check_item_id("aviso-datos") and not check_item_id("../x")


def test_only_the_closest_item_over_the_threshold_is_said() -> None:
    listed = MandatoryList.model_validate(LIST)
    scores = item_scores(listed, [0.65, 0.3, 0.7])
    assert [(s.id, s.score, s.said) for s in scores] == [
        ("aviso-datos", 0.65, False), ("desistimiento", 0.7, True)]
    low = item_scores(listed, [0.5, 0.2, 0.1])
    assert not any(s.said for s in low)
    with pytest.raises(ValueError):
        item_scores(listed, [0.1])


def test_the_items_do_not_change_the_index_version() -> None:
    assert index_version(manifest(), "f") == index_version(
        manifest(LIST), "f")
    assert describe("d", None) == {"domain": "d", "version": 0,
                                   "items": []}


def test_the_seed_domains_list_their_items() -> None:
    for domain, count in (("support.demo", 4), ("interview.generic", 3)):
        path = f"knowledge/domains/{domain}/manifest.yaml"
        with open(path, encoding="utf-8") as handle:
            seed = DomainManifest.model_validate(yaml.safe_load(handle))
        assert seed.mandatory is not None
        assert len(seed.mandatory.items) == count
        assert seed.mandatory.threshold == 0.575


def test_a_sentence_is_checked_against_every_example() -> None:
    embedder = FakeEmbedder()
    found = {"d": manifest(LIST), "plain": manifest()}

    def lookup(domain: str) -> DomainManifest:
        if domain not in found:
            raise NotFoundError("no domain")
        return found[domain]

    items = MandatoryItems(lookup, SerialEmbedder(embedder))
    assert items.items("d")["version"] == 3
    assert items.items("plain")["items"] == []
    out = items.check("d", "Tiene catorce días para desistir")
    said = [i for i in out["items"] if i["said"]]  # type: ignore[attr-defined]
    assert [i["id"] for i in said] == ["desistimiento"]
    assert out["threshold"] == 0.6 and out["version"] == 3
    calls = embedder.calls
    items.check("d", "otra frase")
    assert embedder.calls == calls + 1  # the examples are cached
    for text in ("", "x" * 2001):
        with pytest.raises(InvalidInputError):
            items.check("d", text)
    with pytest.raises(NotFoundError):
        items.check("plain", "hola")
    with pytest.raises(NotFoundError):
        items.check("nope", "hola")


# ---- the @2 templates: two assistance modes ---------------------------------

WINDOW = [
    {"id": "t0.s0", "speaker": "S1", "text": "Cuénteme."},
    {"id": "t1.s0", "speaker": "S2",
     "text": "Me cobraron dos veces y estoy harto."},
]


def request(template: str, mode: Any = None) -> ReasonRequest:
    constraints: Dict[str, Any] = {"max_suggestions": 1}
    if mode:
        constraints["mode"] = mode
    return ReasonRequest.model_validate({
        "request_id": "r", "role": "suggester", "template": template,
        "language": "es", "constraints": constraints,
        "context": {"window": WINDOW,
                    "speaker_roles": {"S1": "agent", "S2": "customer"}},
        "retrieved": [{"id": "kb:1", "text": "Devolución en 5 días."}],
    })


def run(req: ReasonRequest, text: str, signals: Any = None) -> Any:
    item: Dict[str, Any] = {"kind": "answer", "text": text,
                            "cites": ["kb:1"], "about": ["t1.s0"]}
    if signals:
        item["signals"] = signals
    model = FakeChatModel([json.dumps({"suggestions": [item]})])
    return Reasoner(lambda alias: model).reason(req), model


def test_the_mode_is_in_the_user_prompt_not_the_system_prompt() -> None:
    for template in ("suggester.support@2", "suggester.interview@2"):
        pincel = get_template(template, "suggester")
        a, b = request(template, "pincelada"), request(template, "guiado")
        assert pincel.system(a) == pincel.system(b)
        assert "Mode: pincelada" in pincel.user(a)
        assert "Mode: guiado" in pincel.user(b)
        assert "Mode: pincelada" in pincel.user(request(template))
        assert "At most 1 suggestion" in pincel.user(a)


def test_a_long_pincelada_is_cut_and_a_guided_sentence_is_kept() -> None:
    long = "Ofrezca la devolución del cobro duplicado hoy mismo por favor"
    out, _ = run(request("suggester.support@2", "pincelada"), long,
                 ["complaint"])
    text = out.suggestions[0].text
    assert len(text.rstrip("…").split()) == PINCELADA_WORDS
    assert text.endswith("…") and out.suggestions[0].signals == ["complaint"]
    guided, model = run(request("suggester.support@2", "guiado"), long)
    assert guided.suggestions[0].text == long
    assert model.calls[0][2] <= 240


def test_the_interview_modes_never_rate_or_label() -> None:
    template = get_template("suggester.interview@2", "suggester")
    system = template.system(request("suggester.interview@2"))
    for word in ("Never evaluate", "rate", "emotions", "personality"):
        assert word in system
    out, _ = run(request("suggester.interview@2", "guiado"),
                 "¿Qué hizo usted exactamente?", ["complaint"])
    assert out.suggestions[0].signals == []
    short, _ = run(request("suggester.interview@2"),
                   "Pregunte por el detalle técnico del despliegue en "
                   "producción")
    assert short.suggestions[0].text.endswith("…")


def test_an_unknown_mode_breaks_the_contract() -> None:
    with pytest.raises(ValidationError):
        request("suggester.support@2", "loud")
