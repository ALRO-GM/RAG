"""An owner's list checked by the RAG (#294): sent with the sentence,
the closest-item rule unchanged, and the examples close to another
item's (a warning for the editor)."""

from typing import Any, Dict

import pytest

from src.application.knowledge.mandatory import KEEP_LISTS, MandatoryItems
from src.domain.errors import InvalidInputError, NotFoundError
from src.domain.knowledge.domains import DomainManifest
from src.domain.knowledge.mandatory import MandatoryList, close_examples
from tests.fakes import FakeEmbedder

OWNED: Dict[str, Any] = {
    "version": 4, "threshold": 0.6, "remind_seconds": 30,
    "items": [
        {"id": "tarifa", "text": "Precio de la tarifa", "phase": "during",
         "must_say": ["la tarifa cuesta veinte euros al mes",
                      "el precio mensual es de veinte euros"]},
        {"id": "permanencia", "text": "Sin permanencia",
         "must_say": ["no tiene permanencia ninguna"]},
    ],
}


def no_domain(domain_id: str) -> DomainManifest:
    """No domain is known (the sent list does not need one)."""
    raise NotFoundError(domain_id)


def test_a_sent_list_is_checked_like_a_domain_list() -> None:
    embedder = FakeEmbedder()
    items = MandatoryItems(no_domain, embedder)
    found = items.check_list(OWNED, "La tarifa cuesta veinte euros al mes.")
    assert found["version"] == 4 and found["threshold"] == 0.6
    said = {i["id"]: i["said"] for i in found["items"]}  # type: ignore
    assert said == {"tarifa": True, "permanencia": False}
    calls = embedder.calls
    items.check_list(OWNED, "Otra frase cualquiera.")
    assert embedder.calls == calls + 1  # the examples' vectors are kept


def test_only_the_last_lists_are_kept() -> None:
    embedder = FakeEmbedder()
    items = MandatoryItems(no_domain, embedder)
    for n in range(KEEP_LISTS + 2):
        items.check_list({**OWNED, "version": n + 1}, "hola")
    assert len(items._sent) == KEEP_LISTS


@pytest.mark.parametrize("listed, text", [
    ({"items": []}, "hola"),
    ({"version": 1, "items": [{"id": "Bad Id", "text": "x",
                               "must_say": ["y"]}]}, "hola"),
    (OWNED, ""),
    (OWNED, "x" * 2001),
])
def test_bad_sent_lists_and_texts(listed: Dict[str, Any], text: str) -> None:
    with pytest.raises(InvalidInputError):
        MandatoryItems(no_domain, FakeEmbedder()).check_list(listed, text)


def test_a_list_that_is_not_an_object_is_refused() -> None:
    with pytest.raises(InvalidInputError):
        MandatoryItems(no_domain, FakeEmbedder()).review(
            ["not", "a", "list"])  # type: ignore[arg-type]


def test_close_examples_name_the_closest_other_item_once() -> None:
    listed = MandatoryList.model_validate(OWNED)
    matrix = [
        [1.0, 0.9, 0.7],
        [0.9, 1.0, 0.5],
        [0.7, 0.5, 1.0],
    ]
    close = close_examples(listed, matrix)
    assert [(c.item, c.other, c.score) for c in close] == [
        ("tarifa", "permanencia", 0.7), ("permanencia", "tarifa", 0.7)]
    assert close_examples(listed, matrix, threshold=0.75) == []
    with pytest.raises(ValueError):
        close_examples(listed, [[1.0]])


def test_review_uses_the_embedding_of_the_examples() -> None:
    same = {**OWNED, "items": [
        OWNED["items"][0],
        {"id": "otra", "text": "Otra", "must_say": [
            "la tarifa cuesta veinte euros al mes"]},
    ]}
    found = MandatoryItems(no_domain, FakeEmbedder()).review(same)
    assert found["threshold"] == 0.6
    pairs = {(c["item"], c["other"]) for c in found["close"]}  # type: ignore
    assert ("otra", "tarifa") in pairs and ("tarifa", "otra") in pairs
    apart = MandatoryItems(no_domain, FakeEmbedder()).review(OWNED)
    assert apart["close"] == []
