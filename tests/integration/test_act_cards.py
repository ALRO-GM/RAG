"""The theory cards on disk and the retrieval index over them (#94).

Checks the committed cards (every category, every sequence valid, no
example taken from the evaluation dialogues), the JSON loader's errors,
and an index built with the real BM25 adapter and a fake embedder.
"""

import json
import re
import shutil
import unicodedata
from pathlib import Path

import pytest

from src.adapters.outbound.bm25_index import BM25Builder
from src.adapters.outbound.json_act_cards import (
    DEFAULT_THEORY_DIR,
    JsonActCards,
)
from src.application.skills.rag_act_tagger import (
    ActCardIndex,
    RetrievalActTagger,
    RetrievalSettings,
)
from src.domain.errors import InvalidInputError, NotFoundError
from src.domain.skills.act_retrieval import FusionWeights, corpus_chunks
from src.domain.skills.act_tagging import sentence_inputs
from src.domain.skills.dialogue_acts import CATEGORIES
from tests.fakes import FakeEmbedder

DIALOGUES = "examples/skills/dialogue_acts/dialogues.json"


def normal(text: str) -> str:
    """Return lower-case words without accents or punctuation."""
    folded = unicodedata.normalize("NFKD", text.lower())
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    return " ".join(re.findall(r"\w+", folded))


def test_committed_cards_load_and_cover_every_category() -> None:
    """All 13 cards, sequences with pairs and windows, a stable version."""
    cards = JsonActCards()
    corpus = cards.load()
    assert sorted(c.category for c in corpus.cards) == sorted(CATEGORIES)
    assert len(corpus.sequences) >= 15
    views = {c.view for c in corpus_chunks(corpus)}
    assert views == {"single", "pair", "window"}
    for card in corpus.cards:
        assert card.cues and card.not_ and card.examples
    assert cards.version().startswith("theory-")
    assert cards.version() == JsonActCards(DEFAULT_THEORY_DIR).version()


def test_card_examples_are_not_evaluation_sentences() -> None:
    """No card example repeats a sentence of the Spanish set."""
    documents = json.loads(Path(DIALOGUES).read_text(encoding="utf-8"))
    evaluation = {normal(u["text"]) for d in documents
                  for u in d["utterances"]}
    examples = JsonActCards().load().example_texts()
    assert len(examples) > 300
    assert [e for e in examples if normal(e) in evaluation] == []


def test_loader_errors(tmp_path: Path) -> None:
    """Missing files, non-objects and invalid cards are reported."""
    with pytest.raises(NotFoundError):
        JsonActCards(str(tmp_path)).load()
    with pytest.raises(NotFoundError):
        JsonActCards(str(tmp_path)).version()
    (tmp_path / "categories.json").write_text("[]", encoding="utf-8")
    (tmp_path / "sequences.json").write_text("{}", encoding="utf-8")
    with pytest.raises(InvalidInputError):
        JsonActCards(str(tmp_path)).load()
    (tmp_path / "categories.json").write_text("{nope", encoding="utf-8")
    with pytest.raises(InvalidInputError):
        JsonActCards(str(tmp_path)).load()
    (tmp_path / "categories.json").write_text('{"cards": []}',
                                              encoding="utf-8")
    with pytest.raises(InvalidInputError):
        JsonActCards(str(tmp_path)).load()


def test_version_changes_with_a_card(tmp_path: Path) -> None:
    """Editing a card changes the version."""
    copy = tmp_path / "theory"
    shutil.copytree(DEFAULT_THEORY_DIR, copy)
    before = JsonActCards(str(copy)).version()
    path = copy / "sequences.json"
    path.write_text(path.read_text(encoding="utf-8") + "\n",
                    encoding="utf-8")
    assert JsonActCards(str(copy)).version() != before


def test_hybrid_index_with_real_bm25() -> None:
    """The committed cards, BM25 and a fake embedder tag a short talk."""
    corpus = JsonActCards().load()
    index = ActCardIndex(corpus_chunks(corpus), FakeEmbedder(dim=64),
                         RetrievalSettings(lexical_weight=0.5),
                         BM25Builder())
    tagger = RetrievalActTagger(index, FusionWeights(
        views={"single": 1.0, "pair": 1.0, "window": 0.5}))
    out = tagger.predict(sentence_inputs([
        ("a", "Ana", "¿A qué hora abre la oficina?"),
        ("b", "Luis", "A las nueve."),
        ("c", "Ana", "Mhm."),
    ]))
    assert len(out) == 3
    assert all(sum(p.values()) == pytest.approx(1.0) for p in out)
    assert max(out[0], key=lambda c: out[0][c]) == "pregunta"
