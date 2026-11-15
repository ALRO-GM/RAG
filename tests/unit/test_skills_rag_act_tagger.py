"""Retrieval act taggers R and RG with fake parts (#94)."""

from typing import Dict, List, Optional

import pytest

from src.application.skills.rag_act_tagger import (
    ActCardIndex,
    RetrievalActTagger,
    RetrievalJudgeTagger,
    RetrievalSettings,
    answerer_as_question,
    candidate_categories,
    mix_judged,
)
from src.domain.skills.act_rag_prompting import JudgedActs
from src.domain.skills.act_retrieval import (
    CategoryCard,
    FusionWeights,
    SequenceCard,
    SequencePosition,
    TheoryCorpus,
    corpus_chunks,
)
from src.domain.skills.act_tagging import ActInput, sentence_inputs
from src.domain.skills.dialogue_acts import CATEGORIES
from tests.fakes import FakeChatModel, FakeEmbedder, FakeLexicalBuilder

EXAMPLES = {
    "pregunta": ["when does the truck arrive"],
    "respuesta": ["monday morning at nine"],
    "escucha": ["mhm mhm"],
}


def corpus() -> TheoryCorpus:
    """Return 13 cards (three with examples) and one sequence."""
    cards = [CategoryCard(category=c, definition=f"definition {c}",
                          function=f"function {c}",
                          examples=EXAMPLES.get(c, []))
             for c in CATEGORIES]
    seq = SequenceCard(
        id="q-a", description="question then answer",
        positions=[SequencePosition(category="pregunta", speaker="A"),
                   SequencePosition(category="respuesta", speaker="B"),
                   SequencePosition(category="escucha", speaker="A")],
        examples=[["when does the truck arrive", "monday morning at nine",
                   "mhm mhm"]])
    return TheoryCorpus(cards=cards, sequences=[seq])


def items() -> List[ActInput]:
    """Return a question, its answer and a back-channel."""
    return sentence_inputs([
        ("a", "Ana", "when does the truck arrive"),
        ("b", "Luis", "monday morning at nine"),
        ("c", "Ana", "mhm mhm"),
    ])


def index(lexical_weight: float = 0.0) -> ActCardIndex:
    """Return an index over the corpus with the fake embedder."""
    return ActCardIndex(corpus_chunks(corpus(), theory=False),
                        FakeEmbedder(dim=64),
                        RetrievalSettings(k=3,
                                          lexical_weight=lexical_weight),
                        FakeLexicalBuilder())


def test_index_needs_chunks_and_a_lexical_builder() -> None:
    """An empty index or BM25 without a builder is refused."""
    with pytest.raises(ValueError):
        ActCardIndex([], FakeEmbedder())
    with pytest.raises(ValueError):
        ActCardIndex(corpus_chunks(corpus()), FakeEmbedder(),
                     RetrievalSettings(lexical_weight=0.5))


def test_evidence_by_view() -> None:
    """The first sentence only has the single view; the others more."""
    found = index().evidence(items())
    assert set(found[0].votes) == {"single"}
    assert set(found[1].votes) == {"single", "pair"}
    assert set(found[2].votes) == {"single", "pair", "window"}
    assert max(found[1].votes["pair"],
               key=lambda c: found[1].votes["pair"][c]) == "respuesta"
    assert max(found[0].votes["single"],
               key=lambda c: found[0].votes["single"][c]) == "pregunta"


def test_hybrid_evidence_uses_bm25() -> None:
    """With a lexical share the same sentences are still found."""
    found = index(lexical_weight=0.5).evidence(items())
    assert max(found[2].votes["window"],
               key=lambda c: found[2].votes["window"][c]) == "escucha"


def test_retrieval_tagger_predicts_each_item() -> None:
    """R returns a full distribution and names its parts."""
    tagger = RetrievalActTagger(index(), FusionWeights(
        views={"single": 1.0, "pair": 1.0, "window": 1.0}))
    out = tagger.predict(items())
    assert [max(p, key=lambda c: p[c]) for p in out] == [
        "pregunta", "respuesta", "escucha"]
    assert all(sum(p.values()) == pytest.approx(1.0) for p in out)
    assert tagger.name == "rag"
    assert tagger.version == "theory-cards+fake-embedder"


def test_candidates_add_the_best_pair_and_window() -> None:
    """Prompt cards: the top 3 of the vote, then the pair/window target."""
    idx = index()
    ev = idx.evidence(items())[1]
    fused = {c: 0.0 for c in CATEGORIES}
    fused.update({"otro": 0.5, "gestion": 0.3, "relacion": 0.2})
    assert candidate_categories(ev, idx, fused) == [
        "otro", "gestion", "relacion", "respuesta"]


def test_judge_tagger_mixes_and_caches() -> None:
    """RG asks once per sentence, parses and mixes with R."""
    judge = FakeChatModel(replies=[
        '{"labels": [{"act": "pregunta", "weight": 100}], "reason": "r"}',
        "no json",
        '{"labels": [{"act": "escucha", "weight": 60}, '
        '{"act": "aceptacion", "weight": 40}]}'])
    retrieval = RetrievalActTagger(index(), FusionWeights())
    cache = DictCache()
    tagger = RetrievalJudgeTagger(retrieval, corpus(), judge, mix=0.2,
                                  cache=cache)
    out = tagger.predict(items())
    assert len(judge.calls) == 3
    system, user, _ = judge.calls[1]
    assert "OBJETIVO [0], dicha por Luis" in user
    assert "CONTEXTO (no se etiqueta)" in user
    assert max(out[0], key=lambda c: out[0][c]) == "pregunta"
    assert out[0]["pregunta"] >= 0.8
    assert max(out[2], key=lambda c: out[2][c]) == "escucha"
    verdicts, _ = tagger.verdicts(items())
    assert len(judge.calls) == 3
    assert verdicts[1].judged is None
    assert tagger.name == "rag-judge"
    assert tagger.version.endswith("+fake-judge")
    with pytest.raises(ValueError):
        RetrievalJudgeTagger(retrieval, corpus(), judge, mix=2.0)


class DictCache:
    """A cache in a dictionary."""

    def __init__(self) -> None:
        """Start empty."""
        self.data: Dict[str, str] = {}

    def get(self, key: str) -> Optional[str]:
        """Return a stored reply."""
        return self.data.get(key)

    def set(self, key: str, value: str) -> None:
        """Store a reply."""
        self.data[key] = value

    def clear(self) -> int:
        """Forget everything."""
        n = len(self.data)
        self.data.clear()
        return n


def test_mix_judged() -> None:
    """No usable reply keeps R; otherwise the judge dominates."""
    r = {c: 1 / 13 for c in CATEGORIES}
    assert mix_judged(None, r, 0.2) == pytest.approx(r)
    j = JudgedActs(labels=[("rechazo", 100)])
    out = mix_judged(j, r, 0.2)
    assert out["rechazo"] == pytest.approx(0.8 + 0.2 / 13)


def test_answerer_as_question_counts_only_answerers() -> None:
    """Only sentences after another speaker's pregunta are counted."""
    q = {c: 0.0 for c in CATEGORIES}
    q["pregunta"] = 1.0
    a = {c: 0.0 for c in CATEGORIES}
    a["respuesta"] = 1.0
    wrong, total = answerer_as_question(
        ["pregunta", "pregunta", "pregunta", None],
        ["respuesta", "respuesta", "pregunta", "respuesta"],
        [True, True, True, True], [q, a, q, q])
    assert (wrong, total) == (1, 2)
    assert answerer_as_question(["pregunta"], ["respuesta"], [False],
                                [q]) == (0, 0)
