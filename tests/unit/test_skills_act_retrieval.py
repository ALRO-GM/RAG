"""Theory cards, chunks by position, votes and fusion (#94)."""

from typing import Dict, List

import numpy as np
import pytest
from pydantic import ValidationError

from src.domain.skills.act_rag_prompting import (
    judged_distribution,
    parse_weighted_acts,
    rag_system_prompt,
    rag_user_prompt,
)
from src.domain.skills.act_retrieval import (
    CardChunk,
    CategoryCard,
    FusionWeights,
    Hit,
    SequenceCard,
    SequencePosition,
    TheoryCorpus,
    act_terms,
    aligned_scores,
    card_chunks,
    change_agreement,
    closest_pairs,
    corpus_chunks,
    dialogue_chunks,
    fuse_views,
    offset_weights,
    sequence_chunks,
    top_hits,
    view_query,
    view_votes,
)
from src.domain.skills.act_tagging import ActInput, sentence_inputs
from src.domain.skills.dialogue_acts import (
    CATEGORIES,
    ActRecord,
    ContextLine,
    WeightedLabel,
)


def card(category: str, **extra: object) -> CategoryCard:
    """Return a minimal card of ``category``."""
    data: Dict[str, object] = {
        "category": category, "definition": f"def {category}",
        "function": f"fn {category}", "examples": [f"ej {category}"]}
    data.update(extra)
    return CategoryCard.model_validate(data)


def corpus() -> TheoryCorpus:
    """Return a corpus with all 13 cards and one sequence."""
    cards = [card(c) for c in CATEGORIES]
    cards[0] = card("pregunta", cues=["¿...?"], boundary=["frontera"],
                    **{"not": [{"category": "peticion",
                                "contrast": "pide una acción",
                                "examples": ["¿Me pasas la sal?"]}]})
    seq = SequenceCard(
        id="p-r", description="pregunta y respuesta",
        positions=[SequencePosition(category="pregunta", speaker="A"),
                   SequencePosition(category="respuesta", speaker="B"),
                   SequencePosition(category="escucha", speaker="A")],
        examples=[["¿Cuándo?", "El lunes.", "Mhm."]])
    return TheoryCorpus(cards=cards, sequences=[seq])


def test_cards_validate_categories_and_alias() -> None:
    """Unknown categories and self-contrasts are refused; ``not`` loads."""
    with pytest.raises(ValidationError):
        card("nada")
    with pytest.raises(ValidationError):
        card("pregunta", **{"not": [{"category": "pregunta",
                                     "contrast": "x"}]})
    with pytest.raises(ValidationError):
        card("pregunta", **{"not": [{"category": "nada", "contrast": "x"}]})
    c = corpus().card("pregunta")
    assert c.not_[0].category == "peticion"
    with pytest.raises(KeyError):
        TheoryCorpus.card(corpus(), "nada")


def test_corpus_needs_every_category_once() -> None:
    """A missing or repeated card, or a repeated sequence id, fails."""
    cards = [card(c) for c in CATEGORIES]
    with pytest.raises(ValidationError):
        TheoryCorpus(cards=cards[:-1])
    with pytest.raises(ValidationError):
        TheoryCorpus(cards=cards + [card("otro")])
    seq = corpus().sequences[0]
    with pytest.raises(ValidationError):
        TheoryCorpus(cards=cards, sequences=[seq, seq])


def test_sequence_examples_match_positions() -> None:
    """An example with the wrong number of texts is refused."""
    ask = SequencePosition(category="pregunta", speaker="A")
    answer = SequencePosition(category="respuesta", speaker="B")
    with pytest.raises(ValidationError):
        SequenceCard(id="x", description="d", positions=[ask, answer],
                     examples=[["solo una"]])
    with pytest.raises(ValidationError):
        SequenceCard(id="x", description="d", positions=[ask])
    with pytest.raises(ValidationError):
        SequencePosition(category="nada", speaker="A")


def test_card_chunks_vote_for_the_right_category() -> None:
    """Contrast examples vote for the neighbour; theory is optional."""
    chunks = card_chunks(corpus().card("pregunta"))
    kinds = {c.kind for c in chunks}
    assert kinds == {"definition", "cue", "boundary", "example"}
    neighbour = [c for c in chunks if c.texts == ["¿Me pasas la sal?"]]
    assert neighbour[0].category == "peticion"
    assert neighbour[0].source == "pregunta"
    bare = card_chunks(corpus().card("pregunta"), theory=False)
    assert {c.kind for c in bare} == {"example"}
    assert all(c.view == "single" for c in chunks)


def test_sequence_chunks_vote_only_for_the_last_position() -> None:
    """Pairs and windows end at each position; context is kept apart."""
    chunks = sequence_chunks(corpus().sequences[0])
    pairs = [c for c in chunks if c.view == "pair"]
    windows = [c for c in chunks if c.view == "window"]
    assert [(c.context, c.category) for c in pairs] == [
        (["pregunta"], "respuesta"), (["respuesta"], "escucha")]
    assert pairs[0].changes == [True]
    assert len(windows) == 1
    assert windows[0].texts == ["¿Cuándo?", "El lunes.", "Mhm."]
    assert windows[0].context == ["pregunta", "respuesta"]
    assert windows[0].category == "escucha"
    assert windows[0].changes == [True, True]
    every = corpus_chunks(corpus())
    assert {c.view for c in every} == {"single", "pair", "window"}


def test_chunk_shape_is_checked() -> None:
    """Views have their number of positions and aligned metadata."""
    with pytest.raises(ValidationError):
        CardChunk(id="x", view="single", kind="example", texts=["a", "b"],
                  changes=[True], category="pregunta", context=["otro"])
    with pytest.raises(ValidationError):
        CardChunk(id="x", view="pair", kind="example", texts=["a", "b"],
                  category="pregunta")
    with pytest.raises(ValidationError):
        CardChunk(id="x", view="window", kind="example", texts=["a", "b"],
                  changes=[True], category="pregunta", context=["otro"])
    with pytest.raises(ValidationError):
        CardChunk(id="x", view="pair", kind="example", texts=["a", "b"],
                  changes=[True], category="pregunta", context=["nada"])


def record(n: int, speaker: str, text: str, category: str,
           dialogue: str = "d1") -> ActRecord:
    """Return a record of our set."""
    return ActRecord(
        corpus="own", licence="x", non_commercial=False, language="es",
        dialogue=dialogue, id=f"t{n}", order=n, speaker=speaker, text=text,
        context=[], source_label=category, categories=[category],
        labels=[WeightedLabel(label=category, weight=100)])


def test_dialogue_chunks_follow_spoken_order() -> None:
    """Labelled dialogues give single, pair and window chunks in order."""
    rows = [record(2, "A", "Mhm.", "escucha"),
            record(0, "A", "¿Qué tal?", "pregunta"),
            record(1, "B", "Bien.", "respuesta"),
            record(0, "C", "Hola.", "relacion", "d2")]
    chunks = dialogue_chunks(rows)
    singles = [c for c in chunks if c.view == "single"]
    assert len(singles) == 4
    pair = [c for c in chunks if c.view == "pair" and c.source == "d1"]
    assert pair[0].texts == ["¿Qué tal?", "Bien."]
    assert pair[0].context == ["pregunta"]
    assert all(c.kind == "dialogue" for c in chunks)
    unusable = record(3, "B", "x", "otro").model_copy(
        update={"categories": [], "labels": []})
    assert len(dialogue_chunks([unusable])) == 0


def test_view_query_needs_context() -> None:
    """Pair needs one sentence before and window two."""
    first, second, third = sentence_inputs(
        [("a", "Ana", "¿Vienes?"), ("b", "Luis", "Sí."),
         ("c", "Luis", "A las tres.")])
    assert view_query(first, "pair") is None
    assert view_query(second, "window") is None
    pair = view_query(second, "pair")
    assert pair is not None
    assert pair.texts == ["¿Vienes?", "Sí."] and pair.changes == [True]
    window = view_query(third, "window")
    assert window is not None
    assert window.changes == [True, False]
    single = view_query(first, "single")
    assert single is not None and single.texts == ["¿Vienes?"]


def test_act_terms_keep_short_cues_and_question_marks() -> None:
    """Accents go, short words stay, a question mark becomes a term."""
    assert act_terms("¿Sí, vale?") == ["si", "vale", "?"]
    assert act_terms("Información.") == ["informacion"]


def test_offset_weights() -> None:
    """The target takes its share; the rest is split evenly."""
    assert offset_weights(0.6, 1) == [1.0]
    assert offset_weights(0.6, 3) == pytest.approx([0.6, 0.2, 0.2])
    with pytest.raises(ValueError):
        offset_weights(0.0, 2)


def test_aligned_scores_use_only_shared_offsets() -> None:
    """A chunk without the previous position is scored on the target."""
    sim0 = np.array([[1.0, 0.5]], dtype=np.float32)
    sim1 = np.array([[0.0, 1.0]], dtype=np.float32)
    q = [np.array([True]), np.array([True])]
    c = [np.array([True, True]), np.array([False, True])]
    out = aligned_scores([sim0, sim1], q, c, None, [0.5, 0.5])
    assert out[0, 0] == pytest.approx(1.0)
    assert out[0, 1] == pytest.approx(0.75)
    match = np.array([[1.0, 0.0]], dtype=np.float32)
    bonus = aligned_scores([sim0, sim1], q, c, match, [0.5, 0.5], 0.1)
    assert bonus[0, 0] == pytest.approx(1.1)
    with pytest.raises(ValueError):
        aligned_scores([], q, c, None, [1.0])


def test_change_agreement() -> None:
    """Speaker changes are compared from the target backwards."""
    item = ActInput(id="x", speaker="B", text="Sí.",
                    context=[ContextLine(speaker="A", text="¿Vienes?")])
    query = view_query(item, "pair")
    assert query is not None
    pair = sequence_chunks(corpus().sequences[0])[0]
    same = pair.model_copy(update={"changes": [False]})
    single = card_chunks(corpus().card("otro"))[0]
    out = change_agreement([query], [pair, same, single])
    assert out.tolist() == [[1.0, 0.0, 1.0]]


def test_top_hits_and_votes() -> None:
    """The best chunks vote for their category, near ties share."""
    chunks: List[CardChunk] = [
        card_chunks(card("pregunta"), theory=False)[0],
        card_chunks(card("respuesta"), theory=False)[0],
        card_chunks(card("pregunta"), theory=False)[0]]
    scores = np.array([[0.9, 0.8, 0.1]], dtype=np.float32)
    hits = top_hits(scores, 2)[0]
    assert [h.chunk for h in hits] == [0, 1]
    votes = view_votes(hits, chunks, sharpness=20.0)
    assert votes["pregunta"] > votes["respuesta"] > 0
    assert sum(votes.values()) == pytest.approx(1.0)
    assert view_votes([], chunks) == {}
    assert top_hits(np.zeros((1, 0), dtype=np.float32), 3) == [[]]
    with pytest.raises(ValueError):
        top_hits(scores, 0)
    assert Hit(chunk=0, score=1.0).chunk == 0


def test_fuse_views_weighs_and_falls_back() -> None:
    """Weighed views mix; without weighed votes the single view answers."""
    votes = {"single": {"pregunta": 1.0}, "pair": {"respuesta": 1.0}}
    both = fuse_views(votes, FusionWeights(
        views={"single": 1.0, "pair": 1.0}))
    assert both["pregunta"] == pytest.approx(both["respuesta"])
    pair_only = fuse_views({"single": {"pregunta": 1.0}},
                           FusionWeights(views={"pair": 1.0}))
    assert max(pair_only, key=lambda c: pair_only[c]) == "pregunta"
    empty = fuse_views({}, FusionWeights())
    assert empty["otro"] == pytest.approx(1 / 13)
    sharp = fuse_views(votes, FusionWeights(views={"single": 1.0},
                                            temperature=0.5))
    assert sharp["pregunta"] > fuse_views(
        votes, FusionWeights(views={"single": 1.0}))["pregunta"]
    with pytest.raises(ValidationError):
        FusionWeights(views={"single": 0.0})
    with pytest.raises(ValidationError):
        FusionWeights(views={"nada": 1.0})


def test_closest_pairs() -> None:
    """Pairs come sorted by cosine."""
    vectors = np.array([[1, 0], [0.8, 0.6], [0, 1]], dtype=np.float32)
    pairs = closest_pairs(["a", "b", "c"], vectors, top=2)
    assert pairs[0][:2] == ("a", "b")
    assert pairs[0][2] == pytest.approx(0.8)
    with pytest.raises(ValueError):
        closest_pairs(["a"], vectors)


def test_example_texts_cover_cards_contrasts_and_sequences() -> None:
    """Every example sentence is listed (for the disjointness check)."""
    texts = corpus().example_texts()
    assert "¿Me pasas la sal?" in texts and "Mhm." in texts


def test_prompt_marks_target_context_and_positions() -> None:
    """The target and the context are labelled; chunks show positions."""
    c = corpus()
    item = ActInput(id="x", speaker="Luis", text="El lunes.",
                    context=[ContextLine(speaker="Ana", text="¿Cuándo?")])
    pair = sequence_chunks(c.sequences[0])[0]
    window = [x for x in sequence_chunks(c.sequences[0])
              if x.view == "window"]
    single = card_chunks(c.card("respuesta"), theory=False)
    user = rag_user_prompt(item, [c.card("pregunta"), c.card("peticion")],
                           single, [pair], window)
    assert "CONTEXTO (no se etiqueta)" in user
    assert "[-1] Ana: ¿Cuándo?" in user
    assert "OBJETIVO [0], dicha por Luis (otro hablante" in user
    assert "no es peticion: pide una acción" in user
    assert "⇒ objetivo: respuesta" in user
    assert "(otro hablante) respuesta" in user
    assert "VENTANA" in user and "FRASE SOLA" in user
    alone = rag_user_prompt(ActInput(id="y", speaker="A", text="Hola."),
                            [c.card("relacion")], [], [], [])
    assert "(inicio)" in alone and "PAR" not in alone
    assert "respuesta" in rag_system_prompt()


def test_parse_weighted_acts() -> None:
    """Weights are normalised to 100; bad replies give None."""
    j = parse_weighted_acts(
        'Vale: {"labels": [{"act": "compromiso", "weight": 70}, '
        '{"act": "aceptacion", "weight": 20}], "reason": "ficha"}')
    assert j is not None
    assert j.labels == [("compromiso", 78), ("aceptacion", 22)]
    assert j.reason == "ficha"
    dist = judged_distribution(j)
    assert dist["compromiso"] == pytest.approx(0.78)
    single = parse_weighted_acts('{"act": "pregunta"}')
    assert single is not None and single.labels == [("pregunta", 100)]
    many = parse_weighted_acts(
        '{"labels": [{"act": "a"}, {"act": "otro", "weight": 10}, '
        '{"act": "otro", "weight": 5}, {"act": "escucha", "weight": "x"},'
        ' 3, {"act": "relacion", "weight": 30}, '
        '{"act": "gestion", "weight": 40}, {"act": "opinion", '
        '"weight": 20}]}')
    assert many is not None
    assert [a for a, _ in many.labels] == ["gestion", "relacion", "opinion"]
    assert sum(w for _, w in many.labels) == 100
    for bad in ("sin json", "{no es json}", "[1, 2]", '{"labels": 3}',
                '{"labels": [{"act": "nada"}]}'):
        assert parse_weighted_acts(bad) is None
