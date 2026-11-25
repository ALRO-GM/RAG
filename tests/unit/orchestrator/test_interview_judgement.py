"""How accurate the answers were: reading the judge's reply (#259)."""

from typing import Any, Dict, List

from orchestrator.domain.interview import (
    Answer,
    Judgement,
    SheetPoint,
    SheetSlot,
)
from orchestrator.domain.interview_judgement import (
    accuracy_counts,
    capped_score,
    clean_text,
    is_experience,
    judge_slots,
    judgement_from,
    label_of,
)

SAID = {
    "u1": "El drop es la diferencia de altura entre el talón y la puntera.",
    "u2": "Trabajé tres años en una tienda de running en Valencia.",
}


def running(score: int = 80) -> SheetSlot:
    return SheetSlot(
        id="s1", label="Material de running", level="minimo",
        status="filled", score=score, above=True,
        answers=[Answer(utterance="u1", quote="El drop es la diferencia",
                        status="complete"),
                 Answer(utterance="u2", quote="Trabajé tres años",
                        status="partial")],
        points=[SheetPoint(id="p1", text="Amortiguación y drop",
                           status="treated", quote="El drop",
                           utterance="u1"),
                SheetPoint(id="p2", text="Tipos de pisada"),
                SheetPoint(id="p3", text="Cómo elegir la talla",
                           status="unknown")],
    )


def test_labels_are_clamped_to_the_known_ones() -> None:
    assert label_of("solid") == "solid"
    assert label_of("Basic") == "correct_basic"
    assert label_of("off topic") == "off_topic"
    assert label_of("excellent") == "unclear"
    assert label_of(None) is None and label_of("") is None


def test_a_judgement_rests_on_the_candidate_s_own_words() -> None:
    found = judgement_from({"label": "solid", "utterance": "u9",
                            "quote": "diferencia de altura",
                            "reason": "Correcto y preciso.",
                            "cites": ["c1", "c9", 3]}, SAID, {"c1"})
    assert found is not None
    assert found.utterance == "u1" and found.quote == "diferencia de altura"
    assert found.basis == "documents" and found.cites == ["c1"]
    invented = judgement_from({"label": "solid", "utterance": "u1",
                               "quote": "palabras que no dijo"}, SAID, set())
    assert invented is not None and invented.quote == SAID["u1"]
    assert invented.basis == "general" and invented.cites == []
    nobody = judgement_from({"label": "wrong", "quote": "nada"}, SAID, set())
    assert nobody is not None and nobody.utterance is None
    assert nobody.quote == ""
    assert judgement_from("junk", SAID, set()) is None
    assert judgement_from({"reason": "sin etiqueta"}, SAID, set()) is None


def test_an_experience_claim_is_never_wrong() -> None:
    assert is_experience(SAID["u2"]) and not is_experience(SAID["u1"])
    by_kind = judgement_from({"label": "wrong", "utterance": "u1",
                              "kind": "experience"}, SAID, set())
    by_words = judgement_from({"label": "wrong", "utterance": "u2"},
                              SAID, set())
    fact = judgement_from({"label": "wrong", "utterance": "u1"}, SAID, set())
    assert by_kind is not None and by_kind.label == "to_verify"
    assert by_words is not None and by_words.label == "to_verify"
    assert fact is not None and fact.label == "wrong"


def test_a_verdict_on_the_person_is_never_kept() -> None:
    for verdict in ("Miente sobre su experiencia.", "Se lo ha inventado.",
                    "No apto para el puesto.", "Conviene contratarla.",
                    "The candidate is lying.",
                    "Not suitable for the role.",
                    "Parece que inventa los datos.", "Está inventando.",
                    "Inventa cifras.", "Probablemente mienta.",
                    "No la contrataría.", "Descartaría a la candidata.",
                    "Fabricó la respuesta.", "She invented that.",
                    "The answer looks likely dishonest."):
        assert clean_text(verdict) == "", verdict
    assert clean_text("  El drop no es el peso  ") == "El drop no es el peso"
    found = judgement_from({"label": "wrong", "utterance": "u1",
                            "reason": "Miente: el drop no es eso."},
                           SAID, set())
    assert found is not None and found.reason == ""


def test_the_score_rule_never_lets_a_wrong_answer_score_as_covered() -> None:
    slot = running()
    assert capped_score(None, slot) is None
    assert capped_score(90, slot) == 90  # not judged: the model's score
    for label, cap in (("wrong", 30), ("off_topic", 20), ("unclear", 50),
                       ("correct_basic", 75), ("solid", 100),
                       ("to_verify", 100)):
        judged = slot.model_copy(update={
            "judgement": Judgement(label=label)})  # type: ignore[arg-type]
        assert capped_score(100, judged) == cap, label
    point = slot.points[0].model_copy(update={
        "judgement": Judgement(label="wrong")})
    wrong_point = slot.model_copy(update={
        "judgement": Judgement(label="solid"),
        "points": [point, *slot.points[1:]]})
    assert capped_score(95, wrong_point) == 50


def test_counts_cover_labels_and_coverage() -> None:
    slot = running()
    assert accuracy_counts(slot) == {"not_treated": 1, "does_not_know": 1}
    plain = SheetSlot(id="s2", label="Caja", level="minimo",
                      judgement=Judgement(label="solid"))
    assert accuracy_counts(plain) == {"solid": 1}
    assert accuracy_counts(SheetSlot(id="s3", label="x",
                                     level="minimo")) == {}


def reply() -> Dict[str, Any]:
    return {
        "slots": [
            {"id": "s1", "judgement": {
                "label": "wrong", "utterance": "u1", "quote": "El drop",
                "reason": "El drop no es el peso: es la diferencia de "
                          "altura.", "kind": "fact"},
             "points": [
                 {"id": "p1", "label": "solid", "utterance": "u1",
                  "reason": "Define bien el drop."},
                 {"id": "p2", "label": "solid", "utterance": "u1"},
                 {"id": "p3", "label": "wrong", "utterance": "u1"},
                 "junk"]},
            {"id": "s2", "judgement": {"label": "solid"}},
        ],
        "to_check": [
            {"utterance": "u2", "quote": "tres años", "why": "wrong",
             "note": "Comprobar."},
            {"utterance": "u1", "quote": "diferencia de altura",
             "why": "contradiction", "note": "No encaja con lo de antes."},
            {"utterance": "u1", "quote": "invented words", "why": "vague"},
            {"utterance": "u1", "quote": "talón", "why": "lies"},
            "junk",
        ],
    }


def test_judge_slots_merges_caps_and_lists_what_to_check() -> None:
    unanswered = SheetSlot(id="s2", label="Caja", level="minimo", score=60)
    slots, checks = judge_slots(reply(), [running(), unanswered],
                                {"s1": ["c1"]}, SAID)
    first, second = slots
    assert first.judgement is not None and first.judgement.label == "wrong"
    assert first.judgement.quote == "El drop"
    assert [p.judgement.label if p.judgement else None
            for p in first.points] == ["solid", None, None]
    assert first.score == 30 and first.above is False
    assert first.accuracy == {"solid": 1, "not_treated": 1,
                              "does_not_know": 1}
    assert second.judgement is None and second.score == 60  # no answers
    assert [(c.why, c.quote) for c in checks] == [
        ("contradiction", "diferencia de altura"), ("wrong", "El drop")]
    assert checks[0].slot == "s1" and checks[1].note.startswith("El drop")


def test_a_reply_without_judgements_changes_nothing_but_counts() -> None:
    slots, checks = judge_slots({}, [running()], {}, None)
    assert slots[0].judgement is None and slots[0].score == 80
    assert slots[0].accuracy == {"not_treated": 1, "does_not_know": 1}
    assert checks == []
    many: List[Dict[str, Any]] = [
        {"utterance": "u1", "quote": f"{w}", "why": "vague"}
        for w in SAID["u1"].split()]
    _, checks = judge_slots({"to_check": many}, [running()], {}, SAID)
    assert len(checks) == 10


def test_domain_sentences_that_look_like_verdicts_are_kept() -> None:
    """Review of #259: the filter aims at the person, not at words."""
    for fine in ("Confunde el inventario FIFO con LIFO.",
                 "El envase debe ser apto para alimentos.",
                 "Rechazo de pedidos: explica bien el proceso.",
                 "Descarta bien las tallas que no sirven.",
                 "The cable is suitable for outdoor use.",
                 "Habla del engranaje y del enganche del remolque.",
                 "Inventó un método de reposición: no, lo describe bien."):
        assert clean_text(fine) == " ".join(fine.split()), fine
    for verdict in ("Se lo ha inventado todo.", "Está mintiendo.",
                    "No es apta para el puesto.", "Yo no la contrataría; "
                    "no recomiendo contratar.", "Mejor descartarla.",
                    "He made it up.", "She should not be hired.",
                    "Reject this candidate."):
        assert clean_text(verdict) == "", verdict


def test_english_experience_claims_are_to_verify() -> None:
    said = {"e1": "I worked three years at Amazon as a picker.",
            "e2": "I've been a team lead for two years.",
            "e3": "I have 5 years of experience with Django.",
            "e4": "An index is a sorted structure that speeds up reads."}
    for uid in ("e1", "e2", "e3"):
        assert is_experience(said[uid]), uid
        found = judgement_from({"label": "wrong", "utterance": uid},
                               said, set())
        assert found is not None and found.label == "to_verify", uid
    assert not is_experience(said["e4"])


def test_a_short_quote_without_an_id_names_no_utterance() -> None:
    found = judgement_from({"label": "solid", "quote": "e"}, SAID, set())
    assert found is not None and found.utterance is None
    assert found.quote == ""
    longer = judgement_from({"label": "solid", "quote": "entre el talón"},
                            SAID, set())
    assert longer is not None and longer.utterance == "u1"


def test_a_reply_of_the_wrong_shape_never_raises() -> None:
    """Review of #259: every field of the reply is read defensively."""
    for odd in (
        {"slots": [{"id": "s1", "points": 3, "judgement": [1]}]},
        {"slots": [{"id": "s1", "points": True,
                    "judgement": {"label": {"x": 1}, "reason": ["a"],
                                  "utterance": 7, "quote": None,
                                  "kind": [], "cites": "c1"}}]},
        {"slots": [{"id": "s1", "points": [{"id": "p1", "label": 3,
                                            "reason": {"a": 1}}]}],
         "to_check": {"why": "wrong"}},
        {"slots": "nope", "to_check": [{"why": ["wrong"], "quote": 3,
                                        "utterance": {"u": 1}},
                                       {"why": "vague", "quote": ["x"]}]},
        {"slots": [None, 3, {"id": 4}]},
    ):
        slots, checks = judge_slots(odd, [running()], {"s1": ["c1"]}, SAID)
        assert len(slots) == 1 and isinstance(checks, list)
    weird = judgement_from({"label": 3, "reason": {"a": 1}}, SAID, set())
    assert weird is not None and weird.label == "unclear"
    assert weird.reason == ""


def test_more_ways_of_telling_one_s_own_experience() -> None:
    for claim in ("He estado tres años en una tienda de running.",
                  "Llevo más de cinco años como encargada.",
                  "Llevo dos años de dependiente.",
                  "I spent four years at a bike shop.",
                  "Worked at a big supermarket for six years."):
        assert is_experience(claim), claim
    assert not is_experience("Llevo zapatillas con drop bajo.")
