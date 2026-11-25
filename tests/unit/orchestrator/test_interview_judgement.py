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
                    "The candidate is lying.", "Not suitable."):
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
