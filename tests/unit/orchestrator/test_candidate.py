"""The candidate profile across interview rounds (#235): domain and use cases.

Synthetic answers only; no model is called.
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pytest

from orchestrator.application.candidates import CandidateBook, CoreTracker
from orchestrator.application.interview import PlanError
from orchestrator.application.ports import Document
from orchestrator.domain.candidate import (
    CandidateProfile,
    Captured,
    CoreQuestion,
    Round,
    asked_core,
    candidate_id,
    capture,
    core_questions,
    expired,
    new_profile,
    salary,
    start_and_notice,
    age_unknown,
    open_questions,
    view,
    with_capture,
    with_confirmed,
    with_correction,
    with_round,
    without_answer,
    without_old_proposals,
    without_proposal,
    yes_no,
)
from orchestrator.domain.events import LiveEvent
from tests.unit.orchestrator.fixtures import REPO

CORE = REPO / "knowledge" / "interview" / "core-questions.json"
NOW = datetime(2026, 11, 24, 10, 0, 0)


def questions() -> List[CoreQuestion]:
    return core_questions(json.loads(CORE.read_text(encoding="utf-8")))


class MemoryCandidates:
    """The candidate store in memory."""

    def __init__(self) -> None:
        self.kept: Dict[Tuple[str, str], CandidateProfile] = {}
        self.fail = False

    def load(self, plan_id: str, candidate_id: str) -> CandidateProfile:
        return self.kept[(plan_id, candidate_id)]

    def save(self, profile: CandidateProfile) -> None:
        if self.fail:
            raise OSError("disk full")
        self.kept[(profile.plan_id, profile.id)] = profile

    def list(self, plan_id: str) -> List[CandidateProfile]:
        return [p for (plan, _), p in self.kept.items() if plan == plan_id]

    def plans(self) -> List[str]:
        return sorted({plan for plan, _ in self.kept})

    def delete(self, plan_id: str, candidate_id: str) -> bool:
        return self.kept.pop((plan_id, candidate_id), None) is not None


class Core:
    def questions(self) -> List[CoreQuestion]:
        return questions()


class Clock:
    def __init__(self) -> None:
        self.at = NOW

    def __call__(self) -> datetime:
        return self.at


def book(
    days: Optional[int] = 30,
) -> Tuple[CandidateBook, MemoryCandidates, Clock]:
    store, clock = MemoryCandidates(), Clock()
    return (CandidateBook(store, Core(), lambda p: p == "dev", days, clock),
            store, clock)


def final(uid: str, speaker: str, text: str, seq: int = 0,
          at: float = 0.0) -> LiveEvent:
    return LiveEvent.model_validate({
        "schema": "live.event.v1", "session_id": "s1", "seq": seq,
        "type": "final", "at": "2026-11-24T10:00:00Z",
        "stream_time": float(seq),
        "utterance": {"id": uid, "turn": seq, "index": 0, "speaker": speaker,
                      "start": at, "end": at + 1.0, "text": text,
                      "revision": 0},
    })


# ---- the core questions and what asks them ---------------------------


def test_the_committed_core_questions_are_read_in_order() -> None:
    found = questions()
    assert [q.id for q in found] == [
        "work_mode_location", "work_authorization", "start_date_notice",
        "motivation", "salary_expectation",
    ]
    assert found[0].has_blanks and not found[1].has_blanks
    assert found[1].text("en").startswith("Are you authorised")
    assert found[1].text("fr") == found[1].text("es")


def test_a_document_that_is_not_the_core_schema_is_refused() -> None:
    with pytest.raises(ValueError):
        core_questions({"schema": "other", "questions": []})
    with pytest.raises(ValueError):
        core_questions({"schema": "interview.core.v1", "questions": []})
    one = json.loads(CORE.read_text(encoding="utf-8"))["questions"][0]
    with pytest.raises(ValueError):
        core_questions({"schema": "interview.core.v1",
                        "questions": [one, one]})


def test_each_committed_question_is_recognised_when_asked() -> None:
    found = questions()
    for q in found:
        for language in ("es", "en"):
            text = q.text(language).replace("[", "").replace("]", "")
            assert asked_core(text, found) == q.id, (q.id, language)


@pytest.mark.parametrize("text", [
    "Cuéntame un proyecto del que estés orgulloso.",
    "¿Cómo resolviste el problema de rendimiento?",
    "¿Qué tal el viaje hasta aquí?",
    "Tell me about a hard bug you fixed.",
    # Close to a core question, but about another thing (review #235).
    "¿Necesitas autorización para trabajar en España?",
    "¿Necesitas permiso de trabajo?",
    "Do you need a work permit?",
    "¿Cuál es tu salario actual?",
    "¿Cuánto cobras ahora?",
    "¿Qué tal fue tu incorporación en tu último trabajo?",
    "¿Cómo gestionaste el preaviso en tu anterior empresa?",
    "Why did you apply to your last job?",
    "¿Por qué quieres dejar tu trabajo actual?",
    "¿Cómo te integraste al llegar?",
    "¿Cómo organizas tu día cuando trabajas en remoto? ¿Rindes más?",
    "Pregunta cómo trabaja en remoto y si puedes pedir un ejemplo",
])
def test_other_questions_are_not_core_questions(text: str) -> None:
    assert asked_core(text, questions()) is None


def test_only_the_listed_questions_are_looked_for() -> None:
    found = [q for q in questions() if q.id != "work_authorization"]
    assert asked_core("¿Tienes autorización para trabajar en España?",
                      found) is None


# ---- reading the answers -----------------------------------------------


@pytest.mark.parametrize(("text", "expected"), [
    ("Sí, me encaja perfectamente", True),
    ("Sí, no hay problema", True),
    ("Claro", True),
    ("Yes, that works for me", True),
    ("No.", False),
    ("No me encaja", False),
    ("No, no me viene bien porque cuido de mi madre", False),
    ("No, it doesn't work for me", False),
    ("Estoy en trámite", None),
    ("Bueno, depende del horario", None),
    ("Hmm", None),
    ("Claro que no.", False),
    ("Of course not.", False),
    ("No, bueno, sí.", None),
    ("Sí, ¿por qué no?", None),
])
def test_yes_or_no_only_when_clear(text: str,
                                   expected: Optional[bool]) -> None:
    assert yes_no(text) is expected


def test_no_answer_keeps_the_candidates_words() -> None:
    found = capture("work_mode_location",
                    "No, no me viene bien porque cuido de mi madre")
    assert found is not None
    assert found.fields == {"availability.work_mode_ok": False,
                            "availability.location_note": None}
    assert found.summary == ""
    authorised = capture("work_authorization",
                         "Sí, soy española, tengo permiso de trabajo")
    assert authorised is not None and authorised.summary == ""
    assert "español" not in str(authorised)
    paid = capture("salary_expectation", "Soy de Venezuela y pido 33.000")
    assert paid is not None and "venezuela" not in str(paid).lower()
    assert capture("work_authorization", "Está en trámite") is None


@pytest.mark.parametrize(("text", "start", "days"), [
    ("Podría incorporarme en dos semanas, tengo quince días de preaviso",
     "en dos semanas", 15),
    ("De inmediato, no tengo preaviso", "de inmediato", 0),
    ("Tengo un mes de preaviso", None, 30),
    ("A principios de marzo", "a principios de marzo", None),
    ("Pues tendría que dar un preaviso de 15 días", None, 15),
    ("I could start in two weeks", "in two weeks", None),
])
def test_start_and_notice(text: str, start: Optional[str],
                          days: Optional[int]) -> None:
    found = start_and_notice(text)
    assert found is not None
    assert found.fields["availability.earliest_start"] == start
    assert found.fields.get("availability.notice_days") == days


def test_a_vague_start_is_not_captured() -> None:
    assert start_and_notice("Tendría que hablarlo en casa") is None


@pytest.mark.parametrize(("text", "low", "high"), [
    ("Entre 28 y 32 mil euros brutos al año", 28000, 32000),
    ("Unos 30.000 euros", 30000, 30000),
    ("30k", 30000, 30000),
    ("Pues 35 mil", 35000, 35000),
    ("entre 28.000 y 32.000", 28000, 32000),
    ("Around 40k", 40000, 40000),
])
def test_salary_figures_and_ranges(text: str, low: int, high: int) -> None:
    found = salary(text)
    assert found is not None
    assert found.fields == {"salary.expectation_min": low,
                            "salary.expectation_max": high}


@pytest.mark.parametrize("text", [
    "1.800 al mes", "Prefiero no decirlo", "Depende de las condiciones",
    "Tengo 3 años de experiencia", "30 y 40", "un millón",
    "5, 10 o 20 mil",
    # Pay history is never taken as the expectation (review #235).
    "Mi salario actual es de 27.000 y busco 33.000.",
    "Ahora mismo cobro 28.000.",
    "Ahora gano 27.000, me gustaría llegar a 33.000.",
    "En mi empresa actual tengo 30.000 y quiero 34.000",
    "I make 40k now and want 45k",
    "My current salary is 40,000",
    "40.000 de fijo más 5.000 de variable",
])
def test_unsure_salaries_are_not_captured(text: str) -> None:
    assert salary(text) is None


@pytest.mark.parametrize("text", [
    "Me encanta el producto y quiero trabajar en un equipo que cuide la "
    "calidad",
    "Me mudé desde Venezuela hace un año y busco un trabajo estable aquí.",
    "Tengo ansiedad y necesito un ambiente más tranquilo que el actual.",
])
def test_the_motivation_is_never_read_from_the_candidates_words(
    text: str,
) -> None:
    assert capture("motivation", text) is None


def test_unknown_questions_and_empty_answers_capture_nothing() -> None:
    assert capture("shoe_size", "42") is None
    assert capture("work_authorization", "  ") is None


# ---- the profile -------------------------------------------------------


def test_ids_come_from_the_name_and_never_clash() -> None:
    assert candidate_id("Lucía Marín", []) == "lucia-marin"
    assert candidate_id("Lucía Marín", ["lucia-marin"]) == "lucia-marin-2"
    assert candidate_id("!!!", []) == "c"
    with pytest.raises(ValueError):
        new_profile("dev", "   ", [], NOW)


def test_a_capture_is_only_a_proposal_until_confirmed() -> None:
    profile = new_profile("dev", "Lucía", [], NOW)
    first = with_capture(profile, "work_authorization",
                         Captured({"eligibility.work_authorization": True}),
                         "s1", NOW)
    assert first is not None
    assert first.answers == {}
    assert first.proposals["work_authorization"].source == "proposed"
    assert open_questions(first, questions())[1].id == "work_authorization"
    again = with_capture(first, "work_authorization",
                         Captured({"eligibility.work_authorization": False}),
                         "s2", NOW)
    assert again is not None and again.proposals[
        "work_authorization"].fields == {
        "eligibility.work_authorization": False}
    confirmed = with_confirmed(first, ["work_authorization", "nope"], NOW)
    assert confirmed.proposals == {}
    assert confirmed.answers["work_authorization"].source == "confirmed"
    assert with_capture(confirmed, "work_authorization",
                        Captured({"eligibility.work_authorization": False}),
                        "s2", NOW) is None
    dropped = without_proposal(first, "work_authorization", NOW)
    assert dropped.proposals == {} and dropped.answers == {}
    question = questions()[1]
    corrected = with_correction(first, question, "No", NOW)
    assert corrected.proposals == {}
    answer = corrected.answers["work_authorization"]
    assert answer.source == "corrected"
    assert answer.fields == {"eligibility.work_authorization": False}
    assert answer.summary == ""
    assert with_capture(corrected, "work_authorization",
                        Captured({"eligibility.work_authorization": True}),
                        "s3", NOW) is None
    reopened = without_answer(corrected, "work_authorization", NOW)
    assert "work_authorization" not in reopened.answers


def test_a_correction_keeps_typed_words_the_rules_do_not_read() -> None:
    profile = new_profile("dev", "Lucía", [], NOW)
    start = questions()[2]
    typed = with_correction(profile, start, "Tras Semana Santa", NOW)
    assert typed.answers["start_date_notice"].fields == {}
    assert typed.answers["start_date_notice"].summary == "Tras Semana Santa"
    motive = with_correction(profile, questions()[3], "Le gusta el equipo",
                             NOW)
    assert motive.answers["motivation"].fields == {
        "motivation.summary": "Le gusta el equipo"}
    with pytest.raises(ValueError):
        with_correction(profile, start, "  ", NOW)


def test_rounds_are_kept_once_per_session_and_capped() -> None:
    profile = new_profile("dev", "Lucía", [], NOW)
    for n in range(25):
        profile = with_round(profile, Round(session_id=f"s{n}",
                                            at="2026-11-24T10:00:00"), NOW)
    profile = with_round(profile, Round(session_id="s24", at="x",
                                        kind="bot"), NOW)
    assert len(profile.rounds) == 20
    assert profile.rounds[-1].kind == "bot"
    assert [r.session_id for r in profile.rounds].count("s24") == 1


def test_profiles_expire_after_the_retention() -> None:
    profile = new_profile("dev", "Lucía", [], NOW)
    assert not expired(profile, NOW + timedelta(days=30), 30)
    assert expired(profile, NOW + timedelta(days=31), 30)
    assert not expired(profile, NOW + timedelta(days=999), None)
    broken = profile.model_copy(update={"updated_at": "yesterday"})
    assert not expired(broken, NOW, 30)  # kept and reported, never guessed
    assert age_unknown(broken) and not age_unknown(profile)


def test_old_proposals_go_with_the_retention() -> None:
    profile = new_profile("dev", "Lucía", [], NOW)
    old = with_capture(profile, "salary_expectation", Captured({}), "s1",
                       NOW - timedelta(days=40))
    assert old is not None
    fresh = with_capture(old, "work_authorization", Captured({}), "s2", NOW)
    assert fresh is not None
    trimmed = without_old_proposals(fresh, NOW, 30)
    assert trimmed is not None and list(trimmed.proposals) == [
        "work_authorization"]
    assert without_old_proposals(trimmed, NOW, 30) is None
    assert without_old_proposals(fresh, NOW, None) is None


def test_the_view_lists_every_question_with_its_answer() -> None:
    profile = new_profile("dev", "Lucía", [], NOW)
    profile = with_correction(profile, questions()[1], "Sí", NOW)
    shown = view(profile, questions())
    assert shown["answered"] == 1
    assert [r["id"] for r in shown["questions"]][1] == "work_authorization"
    assert shown["questions"][1]["answer"]["source"] == "corrected"
    proposed = with_capture(profile, "salary_expectation", Captured({}),
                            "s1", NOW)
    assert proposed is not None
    seen = view(proposed, questions())
    assert seen["proposed"] == 1 and seen["answered"] == 1
    assert seen["questions"][4]["proposal"]["source"] == "proposed"
    assert shown["questions"][0]["answer"] is None


# ---- the book ----------------------------------------------------------


def test_the_book_creates_corrects_clears_and_deletes() -> None:
    candidates, store, _ = book()
    made = candidates.create("dev", "  Lucía   Marín ")
    assert made["name"] == "Lucía Marín" and made["id"] == "lucia-marin"
    other = candidates.create("dev", "Lucía Marín")
    assert other["id"] == "lucia-marin-2"
    shown = candidates.correct("dev", "lucia-marin", "salary_expectation",
                               "entre 30 y 34 mil")
    row = shown["questions"][4]["answer"]
    assert row["fields"]["salary.expectation_max"] == 34000
    assert candidates.clear("dev", "lucia-marin", "salary_expectation")[
        "answered"] == 0
    listed = candidates.list("dev")
    assert {c["id"] for c in listed["candidates"]} == {
        "lucia-marin", "lucia-marin-2"}
    assert listed["retention_days"] == 30
    assert candidates.delete("dev", "lucia-marin")["deleted"] is True
    assert ("dev", "lucia-marin") not in store.kept
    with pytest.raises(PlanError) as gone:
        candidates.delete("dev", "lucia-marin")
    assert gone.value.code == "not_found"


@pytest.mark.parametrize("call", [
    lambda b: b.create("nope", "Ana"),
    lambda b: b.list("nope"),
    lambda b: b.get("dev", "nobody"),
    lambda b: b.correct("dev", "ana", "shoe_size", "42"),
    lambda b: b.by_session("s9"),
])
def test_the_book_refuses_what_does_not_exist(call: object) -> None:
    candidates, _, _ = book()
    candidates.create("dev", "Ana")
    with pytest.raises(PlanError) as refused:
        call(candidates)  # type: ignore[operator]
    assert refused.value.code == "not_found"


def test_a_name_and_an_answer_are_required() -> None:
    candidates, _, _ = book()
    with pytest.raises(PlanError) as empty:
        candidates.create("dev", "")
    assert empty.value.code == "invalid_request"
    candidates.create("dev", "Ana")
    with pytest.raises(PlanError) as blank:
        candidates.correct("dev", "ana", "motivation", " ")
    assert blank.value.code == "invalid_request"


def test_old_profiles_are_deleted_when_the_retention_is_over() -> None:
    candidates, store, clock = book(days=30)
    candidates.create("dev", "Ana")
    clock.at = NOW + timedelta(days=31)
    with pytest.raises(PlanError):
        candidates.get("dev", "ana")
    assert store.kept == {}
    candidates.create("dev", "Bea")
    clock.at = clock.at + timedelta(days=40)
    assert candidates.list("dev")["candidates"] == []
    forever, _, clock = book(days=None)
    forever.create("dev", "Ana")
    clock.at = NOW + timedelta(days=999)
    assert forever.purge() == 0
    assert forever.get("dev", "ana")["name"] == "Ana"


def test_the_candidate_of_a_session_is_found_by_its_rounds() -> None:
    candidates, _, _ = book()
    candidates.create("dev", "Ana")
    candidates.keep_round("dev", "ana", Round(session_id="s7", at="x"))
    assert candidates.by_session("s7")["id"] == "ana"
    assert candidates.keep_round("dev", "gone", Round(session_id="s",
                                                      at="x")) is None
    assert candidates.keep_capture(
        "dev", "gone", "motivation", Captured({}), "s") is None


# ---- following a session -----------------------------------------------


def tracker(
    roles: Optional[Dict[str, str]] = None,
) -> Tuple[CoreTracker, List[Document], CandidateBook, MemoryCandidates]:
    candidates, store, _ = book()
    candidates.create("dev", "Lucía")
    published: List[Document] = []
    track = CoreTracker(candidates, candidates.profile("dev", "lucia"), "s1",
                        roles or {"S1": "interviewer", "S2": "candidate"},
                        published.append)
    track.start()
    return track, published, candidates, store


def test_an_answer_is_captured_only_after_its_question() -> None:
    track, published, candidates, _ = tracker()
    track.on_event(final("t0.s0", "S2", "Sí, claro"))
    assert track.answered() == set()
    track.on_event(final("t1.s0", "S1",
                         "¿Tienes autorización para trabajar en España?"))
    track.on_event(final("t2.s0", "S2", "Sí, claro, sin problema."))
    # A proposal, not an answer: the question is still open.
    assert track.answered() == set()
    assert published[-1]["captured"] == "work_authorization"
    profile = candidates.profile("dev", "lucia")
    kept = profile.proposals["work_authorization"]
    assert kept.session_id == "s1" and kept.source == "proposed"
    assert profile.answers == {}


def test_an_answer_may_take_a_few_sentences_then_it_is_let_go() -> None:
    track, _, _, _ = tracker()
    track.on_event(final("t0.s0", "S1", "¿Qué expectativa salarial tienes?"))
    track.on_event(final("t1.s0", "S2", "Pues mira,"))
    track.on_event(final("t2.s0", "S2", "unos 32.000 euros al año."))
    assert "salary_expectation" in track.profile.proposals
    track.on_event(final("t3.s0", "S1",
                         "¿Cuándo podrías incorporarte con el preaviso?"))
    for n in range(3):
        track.on_event(final(f"t4.s{n}", "S2", "Bueno, ya veremos."))
    track.on_event(final("t5.s0", "S2", "En dos semanas."))
    assert "start_date_notice" not in track.profile.proposals


def test_an_instruction_or_a_late_answer_lets_the_question_go() -> None:
    track, _, _, _ = tracker()
    track.on_event(final("t0.s0", "S1",
                         "¿Tienes autorización para trabajar en España?"))
    track.on_event(final("t1.s0", "S2", "Soy ciudadana europea."))
    track.on_event(final("t2.s0", "S1", "Vale. Ahora háblame de tu último "
                                        "proyecto con clientes."))
    track.on_event(final("t3.s0", "S2", "Sí, claro, en mi último proyecto "
                                        "lideré un equipo de cinco."))
    assert track.profile.proposals == {}
    track.on_event(final("t4.s0", "S1", "¿Qué expectativa salarial tienes?",
                         at=10.0))
    track.on_event(final("t5.s0", "S2", "Unos 30.000.", at=60.0))
    assert track.profile.proposals == {}
    track.on_event(final("t6.s0", "S1", "Bien."))  # too short to let go
    assert track.pending is None


def test_a_new_question_lets_the_pending_one_go() -> None:
    track, _, _, _ = tracker()
    track.on_event(final("t0.s0", "S1", "¿Qué expectativa salarial tienes?"))
    track.on_event(final("t1.s0", "S1", "Perdona, ¿me oyes bien?"))
    track.on_event(final("t2.s0", "S2", "30.000 euros"))
    assert track.answered() == set()


def test_answered_questions_are_not_captured_again_nor_suggested() -> None:
    track, _, candidates, _ = tracker()
    candidates.correct("dev", "lucia", "work_authorization", "No")
    track.profile = candidates.profile("dev", "lucia")
    track.on_event(final("t0.s0", "S1",
                         "¿Tienes autorización para trabajar en España?"))
    assert track.pending is None
    suggestions = [
        {"text": "Pregunta si tiene permiso",
         "example": "¿Tienes autorización para trabajar en España?"},
        {"text": "Pide un ejemplo", "example": "¿Qué hiciste tú?"},
    ]
    assert track.screen(suggestions) == suggestions[1:]
    assert len(track.open()) == 4
    kept = candidates.profile("dev", "lucia").answers["work_authorization"]
    assert kept.source == "corrected"


def test_nothing_is_screened_before_an_answer() -> None:
    track, _, _, _ = tracker()
    items = [{"text": "¿Qué expectativa salarial tienes?"}]
    assert track.screen(items) == items


def test_a_correction_made_meanwhile_is_never_overwritten() -> None:
    track, _, candidates, _ = tracker()
    track.on_event(final("t0.s0", "S1", "¿Qué te ha llevado a presentarte "
                                        "a este puesto?"))
    track.on_event(final("t1.s0", "S1",
                         "¿Tienes autorización para trabajar en España?"))
    candidates.correct("dev", "lucia", "work_authorization", "No")
    track.on_event(final("t2.s0", "S2", "Sí, claro."))
    profile = candidates.profile("dev", "lucia")
    assert profile.answers["work_authorization"].fields == {
        "eligibility.work_authorization": False}
    assert profile.proposals == {}


def test_the_ai_interviewer_says_which_question_it_asked() -> None:
    track, _, _, _ = tracker(roles={"S1": "candidate"})
    track.expect("work_authorization")
    track.on_event(final("t0.s0", "S1", "Sí, la tengo."))
    assert "work_authorization" in track.profile.proposals
    track.expect("salary_expectation")
    track.expect(None)  # the next turn asked something else
    track.on_event(final("t1.s0", "S1", "30.000 euros"))
    assert "salary_expectation" not in track.profile.proposals


def test_a_failed_write_is_said_once_and_the_session_goes_on() -> None:
    track, published, _, store = tracker()
    store.fail = True
    for n in range(2):
        track.on_event(final(f"t{n}.s0", "S1",
                             "¿Tienes autorización para trabajar?"))
        track.on_event(final(f"t{n}.s1", "S2", "Sí"))
    notices = [m for m in published if m["type"] == "notice"]
    assert len(notices) == 1 and notices[0]["code"] == "core_not_kept"


def test_the_round_is_noted_at_the_end_and_unknown_roles_are_ignored(
) -> None:
    track, _, candidates, _ = tracker(roles={"S1": "unknown"})
    track.on_event(final("t0.s0", "S1", "¿Tienes autorización?"))
    track.on_event(LiveEvent.model_validate({
        "schema": "live.event.v1", "session_id": "s1", "seq": 9,
        "type": "session_ended", "at": "2026-11-24T10:00:00Z",
        "stream_time": 9.0,
    }))
    track.finish()
    assert [r.session_id for r in candidates.profile("dev", "lucia").rounds
            ] == ["s1"]
    assert track.plan_id == "dev"


def test_the_interviewer_confirms_edits_or_discards_proposals() -> None:
    candidates, _, _ = book()
    candidates.create("dev", "Lucía")
    for qid, found in (
        ("work_authorization",
         Captured({"eligibility.work_authorization": True})),
        ("salary_expectation",
         Captured({"salary.expectation_min": 30000,
                   "salary.expectation_max": 30000})),
        ("start_date_notice",
         Captured({"availability.earliest_start": "de inmediato"},
                  summary="de inmediato")),
    ):
        candidates.keep_capture("dev", "lucia", qid, found, "s1")
    shown = candidates.get("dev", "lucia")
    assert shown["proposed"] == 3 and shown["answered"] == 0
    one = candidates.confirm("dev", "lucia", ["work_authorization"])
    assert one["answered"] == 1 and one["proposed"] == 2
    dropped = candidates.discard("dev", "lucia", "start_date_notice")
    assert dropped["proposed"] == 1
    every = candidates.confirm("dev", "lucia")
    assert every["answered"] == 2 and every["proposed"] == 0
    with pytest.raises(PlanError) as missing:
        candidates.confirm("dev", "lucia", ["motivation"])
    assert missing.value.code == "not_found"
    with pytest.raises(PlanError) as nothing:
        candidates.discard("dev", "lucia", "motivation")
    assert nothing.value.code == "not_found"


def test_a_profile_made_for_a_session_that_failed_is_forgotten() -> None:
    candidates, store, _ = book()
    candidates.create("dev", "Ana")
    candidates.forget_new("dev", "ana")
    assert store.kept == {}
    candidates.forget_new("dev", "ana")  # already gone: nothing happens
    candidates.create("dev", "Bea")
    candidates.correct("dev", "bea", "motivation", "Le gusta el sector")
    candidates.forget_new("dev", "bea")
    assert ("dev", "bea") in store.kept  # it has data: kept


def test_the_sweep_drops_old_proposals_and_keeps_unreadable_dates(
    caplog: pytest.LogCaptureFixture,
) -> None:
    candidates, store, clock = book(days=30)
    candidates.create("dev", "Ana")
    candidates.keep_capture("dev", "ana", "work_authorization",
                            Captured({}), "s1")
    clock.at = NOW + timedelta(days=20)
    candidates.correct("dev", "ana", "motivation", "Le gusta el sector")
    clock.at = NOW + timedelta(days=35)
    assert candidates.purge() == 0
    assert store.kept[("dev", "ana")].proposals == {}
    odd = store.kept[("dev", "ana")].model_copy(
        update={"updated_at": "unknown"})
    store.kept[("dev", "ana")] = odd
    clock.at = NOW + timedelta(days=400)
    assert candidates.purge() == 0
    assert ("dev", "ana") in store.kept
    assert "unreadable updated_at" in caplog.text
