"""Interview knowledge files (KN-4, #234): forbidden questions, core questions
and the corrections of the interview-method corpus.

Plain data, no models: the schema
(``src/domain/knowledge/interview_rules.py``),
the invariants a lawyer's reviewer relies on (every item has a legal basis
with a URL, a lawful alternative and its exceptions), the pattern examples,
and that no core question and no lawful alternative trips a pattern.
"""

import copy
import json
import re
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

import pytest
from pydantic import ValidationError

from src.domain.knowledge.interview_rules import (
    MAX_CORE_QUESTIONS, CoreQuestions, ForbiddenRules, core_hits, fold,
    parse_core, parse_forbidden,
)

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "knowledge" / "interview"
CORPUS = (ROOT / "knowledge" / "domains" / "interview.generic"
          / "metodo-entrevista.jsonl")
OFFICIAL_HOSTS = {"www.boe.es", "eur-lex.europa.eu", "www.aepd.es"}
REQUIRED = {
    "age", "family_status", "pregnancy", "health", "disability",
    "religion_ideology", "union", "origin_nationality_language",
    "sexual_orientation_gender", "criminal_record",
}


def load(name: str) -> Dict[str, Any]:
    """Read one JSON file of the knowledge folder."""
    data = json.loads((FOLDER / name).read_text("utf-8"))
    assert isinstance(data, dict)
    return data


@pytest.fixture(scope="module")
def rules() -> ForbiddenRules:
    """The committed forbidden-questions file, parsed."""
    return parse_forbidden(load("forbidden-questions.json"))


@pytest.fixture(scope="module")
def core() -> CoreQuestions:
    """The committed core-questions file, parsed."""
    return parse_core(load("core-questions.json"))


def test_fold_drops_accents_case_and_curly_apostrophes() -> None:
    """The patterns see plain lower-case text."""
    assert fold("¿Cuántos AÑOS tienes?") == "¿cuantos anos tienes?"
    assert fold("What’s your age") == "what's your age"


def test_forbidden_file_covers_the_required_categories(
        rules: ForbiddenRules) -> None:
    """Every category the ticket names is there, and the AI limits too."""
    assert REQUIRED <= {c.id for c in rules.categories}
    assert {"pay_history", "excess_personal_data",
            "indirect_proxies"} <= {c.id for c in rules.categories}
    ids = {a.id for a in rules.ai_limits}
    assert {"emotion-inference", "recruitment-high-risk"} <= ids
    assert "legal advice" in rules.disclaimer.en
    assert "asesoramiento jurídico" in rules.disclaimer.es


def test_every_item_has_basis_alternative_and_why(
        rules: ForbiddenRules) -> None:
    """Basis with an official URL and an article, an alternative, a reason."""
    for cat in rules.categories:
        assert cat.legal_basis, cat.id
        assert cat.lawful_alternative.es and cat.lawful_alternative.en
        assert cat.why.es and cat.why.en
        assert len(cat.examples.es) >= 3 and len(cat.examples.en) >= 3
        for basis in cat.legal_basis:
            assert re.search(r"art\.|anexo|apartado", basis.ref), basis.ref
            assert urlparse(basis.url).hostname in OFFICIAL_HOSTS
            if basis.verified and basis.source != (
                    "aepd-relaciones-laborales"):
                assert basis.quote, (cat.id, basis.ref)
        for exemption in cat.exceptions:
            assert exemption.es and exemption.en


def test_unverified_items_are_flagged_with_a_note(
        rules: ForbiddenRules) -> None:
    """Whatever could not be read in the official text says so."""
    unverified = [
        (c.id, b.ref) for c in rules.categories for b in c.legal_basis
        if not b.verified
    ]
    assert unverified
    for cat in rules.categories:
        for basis in cat.legal_basis:
            assert basis.verified or basis.note
        for exemption in cat.exceptions:
            assert exemption.basis or "UNVERIFIED" in exemption.en.upper() \
                or "NOT checked" in exemption.en or "no verific" in (
                    exemption.es.lower())
    assert not rules.sources["dir-2000-43"].text_read
    # An item is only "verified" when nothing in it is unverified or noted.
    for cat in rules.categories:
        every = list(cat.legal_basis) + [
            b for e in cat.exceptions for b in e.basis]
        clean = all(b.verified and not b.note for b in every) and not (
            cat.status_note) and all(e.basis for e in cat.exceptions)
        assert (cat.verification == "verified") == clean, cat.id
    partly = {c.id for c in rules.categories
              if c.verification == "partly_verified"}
    assert {"pay_history", "origin_nationality_language",
            "criminal_record", "religion_ideology",
            "physical_appearance"} <= partly


def test_sources_are_the_research_log(rules: ForbiddenRules) -> None:
    """Every source has an official URL and says how it was read."""
    for sid, source in rules.sources.items():
        assert urlparse(source.url).hostname in OFFICIAL_HOSTS, sid
        assert source.how_read and source.title
    boe = [s for s in rules.sources.values() if s.kind.startswith("BOE")]
    assert boe and all("BOE-A-" in s.url for s in boe)
    used = {b.source for c in rules.categories for b in c.legal_basis}
    assert used <= set(rules.sources)


def test_examples_match_their_own_patterns(rules: ForbiddenRules) -> None:
    """Each example trips its category, in its language."""
    for cat in rules.categories:
        for lang in ("es", "en"):
            patterns = getattr(cat.patterns, lang)
            for example in getattr(cat.examples, lang):
                folded = fold(example)
                assert any(re.search(p, folded) for p in patterns), (
                    cat.id, example)
                assert cat.id in rules.hits(example), (cat.id, example)


def test_lawful_alternatives_do_not_trip_a_pattern(
        rules: ForbiddenRules) -> None:
    """The questions we recommend are never flagged (the quoted ones)."""
    for cat in rules.categories:
        for lang in ("es", "en"):
            text = getattr(cat.lawful_alternative, lang)
            for quoted in re.findall(r"[«“]([^»”]+)[»”]", text):
                assert rules.hits(quoted) == [], (cat.id, quoted)


def test_core_questions_are_few_and_fill_distinct_profile_fields(
        core: CoreQuestions) -> None:
    """At most six, ordered, each with a profile field (no duplicates)."""
    assert 1 <= len(core.questions) <= MAX_CORE_QUESTIONS
    paths = core.profile_paths()
    assert len(paths) == len(set(paths))
    for question in core.questions:
        assert question.question.es.endswith("?")
        assert question.question.en.endswith("?")
        assert question.question.es.count("?") == 1
        assert question.profile_fields and question.legal_note.es


def test_no_core_question_matches_a_forbidden_pattern(
        rules: ForbiddenRules, core: CoreQuestions) -> None:
    """The questions every candidate gets are lawful by our own rules."""
    assert all(hits == [] for hits in core_hits(rules, core).values())


def test_core_references_exist(rules: ForbiddenRules,
                               core: CoreQuestions) -> None:
    """Related categories and legal sources are in the forbidden file."""
    ids = {c.id for c in rules.categories}
    for question in core.questions:
        assert set(question.related_forbidden) <= ids
        assert set(question.legal_basis) <= set(rules.sources)


def test_work_authorisation_is_asked_without_nationality(
        core: CoreQuestions) -> None:
    """Lawful phrasing: authorisation, never nationality or origin."""
    by_id = {q.id: q for q in core.questions}
    text = fold(by_id["work_authorization"].question.es)
    assert "autorizacion" in text
    assert not re.search(r"nacionalidad|origen|pais", text)
    salary = fold(by_id["salary_expectation"].question.es)
    assert "expectativa" in salary and "cobr" not in salary


def test_the_schema_refuses_bad_files(rules: ForbiddenRules,
                                      core: CoreQuestions) -> None:
    """Unknown source, bad regex, too many core questions, bad order."""
    raw = load("forbidden-questions.json")
    bad = copy.deepcopy(raw)
    bad["categories"][0]["legal_basis"][0]["source"] = "nope"
    with pytest.raises(ValidationError):
        parse_forbidden(bad)
    bad = copy.deepcopy(raw)
    bad["categories"][0]["patterns"]["es"] = ["("]
    with pytest.raises(ValidationError):
        parse_forbidden(bad)
    bad = copy.deepcopy(raw)
    bad["categories"][0]["legal_basis"] = []
    with pytest.raises(ValidationError):
        parse_forbidden(bad)
    bad = copy.deepcopy(raw)
    bad["categories"][1]["id"] = bad["categories"][0]["id"]
    with pytest.raises(ValidationError):
        parse_forbidden(bad)
    bad = copy.deepcopy(raw)
    bad["categories"][0]["legal_basis"][0]["verified"] = False
    bad["categories"][0]["legal_basis"][0]["note"] = None
    with pytest.raises(ValidationError):
        parse_forbidden(bad)
    graw = load("core-questions.json")
    many = copy.deepcopy(graw)
    many["max_questions"] = 7
    with pytest.raises(ValidationError):
        parse_core(many)
    many = copy.deepcopy(graw)
    many["questions"] = many["questions"] * 2
    with pytest.raises(ValidationError):
        parse_core(many)
    many = copy.deepcopy(graw)
    many["questions"].reverse()
    with pytest.raises(ValidationError):
        parse_core(many)
    many = copy.deepcopy(graw)
    many["questions"][0]["profile_fields"][0]["path"] = "nodots"
    with pytest.raises(ValidationError):
        parse_core(many)


def test_a_core_question_that_trips_a_pattern_is_reported(
        rules: ForbiddenRules, core: CoreQuestions) -> None:
    """The invariant is real: a bad wording is caught."""
    raw = load("core-questions.json")
    raw["questions"][0]["question"]["es"] = "¿Cuántos años tienes?"
    worse = parse_core(raw)
    assert core_hits(rules, worse)[worse.questions[0].id] == ["age"]


INNOCENT = [
    "¿Cuántos años tienes de experiencia?",
    "¿Cuántos años tiene la empresa en la que trabajas?",
    "¿Qué edad tiene el proyecto?",
    "How old is the codebase?",
    "¿Cuáles son tus ideas para mejorar el producto?",
    "Cuéntame tus opiniones sobre el diseño de la API",
    "In that negotiation, which party had more leverage?",
    "¿Has gestionado las redes sociales de una marca?",
    "Have you run social media campaigns?",
    "What was the first language you learned to code in?",
    "What are your convictions about code quality?",
    "What do you make of this situation?",
    "¿Cómo gestionas la ansiedad antes de una presentación?",
    "¿Has trabajado en un hospital de maternidad?",
]

MISSES = {
    "¿Cuál fue tu último salario?": "pay_history",
    "¿Eres español?": "origin_nationality_language",
    "Are you a citizen?": "origin_nationality_language",
    "¿Eres de aquí?": "origin_nationality_language",
    "¿Te vas a casar?": "family_status",
    "¿Quieres ser madre?": "family_status",
    "¿Eres madre?": "family_status",
    "¿Eres religioso?": "religion_ideology",
    "¿Perteneces a algún partido político?": "religion_ideology",
    "¿Estás en tratamiento?": "health",
    "¿Estás sana?": "health",
    "¿Fumas?": "health",
    "¿Bebes alcohol?": "health",
    "¿Qué edad tienen tus hijos?": "family_status",
}


def test_innocent_questions_are_not_flagged(rules: ForbiddenRules) -> None:
    """Ordinary interview questions that share words with a pattern."""
    for question in INNOCENT:
        assert rules.hits(question) == [], question


def test_known_misses_are_flagged_in_their_category(
        rules: ForbiddenRules) -> None:
    """Direct questions an earlier version let through."""
    for question, category in MISSES.items():
        assert category in rules.hits(question), question


def test_lisos_doubt_makes_its_categories_partly_verified(
        rules: ForbiddenRules) -> None:
    """Every category citing LISOS 16.1.c carries the open-doubt note."""
    for cat in rules.categories:
        for basis in cat.legal_basis:
            if basis.source == "lisos":
                assert basis.note and "intermediation" in basis.note
                assert cat.verification == "partly_verified", cat.id


def test_work_mode_follow_up_does_not_ask_why(core: CoreQuestions) -> None:
    """A «no» to the mode or location is noted without asking the reason."""
    follow = {q.id: q for q in core.questions}["work_mode_location"]
    assert "sin preguntar el motivo" in follow.follow_up.es
    assert "falta para que encajara" not in follow.follow_up.es


COMPOUND = re.compile(
    r"¿[^?]*\by (qué|cómo|cuál|cuándo|quién|con qué|por qué|a quién|dónde)"
    r"\b[^?]*\?")
"""Two questions joined by «y» inside one quoted question."""


def corpus() -> List[Dict[str, Any]]:
    """Read the interview-method corpus."""
    return [json.loads(x) for x in CORPUS.read_text("utf-8").splitlines()]


def test_corpus_has_one_question_per_quoted_question() -> None:
    """Quoted questions to ask (not the legal guard's examples) are single."""
    for row in corpus():
        if row["intent"] == "legal_guard":
            continue
        for quoted in re.findall(r'"([^"]+)"', row["text"]):
            assert quoted.count("?") <= 1, (row["id"], quoted)
            if row["id"] != "m006":  # m006 quotes the compound to avoid
                assert not COMPOUND.search(quoted), (row["id"], quoted)


def test_corpus_questions_are_not_forbidden_ones(
        rules: ForbiddenRules) -> None:
    """No quoted question of a non-legal record trips a forbidden pattern."""
    for row in corpus():
        if row["intent"] == "legal_guard":
            continue
        for quoted in re.findall(r'"([^"]+)"', row["text"]):
            assert rules.hits(quoted) == [], (row["id"], quoted)


def test_corpus_legal_records_use_current_citations() -> None:
    """The records state the current articles checked in the BOE texts."""
    by_id = {r["id"]: r["text"] for r in corpus()}
    assert "art. 9.5" in by_id["m160"]
    assert "art. 36.1" in by_id["m162"]
    assert "art. 57" in by_id["m163"]
    assert "Ley Orgánica 8/2021" in by_id["m163"]
    assert "art. 16.2" in by_id["m158"]
    assert "2023/970" in by_id["m171"]
    assert "art. 5.1.f" in by_id["m173"]
    assert "con o sin algún ajuste" not in by_id["m160"]
    assert by_id["m161"] != by_id["m005"]
