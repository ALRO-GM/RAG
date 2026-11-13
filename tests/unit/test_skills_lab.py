"""The bridge between evidence retrieval and the experiment lab."""

from typing import Any, List, Mapping, Optional, Sequence, Tuple

import pytest

from src.application.experiments import ExperimentService
from src.application.skills.evidence import EvidenceRetriever
from src.application.skills.gold import (
    AnnotatedEvidence,
    Annotation,
    AnnotationSet,
)
from src.application.skills.lab import (
    SkillsLabPipeline,
    decode_query,
    encode_query,
    lab_dataset,
    load_rubric,
    turn_source,
)
from src.domain.errors import InvalidInputError, NotFoundError
from src.domain.experiments import Suite, TrialSpec
from src.domain.models import AnsweredQuestion
from src.domain.skills.rubric import RubricCatalogue
from src.domain.skills.transcript import Transcript
from tests.fakes import (
    MEETING_ES,
    FakeEmbedder,
    FakeLexicalBuilder,
    InMemoryCatalogueRepository,
    InMemoryDatasetRepository,
    InMemoryTranscriptRepository,
    InMemoryTrialStore,
    make_catalogue,
    make_transcript,
)

GoldCells = Mapping[Tuple[str, str], Tuple[Optional[int], Sequence[int]]]
GOLD: GoldCells = {
    ("Maya", "listening"): (4, [3, 0]),
    ("Tom", "listening"): (2, [1]),
    ("Tom", "planning"): (3, [4, 7]),
    ("Priya", "planning"): (None, [5]),
    ("Priya", "listening"): (None, []),
}
"""The obvious evidence turns of the sample meeting."""


def reference(cells: GoldCells = GOLD, session: str = "s1") -> AnnotationSet:
    """Return a reference set from ``(participant, id) → (rating, turns)``."""
    return AnnotationSet(
        session_id=session, annotator="consensus", catalogue="core",
        catalogue_version="1.0",
        annotations=[
            Annotation(
                participant=p, competency_id=c, rating=rating,
                evidence=[AnnotatedEvidence(turn=t, indicator_id="L1")
                          for t in turns],
            )
            for (p, c), (rating, turns) in cells.items()
        ],
    )


class OnePipelineFactory:
    """A pipeline factory building retrievers from two lab parameters."""

    def __init__(
        self,
        transcripts: InMemoryTranscriptRepository,
        catalogues: InMemoryCatalogueRepository,
    ):
        """Serve the given repositories."""
        self.transcripts = transcripts
        self.catalogues = catalogues
        self.embedder = FakeEmbedder(dim=256)
        self.created: List[Mapping[str, Any]] = []

    def create(self, params: Mapping[str, Any]) -> SkillsLabPipeline:
        """Return a pipeline for ``semantic`` / ``rubric_language``."""
        self.created.append(dict(params))
        retriever = EvidenceRetriever(
            FakeLexicalBuilder(),
            self.embedder if params.get("semantic", True) else None,
            semantic_weight=float(params.get("semantic_weight", 1.0)),
        )
        return SkillsLabPipeline(
            retriever, self.transcripts, self.catalogues, "core",
            str(params.get("rubric_language", "auto")),
        )


def pipeline(
    transcripts: Sequence[Transcript] = (),
    catalogues: Sequence[RubricCatalogue] = (),
    rubric_language: str = "auto",
) -> Tuple[SkillsLabPipeline, InMemoryCatalogueRepository]:
    """Return a lexical pipeline over in-memory repositories."""
    repository = InMemoryCatalogueRepository(
        catalogues or [make_catalogue()]
    )
    return SkillsLabPipeline(
        EvidenceRetriever(FakeLexicalBuilder(), None),
        InMemoryTranscriptRepository(transcripts or [make_transcript()]),
        repository, "core", rubric_language,
    ), repository


# --------------------------------------------------------------- queries


def test_query_round_trip() -> None:
    """A (session, participant, competency) survives encoding."""
    query = encode_query("s 1", "María José", "listening")
    assert query == "s 1\tMaría José\tlistening"
    assert decode_query(query) == ("s 1", "María José", "listening")


@pytest.mark.parametrize("query", [
    "", "s1", "s1\tMaya", "s1\tMaya\tlistening\textra", "\tMaya\tlistening",
    "s1\t\tlistening", "s1\tMaya\t", "s1 Maya listening",
    "How do you load adapters?",
])
def test_decode_query_rejects_malformed_questions(query: str) -> None:
    """Anything but three non-empty tab-separated parts is an error."""
    with pytest.raises(InvalidInputError, match="skills lab question"):
        decode_query(query)


def test_turn_source_is_the_session_and_a_one_turn_range() -> None:
    """A turn is the 'character' range [turn, turn + 1) of its session."""
    source = turn_source("s1", 7)
    assert (source.file_path, source.first_character_index,
            source.last_character_index) == ("s1", 7, 8)


# --------------------------------------------------------------- dataset


def test_lab_dataset_has_one_question_per_unit_with_evidence() -> None:
    """Units with evidence become questions; their turns are sources."""
    dataset = lab_dataset([reference()])
    questions = dataset.rag_questions
    assert [q.question_id for q in questions] == [
        "s1/Maya/listening", "s1/Tom/listening", "s1/Tom/planning",
        "s1/Priya/planning",
    ]
    first = questions[0]
    assert isinstance(first, AnsweredQuestion)
    assert decode_query(first.question) == ("s1", "Maya", "listening")
    assert first.sources == [turn_source("s1", 0), turn_source("s1", 3)]
    assert first.answer == "4"
    last = questions[-1]
    assert isinstance(last, AnsweredQuestion) and last.answer == ""


def test_lab_dataset_pools_several_sessions() -> None:
    """Every reference set contributes its own questions."""
    dataset = lab_dataset([reference(), reference(session="s2")])
    assert len(dataset.rag_questions) == 8
    assert dataset.rag_questions[4].question.startswith("s2\t")


def test_lab_dataset_without_evidence_is_an_error() -> None:
    """A reference with no marked turn cannot train anything."""
    with pytest.raises(InvalidInputError, match="no evidence turn"):
        lab_dataset([reference({("Maya", "listening"): (None, [])})])
    with pytest.raises(InvalidInputError, match="no evidence turn"):
        lab_dataset([])


# -------------------------------------------------------------- pipeline


def test_pipeline_index_is_cold_once_then_reused() -> None:
    """The first index builds every session; the second reuses them."""
    lab, _ = pipeline([make_transcript(), make_transcript(session_id="s2")])
    report, cold = lab.index()
    assert cold and not report.incremental
    assert (report.files_total, report.files_reindexed) == (2, 2)
    assert report.chunks == 16
    again, cold_again = lab.index()
    assert not cold_again and again.incremental
    assert (again.files_reindexed, again.files_reused) == (0, 2)
    assert again.chunks == 16


def test_pipeline_index_of_an_empty_repository() -> None:
    """No session: nothing indexed, not a cold build."""
    lab = SkillsLabPipeline(
        EvidenceRetriever(FakeLexicalBuilder(), None),
        InMemoryTranscriptRepository(), InMemoryCatalogueRepository(), "core",
    )
    report, cold = lab.index()
    assert (report.files_total, report.chunks, cold) == (0, 0, False)
    assert not report.incremental


def test_pipeline_search_returns_turns_as_chunks() -> None:
    """A hit is a one-turn chunk carrying its text and best indicator."""
    lab, _ = pipeline()
    (hits,) = lab.search([encode_query("s1", "Tom", "planning")], 2)
    assert [h.chunk.first_character_index for h in hits] == [4, 7]
    best = hits[0].chunk
    assert (best.file_path, best.kind, best.context) == ("s1", "turn", "P2")
    assert best.last_character_index == 5
    assert best.text.startswith("Planning is a waste of time")
    assert best.to_source() == turn_source("s1", 4)
    assert hits[0].score > hits[1].score


def test_pipeline_search_indexes_each_session_once() -> None:
    """Searching reads a transcript on first use only."""
    transcripts = InMemoryTranscriptRepository([make_transcript()])
    lab = SkillsLabPipeline(
        EvidenceRetriever(FakeLexicalBuilder(), None), transcripts,
        InMemoryCatalogueRepository([make_catalogue()]), "core",
    )
    queries = [encode_query("s1", "Tom", "planning"),
               encode_query("s1", "Maya", "listening")]
    results = lab.search(queries, 5)
    lab.search(queries, 5)
    assert len(results) == 2
    assert transcripts.finds == ["s1"]


def test_pipeline_search_errors() -> None:
    """Bad questions, sessions and competencies are user errors."""
    lab, _ = pipeline()
    with pytest.raises(InvalidInputError, match="skills lab question"):
        lab.search(["What is a LoRA adapter?"], 5)
    with pytest.raises(NotFoundError):
        lab.search([encode_query("s9", "Tom", "planning")], 5)
    with pytest.raises(InvalidInputError, match="Unknown competency"):
        lab.search([encode_query("s1", "Tom", "charisma")], 5)
    assert lab.search([encode_query("s1", "Zoe", "planning")], 5) == [[]]
    assert lab.search([], 5) == []


def test_rubric_language_auto_follows_the_transcript() -> None:
    """A Spanish session is matched against the Spanish catalogue."""
    lab, catalogues = pipeline(
        [make_transcript(MEETING_ES, "es1"), make_transcript()],
        [make_catalogue("en"), make_catalogue("es")],
    )
    query = encode_query("es1", "Luis", "planning")
    (hits,) = lab.search([query], 2)
    assert [h.chunk.first_character_index for h in hits] == [4, 7]
    assert hits[0].score > 0.0
    lab.search([query, encode_query("s1", "Tom", "planning")], 2)
    assert catalogues.loads == [("core", "es"), ("core", "en")]
    assert lab.rubric("es").language == "es"


def test_rubric_language_auto_falls_back_to_english() -> None:
    """Without a Spanish catalogue the English one is used."""
    lab, catalogues = pipeline([make_transcript(MEETING_ES, "es1")])
    (hits,) = lab.search([encode_query("es1", "Luis", "planning")], 3)
    assert catalogues.loads == [("core", "es"), ("core", "en")]
    assert sorted(h.chunk.first_character_index for h in hits) == [1, 4, 7]
    assert all(h.score == 0.0 for h in hits)
    assert lab.rubric("es").language == "en"
    assert len(catalogues.loads) == 2


def test_rubric_language_fixed_ignores_the_transcript() -> None:
    """``rubric_language='en'`` always loads the English catalogue."""
    lab, catalogues = pipeline(
        [make_transcript(MEETING_ES, "es1")],
        [make_catalogue("en"), make_catalogue("es")], rubric_language="en",
    )
    lab.search([encode_query("es1", "Luis", "planning")], 3)
    assert catalogues.loads == [("core", "en")]


def test_load_rubric_fallback_rules() -> None:
    """The fallback is tried once, and only for another language."""
    english = InMemoryCatalogueRepository([make_catalogue("en")])
    assert load_rubric(english, "core", "es").language == "en"
    assert load_rubric(english, "core", "en").language == "en"
    with pytest.raises(NotFoundError):
        load_rubric(english, "core", "es", fallback=None)
    with pytest.raises(NotFoundError):
        load_rubric(english, "other", "es")
    spanish = InMemoryCatalogueRepository([make_catalogue("es")])
    with pytest.raises(NotFoundError):
        load_rubric(spanish, "core", "en")
    assert spanish.loads == [("core", "en")]
    assert load_rubric(spanish, "core", "fr", fallback="es").language == "es"


# ------------------------------------------- driven by ExperimentService


def wire() -> Tuple[ExperimentService, OnePipelineFactory, InMemoryTrialStore]:
    """Return the real experiment service over in-memory skills data."""
    dataset = lab_dataset([reference()])
    datasets = InMemoryDatasetRepository(
        {"evidence.json": dataset.model_dump_json()}
    )
    factory = OnePipelineFactory(
        InMemoryTranscriptRepository([make_transcript()]),
        InMemoryCatalogueRepository([make_catalogue()]),
    )
    store = InMemoryTrialStore()
    return ExperimentService(factory, datasets, store), factory, store


def test_experiment_trial_finds_the_obvious_gold_turns() -> None:
    """A trial on the sample meeting recalls every reference turn."""
    experiments, factory, store = wire()
    questions = experiments.load_questions({"evidence": "evidence.json"})
    assert len(questions["evidence"]) == 4
    result = experiments.run_trial(
        "skills", TrialSpec(name="hybrid", params={}), questions
    )
    metrics = result.metrics["evidence"]
    assert metrics["recall@5"] == pytest.approx(1.0)
    assert metrics["recall@3"] == pytest.approx(1.0)
    assert metrics["mrr"] > 0.7
    assert result.ranks["evidence"] == [[1, 2], [1], [1, 2], [1]]
    assert (result.n_chunks, result.index_cold) == (8, True)
    assert store.appended == 1 and factory.created == [{}]


def test_experiment_suite_compares_lexical_and_hybrid() -> None:
    """A suite runs one trial per parameter set and resumes stored ones."""
    experiments, factory, store = wire()
    suite = Suite(name="modes", base={"semantic": True},
                  factors={"semantic": [False]})
    results = experiments.run_suite(suite, {"evidence": "evidence.json"})
    assert len(results) == 2
    assert [r.params["semantic"] for r in results] == [True, False]
    assert all(r.mean_metric("recall@5") == pytest.approx(1.0)
               for r in results)
    experiments.run_suite(suite, {"evidence": "evidence.json"})
    assert len(factory.created) == 2 and store.appended == 2
