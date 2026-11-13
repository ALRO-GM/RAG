"""Bridge between the soft-skills retriever and the experiment lab.

The lab measures a pipeline on questions with reference sources. Here
a question is "which turns of this participant show this competency"
and a source is a turn, so every lab command (experiment, tune,
analyze) works on evidence retrieval without any change.
"""

import time
from typing import Dict, List, Optional, Sequence, Tuple

from src.application.skills.evidence import (
    EvidenceRetriever,
    TranscriptIndex,
)
from src.application.skills.gold import AnnotationSet
from src.application.skills.ports import (
    CatalogueRepository,
    TranscriptRepository,
)
from src.domain.errors import InvalidInputError, NotFoundError
from src.domain.models import (
    AnsweredQuestion,
    Chunk,
    IndexReport,
    MinimalSource,
    RagDataset,
    ScoredChunk,
)
from src.domain.skills.rubric import RubricCatalogue

SEPARATOR = "\t"


def encode_query(session_id: str, participant: str, competency: str) -> str:
    """Return the lab question of a (session, participant, competency)."""
    return SEPARATOR.join((session_id, participant, competency))


def decode_query(query: str) -> Tuple[str, str, str]:
    """Split a lab question into session, participant and competency."""
    parts = query.split(SEPARATOR)
    if len(parts) != 3 or not all(parts):
        raise InvalidInputError(
            "A skills lab question is 'session<TAB>participant<TAB>"
            f"competency', got {query!r}."
        )
    return parts[0], parts[1], parts[2]


def turn_source(session_id: str, turn: int) -> MinimalSource:
    """Return a turn as a source: the session and the turn number."""
    return MinimalSource(
        file_path=session_id,
        first_character_index=turn,
        last_character_index=turn + 1,
    )


def lab_dataset(reference: Sequence[AnnotationSet]) -> RagDataset:
    """Turn reference annotations into a lab dataset.

    Every annotated (participant, competency) with evidence becomes a
    question whose reference sources are its evidence turns.
    """
    questions: List[AnsweredQuestion] = []
    for item in reference:
        for annotation in item.annotations:
            turns = sorted(annotation.turns())
            if not turns:
                continue
            query = encode_query(
                item.session_id, annotation.participant,
                annotation.competency_id,
            )
            questions.append(AnsweredQuestion(
                question_id=query.replace(SEPARATOR, "/"),
                question=query,
                sources=[turn_source(item.session_id, t) for t in turns],
                answer="" if annotation.rating is None
                else str(annotation.rating),
            ))
    if not questions:
        raise InvalidInputError("The reference has no evidence turn.")
    return RagDataset.model_validate({"rag_questions": questions})


class SkillsLabPipeline:
    """Evidence retrieval seen as an experiment pipeline."""

    def __init__(
        self,
        retriever: EvidenceRetriever,
        transcripts: TranscriptRepository,
        catalogues: CatalogueRepository,
        catalogue: str,
        rubric_language: str = "auto",
    ):
        """Wire the retriever and the repositories."""
        self.retriever = retriever
        self.transcripts = transcripts
        self.catalogues = catalogues
        self.catalogue = catalogue
        self.rubric_language = rubric_language
        self._indexes: Dict[str, TranscriptIndex] = {}
        self._rubrics: Dict[str, RubricCatalogue] = {}

    def _index(self, session_id: str) -> TranscriptIndex:
        """Return the index of a session, building it once."""
        if session_id not in self._indexes:
            self._indexes[session_id] = self.retriever.index(
                self.transcripts.find(session_id)
            )
        return self._indexes[session_id]

    def rubric(self, language: str) -> RubricCatalogue:
        """Return the catalogue used for transcripts in ``language``."""
        wanted = (
            language if self.rubric_language == "auto"
            else self.rubric_language
        )
        if wanted not in self._rubrics:
            self._rubrics[wanted] = load_rubric(
                self.catalogues, self.catalogue, wanted
            )
        return self._rubrics[wanted]

    def index(self) -> Tuple[IndexReport, bool]:
        """Index every stored session; cold when none was ready."""
        start = time.perf_counter()
        sessions = self.transcripts.sessions()
        fresh = [s for s in sessions if s not in self._indexes]
        for session_id in sessions:
            self._index(session_id)
        report = IndexReport(
            files_total=len(sessions),
            files_reindexed=len(fresh),
            files_reused=len(sessions) - len(fresh),
            files_removed=0,
            chunks=sum(
                len(i.transcript.turns) for i in self._indexes.values()
            ),
            seconds=round(time.perf_counter() - start, 3),
            incremental=bool(sessions) and not fresh,
        )
        return report, bool(sessions) and len(fresh) == len(sessions)

    def search(
        self, queries: Sequence[str], k: int
    ) -> List[List[ScoredChunk]]:
        """Return the top-``k`` turns of every encoded question."""
        results: List[List[ScoredChunk]] = []
        for query in queries:
            session_id, participant, competency_id = decode_query(query)
            index = self._index(session_id)
            rubric = self.rubric(index.language)
            found = self.retriever.candidates(
                index, participant, rubric.competency(competency_id), k,
                rubric.language,
            )
            results.append([
                ScoredChunk(
                    chunk=Chunk(
                        file_path=session_id,
                        first_character_index=c.turn,
                        last_character_index=c.turn + 1,
                        kind="turn",
                        context=c.indicator_id,
                        text=index.transcript.turns[c.turn].text,
                    ),
                    score=c.score,
                )
                for c in found
            ])
        return results


def load_rubric(
    catalogues: CatalogueRepository,
    name: str,
    language: str,
    fallback: Optional[str] = "en",
) -> RubricCatalogue:
    """Load a catalogue in ``language``, else in the fallback language."""
    try:
        return catalogues.load(name, language)
    except NotFoundError:
        if not fallback or fallback == language:
            raise
        return catalogues.load(name, fallback)
