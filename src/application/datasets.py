"""Use cases operating on whole datasets: search, answer, evaluate."""

import time
from pathlib import PurePath
from typing import List, Optional, Sequence, Tuple

from src.application.answering import AnswerService
from src.application.ports import (
    DatasetRepository,
    NoProgress,
    Progress,
    SourceReader,
)
from src.application.search import SearchService
from src.domain.evaluation import evaluate
from src.domain.models import (
    DatasetRunReport,
    EvaluationReport,
    MinimalAnswer,
    MinimalSearchResults,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
from src.domain.validation import validate_k


class DatasetService:
    """Batch commands reading and writing evaluator JSON files."""

    def __init__(
        self,
        repository: DatasetRepository,
        search: SearchService,
        answers: AnswerService,
        sources: SourceReader,
        progress: Optional[Progress] = None,
    ):
        """Wire the use cases and ports."""
        self.repository = repository
        self.search = search
        self.answers = answers
        self.sources = sources
        self.progress = progress or NoProgress()

    def search_dataset(
        self, dataset_path: str, k: int, save_directory: str
    ) -> DatasetRunReport:
        """Search every question and write ``StudentSearchResults``."""
        validate_k(k)
        dataset = self.repository.read(dataset_path, RagDataset)
        start = time.perf_counter()
        questions = dataset.rag_questions
        hits = self.search.search_many([q.question for q in questions], k)
        results = StudentSearchResults(
            k=k,
            search_results=[
                MinimalSearchResults(
                    question_id=q.question_id,
                    question=q.question,
                    retrieved_sources=[h.chunk.to_source() for h in found],
                )
                for q, found in zip(questions, hits)
            ],
        )
        target = self.repository.write(
            save_directory, PurePath(dataset_path).name, results
        )
        return DatasetRunReport(
            n_questions=len(questions),
            output_path=target,
            seconds=round(time.perf_counter() - start, 2),
            mode=self.search.mode,
        )

    def answer_dataset(
        self,
        results_path: str,
        save_directory: str,
        k: Optional[int] = None,
    ) -> DatasetRunReport:
        """Answer every question of saved search results.

        Only the first ``k`` sources of each question are used as context
        (all of them by default); the output keeps every source.
        """
        results = self.repository.read(results_path, StudentSearchResults)
        limit = validate_k(k) if k is not None else results.k
        start = time.perf_counter()
        answers: List[MinimalAnswer] = []
        items = results.search_results
        for item in self.progress.track(items, len(items), "Answering"):
            chunks = self.sources.read(item.retrieved_sources[:limit])
            answers.append(MinimalAnswer(
                question_id=item.question_id,
                question=item.question,
                retrieved_sources=item.retrieved_sources,
                answer=self.answers.answer_from(item.question, chunks),
            ))
        output = StudentSearchResultsAndAnswer(
            search_results=answers, k=results.k
        )
        target = self.repository.write(
            save_directory, PurePath(results_path).name, output
        )
        return DatasetRunReport(
            n_questions=len(answers),
            output_path=target,
            seconds=round(time.perf_counter() - start, 2),
        )

    def evaluate(
        self,
        results_path: str,
        dataset_path: str,
        k: int = 10,
        max_context_length: int = 2000,
        ks: Sequence[int] = (1, 3, 5, 10),
    ) -> Tuple[EvaluationReport, List[str]]:
        """Return recall@k and format warnings for saved search results."""
        validate_k(k)
        results = self.repository.read(results_path, StudentSearchResults)
        dataset = self.repository.read(dataset_path, RagDataset)
        warnings: List[str] = []
        too_long = sum(
            1 for r in results.search_results for s in r.retrieved_sources
            if s.last_character_index - s.first_character_index
            > max_context_length
        )
        if too_long:
            warnings.append(
                f"{too_long} source(s) longer than {max_context_length} "
                "characters (the moulinette would reject the file)."
            )
        too_many = sum(
            1 for r in results.search_results if len(r.retrieved_sources) > k
        )
        if too_many:
            warnings.append(
                f"{too_many} question(s) with more than {k} sources."
            )
        selected = sorted({x for x in ks if x <= k} | {k})
        return evaluate(results, dataset, selected), warnings
