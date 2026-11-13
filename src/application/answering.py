"""Use case: answer a question from retrieved context."""

import threading
from typing import Optional, Sequence

from src.application.caching import cache_key
from src.application.ports import Cache, Generator
from src.application.search import SearchService
from src.domain.models import Chunk, MinimalAnswer
from src.domain.prompting import NO_ANSWER
from src.domain.validation import validate_k, validate_query


class AnswerService:
    """Retrieve context and generate a grounded, cached answer."""

    def __init__(
        self,
        search: SearchService,
        generator: Generator,
        cache: Optional[Cache] = None,
    ):
        """Wire the search use case and the generator port."""
        self.search = search
        self.generator = generator
        self.cache = cache
        self._lock = threading.RLock()

    def answer(self, question: str, k: int) -> MinimalAnswer:
        """Retrieve ``k`` chunks for ``question`` and answer it."""
        validate_query(question)
        validate_k(k)
        chunks = [h.chunk for h in self.search.search(question, k)]
        return MinimalAnswer(
            question_id="",
            question=question,
            retrieved_sources=[c.to_source() for c in chunks],
            answer=self.answer_from(question, chunks),
        )

    def answer_from(self, question: str, chunks: Sequence[Chunk]) -> str:
        """Answer ``question`` from already retrieved ``chunks``."""
        if not question.strip():
            return "The question is empty."
        if not chunks:
            return NO_ANSWER
        key = cache_key(
            "answer", self.generator.model_name, question,
            [[c.file_path, c.text] for c in chunks],
        )
        with self._lock:
            cached = self.cache.get(key) if self.cache else None
            if cached is not None:
                return cached
            answer = self.generator.generate(question, chunks)
            if self.cache:
                self.cache.set(key, answer)
            return answer
