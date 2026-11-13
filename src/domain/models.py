"""Domain entities and the pydantic models exchanged between stages."""

import uuid
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """A location in the corpus: a file and a character range."""

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """A question without a reference answer."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """A question with its reference sources and answer."""

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """A dataset of RAG questions."""

    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """The sources retrieved for one question."""

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """The sources retrieved for one question plus the generated answer."""

    answer: str


class StudentSearchResults(BaseModel):
    """Search results for a whole dataset."""

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Search results and generated answers for a whole dataset."""

    search_results: List[MinimalAnswer]
    k: int


class Chunk(BaseModel):
    """A contiguous slice of a corpus file, the unit of retrieval.

    ``first_character_index`` is inclusive and ``last_character_index``
    exclusive, so ``text == file_text[first:last]``.
    """

    file_path: str
    first_character_index: int
    last_character_index: int
    kind: str
    context: str = ""
    text: str

    def to_source(self) -> MinimalSource:
        """Return the chunk location as a :class:`MinimalSource`."""
        return MinimalSource(
            file_path=self.file_path,
            first_character_index=self.first_character_index,
            last_character_index=self.last_character_index,
        )

    def search_text(self) -> str:
        """Return the text that is indexed: path, context and body."""
        return f"{self.file_path}\n{self.context}\n{self.text}"

    def embedding_text(self, style: str = "full") -> str:
        """Return the text given to the embedding model.

        ``full`` is :meth:`search_text`; ``body`` drops the file path,
        leaving more of the model's short input window to the content.
        """
        if style == "body":
            return f"{self.context}\n{self.text}"
        return self.search_text()


class ScoredChunk(BaseModel):
    """A retrieved chunk with its ranking score."""

    chunk: Chunk
    score: float


class IndexManifest(BaseModel):
    """Metadata describing a persisted index."""

    version: int
    max_chunk_size: int
    semantic: bool
    embedding_model: str
    fingerprint: str
    n_chunks: int
    n_files: int


class ProcessedDocument(BaseModel):
    """A corpus file after chunking and lexical analysis."""

    file_path: str
    sha256: str
    chunks: List[Chunk]
    term_counts: List[Dict[str, int]]


class IndexReport(BaseModel):
    """Summary of an indexing run."""

    files_total: int
    files_reindexed: int
    files_reused: int
    files_removed: int
    chunks: int
    seconds: float
    incremental: bool


class EvaluationReport(BaseModel):
    """Recall@k averaged over the evaluated questions."""

    recall: Dict[int, float]
    n_questions: int
    n_missing: int


class DatasetRunReport(BaseModel):
    """Summary of a batch command over a dataset."""

    n_questions: int
    output_path: str
    seconds: float
    mode: Optional[str] = None
