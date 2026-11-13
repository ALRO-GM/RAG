"""Use case: tag every sentence with the rubric indicators it may show.

The rubric (``knowledge/rubrics``) is the operational form of the
research in ``docs/skills/THEORY.md``: each competency lists observable
behaviours ("indicators"). A *tag* links one sentence (who said it,
when) to one indicator. Tags made here are **candidates**: they come
from text similarity alone, with no judgement of whether the behaviour
really occurred. They exist to (1) give annotators and the judge a
short list instead of 110 indicators, and (2) show on a timeline where
each competency may have been exercised. A language model or a person
confirms or rejects them afterwards.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel

from src.application.skills.evidence import EvidenceRetriever
from src.domain.skills.rubric import Polarity, RubricCatalogue
from src.domain.skills.speech import (
    Sentence,
    clock,
    sentence_transcript,
    transcript_sentences,
)
from src.domain.skills.transcript import Transcript


class SentenceTag(BaseModel):
    """A candidate link between a sentence and a rubric indicator.

    ``score`` is the retrieval score (lexical + semantic) that ranked
    it; ``similarity`` is the cosine alone, comparable across
    sentences, or ``None`` without an embedding model.
    """

    indicator_id: str
    competency_id: str
    polarity: Polarity
    score: float
    similarity: Optional[float] = None


class TaggedSentence(BaseModel):
    """A sentence with its candidate tags, best first."""

    sentence: Sentence
    tags: List[SentenceTag]


class SentenceTagger:
    """Attach candidate indicators to the sentences of a transcript.

    Each sentence competes, per competency, with the other sentences of
    the same speaker (the retriever ranks a speaker's sentences against
    the indicators of a competency). A sentence keeps at most ``top``
    tags, each with a cosine of at least ``min_similarity`` when an
    embedding model is available.

    Args:
        retriever: The evidence retriever (its ``min_words`` decides
            which sentences are long enough to be tagged at all).
        top: Most tags kept per sentence.
        min_similarity: Lowest cosine accepted. It depends on the
            embedding model and is provisional until reference data
            exists.
        per_competency: How many of a speaker's sentences are
            considered for each competency.
    """

    def __init__(
        self,
        retriever: EvidenceRetriever,
        top: int = 2,
        min_similarity: float = 0.3,
        per_competency: int = 8,
    ):
        """Wire the retriever and the thresholds."""
        self.retriever = retriever
        self.top = max(top, 1)
        self.min_similarity = min_similarity
        self.per_competency = max(per_competency, 1)

    def tag(
        self, transcript: Transcript, rubric: RubricCatalogue
    ) -> List[TaggedSentence]:
        """Return every sentence of ``transcript`` with its tags."""
        sentences = transcript_sentences(transcript)
        if not sentences:
            return []
        units = sentence_transcript(transcript, sentences)
        index = self.retriever.index(units)
        found: Dict[int, List[SentenceTag]] = {}
        for competency in rubric.competencies:
            for speaker in units.speakers():
                for candidate in self.retriever.candidates(
                    index, speaker, competency, self.per_competency,
                    rubric.language,
                ):
                    indicator = competency.indicator(candidate.indicator_id)
                    if indicator is None or candidate.score <= 0:
                        continue
                    similarity = self.retriever.similarity(
                        index, candidate.turn, indicator
                    )
                    if (
                        similarity is not None
                        and similarity < self.min_similarity
                    ):
                        continue
                    found.setdefault(candidate.turn, []).append(SentenceTag(
                        indicator_id=indicator.id,
                        competency_id=competency.id,
                        polarity=indicator.polarity,
                        score=round(candidate.score, 4),
                        similarity=(
                            None if similarity is None
                            else round(similarity, 4)
                        ),
                    ))
        return [
            TaggedSentence(
                sentence=sentence,
                tags=sorted(
                    found.get(position, []),
                    key=lambda t: (-t.score, t.indicator_id),
                )[:self.top],
            )
            for position, sentence in enumerate(sentences)
        ]


def render_timeline(tagged: List[TaggedSentence]) -> str:
    """Return the tagged sentences as a readable timeline."""
    lines: List[str] = []
    for item in tagged:
        s = item.sentence
        tags = ", ".join(
            f"{'+' if t.polarity == 'positive' else '-'}{t.indicator_id}"
            + ("" if t.similarity is None else f" {t.similarity:.2f}")
            for t in item.tags
        )
        lines.append(
            f"[{clock(s.start)}] {s.id} {s.speaker}: {s.text}"
            + (f"  <{tags}>" if tags else "")
        )
    return "\n".join(lines)
