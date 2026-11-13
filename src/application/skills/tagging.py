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

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel

from src.application.caching import cache_key
from src.application.ports import Cache, Progress
from src.application.skills.evidence import EvidenceRetriever
from src.application.skills.ports import ChatModel
from src.domain.skills.rubric import Competency, Polarity, RubricCatalogue
from src.domain.skills.speech import (
    Sentence,
    clock,
    sentence_transcript,
    transcript_sentences,
)
from src.domain.skills.tag_prompting import (
    confirm_system_prompt,
    confirm_user_prompt,
    parse_confirmation,
)
from src.domain.skills.transcript import Transcript

DecisionStatus = Literal["confirmed", "rejected", "unparsed"]


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


class TagDecision(BaseModel):
    """What the classifier said about a sentence that had candidates.

    ``confirmed``: the sentence shows ``indicator_id``. ``rejected``:
    it shows none of the indicators it was offered, or the classifier
    named one that was not offered (then ``reason`` says so).
    ``unparsed``: the reply could not be read; nothing is concluded.
    """

    status: DecisionStatus
    indicator_id: Optional[str] = None
    competency_id: Optional[str] = None
    polarity: Optional[Polarity] = None
    reason: str = ""
    model: str = ""


class TaggedSentence(BaseModel):
    """A sentence with its candidate tags, best first.

    ``decision`` is set once a classifier has read the sentence;
    sentences without candidates are never sent to it.
    """

    sentence: Sentence
    tags: List[SentenceTag]
    decision: Optional[TagDecision] = None


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


class TagConfirmer:
    """Turn candidate tags into confirmed or rejected ones (call 1).

    For every sentence that has candidates, the judge reads the
    sentence, the ``context`` sentences said just before it (by anyone)
    and **all** the indicators of the candidate competencies, and names
    the one indicator the sentence shows, or none. Candidates thus
    choose the competencies to look at; the judge chooses the
    indicator, which retrieval alone gets wrong (``AUDIO.md`` §4).

    An indicator the judge was not offered is never accepted: the
    sentence is rejected and the reason says what was named.
    """

    def __init__(
        self,
        judge: ChatModel,
        cache: Optional[Cache] = None,
        progress: Optional[Progress] = None,
        context: int = 3,
        max_tokens: int = 120,
    ):
        """Wire the judge model and the optional reply cache."""
        self.judge = judge
        self.cache = cache
        self.progress = progress
        self.context = max(context, 0)
        self.max_tokens = max_tokens

    def _reply(self, system: str, user: str) -> str:
        """Ask the judge, through the cache when there is one."""
        key = cache_key("tag", self.judge.model_name, system, user)
        cached = self.cache.get(key) if self.cache else None
        if cached is not None:
            return cached
        reply = self.judge.complete(system, user, self.max_tokens)
        if self.cache:
            self.cache.set(key, reply)
        return reply

    def confirm(
        self,
        tagged: List[TaggedSentence],
        rubric: RubricCatalogue,
        language: Optional[str] = None,
    ) -> List[TaggedSentence]:
        """Return ``tagged`` with a decision on every candidate sentence."""
        system = confirm_system_prompt(language or rubric.language)
        sentences = [t.sentence for t in tagged]
        todo = [n for n, t in enumerate(tagged) if t.tags]
        positions = (
            self.progress.track(todo, len(todo), "Confirming tags")
            if self.progress else todo
        )
        decisions: Dict[int, TagDecision] = {}
        for position in positions:
            offered: List[Competency] = []
            for tag in tagged[position].tags:
                competency = rubric.competency(tag.competency_id)
                if competency not in offered:
                    offered.append(competency)
            user = confirm_user_prompt(
                sentences[max(position - self.context, 0):position],
                sentences[position], offered,
            )
            decisions[position] = self._decide(
                self._reply(system, user), offered
            )
        return [
            item.model_copy(update={"decision": decisions.get(n)})
            for n, item in enumerate(tagged)
        ]

    def _decide(
        self, reply: str, offered: List[Competency]
    ) -> TagDecision:
        """Check the judge's reply against the indicators it was shown."""
        model = self.judge.model_name
        named, reason, ok = parse_confirmation(reply)
        if not ok:
            return TagDecision(status="unparsed", model=model)
        if named is None:
            return TagDecision(status="rejected", reason=reason, model=model)
        for competency in offered:
            indicator = competency.indicator(named)
            if indicator is not None:
                return TagDecision(
                    status="confirmed", indicator_id=indicator.id,
                    competency_id=competency.id,
                    polarity=indicator.polarity, reason=reason, model=model,
                )
        return TagDecision(
            status="rejected", model=model,
            reason=f"named {named!r}, which was not offered",
        )


def render_timeline(tagged: List[TaggedSentence]) -> str:
    """Return the tagged sentences as a readable timeline.

    Candidates are shown in ``<...>``; once a classifier has decided,
    the line shows ``[indicator]`` (confirmed), ``[none]`` (rejected)
    or ``[?]`` (reply not understood) instead.
    """
    lines: List[str] = []
    for item in tagged:
        s = item.sentence
        if item.decision is None:
            tags = ", ".join(
                f"{'+' if t.polarity == 'positive' else '-'}"
                f"{t.indicator_id}"
                + ("" if t.similarity is None else f" {t.similarity:.2f}")
                for t in item.tags
            )
            mark = f"  <{tags}>" if tags else ""
        elif item.decision.status == "confirmed":
            sign = "+" if item.decision.polarity == "positive" else "-"
            mark = f"  [{sign}{item.decision.indicator_id}]"
        else:
            mark = "  [none]" if item.decision.status == "rejected" else (
                "  [?]"
            )
        lines.append(f"[{clock(s.start)}] {s.id} {s.speaker}: {s.text}{mark}")
    return "\n".join(lines)
