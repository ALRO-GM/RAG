"""Use case: produce evidence-backed scorecards for a session."""

import time
from typing import List, Optional, Sequence

from src.application.caching import cache_key
from src.application.ports import Cache, NoProgress, Progress
from src.application.skills.evidence import (
    EvidenceRetriever,
    TranscriptIndex,
)
from src.application.skills.ports import ChatModel
from src.domain.errors import InvalidInputError
from src.domain.skills.assessment import (
    CompetencyAssessment,
    Evidence,
    ParticipantScorecard,
    SessionAssessment,
)
from src.domain.skills.grounding import decide_rating, ground
from src.domain.skills.interaction import describe
from src.domain.skills.prompting import (
    parse_judgement,
    system_prompt,
    user_prompt,
)
from src.domain.skills.rubric import Competency, RubricCatalogue
from src.domain.skills.transcript import Transcript


class AssessmentService:
    """Retrieve, judge and verify the evidence of every competency.

    For each participant and competency: the retriever proposes the
    turns closest to the rubric, the judge quotes and rates, and the
    grounding rules of the domain discard what the transcript does not
    support. Nothing the judge says reaches the scorecard unverified.
    """

    def __init__(
        self,
        retriever: EvidenceRetriever,
        judge: ChatModel,
        cache: Optional[Cache] = None,
        progress: Optional[Progress] = None,
        candidates: int = 8,
        min_incidents: int = 2,
        max_tokens: int = 700,
        min_similarity: float = 0.0,
        min_episodes: int = 1,
        episode_gap: int = 3,
        min_words: int = 0,
    ):
        """Wire the retriever, the judge model and the cache."""
        if candidates < 1 or min_incidents < 1:
            raise InvalidInputError(
                "candidates and min_incidents must be at least 1."
            )
        self.retriever = retriever
        self.judge = judge
        self.cache = cache
        self.progress = progress or NoProgress()
        self.candidates = candidates
        self.min_incidents = min_incidents
        self.max_tokens = max_tokens
        self.min_similarity = min_similarity
        self.min_episodes = min_episodes
        self.episode_gap = episode_gap
        self.min_words = min_words

    def assess(
        self,
        transcript: Transcript,
        catalogue: RubricCatalogue,
        participants: Optional[Sequence[str]] = None,
        competency_ids: Optional[Sequence[str]] = None,
    ) -> SessionAssessment:
        """Return the scorecards of ``participants`` (default: all)."""
        start = time.perf_counter()
        speakers = transcript.speakers()
        chosen = list(participants) if participants else speakers
        unknown = [p for p in chosen if p not in speakers]
        if unknown:
            raise InvalidInputError(
                f"Unknown participant(s): {', '.join(unknown)}. "
                f"Speakers: {', '.join(speakers)}."
            )
        competencies = catalogue.select(
            list(competency_ids) if competency_ids else None
        )
        index = self.retriever.index(transcript)
        team = describe(transcript)
        pairs = [(p, c) for p in chosen for c in competencies]
        quiet = {
            p for p in chosen if team.of(p).words < self.min_words
        }
        results = {
            (p, c.id): self._too_quiet(c) if p in quiet
            else self._assess_one(index, catalogue, p, c)
            for p, c in self.progress.track(
                pairs, len(pairs), "Assessing competencies"
            )
        }
        embedder = self.retriever.embedder
        return SessionAssessment(
            session_id=transcript.session_id,
            language=index.language,
            catalogue=catalogue.name,
            catalogue_version=catalogue.version,
            judge_model=self.judge.model_name,
            embedding_model=embedder.model_name if embedder else "",
            min_incidents=self.min_incidents,
            rules={
                "min_incidents": self.min_incidents,
                "min_episodes": self.min_episodes,
                "episode_gap": self.episode_gap,
                "min_similarity": self.min_similarity,
                "min_participant_words": self.min_words,
                "candidates": self.candidates,
            },
            team=team,
            scorecards=[
                ParticipantScorecard(
                    participant=p,
                    interaction=team.of(p),
                    assessments=[results[(p, c.id)] for c in competencies],
                )
                for p in chosen
            ],
            seconds=round(time.perf_counter() - start, 3),
        )

    def _too_quiet(self, competency: Competency) -> CompetencyAssessment:
        """Return the result for a participant below the speech floor.

        Someone who barely spoke gets no rating at all: silence is
        not evidence of a low level.
        """
        return CompetencyAssessment(
            competency_id=competency.id,
            status="insufficient_evidence",
            flags=["below_participation_floor"],
        )

    def _reply(self, system: str, user: str) -> str:
        """Ask the judge, through the cache when there is one."""
        key = cache_key("judge", self.judge.model_name, system, user)
        cached = self.cache.get(key) if self.cache else None
        if cached is not None:
            return cached
        reply = self.judge.complete(system, user, self.max_tokens)
        if self.cache:
            self.cache.set(key, reply)
        return reply

    def _assess_one(
        self,
        index: TranscriptIndex,
        catalogue: RubricCatalogue,
        participant: str,
        competency: Competency,
    ) -> CompetencyAssessment:
        """Assess one competency of one participant."""
        found = self.retriever.candidates(
            index, participant, competency, self.candidates,
            catalogue.language,
        )
        turn_ids: List[int] = sorted(c.turn for c in found)
        if len(found) < self.min_incidents:
            return CompetencyAssessment(
                competency_id=competency.id,
                status="insufficient_evidence",
                candidate_turns=turn_ids,
                flags=["too_few_turns"],
            )
        turns = self.retriever.turns(index, found)
        reply = self._reply(
            system_prompt(
                catalogue.scale, self.min_incidents, index.language
            ),
            user_prompt(
                index.transcript, turns, participant, competency,
                catalogue.scale,
            ),
        )
        judgement = parse_judgement(reply)
        verified, rejected = ground(
            judgement.evidence, turns, participant, competency
        )
        evidence = self._supported(index, competency, verified)
        rating, flags = decide_rating(
            judgement.rating, evidence, catalogue.scale, self.min_incidents,
            self.min_episodes, self.episode_gap,
        )
        if judgement.parse_error:
            flags.append("unparseable_reply")
        return CompetencyAssessment(
            competency_id=competency.id,
            status="rated" if rating is not None else "insufficient_evidence",
            rating=rating,
            evidence=evidence,
            rationale=judgement.rationale,
            candidate_turns=turn_ids,
            proposed=len(judgement.evidence),
            rejected=rejected,
            unsupported=len(verified) - len(evidence),
            flags=flags,
        )

    def _supported(
        self,
        index: TranscriptIndex,
        competency: Competency,
        verified: List[Evidence],
    ) -> List[Evidence]:
        """Score each quote against its indicator; drop the distant ones.

        The judge decides which indicator a quote shows; the embedding
        model checks that decision. Below ``min_similarity`` the quote
        is real but does not look like the behaviour, so it is dropped.
        """
        kept: List[Evidence] = []
        for item in verified:
            indicator = competency.indicator(item.indicator_id)
            assert indicator is not None
            score = self.retriever.similarity(index, item.turn, indicator)
            if score is not None and score < self.min_similarity:
                continue
            kept.append(item.model_copy(update={
                "similarity": None if score is None else round(score, 4)
            }))
        return kept
