"""Use case: produce evidence-backed scorecards for a session."""

import time
from typing import (
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

from src.application.caching import cache_key
from src.application.ports import Cache, NoProgress, Progress
from src.application.skills.evidence import (
    EvidenceRetriever,
    TranscriptIndex,
)
from src.application.skills.ports import ChatModel
from src.domain.errors import InvalidInputError
from src.domain.skills.assessment import (
    FORMAT_VERSION,
    CompetencyAssessment,
    Evidence,
    Observation,
    ParticipantScorecard,
    SessionAssessment,
)
from src.domain.skills.grounding import decide_rating, ground
from src.domain.skills.interaction import describe
from src.domain.skills.prompting import (
    PromptStyle,
    parse_judgement,
    system_prompt,
    user_prompt,
)
from src.domain.skills.provisional import estimate
from src.domain.skills.rubric import Competency, RubricCatalogue
from src.domain.skills.transcript import Transcript


class JudgeTuning(NamedTuple):
    """How one judge is asked and how far its quotes are trusted.

    ``style`` is the prompt it gets; ``min_similarity`` is the lowest
    cosine between a cited turn and its indicator that keeps the quote
    as evidence (the second opinion of the embedding model). A quote
    below it is kept only as a ``weak_match`` observation (#103).
    """

    style: PromptStyle
    min_similarity: float


class AssessmentService:
    """Retrieve, judge and verify the evidence of every competency.

    For each participant and competency: the retriever proposes the
    turns closest to the rubric, the judge quotes and rates, and the
    grounding rules of the domain discard what the transcript does not
    support. Nothing the judge says reaches the scorecard unverified.

    Without a firm level, what was verified still counts (#103): the
    quotes, those the similarity check set apart and the sentences
    step f confirmed become a provisional estimate and observations
    for a reviewer, with no further call to the judge.

    ``tuning`` gives the prompt style and the similarity floor from
    the name of the judge that answers (#104: the local models and
    Claude are tuned apart); without it every judge gets the standard
    prompt and ``min_similarity``. It is asked on every call because a
    fallback judge may take over in the middle of a session, and the
    floor it gives is the one that sets ``weak_match`` observations
    apart.
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
        tuning: Optional[Callable[[str], JudgeTuning]] = None,
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
        self.tuning = tuning

    def tuned(self) -> JudgeTuning:
        """Return the tuning of the judge answering now."""
        if self.tuning is None:
            return JudgeTuning("standard", self.min_similarity)
        return self.tuning(self.judge.model_name)

    def assess(
        self,
        transcript: Transcript,
        catalogue: RubricCatalogue,
        participants: Optional[Sequence[str]] = None,
        competency_ids: Optional[Sequence[str]] = None,
        confirmed: Sequence[Observation] = (),
    ) -> SessionAssessment:
        """Return the scorecards of ``participants`` (default: all).

        ``confirmed`` are the sentences the judge of step f confirmed,
        as observations; they feed provisional estimates and never
        count towards a firm level.
        """
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
        seen = by_person_and_competency(confirmed, transcript, catalogue)
        results = {
            (p, c.id): self._too_quiet(c, seen.get((p, c.id), []))
            if p in quiet
            else self._assess_one(
                index, catalogue, p, c, seen.get((p, c.id), []),
                team.of(p).words,
            )
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
                "min_similarity": self.tuned().min_similarity,
                "min_participant_words": self.min_words,
                "candidates": self.candidates,
            },
            team=team,
            scorecards=[
                ParticipantScorecard(
                    participant=p,
                    person_id=transcript.people.get(p),
                    interaction=team.of(p),
                    assessments=[results[(p, c.id)] for c in competencies],
                )
                for p in chosen
            ],
            seconds=round(time.perf_counter() - start, 3),
            started_at=transcript.started_at,
            conversation_type=transcript.conversation_type,
            format_version=FORMAT_VERSION,
        )

    def _too_quiet(
        self, competency: Competency, confirmed: List[Observation]
    ) -> CompetencyAssessment:
        """Return the result for a participant below the speech floor.

        Someone who barely spoke gets no rating at all, not even a
        provisional one: silence is not evidence of a low level. What
        step f confirmed is kept as observations.
        """
        return CompetencyAssessment(
            competency_id=competency.id,
            status="insufficient_evidence",
            flags=["below_participation_floor"],
            observations=confirmed,
        )

    def _reply(self, system: str, user: str) -> str:
        """Ask the judge, through the cache when there is one."""
        key = cache_key("judge", self.judge.model_name, system, user)
        cached = self.cache.get(key) if self.cache else None
        if cached is not None:
            return cached
        reply = self.judge.complete(system, user, self.max_tokens)
        if self.cache:
            # Keyed by the judge that answered: a fallback judge may have
            # taken over during the call.
            self.cache.set(
                cache_key("judge", self.judge.model_name, system, user),
                reply,
            )
        return reply

    def _assess_one(
        self,
        index: TranscriptIndex,
        catalogue: RubricCatalogue,
        participant: str,
        competency: Competency,
        confirmed: List[Observation],
        words: int,
    ) -> CompetencyAssessment:
        """Assess one competency of one participant."""
        found = self.retriever.candidates(
            index, participant, competency, self.candidates,
            catalogue.language,
        )
        turn_ids: List[int] = sorted(c.turn for c in found)
        if len(found) < self.min_incidents:
            guess, more = estimate(
                None, [], [], confirmed, catalogue.scale,
                self.min_incidents, words, self.episode_gap,
            )
            return CompetencyAssessment(
                competency_id=competency.id,
                status="provisional" if guess else "insufficient_evidence",
                candidate_turns=turn_ids,
                flags=["too_few_turns", *more],
                estimate=guess,
                observations=confirmed,
            )
        turns = self.retriever.turns(index, found)
        tuning = self.tuned()
        reply = self._reply(
            system_prompt(
                catalogue.scale, self.min_incidents, index.language,
                tuning.style,
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
        evidence, weak = self._supported(
            index, competency, verified, tuning.min_similarity
        )
        rating, flags = decide_rating(
            judgement.rating, evidence, catalogue.scale, self.min_incidents,
            self.min_episodes, self.episode_gap,
        )
        if judgement.parse_error:
            flags.append("unparseable_reply")
        guess = None
        observations: List[Observation] = []
        if rating is None:
            quoted = {(e.turn, e.indicator_id) for e in verified}
            extra = [
                o for o in confirmed
                if (o.turn, o.indicator_id) not in quoted
            ]
            guess, more = estimate(
                judgement.rating, evidence, weak, extra, catalogue.scale,
                self.min_incidents, words, self.episode_gap,
            )
            flags += more
            observations = sorted(weak + extra, key=lambda o: o.turn)
        return CompetencyAssessment(
            competency_id=competency.id,
            status=(
                "rated" if rating is not None
                else "provisional" if guess is not None
                else "insufficient_evidence"
            ),
            rating=rating,
            evidence=evidence,
            rationale=judgement.rationale,
            candidate_turns=turn_ids,
            proposed=len(judgement.evidence),
            rejected=rejected,
            unsupported=len(verified) - len(evidence),
            flags=flags,
            estimate=guess,
            observations=observations,
        )

    def _supported(
        self,
        index: TranscriptIndex,
        competency: Competency,
        verified: List[Evidence],
        min_similarity: float,
    ) -> Tuple[List[Evidence], List[Observation]]:
        """Score each quote against its indicator; set the distant apart.

        The judge decides which indicator a quote shows; the embedding
        model checks that decision. Below ``min_similarity`` (the
        floor of the judge that answered) the quote is real but does
        not look like the behaviour: it is not evidence, only an
        observation (``weak_match``) for a reviewer.
        """
        kept: List[Evidence] = []
        weak: List[Observation] = []
        for item in verified:
            indicator = competency.indicator(item.indicator_id)
            assert indicator is not None
            score = self.retriever.similarity(index, item.turn, indicator)
            rounded = None if score is None else round(score, 4)
            if score is not None and score < min_similarity:
                weak.append(Observation(
                    turn=item.turn, quote=item.quote,
                    indicator_id=item.indicator_id, polarity=item.polarity,
                    source="weak_match", start=item.start,
                    similarity=rounded,
                ))
                continue
            kept.append(item.model_copy(update={"similarity": rounded}))
        return kept, weak


def by_person_and_competency(
    confirmed: Sequence[Observation],
    transcript: Transcript,
    catalogue: RubricCatalogue,
) -> Dict[Tuple[str, str], List[Observation]]:
    """Group step f's confirmed sentences by speaker and competency.

    The speaker is read from the transcript turn, so a sentence of a
    turn the transcript does not have, or an indicator the catalogue
    does not know, is left out. One sentence per turn and indicator.
    """
    owner = {
        i.id: c.id for c in catalogue.competencies for i in c.indicators
    }
    grouped: Dict[Tuple[str, str], List[Observation]] = {}
    seen: set[Tuple[int, str]] = set()
    for item in confirmed:
        competency = owner.get(item.indicator_id)
        if competency is None or not 0 <= item.turn < len(transcript.turns):
            continue
        if (item.turn, item.indicator_id) in seen:
            continue
        seen.add((item.turn, item.indicator_id))
        speaker = transcript.turns[item.turn].speaker
        grouped.setdefault((speaker, competency), []).append(item)
    return grouped
