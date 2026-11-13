"""Interaction descriptors computed from the transcript alone.

These are counts, not judgements: who spoke, how much, how evenly.
They give context to the competency ratings (a rating built on three
turns is weaker than one built on sixty) and describe the team.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel

from src.domain.skills.transcript import Transcript

OVERLAP_SECONDS = 0.3


class SpeakerInteraction(BaseModel):
    """Participation counts of one speaker."""

    speaker: str
    turns: int
    words: int
    word_share: float
    turn_share: float
    mean_words_per_turn: float
    questions: int
    seconds: Optional[float] = None
    overlapping_starts: Optional[int] = None


class TeamInteraction(BaseModel):
    """Participation of the whole group.

    ``participation_equality`` is ``1 - Gini`` of the word shares
    normalised to [0, 1]: 1 when everybody speaks the same amount, 0
    when one person does all the talking.
    """

    speakers: int
    turns: int
    words: int
    participation_equality: float
    dominant_speaker: str
    dominant_share: float
    has_timestamps: bool
    per_speaker: List[SpeakerInteraction]

    def of(self, speaker: str) -> SpeakerInteraction:
        """Return the counts of ``speaker``."""
        for item in self.per_speaker:
            if item.speaker == speaker:
                return item
        raise KeyError(speaker)


def gini(values: List[float]) -> float:
    """Return the Gini coefficient of non-negative values (0 = equal)."""
    n = len(values)
    total = sum(values)
    if n == 0 or total <= 0:
        return 0.0
    ordered = sorted(values)
    weighted = sum((i + 1) * v for i, v in enumerate(ordered))
    return (2 * weighted) / (n * total) - (n + 1) / n


def equality(values: List[float]) -> float:
    """Return ``1 - Gini`` rescaled so one talker alone gives 0."""
    n = len(values)
    if n <= 1:
        return 1.0
    return max(0.0, 1.0 - gini(values) * n / (n - 1))


def count_questions(text: str) -> int:
    """Count the questions of a turn by its question marks."""
    marks = text.count("?")
    return marks if marks else text.count("¿")


def describe(transcript: Transcript) -> TeamInteraction:
    """Compute the participation counts of a transcript.

    Durations and overlapping starts are only reported when every turn
    has timestamps. An overlapping start is a turn that begins more
    than ``OVERLAP_SECONDS`` before the previous speaker stops; it is
    a count of simultaneous speech, not a judgement of interruption.
    """
    timed = transcript.has_timestamps()
    turns: Dict[str, int] = {}
    words: Dict[str, int] = {}
    questions: Dict[str, int] = {}
    seconds: Dict[str, float] = {}
    overlaps: Dict[str, int] = {}
    previous = None
    for turn in transcript.turns:
        name = turn.speaker
        turns[name] = turns.get(name, 0) + 1
        words[name] = words.get(name, 0) + turn.n_words()
        questions[name] = questions.get(name, 0) + count_questions(turn.text)
        if timed:
            assert turn.start is not None and turn.end is not None
            seconds[name] = seconds.get(name, 0.0) + max(
                0.0, turn.end - turn.start
            )
            overlaps.setdefault(name, 0)
            if (
                previous is not None and previous.speaker != name
                and previous.end is not None
                and turn.start < previous.end - OVERLAP_SECONDS
            ):
                overlaps[name] += 1
        previous = turn
    total_words = sum(words.values())
    total_turns = sum(turns.values())
    per_speaker = [
        SpeakerInteraction(
            speaker=name,
            turns=turns[name],
            words=words[name],
            word_share=words[name] / total_words if total_words else 0.0,
            turn_share=turns[name] / total_turns if total_turns else 0.0,
            mean_words_per_turn=words[name] / turns[name],
            questions=questions[name],
            seconds=round(seconds[name], 3) if timed else None,
            overlapping_starts=overlaps[name] if timed else None,
        )
        for name in transcript.speakers()
    ]
    dominant = max(per_speaker, key=lambda s: s.words)
    return TeamInteraction(
        speakers=len(per_speaker),
        turns=total_turns,
        words=total_words,
        participation_equality=equality(
            [float(s.words) for s in per_speaker]
        ),
        dominant_speaker=dominant.speaker,
        dominant_share=dominant.word_share,
        has_timestamps=timed,
        per_speaker=per_speaker,
    )
