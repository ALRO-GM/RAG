"""Speakers: from timed words and voice windows to speaker turns.

A speech recogniser returns words with timestamps; diarization labels
short analysis windows of the audio with a voice group. This module
turns the windows into "who spoke when" spans, gives every word the
speaker of the span it falls in, undoes implausible one-word flips and
groups the words into :class:`Turn` objects. Everything here is plain
arithmetic on times and labels: no model, no I/O.

Copied from ``src/domain/skills/speech.py``; the behaviour is the same,
adapted to the transcriber's frozen models.
"""

from typing import Dict, List, Mapping, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field

from transcriber.domain.errors import InvalidInputError
from transcriber.domain.models import Turn, Word

UNKNOWN_SPEAKER = "SPEAKER_00"


class SpeakerSpan(BaseModel):
    """A stretch of audio attributed to one speaker."""

    model_config = ConfigDict(frozen=True)

    speaker: str
    start: float = Field(ge=0)
    end: float = Field(ge=0)


def _middle(word: Word) -> float:
    """Return the midpoint of ``word`` in seconds."""
    return (word.start + word.end) / 2


def assign_speakers(
    words: Sequence[Word], spans: Sequence[SpeakerSpan]
) -> List[Word]:
    """Give every word the speaker whose span overlaps it most.

    A word that overlaps no span (a diarizer gap) takes the speaker of
    the nearest span in time. Without any span every word belongs to
    one unknown speaker.
    """
    labelled: List[Word] = []
    for word in words:
        speaker = UNKNOWN_SPEAKER
        if spans:
            best = max(
                spans,
                key=lambda s: (
                    min(word.end, s.end) - max(word.start, s.start),
                    -min(abs(_middle(word) - s.start),
                         abs(_middle(word) - s.end)),
                ),
            )
            speaker = best.speaker
        labelled.append(word.model_copy(update={"speaker": speaker}))
    return labelled


def smooth_speakers(words: Sequence[Word], max_run: int = 1) -> List[Word]:
    """Undo implausibly short speaker changes.

    A run of at most ``max_run`` words whose neighbours on both sides
    belong to one other speaker takes that speaker: a diarizer window
    boundary in the middle of a sentence otherwise credits a stray word
    to the wrong person. Runs at the very start or end are kept.
    """
    out = list(words)
    runs: List[Tuple[int, int]] = []
    for i, word in enumerate(out):
        if runs and out[runs[-1][0]].speaker == word.speaker:
            runs[-1] = (runs[-1][0], i + 1)
        else:
            runs.append((i, i + 1))
    for n in range(1, len(runs) - 1):
        start, end = runs[n]
        before = out[runs[n - 1][0]].speaker
        after = out[runs[n + 1][0]].speaker
        if end - start <= max_run and before == after:
            for i in range(start, end):
                out[i] = out[i].model_copy(update={"speaker": before})
    return out


def turns_from_words(
    words: Sequence[Word], max_gap: float = 1.0
) -> List[Turn]:
    """Group consecutive words of one speaker into turns.

    A new turn starts when the speaker changes or when the silence
    since the previous word exceeds ``max_gap`` seconds. Words with
    blank text are dropped; a word without a speaker belongs to
    :data:`UNKNOWN_SPEAKER`. Each turn keeps its words.
    """
    groups: List[Tuple[str, List[Word]]] = []
    for word in words:
        if not word.text.strip():
            continue
        speaker = word.speaker or UNKNOWN_SPEAKER
        if groups:
            last_speaker, members = groups[-1]
            last_end = max(w.end for w in members)
            if last_speaker == speaker and word.start - last_end <= max_gap:
                members.append(word)
                continue
        groups.append((speaker, [word]))
    return [
        Turn(
            speaker=speaker,
            text=" ".join(w.text.strip() for w in members),
            start=members[0].start,
            end=max(w.end for w in members),
            words=members,
        )
        for speaker, members in groups
    ]


def speaker_label(number: int) -> str:
    """Return the diarizer label of speaker ``number`` (0-based)."""
    return f"SPEAKER_{number + 1:02d}"


def spans_from_windows(
    windows: Sequence[Tuple[float, float]], labels: Sequence[int]
) -> List[SpeakerSpan]:
    """Turn labelled analysis windows into speaker spans.

    Consecutive windows of one speaker are merged. Where two windows
    of different speakers overlap, the boundary is put halfway.
    """
    if len(windows) != len(labels):
        raise ValueError("One label per window is required.")
    spans: List[SpeakerSpan] = []
    for (start, end), label in zip(windows, labels):
        speaker = speaker_label(label)
        last = spans[-1] if spans else None
        if last is not None and last.speaker == speaker:
            spans[-1] = last.model_copy(update={"end": max(last.end, end)})
            continue
        if last is not None and start < last.end:
            cut = (start + last.end) / 2
            spans[-1] = last.model_copy(update={"end": cut})
            start = cut
        spans.append(SpeakerSpan(speaker=speaker, start=start, end=end))
    return spans


def analysis_windows(
    regions: Sequence[Tuple[float, float]],
    length: float = 1.5,
    step: float = 0.75,
    minimum: float = 0.4,
) -> List[Tuple[float, float]]:
    """Cut speech regions into overlapping windows for voice embeddings.

    A region no longer than ``length`` is one window. Regions shorter
    than ``minimum`` seconds are skipped (too little voice to compare).
    When the last full window leaves at least ``minimum`` seconds
    uncovered, one more window is anchored at the end of the region.
    """
    if length <= 0 or step <= 0:
        raise InvalidInputError("Window length and step must be positive.")
    windows: List[Tuple[float, float]] = []
    for start, end in regions:
        if end - start < minimum:
            continue
        if end - start <= length:
            windows.append((round(start, 3), round(end, 3)))
            continue
        cursor = start
        while cursor + length <= end:
            windows.append((round(cursor, 3), round(cursor + length, 3)))
            cursor += step
        if end - windows[-1][1] >= minimum:
            windows.append((round(end - length, 3), round(end, 3)))
    return windows


def speech_regions(
    words: Sequence[Word], max_gap: float = 0.5
) -> List[Tuple[float, float]]:
    """Return the stretches of continuous speech covered by ``words``.

    Words closer than ``max_gap`` seconds belong to one region. The
    recogniser's word times thus double as voice activity detection.
    """
    regions: List[Tuple[float, float]] = []
    for word in sorted(words, key=lambda w: w.start):
        if regions and word.start - regions[-1][1] <= max_gap:
            regions[-1] = (regions[-1][0], max(regions[-1][1], word.end))
        else:
            regions.append((word.start, word.end))
    return regions


def speech_seconds(regions: Sequence[Tuple[float, float]]) -> float:
    """Return the seconds covered by ``regions``, overlaps counted once."""
    total = 0.0
    reach = float("-inf")
    for start, end in sorted(regions):
        if end <= reach:
            continue
        total += end - max(start, reach)
        reach = end
    return total


def speakers_in_order(turns: Sequence[Turn]) -> List[str]:
    """Return the speakers of ``turns`` in order of first appearance."""
    seen: Dict[str, None] = {}
    for turn in turns:
        seen.setdefault(turn.speaker, None)
    return list(seen)


def rename_speakers(
    turns: Sequence[Turn], names: Mapping[str, str]
) -> List[Turn]:
    """Replace speaker labels in ``turns`` and their words.

    Labels missing from ``names`` are kept. Two labels may not map to
    the same name: that would silently merge two people.
    """
    used = [names.get(s, s) for s in speakers_in_order(turns)]
    duplicated = sorted({n for n in used if used.count(n) > 1})
    if duplicated:
        raise InvalidInputError(
            f"Two speakers would be named {', '.join(duplicated)}."
        )
    renamed: List[Turn] = []
    for turn in turns:
        name = names.get(turn.speaker, turn.speaker)
        renamed.append(turn.model_copy(update={
            "speaker": name,
            "words": [
                w.model_copy(update={"speaker": name}) for w in turn.words
            ],
        }))
    return renamed
