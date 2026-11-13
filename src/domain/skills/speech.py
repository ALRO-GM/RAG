"""Speech: from timed words and speaker spans to turns and sentences.

A speech recogniser returns words with timestamps; a diarizer returns
"who spoke when" spans. This module joins both into the diarized
transcript the rest of the analyzer reads, and splits turns into
sentences that keep their speaker and their time, so that a rubric
indicator can be attached to one sentence instead of a whole turn.
Everything here is plain arithmetic on text and times: no model.
"""

import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field

from src.domain.errors import InvalidInputError
from src.domain.skills.transcript import Transcript, Turn, Word

FloatArray = npt.NDArray[np.float32]

UNKNOWN_SPEAKER = "SPEAKER_00"


class SpeakerSpan(BaseModel):
    """A stretch of audio attributed to one speaker."""

    speaker: str
    start: float = Field(ge=0)
    end: float = Field(ge=0)


class Sentence(BaseModel):
    """One sentence of a turn, with its speaker and time.

    ``id`` is ``t<turn>.s<n>`` and is stable for a given transcript, so
    tags and annotations can point at it. Times are ``None`` when the
    transcript has no timestamps.
    """

    id: str
    turn: int = Field(ge=0)
    index: int = Field(ge=0)
    speaker: str
    text: str
    start: Optional[float] = None
    end: Optional[float] = None

    def n_words(self) -> int:
        """Return the number of words of the sentence."""
        return len(_WORD.findall(self.text))


_WORD = re.compile(r"\w+", re.UNICODE)

# Abbreviations whose final period does not end a sentence (lower case).
ABBREVIATIONS = frozenset({
    "sr", "sra", "srta", "dr", "dra", "d", "dª", "ud", "uds", "vd", "etc",
    "pág", "pag", "p", "ej", "art", "núm", "num", "nº", "aprox", "tel",
    "mr", "mrs", "ms", "prof", "vs", "e.g", "i.e", "no", "fig", "st",
})

# A sentence ends at . ! ? … (possibly repeated, possibly followed by a
# closing quote or bracket) when whitespace and a plausible sentence
# start follow.
_END = re.compile(
    r"([.!?…]+[\"'»”’)\]]*)\s+(?=[\"'«“‘(\[¿¡]*[A-ZÁÉÍÓÚÜÑ0-9])"
)


def split_sentences(text: str) -> List[str]:
    """Split ``text`` into sentences (Spanish and English punctuation).

    A period after a known abbreviation ("Sr.", "etc.", "e.g.") or a
    single capital letter (an initial) does not end a sentence, and
    neither does one inside a number ("3.5"). Text without final
    punctuation is one sentence: speech recognisers often omit it.
    """
    clean = " ".join(text.split())
    if not clean:
        return []
    sentences: List[str] = []
    start = 0
    for match in _END.finditer(clean):
        head = clean[start:match.end(1)]
        if match.group(1).startswith(".") and _is_abbreviation(head):
            continue
        sentences.append(head.strip())
        start = match.end()
    tail = clean[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def _is_abbreviation(head: str) -> bool:
    """Tell whether ``head`` ends with an abbreviation and its period."""
    stripped = head.rstrip("\"'»”’)]")
    if stripped.endswith(".."):  # an ellipsis ends the sentence
        return False
    last = stripped[:-1].split(" ")[-1].lstrip("\"'«“‘([¿¡").lower()
    return last in ABBREVIATIONS or (len(last) == 1 and last.isalpha())


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
                    -min(abs(word.middle() - s.start),
                         abs(word.middle() - s.end)),
                ),
            )
            speaker = best.speaker
        labelled.append(word.model_copy(update={"speaker": speaker}))
    return labelled


def segments_from_words(
    words: Sequence[Word], max_gap: float = 1.0
) -> List[Dict[str, Any]]:
    """Group consecutive words of one speaker into transcript segments.

    A new segment starts when the speaker changes or when the silence
    since the previous word exceeds ``max_gap`` seconds. The result has
    the ``speaker`` / ``text`` / ``start`` / ``end`` shape that
    :func:`~src.domain.skills.transcript.build_transcript` reads; each
    segment also carries its ``words``.
    """
    segments: List[Dict[str, Any]] = []
    for word in words:
        text = word.text.strip()
        if not text:
            continue
        speaker = word.speaker or UNKNOWN_SPEAKER
        last = segments[-1] if segments else None
        if (
            last is not None and last["speaker"] == speaker
            and word.start - last["end"] <= max_gap
        ):
            last["words"].append(word)
            last["end"] = max(last["end"], word.end)
            continue
        segments.append({
            "speaker": speaker, "start": word.start, "end": word.end,
            "words": [word],
        })
    for segment in segments:
        members: List[Word] = segment["words"]
        segment["text"] = " ".join(w.text.strip() for w in members)
    return segments


def turn_sentences(
    turn: Turn, words: Optional[Sequence[Word]] = None
) -> List[Sentence]:
    """Split one turn into sentences with times.

    With timed words (``words``, or the turn's own) each sentence takes
    the time of its first and last word. Without them, the turn's time
    span is shared out in proportion to the characters of each sentence
    (an estimate: people do not speak at a constant rate). Without turn
    times, no times.
    """
    texts = split_sentences(turn.text)
    words = words if words is not None else turn.words
    spans: List[Tuple[Optional[float], Optional[float]]]
    if words and sum(len(t.split()) for t in texts) == len(words):
        spans, position = [], 0
        for text in texts:
            count = len(text.split())
            spans.append(
                (words[position].start, words[position + count - 1].end)
            )
            position += count
    elif turn.start is not None and turn.end is not None and texts:
        total = sum(len(t) for t in texts) or 1
        length = max(turn.end - turn.start, 0.0)
        spans, cursor = [], turn.start
        for text in texts:
            end = cursor + length * len(text) / total
            spans.append((round(cursor, 3), round(end, 3)))
            cursor = end
    else:
        spans = [(None, None)] * len(texts)
    return [
        Sentence(
            id=f"t{turn.index}.s{n}", turn=turn.index, index=n,
            speaker=turn.speaker, text=text, start=start, end=end,
        )
        for n, (text, (start, end)) in enumerate(zip(texts, spans))
    ]


def transcript_sentences(transcript: Transcript) -> List[Sentence]:
    """Split every turn of ``transcript`` into sentences."""
    return [
        sentence
        for turn in transcript.turns
        for sentence in turn_sentences(turn)
    ]


def rename_speakers(
    transcript: Transcript, names: Mapping[str, str]
) -> Transcript:
    """Replace diarizer labels (``SPEAKER_01``) with participant names.

    Labels missing from ``names`` are kept. Two labels may not map to
    the same name: that would silently merge two people.
    """
    used = [names.get(s, s) for s in transcript.speakers()]
    duplicated = sorted({n for n in used if used.count(n) > 1})
    if duplicated:
        raise InvalidInputError(
            f"Two speakers would be named {', '.join(duplicated)}."
        )
    return transcript.model_copy(update={"turns": [
        turn.model_copy(update={
            "speaker": names.get(turn.speaker, turn.speaker),
            "words": [
                w.model_copy(update={
                    "speaker": names.get(turn.speaker, turn.speaker)
                })
                for w in turn.words
            ],
        })
        for turn in transcript.turns
    ]})


def cluster_speakers(
    vectors: FloatArray,
    num_speakers: Optional[int] = None,
    threshold: float = 0.5,
    max_speakers: int = 10,
) -> List[int]:
    """Group voice embeddings into speakers (agglomerative, cosine).

    Clusters are merged by average similarity until ``num_speakers``
    remain or, when the number is unknown, until no two clusters are
    more similar than ``threshold`` (at most ``max_speakers`` remain).
    Labels are numbered by first appearance, so the first voice heard
    is speaker 0.
    """
    count = int(vectors.shape[0])
    if count == 0:
        return []
    if num_speakers is not None and num_speakers < 1:
        raise InvalidInputError("num_speakers must be at least 1.")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit = vectors / np.where(norms > 0, norms, 1.0)
    similarity = (unit @ unit.T).astype(np.float64)
    clusters: Dict[int, List[int]] = {i: [i] for i in range(count)}
    target = num_speakers if num_speakers is not None else 1
    while len(clusters) > target:
        keys = sorted(clusters)
        best: Tuple[float, int, int] = (-2.0, -1, -1)
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                value = float(
                    similarity[np.ix_(clusters[a], clusters[b])].mean()
                )
                if value > best[0]:
                    best = (value, a, b)
        value, a, b = best
        if (
            num_speakers is None and value < threshold
            and len(clusters) <= max_speakers
        ):
            break
        clusters[a] = clusters[a] + clusters.pop(b)
    labels = [0] * count
    order = sorted(clusters.values(), key=min)
    for label, members in enumerate(order):
        for member in members:
            labels[member] = label
    return labels


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
            last.end = max(last.end, end)
            continue
        if last is not None and start < last.end:
            cut = (start + last.end) / 2
            last.end = cut
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
