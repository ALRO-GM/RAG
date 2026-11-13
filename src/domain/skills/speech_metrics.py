"""Metrics for transcription: word errors and speaker attribution."""

import re
import unicodedata
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.domain.skills.speech import SpeakerSpan
from src.domain.skills.transcript import Word

_TOKEN = re.compile(r"\w+", re.UNICODE)


def normalise_words(text: str) -> List[str]:
    """Lower-case words without punctuation or accents.

    Accents are folded because references and recognisers disagree on
    spelling conventions ("á" against "a" in older Spanish texts); the
    measure is about words heard, not orthography.
    """
    folded = "".join(
        c for c in unicodedata.normalize("NFD", text.lower())
        if unicodedata.category(c) != "Mn"
    )
    return _TOKEN.findall(folded)


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Return ``(substitutions + deletions + insertions) / reference``.

    Both texts are normalised with :func:`normalise_words`. An empty
    reference gives 0 when the hypothesis is empty too, else 1.
    """
    ref, hyp = normalise_words(reference), normalise_words(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    previous = list(range(len(hyp) + 1))
    for i, expected in enumerate(ref, start=1):
        current = [i] + [0] * len(hyp)
        for j, heard in enumerate(hyp, start=1):
            current[j] = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (expected != heard),
            )
        previous = current
    return previous[-1] / len(ref)


def reference_speaker(
    time: float, spans: Sequence[SpeakerSpan]
) -> str:
    """Return the reference speaker at ``time`` (nearest span if none)."""
    inside = [s for s in spans if s.start <= time <= s.end]
    if inside:
        return inside[0].speaker
    return min(
        spans, key=lambda s: min(abs(time - s.start), abs(time - s.end))
    ).speaker


def speaker_mapping(
    words: Sequence[Word], spans: Sequence[SpeakerSpan]
) -> Dict[str, str]:
    """Map diarizer labels to reference speakers, one to one.

    The mapping maximises the number of words whose label maps to the
    speaker who was talking (Hungarian assignment), which is how
    diarization is conventionally scored: labels have no meaning of
    their own.
    """
    labels = sorted({w.speaker or "" for w in words})
    people = sorted({s.speaker for s in spans})
    if not labels or not people:
        return {}
    counts = np.zeros((len(labels), len(people)), dtype=np.int64)
    for word in words:
        truth = reference_speaker(word.middle(), spans)
        counts[labels.index(word.speaker or ""), people.index(truth)] += 1
    rows, columns = linear_sum_assignment(-counts)
    return {labels[r]: people[c] for r, c in zip(rows, columns)}


def speaker_accuracy(
    words: Sequence[Word], spans: Sequence[SpeakerSpan]
) -> Tuple[float, Dict[str, str]]:
    """Share of words attributed to the person who said them.

    Returns the accuracy under the best one-to-one label mapping and
    that mapping. Labels left unmapped (more voices found than people)
    count as errors.
    """
    if not words or not spans:
        return 0.0, {}
    mapping = speaker_mapping(words, spans)
    right = sum(
        1 for w in words
        if mapping.get(w.speaker or "")
        == reference_speaker(w.middle(), spans)
    )
    return right / len(words), mapping
