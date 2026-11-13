"""Sentences: from speaker turns to utterances with ids and times.

A turn is split into sentences; each becomes an :class:`Utterance`
with the id ``t<turn>.s<n>`` (the one the analyzer uses for sentence
tags), the time of its first and last word, and the speaker who spoke
before it. Everything here is plain arithmetic on text and times.
"""

import re
from typing import List, Optional, Sequence, Tuple

from transcriber.domain.models import Turn, Utterance, Word

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


def utterance_id(turn: int, index: int) -> str:
    """Return the id of sentence ``index`` of turn ``turn``."""
    return f"t{turn}.s{index}"


def merge_turns(turns: Sequence[Turn]) -> List[Turn]:
    """Join consecutive turns of one speaker and drop empty ones.

    Diarization cuts one contribution into several pieces; the previous
    speaker of a turn must be someone else, so the pieces are joined
    first. The merged turn keeps the lower ``speaker_confidence``: the
    attribution is only as sure as its weakest piece.
    """
    merged: List[Turn] = []
    for turn in turns:
        text = " ".join(turn.text.split())
        if not text:
            continue
        last = merged[-1] if merged else None
        if last is None or last.speaker != turn.speaker:
            merged.append(turn.model_copy(update={"text": text}))
            continue
        merged[-1] = Turn(
            speaker=last.speaker,
            text=f"{last.text} {text}",
            start=min(last.start, turn.start),
            end=max(last.end, turn.end),
            words=[*last.words, *turn.words],
            speaker_confidence=_lower(
                last.speaker_confidence, turn.speaker_confidence
            ),
        )
    return merged


def build_utterances(turns: Sequence[Turn]) -> List[Utterance]:
    """Split ``turns`` into utterances linked to the previous speaker.

    Consecutive turns of one speaker are merged first
    (:func:`merge_turns`), so turn numbers count changes of speaker.
    Every sentence of turn ``n`` gets ``previous_speaker`` = the speaker
    of turn ``n - 1`` and ``previous_utterance`` = its last sentence;
    both are ``None`` in the first turn.
    """
    utterances: List[Utterance] = []
    previous_speaker: Optional[str] = None
    previous_utterance: Optional[str] = None
    for number, turn in enumerate(merge_turns(turns)):
        texts = split_sentences(turn.text)
        current = [
            Utterance(
                id=utterance_id(number, n), turn=number, index=n,
                speaker=turn.speaker, start=start, end=end, text=text,
                previous_speaker=previous_speaker,
                previous_utterance=previous_utterance,
                confidence=_mean_probability(words),
                speaker_confidence=turn.speaker_confidence,
                words=words,
            )
            for n, (text, (start, end, words)) in enumerate(
                zip(texts, _sentence_times(turn, texts))
            )
        ]
        utterances.extend(current)
        previous_speaker = turn.speaker
        previous_utterance = current[-1].id
    return utterances


def _sentence_times(
    turn: Turn, texts: Sequence[str]
) -> List[Tuple[float, float, List[Word]]]:
    """Return the start, end and words of each sentence of ``turn``.

    When the turn's words match the sentences one to one, a sentence
    takes the time of its first and last word. Otherwise the turn's time
    is shared out in proportion to the characters of each sentence (an
    estimate: people do not speak at a constant rate) and no words are
    attached.
    """
    words = turn.words
    if words and sum(len(t.split()) for t in texts) == len(words):
        spans: List[Tuple[float, float, List[Word]]] = []
        position = 0
        for text in texts:
            count = len(text.split())
            members = list(words[position:position + count])
            spans.append((members[0].start, members[-1].end, members))
            position += count
        return spans
    total = sum(len(t) for t in texts) or 1
    length = max(turn.end - turn.start, 0.0)
    spans, cursor = [], turn.start
    for text in texts:
        end = cursor + length * len(text) / total
        spans.append((round(cursor, 3), round(end, 3), []))
        cursor = end
    return spans


def _mean_probability(words: Sequence[Word]) -> Optional[float]:
    """Mean recogniser confidence of ``words``, when they report it."""
    known = [w.probability for w in words if w.probability is not None]
    return round(sum(known) / len(known), 4) if known else None


def _lower(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """Return the smaller of two optional numbers (``None`` if both are)."""
    known = [x for x in (a, b) if x is not None]
    return min(known) if known else None
