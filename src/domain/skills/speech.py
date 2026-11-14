"""Sentences: turns split into sentences that keep speaker and time.

A rubric indicator is attached to one sentence instead of a whole turn,
so every turn of a transcript is split into sentences with a stable id
(``t<turn>.s<n>``), its speaker and its time. Transcripts from the
transcriber (``transcript.v1``) arrive already split: their sentences
and ids are kept as published. Recognition, diarization and voice
matching live in the separate ``transcriber`` package
(``docs/transcriber/ARCHITECTURE.md``). Everything here is plain
arithmetic on text and times: no model.
"""

import re
from typing import List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from src.domain.skills.transcript import Transcript, Turn, Word


class Sentence(BaseModel):
    """One sentence of a turn, with its speaker and time.

    ``id`` is ``t<turn>.s<n>`` and is stable for a given transcript, so
    tags and annotations can point at it. Times are ``None`` when the
    transcript has no timestamps. ``previous_speaker`` is who spoke in
    the turn before, when the transcript says so (``transcript.v1``).
    """

    id: str
    turn: int = Field(ge=0)
    index: int = Field(ge=0)
    speaker: str
    text: str
    start: Optional[float] = None
    end: Optional[float] = None
    previous_speaker: Optional[str] = None

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


def turn_sentences(
    turn: Turn, words: Optional[Sequence[Word]] = None
) -> List[Sentence]:
    """Split one turn into sentences with times.

    A turn that carries its sentences (``transcript.v1``) is not split
    again: they keep their ids, texts and times. Otherwise, with timed
    words (``words``, or the turn's own) each sentence takes
    the time of its first and last word. Without them, the turn's time
    span is shared out in proportion to the characters of each sentence
    (an estimate: people do not speak at a constant rate). Without turn
    times, no times.
    """
    if turn.sentences:
        return [
            Sentence(
                id=s.id, turn=turn.index, index=n, speaker=turn.speaker,
                text=s.text, start=s.start, end=s.end,
                previous_speaker=turn.previous_speaker,
            )
            for n, s in enumerate(turn.sentences)
        ]
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


def sentence_transcript(
    transcript: Transcript, sentences: Optional[Sequence[Sentence]] = None
) -> Transcript:
    """Return a transcript whose turns are the sentences of another.

    Turn ``n`` of the result is sentence ``n`` (in order), so every
    tool written for turns (evidence retrieval, the judge, grounding)
    works at sentence granularity without change.
    """
    items = (
        list(sentences) if sentences is not None
        else transcript_sentences(transcript)
    )
    return Transcript(
        session_id=transcript.session_id,
        language=transcript.language,
        turns=[
            Turn(index=n, speaker=s.speaker, text=s.text,
                 start=s.start, end=s.end)
            for n, s in enumerate(items)
        ],
    )


def clock(seconds: Optional[float]) -> str:
    """Format seconds as ``hh:mm:ss`` (``--:--:--`` when unknown)."""
    if seconds is None:
        return "--:--:--"
    total = int(max(seconds, 0.0))
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"
