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
    "mr", "mrs", "ms", "prof", "vs", "e.g", "i.e", "fig", "st",
})

# Abbreviations that are also words and only abbreviate before a
# number: "No. 5" is "number 5", but "Creo que no. Pero sigo." ends.
NUMBER_ABBREVIATIONS = frozenset({"no"})

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
    neither does one inside a number ("3.5"). "No." is an abbreviation
    only when a digit follows ("No. 5"). Text without final
    punctuation is one sentence: speech recognisers often omit it.
    """
    clean = " ".join(text.split())
    if not clean:
        return []
    sentences: List[str] = []
    start = 0
    for match in _END.finditer(clean):
        head = clean[start:match.end(1)]
        if match.group(1).startswith(".") and _is_abbreviation(
            head, clean[match.end():]
        ):
            continue
        sentences.append(head.strip())
        start = match.end()
    tail = clean[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def _is_abbreviation(head: str, following: str) -> bool:
    """Tell whether ``head`` ends with an abbreviation and its period.

    ``following`` is the text after the period and its whitespace.
    """
    stripped = head.rstrip("\"'»”’)]")
    if stripped.endswith(".."):  # an ellipsis ends the sentence
        return False
    last = stripped[:-1].split(" ")[-1].lstrip("\"'«“‘([¿¡").lower()
    if last in NUMBER_ABBREVIATIONS:
        return following[:1].isdigit()
    return last in ABBREVIATIONS or (len(last) == 1 and last.isalpha())


def utterance_id(turn: int, index: int) -> str:
    """Return the id of sentence ``index`` of turn ``turn``."""
    return f"t{turn}.s{index}"


def _runs(turns: Sequence[Turn]) -> List[List[Turn]]:
    """Group ``turns`` into runs of consecutive pieces of one speaker.

    Turns are put in time order first (by ``start``; the sort is stable,
    so pieces that start together keep their given order) and turns
    without text are dropped. Each piece's text is normalised to single
    spaces.
    """
    runs: List[List[Turn]] = []
    for turn in sorted(turns, key=lambda t: t.start):
        text = " ".join(turn.text.split())
        if not text:
            continue
        piece = turn.model_copy(update={"text": text})
        if runs and runs[-1][-1].speaker == turn.speaker:
            runs[-1].append(piece)
        else:
            runs.append([piece])
    return runs


def merge_turns(turns: Sequence[Turn]) -> List[Turn]:
    """Join consecutive turns of one speaker and drop empty ones.

    Turns are taken in time order (sorted by ``start``), whatever order
    they are given in, so the merged turns and their words run forward.
    Diarization cuts one contribution into several pieces; the previous
    speaker of a turn must be someone else, so the pieces are joined
    first. The merged turn keeps the lower ``speaker_confidence``: the
    attribution is only as sure as its weakest piece.
    """
    merged: List[Turn] = []
    for pieces in _runs(turns):
        if len(pieces) == 1:
            merged.append(pieces[0])
            continue
        merged.append(Turn(
            speaker=pieces[0].speaker,
            text=" ".join(p.text for p in pieces),
            start=min(p.start for p in pieces),
            end=max(p.end for p in pieces),
            words=[w for p in pieces for w in p.words],
            speaker_confidence=_weakest(pieces),
        ))
    return merged


def build_utterances(turns: Sequence[Turn]) -> List[Utterance]:
    """Split ``turns`` into utterances linked to the previous speaker.

    Turns are taken in time order and consecutive turns of one speaker
    are merged first (:func:`merge_turns`), so turn numbers count
    changes of speaker. Every sentence of turn ``n`` gets
    ``previous_speaker`` = the speaker of turn ``n - 1`` and
    ``previous_utterance`` = its last sentence; both are ``None`` in the
    first turn. Times come from the words (see :func:`_sentence_times`).
    """
    utterances: List[Utterance] = []
    previous_speaker: Optional[str] = None
    previous_utterance: Optional[str] = None
    for number, pieces in enumerate(_runs(turns)):
        speaker = pieces[0].speaker
        confidence = _weakest(pieces)
        texts = split_sentences(" ".join(p.text for p in pieces))
        current = [
            Utterance(
                id=utterance_id(number, n), turn=number, index=n,
                speaker=speaker, start=start, end=end, text=text,
                previous_speaker=previous_speaker,
                previous_utterance=previous_utterance,
                confidence=_mean_probability(words),
                speaker_confidence=confidence,
                words=words,
            )
            for n, (text, (start, end, words)) in enumerate(
                zip(texts, _sentence_times(pieces, texts))
            )
        ]
        utterances.extend(current)
        previous_speaker = speaker
        previous_utterance = current[-1].id
    return utterances


# One token of a turn's text: its start, its end and the recognised
# word it came from (``None`` when its time is interpolated).
_Token = Tuple[float, float, Optional[Word]]


def _sentence_times(
    pieces: Sequence[Turn], texts: Sequence[str]
) -> List[Tuple[float, float, List[Word]]]:
    """Return the start, end and words of each sentence of a turn.

    ``pieces`` are the turns merged into one (a single turn when nothing
    was merged) and ``texts`` the sentences of their joined text. Each
    piece is aligned on its own: when its words (ignoring words with no
    text) cover its whitespace tokens one to one, every token takes the
    time of its word, and a sentence runs from its first word to its
    last. Interpolation is only a fallback for a piece with no usable
    words: its time is shared out among the parts of sentences it holds
    in proportion to their characters (an estimate: people do not speak
    at a constant rate), and those parts carry no words.
    """
    counts = [len(t.split()) for t in texts]
    sentence_of = [n for n, count in enumerate(counts) for _ in range(count)]
    tokens: List[_Token] = []
    for piece in pieces:
        start = len(tokens)
        size = len(piece.text.split())
        tokens.extend(
            _piece_tokens(piece, sentence_of[start:start + size])
        )
    spans: List[Tuple[float, float, List[Word]]] = []
    position = 0
    for count in counts:
        members = tokens[position:position + count]
        position += count
        words: List[Word] = []
        for _, _, word in members:
            if word is not None and (not words or words[-1] is not word):
                words.append(word)
        spans.append((
            min(t[0] for t in members), max(t[1] for t in members), words,
        ))
    return spans


def _piece_tokens(piece: Turn, sentence_of: Sequence[int]) -> List[_Token]:
    """Return the timed tokens of ``piece``.

    ``sentence_of`` gives, for each token of the piece, the sentence it
    belongs to. Tokens take their word's time when the piece's words
    align with its text. Otherwise each run of tokens of one sentence
    gets a share of the piece's time proportional to its characters,
    and every token of the run takes that span.
    """
    own = piece.text.split()
    aligned = _align(own, piece.words)
    if aligned is not None:
        return aligned
    groups: List[List[str]] = []
    for n, token in enumerate(own):
        if n == 0 or sentence_of[n] != sentence_of[n - 1]:
            groups.append([])
        groups[-1].append(token)
    total = sum(len(" ".join(g)) for g in groups) or 1
    length = piece.end - piece.start
    out: List[_Token] = []
    cursor = piece.start
    for group in groups:
        end = cursor + length * len(" ".join(group)) / total
        out.extend((round(cursor, 3), round(end, 3), None) for _ in group)
        cursor = end
    return out


def _align(
    tokens: Sequence[str], words: Sequence[Word]
) -> Optional[List[_Token]]:
    """Give each token the time of its word, or ``None`` if they differ.

    Words with no text are ignored (recognisers emit them); a word whose
    text holds several tokens gives its time to each of them.
    """
    out: List[_Token] = []
    for word in words:
        out.extend((word.start, word.end, word) for _ in word.text.split())
    return out if out and len(out) == len(tokens) else None


def _mean_probability(words: Sequence[Word]) -> Optional[float]:
    """Mean recogniser confidence of ``words``, when they report it."""
    known = [w.probability for w in words if w.probability is not None]
    return round(sum(known) / len(known), 4) if known else None


def _weakest(pieces: Sequence[Turn]) -> Optional[float]:
    """The lowest known ``speaker_confidence`` of ``pieces``, if any."""
    known = [
        p.speaker_confidence for p in pieces
        if p.speaker_confidence is not None
    ]
    return min(known) if known else None
