"""Our Spanish dialogue-act set: its plain-text source and its checks.

The set (``examples/skills/dialogue_acts/``) is written as plain text so
a reader can review every label beside its sentence::

    === neg-01 | negociacion | Renovación del contrato de limpieza
    -- fase: apertura
    Marta: Bueno, pues si os parece empezamos. | gestion
    Marta: ¿Habéis podido ver la oferta? | pregunta #q1
    Luis: Sí, la vimos ayer. | respuesta >q1

- ``===`` opens a dialogue: session id, conversation type
  (:data:`~src.domain.skills.dialogue_acts.SESSION_TYPES`) and title.
- ``-- fase: <name>`` sets the phase of the sentences that follow.
- Every other non-blank line is one sentence: ``speaker: text | label``,
  where ``label`` is one category or up to three weighted ones
  (``compromiso:70 aceptacion:30``: whole %, steps of 10, primary first
  and at least 40, summing to 100), optionally followed by ``#anchor``
  (a name other lines can point at) and ``>anchor`` (the sentence it
  replies to). Consecutive sentences of the same speaker form one turn,
  as in ``transcript.v1``.
- Lines starting with ``#`` are comments.

:func:`parse_set` checks what the labels promise: a known category,
a reply link on every answer, acceptance and rejection, links only
backwards and to another speaker, an answer only to a question (or to
a turn-giving management move), anchors defined once, weights that
follow the annotation guide. A reply rule holds for every label of a
sentence, not only the primary one: an ``aceptacion:30`` still needs the
``>anchor`` of what it accepts.
"""

from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from src.domain.errors import InvalidInputError
from src.domain.skills.dialogue_acts import (
    CATEGORIES,
    SECOND_PARTS,
    SESSION_TYPES,
    WeightedLabel,
    check_annotation,
    parse_labels,
)

ANSWERABLE: Tuple[str, ...] = ("pregunta", "gestion")
"""Categories a ``respuesta`` may reply to."""

NOT_REPLIED_TO: Tuple[str, ...] = ("escucha", "otro")
"""Categories no acceptance or rejection may reply to."""


class _Frozen(BaseModel):
    """Immutable model that refuses unknown fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class SetSession(_Frozen):
    """The declared context of one written dialogue."""

    id: str = Field(min_length=1)
    type: str
    title: str
    language: str = "es"


class SetUtterance(_Frozen):
    """One sentence of the set with its intended label.

    ``id`` is ``t<turn>.s<index>`` as in ``transcript.v1``;
    ``responds_to`` is the id of the sentence it replies to; ``act``
    is the primary label and ``labels`` all of them with their weights
    (a single label carries 100).
    """

    id: str
    turn: int = Field(ge=0)
    index: int = Field(ge=0)
    speaker: str
    text: str
    act: str
    labels: List[WeightedLabel]
    responds_to: Optional[str] = None
    phase: Optional[str] = None
    line: int = Field(ge=1)


class SetDialogue(_Frozen):
    """A written dialogue: session, speakers and labelled sentences."""

    session: SetSession
    speakers: List[str]
    utterances: List[SetUtterance]


class _Builder:
    """Accumulate the sentences of the dialogue being read."""

    def __init__(self, session: SetSession, line: int) -> None:
        """Start the dialogue opened at ``line``."""
        self.session = session
        self.line = line
        self.phase: Optional[str] = None
        self.speakers: List[str] = []
        self.utterances: List[SetUtterance] = []
        self.anchors: Dict[str, SetUtterance] = {}
        self.turn = -1
        self.index = 0

    def add(self, line: int, raw: str, where: str) -> None:
        """Add the sentence written on ``line``."""
        speaker, text, labels, anchor, target = _split_sentence(raw, where)
        acts = [lab.label for lab in labels]
        previous = self.utterances[-1] if self.utterances else None
        if previous is None or previous.speaker != speaker:
            self.turn += 1
            self.index = 0
        else:
            self.index += 1
        responds_to = None
        if target is not None:
            first = self.anchors.get(target)
            if first is None:
                raise InvalidInputError(
                    f"{where}: '>{target}' points at no earlier sentence.")
            for act in acts:
                _check_reply(act, first, speaker, where)
            responds_to = first.id
        else:
            for act in acts:
                if act in SECOND_PARTS:
                    raise InvalidInputError(
                        f"{where}: a {act} needs '>anchor' to what it "
                        "replies to.")
        utterance = SetUtterance(
            id=f"t{self.turn}.s{self.index}", turn=self.turn,
            index=self.index, speaker=speaker, text=text, act=acts[0],
            labels=labels, responds_to=responds_to, phase=self.phase,
            line=line,
        )
        if anchor is not None:
            if anchor in self.anchors:
                raise InvalidInputError(
                    f"{where}: anchor '#{anchor}' is defined twice.")
            self.anchors[anchor] = utterance
        if speaker not in self.speakers:
            self.speakers.append(speaker)
        self.utterances.append(utterance)

    def build(self) -> SetDialogue:
        """Return the finished dialogue; refuse an empty one."""
        if not self.utterances:
            raise InvalidInputError(
                f"line {self.line}: dialogue {self.session.id} is empty.")
        return SetDialogue(session=self.session, speakers=self.speakers,
                           utterances=self.utterances)


def _check_reply(
    act: str, first: SetUtterance, speaker: str, where: str
) -> None:
    """Refuse a reply link the label cannot have.

    Any label of the earlier sentence counts: an answer may reply to a
    ``gestion:60 pregunta:40``.
    """
    if first.speaker == speaker:
        raise InvalidInputError(
            f"{where}: a sentence cannot reply to its own speaker.")
    earlier = {lab.label for lab in first.labels}
    if act == "respuesta" and not earlier & set(ANSWERABLE):
        raise InvalidInputError(
            f"{where}: a respuesta replies to a {first.act}, "
            "not to a pregunta.")
    if act in ("aceptacion", "rechazo") and earlier <= set(NOT_REPLIED_TO):
        raise InvalidInputError(
            f"{where}: a {act} cannot reply to a {first.act}.")


def _split_sentence(
    raw: str, where: str
) -> Tuple[str, str, List[WeightedLabel], Optional[str], Optional[str]]:
    """Split ``speaker: text | label[:weight]... #anchor >target``."""
    head, bar, tail = raw.rpartition(" | ")
    speaker, colon, text = head.partition(":")
    if not bar or not colon or not speaker.strip() or not text.strip():
        raise InvalidInputError(
            f"{where}: expected 'speaker: text | label'.")
    tokens = tail.split()
    if not tokens or tokens[0].partition(":")[0] not in CATEGORIES:
        raise InvalidInputError(
            f"{where}: unknown label {tail.strip()!r}.")
    written = [tokens[0]]
    rest = tokens[1:]
    while rest and ":" in rest[0] and rest[0][0] not in "#>":
        written.append(rest.pop(0))
    try:
        labels = parse_labels(" ".join(written))
        check_annotation(labels)
    except ValueError as error:
        raise InvalidInputError(f"{where}: {error}.") from error
    anchor = target = None
    for token in rest:
        if token.startswith("#") and len(token) > 1 and anchor is None:
            anchor = token[1:]
        elif token.startswith(">") and len(token) > 1 and target is None:
            target = token[1:]
        else:
            raise InvalidInputError(f"{where}: unexpected {token!r}.")
    return speaker.strip(), text.strip(), labels, anchor, target


def _open(raw: str, where: str) -> SetSession:
    """Read a ``=== id | type | title`` header."""
    parts = [p.strip() for p in raw[3:].split("|")]
    if len(parts) != 3 or not all(parts):
        raise InvalidInputError(f"{where}: expected '=== id | type | title'.")
    if parts[1] not in SESSION_TYPES:
        raise InvalidInputError(
            f"{where}: unknown conversation type {parts[1]!r}.")
    return SetSession(id=parts[0], type=parts[1], title=parts[2])


def parse_set(text: str, source: str = "set") -> List[SetDialogue]:
    """Parse and check the dialogues written in ``text``.

    Raises :class:`~src.domain.errors.InvalidInputError` naming
    ``source`` and the line of the first problem found.
    """
    dialogues: List[SetDialogue] = []
    seen: Dict[str, int] = {}
    current: Optional[_Builder] = None
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        where = f"{source}:{number}"
        if not line or line.startswith("#"):
            continue
        if line.startswith("==="):
            if current is not None:
                dialogues.append(current.build())
            session = _open(line, where)
            if session.id in seen:
                raise InvalidInputError(
                    f"{where}: dialogue {session.id} already defined "
                    f"on line {seen[session.id]}.")
            seen[session.id] = number
            current = _Builder(session, number)
        elif current is None:
            raise InvalidInputError(f"{where}: sentence before any '==='.")
        elif line.startswith("--"):
            name, _, value = line[2:].partition(":")
            if name.strip() != "fase" or not value.strip():
                raise InvalidInputError(
                    f"{where}: expected '-- fase: <name>'.")
            current.phase = value.strip()
        else:
            current.add(number, line, where)
    if current is not None:
        dialogues.append(current.build())
    return dialogues


def set_distribution(dialogues: List[SetDialogue]) -> Dict[str, int]:
    """Count the sentences of the set per primary label, in category order.
    """
    counts = {c: 0 for c in CATEGORIES}
    for dialogue in dialogues:
        for utterance in dialogue.utterances:
            counts[utterance.act] += 1
    return counts


def weighted_count(dialogues: List[SetDialogue]) -> int:
    """Count the sentences that carry more than one label."""
    return sum(len(u.labels) > 1 for d in dialogues for u in d.utterances)


def _utterance_json(utterance: SetUtterance) -> Dict[str, object]:
    """Return one sentence for :func:`to_transcript_like`."""
    document = utterance.model_dump(exclude={"line", "labels"})
    if len(utterance.labels) > 1:
        document["labels"] = [
            {"label": lab.label, "weight": lab.weight}
            for lab in utterance.labels
        ]
    return document


def to_transcript_like(dialogue: SetDialogue) -> Dict[str, object]:
    """Return the dialogue in a ``transcript.v1``-like JSON shape.

    No times or audio (the set is written, not recorded): sessions carry
    ``type``, sentences carry ``act`` (the primary label),
    ``responds_to`` and ``phase``, and ``labels`` (``label`` and
    ``weight``) only when they have more than one label, so single-label
    sentences keep the shape they always had.
    """
    return {
        "schema": "dialogue_acts.es.v1",
        "session": dialogue.session.model_dump(),
        "speakers": [{"id": s, "name": s, "enrolled": False}
                     for s in dialogue.speakers],
        "utterances": [_utterance_json(u) for u in dialogue.utterances],
        "synthetic": True,
    }
