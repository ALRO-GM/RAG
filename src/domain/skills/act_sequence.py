"""The input of the second layer: a session's sentences with their acts.

Conversation events (``docs/skills/CONVERSATION_EVENTS.md``) are derived
from first-layer labels alone, whoever produced them: the intended
labels of our Spanish set (``examples/skills/dialogue_acts/``) or an
automatic tagger's ``<session>.acts.json``. Both are read into one small
model, :class:`ActSession`: per sentence its id, speaker, turn, position,
optional time, weighted labels and optional reply link.

The JSON shape read by :func:`session_from_json` is the one
``to_transcript_like`` writes for the set (``dialogue_acts.es.v1``),
which is also the shape expected from the tagger::

    {
      "schema": "acts.v1",
      "session": {"id": "demo_es"},
      "utterances": [
        {"id": "t3.s1", "turn": 3, "speaker": "Ana", "text": "...",
         "start": 12.4, "end": 14.0,
         "labels": [{"label": "pregunta", "weight": 72},
                    {"label": "gestion", "weight": 20, "doubtful": true}],
         "responds_to": null}
      ]
    }

``labels`` may be replaced by ``act`` (one label, weight 100); ``turn``,
``start``, ``end`` and ``responds_to`` are optional; unknown keys (the
tagger's model name, ``rest``, ``phase``) are ignored. A missing
``turn`` is counted from speaker changes, as in ``transcript.v1``.
"""

from typing import Any, Dict, List, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.domain.errors import InvalidInputError
from src.domain.skills.dialogue_act_set import SetDialogue
from src.domain.skills.dialogue_acts import WeightedLabel

ACTS_SCHEMA = "acts.v1"
"""Schema name the tagger is expected to write in ``<session>.acts.json``.
"""


class _Frozen(BaseModel):
    """Immutable model that refuses unknown fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ActSentence(_Frozen):
    """One sentence with its first-layer labels.

    ``order`` is its position in the session (0-based); ``turn`` the
    speaker turn it belongs to; ``responds_to`` the id of the sentence
    it replies to, when the source gives one.
    """

    id: str = Field(min_length=1)
    order: int = Field(ge=0)
    turn: int = Field(ge=0)
    speaker: str = Field(min_length=1)
    text: str
    labels: List[WeightedLabel] = Field(min_length=1)
    responds_to: Optional[str] = None
    start: Optional[float] = None
    end: Optional[float] = None

    def weight(self, label: str) -> int:
        """Return the weight of ``label`` on this sentence (0 if absent)."""
        for item in self.labels:
            if item.label == label:
                return item.weight
        return 0


class ActSession(_Frozen):
    """The labelled sentences of one session, in spoken order."""

    id: str = Field(min_length=1)
    sentences: List[ActSentence]
    synthetic: bool = False

    def sentence(self, sentence_id: str) -> ActSentence:
        """Return the sentence with ``sentence_id``."""
        for item in self.sentences:
            if item.id == sentence_id:
                return item
        raise KeyError(sentence_id)

    def speakers(self) -> List[str]:
        """Return the speakers in order of first appearance."""
        seen: List[str] = []
        for item in self.sentences:
            if item.speaker not in seen:
                seen.append(item.speaker)
        return seen


def session_from_set(dialogue: SetDialogue) -> ActSession:
    """Return a dialogue of the Spanish set as an :class:`ActSession`."""
    return ActSession(
        id=dialogue.session.id, synthetic=True,
        sentences=[
            ActSentence(
                id=u.id, order=position, turn=u.turn, speaker=u.speaker,
                text=u.text, labels=u.labels, responds_to=u.responds_to,
            )
            for position, u in enumerate(dialogue.utterances)
        ],
    )


def _labels(raw: Mapping[str, Any], where: str) -> List[WeightedLabel]:
    """Read ``labels`` (or a single ``act``) of one sentence."""
    try:
        if raw.get("labels"):
            return [
                WeightedLabel(
                    label=str(item["label"]), weight=int(item["weight"]),
                    doubtful=bool(item.get("doubtful", False)),
                )
                for item in raw["labels"]
            ]
        if raw.get("act"):
            return [WeightedLabel(label=str(raw["act"]), weight=100)]
    except (KeyError, TypeError, ValueError, ValidationError) as error:
        raise InvalidInputError(f"{where}: bad labels ({error}).") from error
    raise InvalidInputError(f"{where}: no 'labels' nor 'act'.")


def _number(value: Any) -> Optional[float]:
    """Return ``value`` as a float, keeping ``None``."""
    return None if value is None else float(value)


def session_from_json(
    document: Mapping[str, Any], source: str = "acts"
) -> ActSession:
    """Read one session in the ``acts.v1`` / ``dialogue_acts.es.v1`` shape.

    Raises :class:`~src.domain.errors.InvalidInputError` naming
    ``source`` when a required field is missing or a label is unknown.
    """
    session = document.get("session")
    session_id = (
        session.get("id") if isinstance(session, Mapping)
        else document.get("session_id")
    )
    rows = document.get("utterances")
    if not session_id or not isinstance(rows, list):
        raise InvalidInputError(
            f"{source}: expected 'session.id' and a list of 'utterances'.")
    sentences: List[ActSentence] = []
    turn = -1
    previous: Optional[str] = None
    for position, raw in enumerate(rows):
        where = f"{source}: utterance {position}"
        if not isinstance(raw, Mapping):
            raise InvalidInputError(f"{where}: not an object.")
        speaker = str(raw.get("speaker") or "")
        if speaker != previous:
            turn += 1
            previous = speaker
        try:
            sentences.append(ActSentence(
                id=str(raw.get("id") or f"u{position}"), order=position,
                turn=int(raw["turn"]) if raw.get("turn") is not None
                else turn,
                speaker=speaker, text=str(raw.get("text") or ""),
                labels=_labels(raw, where),
                responds_to=raw.get("responds_to"),
                start=_number(raw.get("start")), end=_number(raw.get("end")),
            ))
        except (TypeError, ValueError) as error:
            raise InvalidInputError(f"{where}: {error}") from error
    ids = [s.id for s in sentences]
    if len(set(ids)) != len(ids):
        raise InvalidInputError(f"{source}: sentence ids are not unique.")
    return ActSession(
        id=str(session_id), sentences=sentences,
        synthetic=bool(document.get("synthetic", False)),
    )


def sessions_from_json(
    document: Any, source: str = "acts"
) -> List[ActSession]:
    """Read a session, or a list of them (the set's ``dialogues.json``)."""
    if isinstance(document, list):
        return [
            session_from_json(item, f"{source}[{n}]")
            for n, item in enumerate(document) if isinstance(item, Mapping)
        ]
    if isinstance(document, Mapping):
        return [session_from_json(document, source)]
    raise InvalidInputError(f"{source}: expected a JSON object or list.")


def without_links(session: ActSession) -> ActSession:
    """Return ``session`` with every reply link removed.

    Used to measure the reply linking on a set whose links are known.
    """
    return session.model_copy(update={"sentences": [
        s.model_copy(update={"responds_to": None}) for s in session.sentences
    ]})


def link_map(session: ActSession) -> Dict[str, str]:
    """Return ``{sentence id: id it responds to}`` for the given links."""
    return {
        s.id: s.responds_to for s in session.sentences
        if s.responds_to is not None
    }
