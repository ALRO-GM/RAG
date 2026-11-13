"""Sessions: the life of a recording, and the chunks it is uploaded in.

A session moves ``recording → uploaded → queued → running → done |
failed`` (:data:`TRANSITIONS`). A failed transcription can be queued
again; a finished one is final (its transcript is edited by review, not
recomputed). Recording from the phone arrives in numbered chunks:
:func:`accept_chunk` decides whether a chunk is the next one, a retried
copy of the last one, or an error.
"""

import enum
from typing import FrozenSet, Mapping, Optional

from transcriber.domain.errors import InvalidInputError, StateError
from transcriber.domain.models import Session, SessionState

S = SessionState

TRANSITIONS: Mapping[SessionState, FrozenSet[SessionState]] = {
    S.RECORDING: frozenset({S.UPLOADED}),
    S.UPLOADED: frozenset({S.QUEUED}),
    S.QUEUED: frozenset({S.RUNNING, S.FAILED}),
    S.RUNNING: frozenset({S.DONE, S.FAILED}),
    S.DONE: frozenset(),
    S.FAILED: frozenset({S.QUEUED}),
}
"""The states each state may move to; any other move is an error."""


def can_move(current: SessionState, target: SessionState) -> bool:
    """Tell whether a session in ``current`` may move to ``target``."""
    return target in TRANSITIONS[current]


def move(
    session: Session, target: SessionState, failure: Optional[str] = None
) -> Session:
    """Return ``session`` in state ``target``.

    Moving to ``failed`` needs the reason (``failure``); any other move
    clears the previous reason. Raises :class:`StateError` for a move
    not in :data:`TRANSITIONS` and :class:`InvalidInputError` for a
    failure without a reason, or a reason on another move.
    """
    if not can_move(session.state, target):
        raise StateError(
            f"Session {session.id} cannot go from {session.state.value}"
            f" to {target.value}."
        )
    if target is S.FAILED:
        if not failure or not failure.strip():
            raise InvalidInputError("A failed session needs a reason.")
    elif failure is not None:
        raise InvalidInputError("Only a failed session has a reason.")
    return session.model_copy(update={"state": target, "failure": failure})


def require_state(session: Session, *states: SessionState) -> None:
    """Raise :class:`StateError` unless ``session`` is in ``states``."""
    if session.state not in states:
        wanted = " or ".join(s.value for s in states)
        raise StateError(
            f"Session {session.id} is {session.state.value}; it must be"
            f" {wanted}."
        )


class ChunkAction(str, enum.Enum):
    """What to do with an uploaded audio chunk."""

    APPEND = "append"
    DUPLICATE = "duplicate"


def accept_chunk(session: Session, number: int, sha256: str) -> ChunkAction:
    """Decide what to do with chunk ``number`` whose bytes hash to ``sha256``.

    Chunks are numbered from 0, only accepted while ``recording``. The
    next number is appended. Repeating the last accepted chunk with the
    same bytes is a no-op (a retried upload whose answer was lost).
    Anything else is an error: :class:`InvalidInputError` for a gap, a
    negative number, an earlier chunk, or the last number with other
    bytes; :class:`StateError` outside ``recording``.
    """
    require_state(session, S.RECORDING)
    if number < 0:
        raise InvalidInputError("Chunks are numbered from 0.")
    if number == session.chunks:
        return ChunkAction.APPEND
    if number > session.chunks:
        raise InvalidInputError(
            f"Chunk {number} leaves a gap: the next chunk is"
            f" {session.chunks}."
        )
    if number == session.chunks - 1 and sha256 == session.last_chunk_sha256:
        return ChunkAction.DUPLICATE
    raise InvalidInputError(
        f"Chunk {number} was already accepted with other content."
        if number == session.chunks - 1 else
        f"Chunk {number} was already accepted; the next chunk is"
        f" {session.chunks}."
    )


def with_chunk(session: Session, sha256: str) -> Session:
    """Return ``session`` after appending one chunk hashing to ``sha256``."""
    return session.model_copy(update={
        "chunks": session.chunks + 1, "last_chunk_sha256": sha256,
    })
