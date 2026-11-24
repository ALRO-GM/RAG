"""Who speaks in a mono live session: frames to sentences (LV-6).

A mono live session that asks for diarization hears every voice on one
channel. A streaming diarizer (behind the ``StreamDiarizer`` port;
NVIDIA streaming Sortformer v2) gives, a little after the audio, the
probability that each of up to four speakers ("slots", numbered in the
order they are first heard) talks in every 80 ms frame. This module
turns those frames into a speaker per final sentence; no I/O:

* :class:`SpeakerFrames`: the frames received so far.
* :func:`pick_slot`: the slot of one stretch of time, how much of the
  stretch the frames already cover, and how pure it is.
* :class:`MonoSpeakers`: slot labels (``S1`` .. ``S4``, or the id of
  an enrolled person once the voiceprints name the slot), the finals
  already sent, and the revisions due when later frames or a name
  change a final's speaker (``speaker_update`` events).
* :class:`NamingPlan`: how many seconds of clean speech each slot has
  gathered and when to compare it with the voiceprints again;
  :func:`name_slots` does the comparison (one person per slot at most).

A final is sent with the frames there are (they lag the audio by about
a second); its speaker is then ``provisional`` until the frames cover
the whole sentence, and revised when they say otherwise.
"""

from dataclasses import dataclass
from typing import (
    AbstractSet,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from transcriber.domain.errors import InvalidInputError
from transcriber.domain.live import LiveUtterance
from transcriber.domain.models import Voiceprint
from transcriber.domain.voices import (
    DEFAULT_MATCH_THRESHOLD,
    anonymous_label,
    match_voices,
)

FloatArray = npt.NDArray[np.float32]

MAX_SLOTS = 4
"""Speakers the streaming diarizer tells apart (Sortformer 4spk)."""


class MonoDiarizationSettings(BaseModel):
    """How a mono live session names the speaker of each sentence.

    A frame counts as a slot's speech at ``threshold``. The final of a
    segment waits up to ``wait_ms`` for the frames of its last audio
    (§6 gives attribution 150 ms in mono); what is still missing then
    makes the speaker provisional. A sentence feeds a slot's voice for
    naming when its frames are complete and at least ``purity`` of its
    speech frames are that slot's alone. Naming is first tried after
    ``naming_seconds`` of such speech, again each time the slot has
    ``naming_growth`` times more, until ``naming_max_seconds``. A slot
    takes an enrolled id at a cosine of ``match_threshold`` or more
    (:func:`~transcriber.domain.voices.match_voices`).
    """

    model_config = ConfigDict(frozen=True)

    threshold: float = Field(default=0.5, gt=0, lt=1)
    wait_ms: int = Field(default=150, ge=0, le=5000)
    purity: float = Field(default=0.8, ge=0, le=1)
    naming_seconds: float = Field(default=4.0, gt=0)
    naming_growth: float = Field(default=2.0, gt=1)
    naming_max_seconds: float = Field(default=30.0, gt=0)
    match_threshold: float = Field(
        default=DEFAULT_MATCH_THRESHOLD, ge=-1, le=1
    )


class SpeakerFrames:
    """The per-frame speaker probabilities received so far.

    Frame ``k`` covers ``[k * step, (k + 1) * step)`` seconds of the
    stream; each row holds one probability per slot.
    """

    def __init__(self, step: float, slots: int = MAX_SLOTS) -> None:
        """Start with no frames."""
        if step <= 0 or not 1 <= slots <= MAX_SLOTS:
            raise InvalidInputError(
                f"frames need a positive step and 1-{MAX_SLOTS} slots"
            )
        self.step = step
        self.slots = slots
        self._rows: List[List[float]] = []

    def __len__(self) -> int:
        """Return the number of frames received."""
        return len(self._rows)

    @property
    def covered(self) -> float:
        """Return the stream time up to which frames have arrived."""
        return len(self._rows) * self.step

    def extend(self, first: int, rows: Sequence[Sequence[float]]) -> None:
        """Add ``rows``, the frames from number ``first`` on.

        Raises :class:`InvalidInputError` for a gap or an overlap with
        the frames already there, or a row of the wrong width.
        """
        if first != len(self._rows):
            raise InvalidInputError(
                f"frame {first} does not follow frame {len(self._rows) - 1}"
            )
        for row in rows:
            if len(row) != self.slots:
                raise InvalidInputError(
                    f"a frame has {len(row)} speakers, not {self.slots}"
                )
            self._rows.append([float(p) for p in row])

    def window(self, start: float, end: float) -> Tuple[int, int]:
        """Return the frame numbers ``[a, b)`` whose centres lie in a span.

        A span shorter than a frame gets the frame holding its middle.
        """
        a = int(np.ceil(start / self.step - 0.5))
        b = int(np.ceil(end / self.step - 0.5))
        if b <= a:
            a = int(max((start + end) / 2, 0.0) // self.step)
            b = a + 1
        return max(a, 0), max(b, 0)

    def rows(self, a: int, b: int) -> FloatArray:
        """Return frames ``a`` to ``b`` that have arrived (2-D array)."""
        picked = self._rows[a:min(b, len(self._rows))]
        if not picked:
            return np.zeros((0, self.slots), dtype=np.float32)
        return np.asarray(picked, dtype=np.float32)


@dataclass(frozen=True)
class SlotPick:
    """The slot of a stretch of time, as far as the frames tell.

    ``slot`` is ``None`` when no frame of the stretch holds speech.
    ``coverage`` is the share of the stretch's frames received (1:
    complete, the pick will not change). ``purity`` is the share of
    the frames with speech where the slot talks alone.
    """

    slot: Optional[int]
    coverage: float
    purity: float

    @property
    def complete(self) -> bool:
        """Tell whether every frame of the stretch has arrived."""
        return self.coverage >= 1.0


def pick_slot(
    frames: SpeakerFrames, start: float, end: float, threshold: float
) -> SlotPick:
    """Return the slot that talks most in ``start``-``end``.

    Each slot scores the sum of its probabilities over the frames where
    it is active (at or above ``threshold``); the best score wins, the
    lower slot on a tie.
    """
    a, b = frames.window(start, end)
    rows = frames.rows(a, b)
    coverage = len(rows) / (b - a)
    active = rows >= threshold
    scores = np.where(active, rows, 0.0).sum(axis=0)
    if len(rows) == 0 or float(scores.max()) <= 0.0:
        return SlotPick(None, coverage, 0.0)
    slot = int(np.argmax(scores))
    voiced = active.any(axis=1)
    alone = active[:, slot] & (active.sum(axis=1) == 1)
    purity = float(alone.sum()) / float(voiced.sum())
    return SlotPick(slot, coverage, purity)


def slot_labels(
    given: Sequence[str], slots: int = MAX_SLOTS,
    taken: AbstractSet[str] = frozenset(),
) -> List[str]:
    """Return the id of each slot: ``given`` first, then ``S<n>``.

    Slot ``n`` is the ``n``-th voice heard; the first ones take the ids
    the session was given (a product config lists the agent first, who
    answers the call). The others get the anonymous labels ``S1``,
    ``S2``, ... skipping ids already used and ``taken`` (enrolled ids:
    an anonymous voice never wears a person's id).
    """
    labels = list(given[:slots])
    used = set(labels) | set(taken)
    number = 1
    while len(labels) < slots:
        label = anonymous_label(number)
        if label not in used:
            labels.append(label)
            used.add(label)
        number += 1
    return labels


@dataclass
class _Sent:
    """A final already sent: the utterance, its slot and if complete.

    ``pinned`` is a speaker the sentence keeps whatever its slot says
    (the employee's voice check, or a correction by hand, #249).
    """

    utterance: LiveUtterance
    slot: Optional[int]
    complete: bool
    pinned: Optional[str] = None


class MonoSpeakers:
    """Slot labels and the finals sent, with the revisions they need.

    ``labels`` are the anonymous ids of the slots, in slot order
    (:func:`slot_labels`). A two-party session (#249) gives every slot
    a name (the employee's or the other person's id), so no anonymous
    label is ever shown, and marks the slots whose voice has not been
    compared with the employee's voice check yet (``speaker_unsure``).
    """

    def __init__(
        self,
        labels: Sequence[str],
        settings: MonoDiarizationSettings = MonoDiarizationSettings(),
    ) -> None:
        """Start with anonymous slots and no finals."""
        if not 1 <= len(labels) <= MAX_SLOTS or len(set(labels)) != len(
            labels
        ):
            raise InvalidInputError(
                f"a mono session has 1-{MAX_SLOTS} distinct slot labels"
            )
        self._s = settings
        self.slots = len(labels)
        self._names: Dict[int, str] = {}
        self._anonymous = list(labels)
        self._sent: Dict[str, _Sent] = {}
        self._pending: List[str] = []
        self._unsure: AbstractSet[int] = frozenset()
        self.last: Optional[str] = None

    def anonymous(self) -> List[str]:
        """Return the anonymous label of every slot, in slot order."""
        return list(self._anonymous)

    def label(self, slot: int) -> str:
        """Return the id of ``slot``: a name if given, else its label."""
        return self._names.get(slot, self._anonymous[slot])

    @property
    def names(self) -> Dict[int, str]:
        """Return the enrolled id given to each named slot."""
        return dict(self._names)

    def speaker(
        self, pick: SlotPick, fallback: str
    ) -> Tuple[str, bool]:
        """Return the speaker id of ``pick`` and whether it is provisional.

        Without a slot (no speech in the frames yet) the ``fallback``
        speaker is used.
        """
        speaker = fallback if pick.slot is None else self.label(pick.slot)
        return speaker, not pick.complete

    def unsure(self, slot: Optional[int]) -> bool:
        """Tell whether ``slot``'s voice has not been compared with the
        employee's voice check yet (two-party, ``speaker_unsure``)."""
        return slot is not None and slot in self._unsure

    def sent(
        self, utterance: LiveUtterance, pick: SlotPick,
        pinned: Optional[str] = None,
    ) -> None:
        """Remember a final as sent with ``pick``'s speaker.

        A ``pinned`` final keeps that speaker through every revision.
        """
        known = utterance.id in self._sent
        self._sent[utterance.id] = _Sent(
            utterance, pick.slot, pick.complete, pinned
        )
        if not pick.complete and not known:
            self._pending.append(utterance.id)
        self.last = utterance.speaker

    def slot_of(self, ident: str) -> Optional[int]:
        """Return the slot of the final ``ident`` (``None``: unknown)."""
        sent = self._sent.get(ident)
        return sent.slot if sent is not None else None

    def settled_slot(self, ident: str) -> Optional[int]:
        """Return the slot of final ``ident`` once its frames are complete."""
        sent = self._sent.get(ident)
        if sent is None or not sent.complete:
            return None
        return sent.slot

    def pinned(self) -> Dict[str, str]:
        """Return the speaker of every pinned final."""
        return {
            ident: sent.pinned for ident, sent in self._sent.items()
            if sent.pinned is not None
        }

    def pin(self, ident: str, speaker: str) -> List[LiveUtterance]:
        """Give the final ``ident`` the speaker ``speaker`` for good.

        Returns it when that changes what was sent (empty otherwise,
        and for an unknown id).
        """
        sent = self._sent.get(ident)
        if sent is None:
            return []
        sent.pinned = speaker
        old = sent.utterance
        if (old.speaker, old.speaker_unsure) == (speaker, False):
            return []
        sent.utterance = old.model_copy(update={
            "speaker": speaker, "speaker_unsure": False,
            "revision": old.revision + 1,
        })
        return [sent.utterance]

    def sent_utterances(self) -> List[LiveUtterance]:
        """Return the latest version of every final sent."""
        return [sent.utterance for sent in self._sent.values()]

    def pending(self) -> List[LiveUtterance]:
        """Return the finals whose frames are not complete yet."""
        return [self._sent[i].utterance for i in self._pending]

    def revise(self, frames: SpeakerFrames) -> List[LiveUtterance]:
        """Return the finals whose speaker the new frames change.

        A provisional final is picked again; once its frames are
        complete it stops being provisional (which is a revision too,
        even with the same speaker). Each revised utterance has its
        ``revision`` one higher.
        """
        out: List[LiveUtterance] = []
        still: List[str] = []
        for ident in self._pending:
            sent = self._sent[ident]
            old = sent.utterance
            pick = pick_slot(frames, old.start, old.end, self._s.threshold)
            speaker, provisional = self.speaker(pick, old.speaker)
            unsure = self.unsure(pick.slot)
            if sent.pinned is not None:
                speaker, unsure = sent.pinned, False
            if provisional:
                still.append(ident)
            if (speaker, provisional, unsure) != (
                old.speaker, old.speaker_provisional, old.speaker_unsure
            ):
                new = old.model_copy(update={
                    "speaker": speaker, "speaker_provisional": provisional,
                    "speaker_unsure": unsure,
                    "revision": old.revision + 1,
                })
                out.append(new)
                sent.utterance = new
            sent.slot = pick.slot if pick.slot is not None else sent.slot
            sent.complete = pick.complete
        self._pending = still
        return out

    def rename(
        self, names: Mapping[int, str],
        unsure: AbstractSet[int] = frozenset(),
    ) -> List[LiveUtterance]:
        """Give slots the enrolled ids ``names``; return the finals changed.

        Slots missing from ``names`` go back to their anonymous label.
        ``unsure`` marks the slots whose voice has not been compared with
        the voice check yet (``speaker_unsure``). A
        pinned final keeps its speaker.
        """
        self._names = {
            slot: ident for slot, ident in names.items()
            if 0 <= slot < self.slots
        }
        self._unsure = frozenset(unsure)
        out: List[LiveUtterance] = []
        for sent in self._sent.values():
            if sent.slot is None or sent.pinned is not None:
                continue
            speaker = self.label(sent.slot)
            doubt = self.unsure(sent.slot)
            if (speaker, doubt) != (
                sent.utterance.speaker, sent.utterance.speaker_unsure
            ):
                sent.utterance = sent.utterance.model_copy(update={
                    "speaker": speaker, "speaker_unsure": doubt,
                    "revision": sent.utterance.revision + 1,
                })
                out.append(sent.utterance)
        return out


class NamingPlan:
    """Clean speech gathered per slot and when to try naming it.

    The caller adds the seconds of each clean sentence (:meth:`add`);
    :meth:`due` lists the slots that reached their next threshold.
    """

    def __init__(
        self, settings: MonoDiarizationSettings = MonoDiarizationSettings()
    ) -> None:
        """Start with no speech."""
        self._s = settings
        self.seconds: Dict[int, float] = {}
        self._next: Dict[int, float] = {}

    def wants(self, slot: int) -> bool:
        """Tell whether ``slot`` still gathers speech."""
        return self.seconds.get(slot, 0.0) < self._s.naming_max_seconds

    def add(self, slot: int, seconds: float) -> None:
        """Count ``seconds`` more of ``slot``'s clean speech."""
        self.seconds[slot] = self.seconds.get(slot, 0.0) + max(seconds, 0.0)

    def due(self) -> List[int]:
        """Return the slots to compare now (and move their threshold)."""
        out: List[int] = []
        for slot, seconds in sorted(self.seconds.items()):
            target = self._next.get(slot, self._s.naming_seconds)
            if seconds >= target:
                out.append(slot)
                self._next[slot] = seconds * self._s.naming_growth
        return out


def unit_sum(vectors: FloatArray) -> Optional[Tuple[FloatArray, int]]:
    """Return the sum and the count of the non-zero rows, made unit.

    A slot's voice is the direction of the sum of all its window
    vectors, so new windows are added without embedding the old ones
    again. ``None`` when no row carries a voice.
    """
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        return None
    norms = np.linalg.norm(vectors, axis=1)
    kept = vectors[norms > 1e-6] / norms[norms > 1e-6][:, None]
    if kept.shape[0] == 0:
        return None
    total: FloatArray = kept.sum(axis=0).astype(np.float32)
    return total, int(kept.shape[0])


def name_slots(
    voices: Mapping[int, FloatArray],
    voiceprints: Sequence[Voiceprint],
    threshold: float,
    model: Optional[str] = None,
) -> Tuple[Dict[int, str], Dict[int, Tuple[Optional[str], Optional[float]]]]:
    """Return the enrolled id of each slot that matches a voiceprint.

    ``voices`` holds one unit vector per slot. One person goes to one
    slot at most (the assignment of
    :func:`~transcriber.domain.voices.match_voices`). The second
    result gives, per slot, the nearest voiceprint and its cosine
    (recorded to study the threshold).
    """
    slots = sorted(voices)
    if not slots or not voiceprints:
        return {}, {slot: (None, None) for slot in slots}
    groups = np.vstack([voices[slot] for slot in slots]).astype(np.float32)
    found = match_voices(groups, voiceprints, threshold, model=model)
    names = {
        slots[a.group]: a.speaker for a in found if a.enrolled
    }
    nearest = {
        slots[a.group]: (a.nearest, a.nearest_similarity) for a in found
    }
    return names, nearest
