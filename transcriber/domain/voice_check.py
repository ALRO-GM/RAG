"""Two people on one microphone: the employee's voice check (#249).

A one-microphone live session of a two-party product (an interview, a
support call) starts with a short voice check: the employee says a
sentence or two, guided by the page. That speech is the session's
reference of the employee's voice, with this microphone and this room.
From then on every sentence is the employee's or the other person's;
the diarizer's extra voices never become a third person.

Plain rules, no I/O:

* :class:`VoiceCheckSettings`: how long the check is, and the
  thresholds of the attribution (chosen from the measurement in
  ``docs/live/MONO_DIARIZATION.md`` §"Voice check").
* :class:`VoiceCheck`: the check's state (listening, recognised, short,
  skipped, failed) and the speech heard so far.
* :class:`PartySides`: which side each diarizer slot is on: one slot
  per person, the one whose voice is nearest to the check's (the slot
  that spoke the check stands in until voices can be compared); the
  corrections of the employee («Esta soy yo» / «Esta no soy yo») pin a
  slot to a side.

The reference lives in memory only: it is never written and never
becomes an enrolment.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Mapping, Optional, Tuple

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

FloatArray = npt.NDArray[np.float32]

CheckState = Literal["listening", "recognised", "short", "skipped", "failed"]
"""The states of a voice check (``live.event.v1`` ``voice_check``)."""

Side = Literal["employee", "other"]
"""The two sides of a two-party session."""


class VoiceCheckSettings(BaseModel):
    """How the voice check runs and how its reference is used.

    The check takes the employee's first ``seconds`` of speech (whole
    sentences, so a little more); when the session's first sentences
    stop before that, it settles for ``min_seconds`` once
    ``wait_seconds`` of the stream have passed since the first one
    began, else it is ``short`` and the session keeps the diarizer's
    voices as before. A slot whose voice (its clean speech so far, at
    least ``slot_seconds``) is compared with the check: the nearest
    slot, at ``match`` or more, is the employee's; with ``split`` set,
    so is another slot at ``split`` or more (one person split in two
    slots: off by default, it cost more than it gained in the
    measurement). The values come from ``experiments/live/voice_check.py`` (see
    ``docs/live/MONO_DIARIZATION.md`` §"Voice check").
    """

    model_config = ConfigDict(frozen=True)

    seconds: float = Field(default=5.0, gt=0, le=30)
    min_seconds: float = Field(default=3.0, gt=0, le=30)
    wait_seconds: float = Field(default=20.0, gt=0, le=120)
    slot_seconds: float = Field(default=2.0, ge=0, le=60)
    match: float = Field(default=0.86, ge=-1, le=1)
    split: Optional[float] = Field(default=None, ge=-1, le=1)


class VoiceCheck:
    """The state of the employee's voice check.

    :meth:`hear` adds the seconds of a sentence spoken during the
    check; :meth:`settle` decides when the check is over.
    """

    def __init__(
        self, settings: VoiceCheckSettings = VoiceCheckSettings()
    ) -> None:
        """Start listening, with no speech heard."""
        self._s = settings
        self.state: CheckState = "listening"
        self.heard = 0.0
        self.began: Optional[float] = None

    @property
    def needed(self) -> float:
        """Return the seconds of speech the check asks for."""
        return self._s.seconds

    @property
    def listening(self) -> bool:
        """Tell whether the check still takes speech."""
        return self.state == "listening"

    def hear(self, start: float, seconds: float) -> None:
        """Count a sentence of the check that began at ``start``."""
        if not self.listening:
            return
        if self.began is None:
            self.began = start
        self.heard += max(seconds, 0.0)

    def settle(self, now: float) -> Optional[CheckState]:
        """Return the check's new state at stream time ``now``, if any.

        ``recognised`` once enough speech was heard, or at least
        ``min_seconds`` when ``wait_seconds`` have passed since the
        check's first sentence; ``short`` when less was heard by then.
        ``None`` while it keeps listening.
        """
        if not self.listening:
            return None
        if self.heard >= self._s.seconds:
            self.state = "recognised"
        elif self.began is not None \
                and now - self.began >= self._s.wait_seconds:
            self.state = (
                "recognised" if self.heard >= self._s.min_seconds
                else "short"
            )
        else:
            return None
        return self.state

    def skip(self) -> bool:
        """Skip the check (the employee's choice); tell if it changed."""
        if not self.listening:
            return False
        self.state = "skipped"
        return True

    def fail(self) -> None:
        """Give up the check (its voice could not be computed)."""
        self.state = "failed"


def unit(vector: FloatArray) -> Optional[FloatArray]:
    """Return ``vector`` made unit; ``None`` when it is (nearly) zero."""
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < 1e-6:
        return None
    out: FloatArray = (vector / norm).astype(np.float32)
    return out


@dataclass
class SlotSide:
    """What is known of one slot's side.

    ``score`` is the cosine of its voice with the check (``None``
    before it has ``slot_seconds`` of voice); ``pinned`` a side the
    employee chose by hand.
    """

    side: Side = "other"
    score: Optional[float] = None
    seconds: float = 0.0
    pinned: Optional[Side] = None
    source: str = "default"


@dataclass
class PartySides:
    """The side of each diarizer slot in a two-party session.

    One person per slot, as the measurement chose (``MONO_DIARIZATION.md``
    §"Voice check"): once slots have voice enough to compare
    (:meth:`voices`), the slot nearest to the check, at ``match`` or
    more, is the employee's, and every other slot the other person's,
    unless ``split`` is set and its voice is that close (the diarizer
    split the employee in two). Until a slot is compared, the slot that spoke
    the check stands in for the employee (:meth:`recognise`). Before
    the check every slot is the other person's. A slot the employee
    corrected keeps that side.
    """

    settings: VoiceCheckSettings = field(
        default_factory=VoiceCheckSettings
    )
    reference: Optional[FloatArray] = None
    check_slot: Optional[int] = None
    slots: Dict[int, SlotSide] = field(default_factory=dict)

    def _slot(self, slot: int) -> SlotSide:
        """Return the record of ``slot`` (made on first use)."""
        return self.slots.setdefault(slot, SlotSide())

    def side(self, slot: Optional[int]) -> Side:
        """Return the side of ``slot`` (the other person when unknown)."""
        if slot is None or slot not in self.slots:
            return "other"
        return self.slots[slot].side

    def unsure(self, slot: Optional[int]) -> bool:
        """Tell whether ``slot``'s side is not known from its voice yet.

        After the check, a slot whose voice has not been compared with
        it (less than ``slot_seconds`` of clean speech) is unsure, but
        for the slot that spoke the check. Pinned slots never are.
        """
        if slot is None or self.reference is None:
            return False
        known = self.slots.get(slot)
        if known is None:
            return True
        return known.pinned is None and known.score is None \
            and slot != self.check_slot

    def recognise(
        self, reference: FloatArray, check_slot: Optional[int]
    ) -> None:
        """Take the check's voice and the slot that spoke it."""
        self.reference = unit(reference)
        self.check_slot = check_slot
        if check_slot is not None:
            self._slot(check_slot)
        self._refresh()

    def set_check_slot(self, slot: int) -> None:
        """Say which slot spoke the check (its frames came late)."""
        self.check_slot = slot
        self._slot(slot)
        self._refresh()

    def voices(
        self, voices: Mapping[int, Tuple[FloatArray, float]]
    ) -> None:
        """Compare the slots' voices (unit vector, seconds) with the check."""
        for slot, (voice, seconds) in voices.items():
            known = self._slot(slot)
            known.seconds = seconds
            if self.reference is not None \
                    and seconds >= self.settings.slot_seconds:
                known.score = float(voice @ self.reference)
        self._refresh()

    def _refresh(self) -> None:
        """Set every slot's side from the pins, the voices and the check."""
        free = {
            slot: known.score for slot, known in self.slots.items()
            if known.pinned is None and known.score is not None
        }
        best = max(sorted(free), key=lambda k: free[k] or 0.0) \
            if free else None
        named = best is not None and (free[best] or 0.0) >= \
            self.settings.match
        for slot, known in self.slots.items():
            if known.pinned is not None:
                known.side, known.source = known.pinned, "pinned"
            elif named and slot == best:
                known.side, known.source = "employee", "voice"
            elif named and known.score is not None \
                    and self.settings.split is not None \
                    and known.score >= self.settings.split:
                known.side, known.source = "employee", "split"
            elif not named and slot == self.check_slot \
                    and self.reference is not None:
                known.side, known.source = "employee", "check"
            elif known.score is not None:
                known.side, known.source = "other", "voice"
            else:
                known.side, known.source = "other", "default"

    def pin(self, slot: int, side: Side) -> None:
        """Give ``slot`` the side the employee chose, from now on."""
        known = self._slot(slot)
        known.pinned = side
        if side == "other" and slot == self.check_slot:
            self.check_slot = None
        self._refresh()

    def sides(self) -> Dict[int, Side]:
        """Return the side of every known slot."""
        return {slot: known.side for slot, known in self.slots.items()}

    def diagnosis(self) -> List[Dict[str, object]]:
        """Return each slot's numbers and decision (no audio, no text)."""
        return [
            {"slot": slot, "side": k.side, "source": k.source,
             "score": None if k.score is None else round(k.score, 4),
             "seconds": round(k.seconds, 2), "pinned": k.pinned}
            for slot, k in sorted(self.slots.items())
        ]
