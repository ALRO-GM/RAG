"""Use case piece: the speaker of each sentence of a mono live session (LV-6).

:class:`MonoAttributor` sits between the live engine and a
:class:`~transcriber.application.ports.StreamDiarizer`. The engine
gives it every 32 ms frame of audio (:meth:`MonoAttributor.feed`), the
sentences of each final (:meth:`MonoAttributor.attribute`) and asks it,
after every frame, for the finals whose speaker changed since they
were sent (:meth:`MonoAttributor.poll`): those go out as
``speaker_update`` events.

Naming: the clean speech of each slot (sentences whose frames are
complete and nearly all that slot's alone) is kept in memory, never
written. Once a slot holds a few seconds of it, its new speech is
embedded, in a worker thread, with the same encoder as the enrolment
(:class:`~transcriber.application.ports.VoiceEncoder`) and compared
with the candidates' voiceprints
(:func:`~transcriber.domain.live_diarization.name_slots`); a slot that
matches wears the person's id from then on, and its earlier finals are
revised.

The diarizer may fail (not installed, crashed): attribution then stops,
:attr:`MonoAttributor.problem` says why (the engine warns once) and
every later sentence keeps the last speaker known.

Two parties (#249): a :class:`PartyPlan` makes every sentence the
employee's or the other person's. With a voice check the first
sentences of the session are the employee's by definition; their audio
(kept in memory only, never written, never an enrolment) becomes the
session's reference of the employee's voice, and the slot that spoke
them is the employee's. Every slot's voice is then compared with that
reference (:class:`~transcriber.domain.voice_check.PartySides`); the
employee's corrections («Esta soy yo» / «Esta no soy yo») pin a slot
and the sentence to a side. The scores and decisions are kept per
session as numbers and ids (:meth:`MonoAttributor.diagnosis`).
"""

import logging
import queue
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np

from transcriber.application.ports import (
    Audio,
    StreamDiarizer,
    Timer,
    VoiceEncoder,
)
from transcriber.domain.errors import (
    DiarizerError,
    InvalidInputError,
    TranscriberError,
)
from transcriber.domain.live import SAMPLE_RATE, LiveSpeaker, LiveUtterance
from transcriber.domain.live_diarization import (
    MonoDiarizationSettings,
    MonoSpeakers,
    NamingPlan,
    SlotPick,
    SpeakerFrames,
    name_slots,
    pick_slot,
    slot_labels,
    unit_sum,
)
from transcriber.domain.models import Voiceprint
from transcriber.domain.speakers import analysis_windows
from transcriber.domain.voice_check import (
    PartySides,
    Side,
    VoiceCheck,
    VoiceCheckSettings,
)
from transcriber.domain.voices import FloatArray

LOG = logging.getLogger(__name__)

FEED_SECONDS = 0.16
"""Audio gathered before it is sent to the diarizer (five frames)."""

KEEP_SECONDS = 30.0
"""Recent audio kept in memory to cut the clean speech of a slot."""

FINISH_TIMEOUT = 5.0
"""Longest wait for the last frames once the stream has ended."""

NAMING_TIMEOUT = 30.0
"""Longest wait for the voice encoder at the end of the session."""


@dataclass(frozen=True)
class LiveNaming:
    """What names the slots: a voice encoder and the candidates.

    ``voiceprints`` are the enrolled people who may speak (of the
    encoder's model); ``names`` gives each one's display name.
    ``release`` frees the encoder's model when the session ends.
    """

    encoder: VoiceEncoder
    voiceprints: Sequence[Voiceprint]
    names: Mapping[str, str] = field(default_factory=dict)
    release: Optional[Callable[[], None]] = None


@dataclass(frozen=True)
class PartyPlan:
    """The two people of a two-party mono session (#249).

    ``employee`` and ``other`` are the session's two speakers (id,
    role, name); ``voice_check`` starts the session by listening to
    the employee.
    """

    employee: LiveSpeaker
    other: LiveSpeaker
    voice_check: bool = True
    settings: VoiceCheckSettings = field(
        default_factory=VoiceCheckSettings
    )


@dataclass
class AttributionStats:
    """What attribution did in a session (for the summary)."""

    attribution_ms: List[float] = field(default_factory=list)
    provisional: int = 0
    updates: int = 0
    naming_ms: List[float] = field(default_factory=list)
    nearest: Dict[str, Tuple[Optional[str], Optional[float]]] = field(
        default_factory=dict
    )


class MonoAttributor:
    """Speakers of a mono stream's sentences (see the module)."""

    def __init__(
        self,
        diarizer: StreamDiarizer,
        timer: Timer,
        settings: MonoDiarizationSettings = MonoDiarizationSettings(),
        naming: Optional[LiveNaming] = None,
        given: Sequence[LiveSpeaker] = (),
        background: bool = True,
        parties: Optional[PartyPlan] = None,
    ) -> None:
        """Keep the ports; ``given`` are the ids and roles of the slots.

        Slot ``n`` (the ``n``-th voice heard) takes ``given[n]``; the
        other slots get ``S<n>`` labels. ``background`` runs the voice
        encoder in a worker thread (else in the caller's). ``parties``
        makes it a two-party session (see the module).
        """
        self._d = diarizer
        self._timer = timer
        self._s = settings
        self._naming = naming
        prints = list(naming.voiceprints) if naming is not None else []
        self._prints = [
            v for v in prints
            if naming is None or v.model == naming.encoder.model_name
        ]
        enrolled = {v.speaker_id for v in self._prints}
        self._parties = parties
        if parties is not None:
            # Anonymous labels never show: every slot is named a side.
            given = ()
            enrolled |= {parties.employee.id, parties.other.id}
            settings = settings.model_copy(update={
                "naming_seconds": parties.settings.slot_seconds or
                settings.naming_seconds,
            })
            self._s = settings
        labels = slot_labels(
            [s.id for s in given], diarizer.slots, enrolled
        )
        self._given = {s.id: s for s in given}
        self.speakers = MonoSpeakers(labels, settings)
        self.frames = SpeakerFrames(diarizer.frame_seconds, diarizer.slots)
        self._plan = NamingPlan(settings)
        self.sides: Optional[PartySides] = None
        self.check: Optional[VoiceCheck] = None
        self._check_ids: List[str] = []
        self._check_clips: List[FloatArray] = []
        self._check_job: Optional["Future[Optional[FloatArray]]"] = None
        self._inbox: "queue.SimpleQueue[Tuple[str, str, bool]]" = (
            queue.SimpleQueue()
        )
        self._notices: List[str] = []
        self._held: List[LiveUtterance] = []
        self._waiting_pins: Dict[str, Side] = {}
        self._log: List[Dict[str, Any]] = []
        self._voiceprint_names: Dict[int, str] = {}
        if parties is not None:
            self.sides = PartySides(parties.settings)
            if parties.voice_check:
                self.check = VoiceCheck(parties.settings)
                self._notices.append("listening")
            self.speakers.rename(self._party_names())
        self._clips: Dict[int, List[FloatArray]] = {}
        self._sums: Dict[int, FloatArray] = {}
        self._counts: Dict[int, int] = {}
        self._background = background
        self._executor: Optional[ThreadPoolExecutor] = None
        self._job: Optional[
            "Future[Dict[int, Tuple[FloatArray, int]]]"
        ] = None
        self._recent: FloatArray = np.zeros(0, dtype=np.float32)
        self._recent_offset = 0
        self._unsent: List[FloatArray] = []
        self._unsent_samples = 0
        self._fed = 0
        self.problem: Optional[str] = None
        self.stats = AttributionStats()

    @property
    def model_name(self) -> str:
        """Return the diarization model's identifier."""
        return self._d.model_name

    def roster(self) -> List[LiveSpeaker]:
        """Return the voices of ``session_started``: slots, then enrolled.

        A slot named in the start request keeps its role and name; an
        enrolled candidate is listed with ``enrolled`` set. A two-party
        session lists its two people only (the employee as enrolled
        when their voiceprint is a candidate).
        """
        parties = self._parties
        if parties is not None:
            mine = parties.employee
            return [
                LiveSpeaker(
                    id=mine.id, role=mine.role, name=mine.name,
                    enrolled=any(
                        v.speaker_id == mine.id for v in self._prints),
                ),
                LiveSpeaker(id=parties.other.id, role=parties.other.role,
                            name=parties.other.name),
            ]
        out: List[LiveSpeaker] = []
        for label in self.speakers.anonymous():
            known = self._given.get(label)
            out.append(LiveSpeaker(
                id=label, role=known.role if known else "unknown",
                name=known.name if known else None,
            ))
        names = self._naming.names if self._naming is not None else {}
        for voice in self._prints:
            out.append(LiveSpeaker(
                id=voice.speaker_id, name=names.get(voice.speaker_id),
                enrolled=True,
            ))
        return out

    def start(self) -> None:
        """Start the diarizer; a failure only stops attribution."""
        try:
            self._d.start()
        except TranscriberError as exc:
            self._fail(exc)

    def _fail(self, exc: Exception) -> None:
        """Stop attributing for the rest of the session."""
        if self.problem is None:
            LOG.warning("Live speaker attribution stopped: %s", exc)
            self.problem = str(exc) or type(exc).__name__

    # ------------------------------------------------------------ audio

    def feed(self, samples: FloatArray) -> None:
        """Take the next frame of audio; send it on in small pieces."""
        self._recent = np.concatenate([self._recent, samples])
        extra = len(self._recent) - int(KEEP_SECONDS * SAMPLE_RATE)
        if extra > 0:
            self._recent = self._recent[extra:]
            self._recent_offset += extra
        self._unsent.append(samples)
        self._unsent_samples += len(samples)
        if self._unsent_samples >= FEED_SECONDS * SAMPLE_RATE:
            self._send()

    def _send(self) -> None:
        """Send the gathered audio to the diarizer."""
        if not self._unsent:
            return
        piece = np.concatenate(self._unsent).astype(np.float32)
        self._unsent, self._unsent_samples = [], 0
        self._fed += len(piece)
        if self.problem is not None:
            return
        try:
            self._d.feed(piece)
        except TranscriberError as exc:
            self._fail(exc)

    def _cut(self, start: float, end: float) -> Optional[FloatArray]:
        """Return the recent audio of ``start``-``end``, if still kept."""
        a = int(round(start * SAMPLE_RATE)) - self._recent_offset
        b = int(round(end * SAMPLE_RATE)) - self._recent_offset
        if a < 0 or b > len(self._recent) or b <= a:
            return None
        return self._recent[a:b].copy()

    # ------------------------------------------------------------ frames

    def _pull(self, until: float = 0.0, timeout: float = 0.0) -> None:
        """Add the frames that arrived (waiting as asked)."""
        if self.problem is not None:
            return
        try:
            first, rows = self._d.frames(until, timeout)
            self.frames.extend(first, rows)
        except (DiarizerError, InvalidInputError) as exc:
            self._fail(exc)

    def _pick(self, start: float, end: float) -> SlotPick:
        """Return the slot of ``start``-``end`` from the frames there are."""
        return pick_slot(self.frames, start, end, self._s.threshold)

    def guess(self, start: float, end: float) -> str:
        """Return the likely speaker of ``start``-``end`` (for partials)."""
        if self.check is not None and self.check.listening \
                and self._parties is not None:
            return self._parties.employee.id
        pick = self._pick(start, end)
        if pick.slot is not None:
            return self.speakers.label(pick.slot)
        return self.speakers.last or self.speakers.label(0)

    def attribute(
        self, utterances: Sequence[LiveUtterance]
    ) -> List[LiveUtterance]:
        """Give each final sentence its speaker; remember them.

        Waits up to ``wait_ms`` for the frames of the last sentence.
        """
        if not utterances:
            return []
        began = self._timer.monotonic()
        self._send()
        self._pull(
            max(u.end for u in utterances), self._s.wait_ms / 1000
        )
        out: List[LiveUtterance] = []
        for utterance in utterances:
            pick = self._pick(utterance.start, utterance.end)
            fallback = self.speakers.last or utterance.speaker
            speaker, provisional = self.speakers.speaker(pick, fallback)
            unsure = self.speakers.unsure(pick.slot)
            pinned: Optional[str] = None
            if self._checking(utterance):
                assert self._parties is not None
                pinned = speaker = self._parties.employee.id
                provisional, unsure = False, False
            if self.problem is not None:
                provisional = False
            new = utterance.model_copy(update={
                "speaker": speaker, "speaker_provisional": provisional,
                "speaker_unsure": unsure,
            })
            self.speakers.sent(new, pick, pinned)
            if pick.complete:
                self._keep_clean(new, pick)
            if provisional:
                self.stats.provisional += 1
            self._note("final", id=new.id, start=round(new.start, 2),
                       end=round(new.end, 2), slot=pick.slot,
                       coverage=round(pick.coverage, 2),
                       purity=round(pick.purity, 2), speaker=speaker,
                       provisional=provisional, unsure=unsure,
                       check=pinned is not None)
            out.append(new)
        self._settle_check(max(u.end for u in utterances))
        self.stats.attribution_ms.append(
            (self._timer.monotonic() - began) * 1000
        )
        return out

    # ------------------------------------------------------------ parties

    def _checking(self, utterance: LiveUtterance) -> bool:
        """Take a sentence for the voice check while it listens."""
        check = self.check
        if check is None or not check.listening:
            return False
        if not utterance.text.strip():
            return True
        clip = self._cut(utterance.start, utterance.end)
        if clip is not None and len(clip):
            self._check_clips.append(clip)
            self._check_ids.append(utterance.id)
            check.hear(utterance.start, len(clip) / SAMPLE_RATE)
            self._notices.append("listening")
        return True

    def _settle_check(self, now: float) -> None:
        """End the check when it heard enough (or waited long enough)."""
        check = self.check
        if check is None:
            return
        state = check.settle(now)
        if state is None:
            return
        self._note("check", state=state, heard=round(check.heard, 2),
                   needed=check.needed, sentences=len(self._check_ids))
        if state == "short":
            self._check_clips = []
            self._notices.append("short")
            return
        # Recognised: the reference is computed off the stream; the
        # page says «Voz reconocida» once it is there.
        self._start_reference()

    def _start_reference(self) -> None:
        """Embed the check's audio (worker thread) into the reference."""
        naming = self._naming
        clips, self._check_clips = self._check_clips, []
        if naming is None or not clips:
            self._check_failed("no voice encoder" if naming is None
                               else "no audio")
            return
        encoder = naming.encoder
        if self._background:
            self._check_job = self._executor_ready().submit(
                self._reference, encoder, clips, self._timer, self.stats,
            )
            return
        try:
            reference = self._reference(
                encoder, clips, self._timer, self.stats)
        except TranscriberError as exc:
            reference = None
            LOG.warning("The voice check failed: %s", exc)
        # Sent with the next poll (this may run inside attribute()).
        self._held.extend(self._recognised(reference))

    def _executor_ready(self) -> ThreadPoolExecutor:
        """Return the naming thread (made on first use)."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                1, thread_name_prefix="live-naming"
            )
        return self._executor

    @staticmethod
    def _reference(
        encoder: VoiceEncoder, clips: Sequence[FloatArray], timer: Timer,
        stats: AttributionStats,
    ) -> Optional[FloatArray]:
        """Return the sum of the check's unit window vectors, if any."""
        found = MonoAttributor._embed(encoder, {0: clips}, timer, stats)
        return found[0][0] if 0 in found else None

    def _check_failed(self, why: str) -> None:
        """The check's voice could not be computed: say so, go on."""
        if self.check is not None:
            self.check.fail()
        self._note("check", state="failed", why=why)
        self._notices.append("failed")

    def _recognised(
        self, reference: Optional[FloatArray]
    ) -> List[LiveUtterance]:
        """Take the check's voice as the employee's; return the revisions."""
        sides = self.sides
        if sides is None or reference is None:
            self._check_failed("no voice in the check")
            return []
        sides.recognise(reference, self._check_slot())
        self._note("reference", check_slot=sides.check_slot,
                   seconds=round(self.check.heard, 2)
                   if self.check is not None else None)
        self._notices.append("recognised")
        self._compare_slots()
        return self._apply_sides()

    def _check_slot(self) -> Optional[int]:
        """Return the slot that spoke most of the check (if known)."""
        seconds: Dict[int, float] = {}
        for utterance in self.speakers.sent_utterances():
            if utterance.id not in self._check_ids:
                continue
            slot = self.speakers.slot_of(utterance.id)
            if slot is not None:
                seconds[slot] = seconds.get(slot, 0.0) + (
                    utterance.end - utterance.start)
        if not seconds:
            return None
        return max(sorted(seconds), key=lambda k: seconds[k])

    def _party_names(self) -> Dict[int, str]:
        """Return the id of every slot: the employee's or the other's."""
        parties, sides = self._parties, self.sides
        assert parties is not None and sides is not None
        out: Dict[int, str] = {}
        for slot in range(self.speakers.slots):
            # An enrolment that names a slot still helps (#249), unless
            # the employee said that voice is not theirs.
            known = sides.slots.get(slot)
            named = self._voiceprint_names.get(slot) == parties.employee.id \
                and (known is None or known.pinned != "other")
            mine = sides.side(slot) == "employee" or named
            out[slot] = parties.employee.id if mine else parties.other.id
        return out

    def _apply_sides(self) -> List[LiveUtterance]:
        """Rename every slot to its side; return the finals changed."""
        sides = self.sides
        if sides is None:
            return []
        if sides.reference is not None and sides.check_slot is None:
            # The frames of the check's last sentences come a second late.
            found = self._check_slot()
            if found is not None:
                sides.set_check_slot(found)
        unsure = {
            slot for slot in range(self.speakers.slots)
            if sides.unsure(slot)
        }
        return self.speakers.rename(self._party_names(), unsure)

    def _compare_slots(self) -> None:
        """Compare each slot's voice so far with the check's."""
        sides = self.sides
        if sides is None or sides.reference is None:
            return
        voices: Dict[int, Tuple[FloatArray, float]] = {}
        for slot, total in self._sums.items():
            norm = float(np.linalg.norm(total))
            if self._counts.get(slot) and norm > 1e-6:
                voices[slot] = (
                    (total / norm).astype(np.float32),
                    self._plan.seconds.get(slot, 0.0),
                )
        sides.voices(voices)
        self._note("slots", slots=sides.diagnosis())

    def _requests(self) -> List[LiveUtterance]:
        """Apply the page's requests (skip, corrections); return revisions."""
        out: Dict[str, LiveUtterance] = {}
        while True:
            try:
                action, ident, mine = self._inbox.get_nowait()
            except queue.Empty:
                break
            if action == "skip":
                if self.check is not None and self.check.skip():
                    self._check_clips = []
                    self._note("check", state="skipped")
                    self._notices.append("skipped")
                continue
            out.update((u.id, u) for u in self._correct(ident, mine))
        return list(out.values())

    def _correct(self, ident: str, mine: bool) -> List[LiveUtterance]:
        """Give the voice of final ``ident`` to a side, from now on."""
        parties, sides = self._parties, self.sides
        if parties is None or sides is None:
            return []
        side: Side = "employee" if mine else "other"
        # A slot guessed from part of the line's frames may be wrong: the
        # voice is moved once they are complete.
        slot = self.speakers.settled_slot(ident)
        self._note("correction", id=ident, side=side, slot=slot)
        changed: Dict[str, LiveUtterance] = {}
        if slot is not None:
            sides.pin(slot, side)
            changed.update((u.id, u) for u in self._apply_sides())
        else:
            # Its frames have not come yet: the voice follows once they do.
            self._waiting_pins[ident] = side
        speaker = parties.employee.id if mine else parties.other.id
        changed.update((u.id, u) for u in self.speakers.pin(ident, speaker))
        return list(changed.values())

    def _late_pins(self) -> List[LiveUtterance]:
        """Pin the slots of corrections whose frames came after them."""
        sides = self.sides
        if sides is None or not self._waiting_pins:
            return []
        done = False
        for ident, side in list(self._waiting_pins.items()):
            slot = self.speakers.settled_slot(ident)
            if slot is None:
                continue
            del self._waiting_pins[ident]
            sides.pin(slot, side)
            self._note("correction_slot", id=ident, side=side, slot=slot)
            done = True
        return self._apply_sides() if done else []

    def skip(self) -> None:
        """Skip the employee's voice check (thread-safe, see the engine)."""
        self._inbox.put(("skip", "", False))

    def correct(self, utterance: str, mine: bool) -> None:
        """Say final ``utterance``'s voice is (not) the employee's.

        Thread-safe: applied with the next frame of audio.
        """
        self._inbox.put(("correct", utterance, mine))

    @property
    def voice_check(self) -> bool:
        """Tell whether this is a two-party session (it takes requests)."""
        return self._parties is not None

    def notices(self) -> List[Tuple[str, float, float]]:
        """Return the voice check's news since the last call.

        Each is ``(state, seconds heard, seconds needed)``; the engine
        sends them as ``voice_check`` events.
        """
        check = self.check
        out = [
            (state, check.heard if check else 0.0,
             check.needed if check else 0.0)
            for state in self._notices
        ]
        self._notices = []
        return out

    @property
    def employee_id(self) -> Optional[str]:
        """Return the employee's id in a two-party session."""
        return self._parties.employee.id if self._parties else None

    def _note(self, kind: str, **fields: Any) -> None:
        """Keep one diagnosis record (numbers and ids only)."""
        if self._parties is None:
            return
        record: Dict[str, Any] = {
            "event": kind,
            "stream_time": round(self._fed / SAMPLE_RATE, 2),
        }
        record.update(fields)
        self._log.append(record)

    def diagnosis(self) -> List[Dict[str, Any]]:
        """Return the attribution's records of the session (#249).

        Numbers and ids only (no text, no audio): every final's slot,
        coverage, purity and side; the check; every comparison of the
        slots' voices with it (cosines); the corrections; and the
        final side of every sentence.
        """
        if self._parties is None:
            return []
        final = [
            {"event": "sentence", "id": u.id, "speaker": u.speaker,
             "slot": self.speakers.slot_of(u.id),
             "unsure": u.speaker_unsure}
            for u in self.speakers.sent_utterances()
        ]
        return self._log + final

    def poll(self) -> List[LiveUtterance]:
        """Return the finals revised by new frames or by a new name.

        A two-party session also applies the page's requests and ends
        the voice check when it waited long enough; those work even
        once the diarizer has failed.
        """
        changed: Dict[str, LiveUtterance] = {}
        if self._parties is not None:
            changed.update((u.id, u) for u in self._requests())
            self._settle_check(self._fed / SAMPLE_RATE)
            changed.update((u.id, u) for u in self._reference_step())
            held, self._held = self._held, []
            changed.update((u.id, u) for u in held)
        if self.problem is not None:
            return self._count(changed)
        before = len(self.frames)
        self._pull()
        if len(self.frames) != before:
            changed.update((u.id, u) for u in self._revisions())
            changed.update((u.id, u) for u in self._late_pins())
            if self.sides is not None and self.sides.reference is not None:
                changed.update((u.id, u) for u in self._apply_sides())
        changed.update((u.id, u) for u in self._naming_step())
        return self._count(changed)

    def _reference_step(self, wait: bool = False) -> List[LiveUtterance]:
        """Take the check's voice once its embedding is done."""
        job = self._check_job
        if job is None or not (wait or job.done()):
            return []
        self._check_job = None
        try:
            reference = job.result(timeout=NAMING_TIMEOUT if wait else 0)
        except (TranscriberError, FutureTimeout) as exc:
            LOG.warning("The voice check failed: %s", exc)
            reference = None
        return self._recognised(reference)

    def _count(
        self, changed: Mapping[str, LiveUtterance]
    ) -> List[LiveUtterance]:
        """Count the revisions; return them in time order."""
        self.stats.updates += len(changed)
        return sorted(changed.values(), key=lambda u: (u.start, u.id))

    def finish(self) -> List[LiveUtterance]:
        """Read the last frames of the ended stream; return the revisions.

        Stops the diarizer.
        """
        try:
            self._send()
            if self.problem is None:
                try:
                    self._d.finish()
                except TranscriberError as exc:
                    self._fail(exc)
            self._pull(self._fed / SAMPLE_RATE, FINISH_TIMEOUT)
            changed: Dict[str, LiveUtterance] = {}
            if self._parties is not None:
                # The page's last requests and the check's voice count
                # even when the diarizer failed.
                changed.update((u.id, u) for u in self._requests())
                changed.update(
                    (u.id, u) for u in self._reference_step(True))
                held, self._held = self._held, []
                changed.update((u.id, u) for u in held)
            if self.problem is not None:
                return self._count(changed)
            changed.update((u.id, u) for u in self._revisions())
            changed.update((u.id, u) for u in self._late_pins())
            changed.update((u.id, u) for u in self._naming_step(True))
            changed.update((u.id, u) for u in self._naming_step(True))
            if self.sides is not None:
                changed.update((u.id, u) for u in self._apply_sides())
            return self._count(changed)
        finally:
            self.close()

    def close(self) -> None:
        """Stop the diarizer and the naming; free the voice encoder."""
        self._d.close()
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        if self._naming is not None and self._naming.release is not None:
            self._naming.release()

    def _revisions(self) -> List[LiveUtterance]:
        """Revise pending finals, keep clean speech, maybe name slots."""
        waiting = {u.id for u in self.speakers.pending()}
        changed: Dict[str, LiveUtterance] = {
            u.id: u for u in self.speakers.revise(self.frames)
        }
        still = {u.id for u in self.speakers.pending()}
        for utterance in self._finished(waiting - still):
            pick = self._pick(utterance.start, utterance.end)
            self._keep_clean(utterance, pick)
        return list(changed.values())

    def _finished(self, ids: set[str]) -> List[LiveUtterance]:
        """Return the sent finals of ``ids`` (their frames just completed)."""
        return [
            u for u in self.speakers.sent_utterances() if u.id in ids
        ]

    # ------------------------------------------------------------ naming

    def _keep_clean(self, utterance: LiveUtterance, pick: SlotPick) -> None:
        """Keep a sentence's audio as its slot's voice, if clean enough."""
        slot = pick.slot
        wanted = bool(self._prints) or self._parties is not None
        if self._naming is None or not wanted or slot is None \
                or pick.purity < self._s.purity \
                or not self._plan.wants(slot):
            return
        clip = self._cut(utterance.start, utterance.end)
        if clip is None or len(clip) == 0:
            return
        self._clips.setdefault(slot, []).append(clip)
        self._plan.add(slot, len(clip) / SAMPLE_RATE)

    def _naming_step(self, wait: bool = False) -> List[LiveUtterance]:
        """Apply a finished embedding; start the next one when due.

        The voice encoder runs in a worker thread (on the CPU it takes
        up to a second), so the stream never waits for it; ``wait``
        waits for it (the end of the session).
        """
        out: List[LiveUtterance] = []
        job = self._job
        if job is not None and (wait or job.done()):
            self._job = None
            try:
                found = job.result(timeout=NAMING_TIMEOUT if wait else 0)
            except TranscriberError as exc:
                LOG.warning("Naming the live voices failed: %s", exc)
                found = {}
            except FutureTimeout:
                LOG.warning("Naming the live voices took too long")
                found = {}
            out = self._apply(found)
        if self._job is None and self._naming is not None \
                and self._plan.due():
            clips, self._clips = self._clips, {}
            encoder = self._naming.encoder
            if self._background:
                if self._executor is None:
                    self._executor = ThreadPoolExecutor(
                        1, thread_name_prefix="live-naming"
                    )
                self._job = self._executor.submit(
                    self._embed, encoder, clips, self._timer, self.stats,
                )
            else:
                try:
                    found = self._embed(encoder, clips, self._timer,
                                        self.stats)
                except TranscriberError as exc:
                    LOG.warning("Naming the live voices failed: %s", exc)
                    found = {}
                changed = {u.id: u for u in out}
                changed.update((u.id, u) for u in self._apply(found))
                out = list(changed.values())
        return out

    def _apply(
        self, found: Mapping[int, Tuple[FloatArray, int]]
    ) -> List[LiveUtterance]:
        """Add new window vectors to the slots' voices; rename slots."""
        naming = self._naming
        if naming is None or not found:
            return []
        for slot, (total, count) in found.items():
            known = self._sums.get(slot)
            self._sums[slot] = total if known is None else known + total
            self._counts[slot] = self._counts.get(slot, 0) + count
        voices: Dict[int, FloatArray] = {}
        for slot, total in self._sums.items():
            norm = float(np.linalg.norm(total))
            if self._counts.get(slot) and norm > 1e-6:
                voices[slot] = (total / norm).astype(np.float32)
        try:
            names, nearest = name_slots(
                voices, self._prints, self._s.match_threshold,
                naming.encoder.model_name,
            )
        except TranscriberError as exc:
            LOG.warning("Naming the live voices failed: %s", exc)
            return []
        for slot, near in nearest.items():
            self.stats.nearest[self.speakers.anonymous()[slot]] = near
        if self._parties is not None:
            if self._prints:
                self._note("voiceprint", nearest={
                    str(k): None if v[1] is None else round(v[1], 4)
                    for k, v in nearest.items()}, named=sorted(names))
            self._voiceprint_names = dict(names)
            self._compare_slots()
            return self._apply_sides()
        if names == self.speakers.names:
            return []
        return self.speakers.rename(names)

    @staticmethod
    def _embed(
        encoder: VoiceEncoder, clips: Mapping[int, Sequence[FloatArray]],
        timer: Timer, stats: AttributionStats,
    ) -> Dict[int, Tuple[FloatArray, int]]:
        """Return the sum and count of the unit window vectors per slot."""
        began = timer.monotonic()
        out: Dict[int, Tuple[FloatArray, int]] = {}
        try:
            for slot, pieces in sorted(clips.items()):
                regions: List[Tuple[float, float]] = []
                cursor = 0.0
                for clip in pieces:
                    seconds = len(clip) / SAMPLE_RATE
                    regions.append((cursor, cursor + seconds))
                    cursor += seconds
                windows = analysis_windows(regions)
                if not windows:
                    continue
                audio = Audio(
                    samples=np.concatenate(list(pieces)).astype(np.float32),
                    sample_rate=SAMPLE_RATE,
                )
                summed = unit_sum(encoder.embed(audio, windows))
                if summed is not None:
                    out[slot] = summed
        finally:
            stats.naming_ms.append((timer.monotonic() - began) * 1000)
        return out
