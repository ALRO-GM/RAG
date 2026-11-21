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
"""

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

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
    ) -> None:
        """Keep the ports; ``given`` are the ids and roles of the slots.

        Slot ``n`` (the ``n``-th voice heard) takes ``given[n]``; the
        other slots get ``S<n>`` labels. ``background`` runs the voice
        encoder in a worker thread (else in the caller's).
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
        labels = slot_labels(
            [s.id for s in given], diarizer.slots, enrolled
        )
        self._given = {s.id: s for s in given}
        self.speakers = MonoSpeakers(labels, settings)
        self.frames = SpeakerFrames(diarizer.frame_seconds, diarizer.slots)
        self._plan = NamingPlan(settings)
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
        enrolled candidate is listed with ``enrolled`` set.
        """
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
            if self.problem is not None:
                provisional = False
            new = utterance.model_copy(update={
                "speaker": speaker, "speaker_provisional": provisional,
            })
            self.speakers.sent(new, pick)
            if pick.complete:
                self._keep_clean(new, pick)
            if provisional:
                self.stats.provisional += 1
            out.append(new)
        self.stats.attribution_ms.append(
            (self._timer.monotonic() - began) * 1000
        )
        return out

    def poll(self) -> List[LiveUtterance]:
        """Return the finals revised by new frames or by a new name."""
        if self.problem is not None:
            return []
        before = len(self.frames)
        self._pull()
        changed: Dict[str, LiveUtterance] = {}
        if len(self.frames) != before:
            changed.update((u.id, u) for u in self._revisions())
        changed.update((u.id, u) for u in self._naming_step())
        return self._count(changed)

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
            if self.problem is not None:
                return []
            changed = {u.id: u for u in self._revisions()}
            changed.update((u.id, u) for u in self._naming_step(True))
            changed.update((u.id, u) for u in self._naming_step(True))
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
        if self._naming is None or not self._prints or slot is None \
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
