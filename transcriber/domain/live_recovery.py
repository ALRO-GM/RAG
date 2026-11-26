"""Recovering the audio of a live session that never reached the server (#265).

When the connection drops for longer than the page's memory can hold,
when the tab is closed or crashes, or when «Terminar» is pressed with no
connection, the browser keeps the unacknowledged audio (IndexedDB) and
sends it later. The server transcribes it with the batch pipeline and
merges it into the live transcript in place of the gap. Everything here
is pure: spans of samples, the speaker mapping, the merge and the job's
state machine. No I/O, no audio.

**Spans.** Offsets are samples per channel of the session's stream
(16 kHz), the same numbers as the live socket's ``received_samples``.
The server knows which parts of its stream were silence it put in for
audio the source lost (``gap``, ``align``) and where its stream ended
(``received``). Only those parts are taken from the recovered audio
(:func:`recovered_spans`): what the server heard itself is never
replaced, even when the page sends it again.

**Merge rule** (:func:`merge_recovered`), deterministic:

1. a live sentence whose midpoint falls inside a recovered span is
   dropped (it was heard over the silence that stood in for the gap);
2. a recovered sentence (its times moved by the offset its audio
   starts at) is kept only when its midpoint falls inside a recovered
   span, so overlap with what the server heard counts once;
3. both lists are put in time order (start, end; live before recovered
   on a tie) and regrouped into turns, ids renumbered without gaps.

**Speakers** (:func:`speaker_map`): a stereo part takes the speaker of
its channel; a mono session that was not diarized has one voice; in a
diarized session a voice the batch pipeline recognised as an enrolled
person keeps that person's id; with two parties, once the employee is
recognised every other voice is the other party; voices that cannot be
told are listed as ``R1``, ``R2``… (role unknown), never guessed.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import (
    Any, Dict, List, Mapping, Optional, Sequence, Tuple, cast, get_args,
)

from pydantic import BaseModel, ConfigDict, Field

from transcriber.domain.live import LiveSpeaker
from transcriber.domain.models import (
    Speaker, SpeakerRole, Transcript, Utterance,
)
from transcriber.domain.sentences import regroup_utterances

SAMPLE_RATE = 16000
"""Samples per second of the live stream."""
END = 2 ** 62
"""An offset past any stream (the open end of the tail span)."""

STATES = (
    "uploading", "queued", "transcribing", "merging", "publishing",
    "analysing", "rejudging", "done", "failed",
)
"""The recovery job's states, in order (``done`` and ``failed`` end it)."""
FINAL_STATES = ("done", "failed")
RETRY_STATES = ("transcribing", "merging", "publishing", "analysing",
                "rejudging")
"""States whose step may fail and be tried again."""
MAX_TRIES = 3
"""Tries of one step before the job fails."""
BACKOFF_SECONDS = (30.0, 120.0, 480.0)
"""Wait before the second, third… try of a step."""
ROLES = get_args(SpeakerRole)
"""Roles a transcript speaker may carry."""
RECOVERED_PREFIX = "R"
"""Ids of voices of the recovered audio that could not be told apart."""

Span = Tuple[int, int]
"""``[start, end)`` in samples."""
JOIN_SAMPLES = 30 * SAMPLE_RATE
"""Recovered spans closer than this are transcribed together."""
MAX_PART_SAMPLES = 2 * 3600 * SAMPLE_RATE
"""The longest part transcribed at once (two hours)."""


def merge_spans(spans: Sequence[Span]) -> List[Span]:
    """Return ``spans`` sorted, empty ones dropped, touching ones joined."""
    merged: List[Span] = []
    for start, end in sorted(s for s in spans if s[1] > s[0]):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def intersect(first: Sequence[Span], second: Sequence[Span]) -> List[Span]:
    """Return the parts that are in both lists of spans."""
    a, b = merge_spans(first), merge_spans(second)
    out: List[Span] = []
    i = j = 0
    while i < len(a) and j < len(b):
        start = max(a[i][0], b[j][0])
        end = min(a[i][1], b[j][1])
        if end > start:
            out.append((start, end))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def recovered_spans(
    uploaded: Sequence[Span], silences: Sequence[Span], received: int,
) -> List[Span]:
    """The parts of the uploaded audio the live transcript lacks.

    ``silences``: the silence the server put in for lost audio;
    ``received``: where the server's stream ended (everything after it
    is missing too).
    """
    needed = merge_spans(list(silences) + [(max(received, 0), END)])
    return intersect(uploaded, needed)


def group_spans(
    spans: Sequence[Span], join: int = JOIN_SAMPLES,
    longest: int = MAX_PART_SAMPLES,
) -> List[Span]:
    """Group recovered spans into the parts transcribed one by one.

    Spans less than ``join`` apart share one part (the silence between
    them is short); a part never grows past ``longest`` samples, so the
    audio built for it is bounded by what was uploaded and never spans
    a long gap.
    """
    groups: List[Span] = []
    for start, end in merge_spans(spans):
        if groups and start - groups[-1][1] < join \
                and end - groups[-1][0] <= longest:
            groups[-1] = (groups[-1][0], end)
        else:
            groups.append((start, end))
    return groups


def seconds_of(spans: Sequence[Span]) -> float:
    """The length of ``spans`` in seconds (the tail counted as given)."""
    return sum(end - start for start, end in merge_spans(spans)) \
        / SAMPLE_RATE


def _inside(spans: Sequence[Span], seconds: float) -> bool:
    """Tell whether the time ``seconds`` falls inside one of ``spans``."""
    sample = seconds * SAMPLE_RATE
    return any(start <= sample < end for start, end in spans)


def _middle(utterance: Utterance) -> float:
    """The midpoint of a sentence, in seconds."""
    return (utterance.start + utterance.end) / 2


def speaker_map(
    utterances: Sequence[Utterance],
    heard: Sequence[Speaker],
    live: Sequence[LiveSpeaker],
    channels: int,
    diarized: bool,
    channel: Optional[int] = None,
    parties: Optional[Tuple[str, str]] = None,
) -> Dict[str, str]:
    """Map the batch transcript's speakers to the live session's.

    ``utterances`` give the order voices are first heard in;
    ``heard`` the batch speakers; ``live`` the session's voices;
    ``channel`` the channel a stereo part came from; ``parties`` the
    (employee, other) pair of a two-party session. See the module.
    """
    ids = [s.id for s in heard]
    for utterance in utterances:
        if utterance.speaker not in ids:
            ids.append(utterance.speaker)
    order = sorted(ids, key=lambda i: next(
        (n for n, u in enumerate(utterances) if u.speaker == i),
        len(utterances)))
    if channels == 2 and channel is not None:
        target = next((s.id for s in live if s.channel == channel),
                      f"S{channel + 1}")
        return {i: target for i in order}
    if not diarized:
        target = live[0].id if live else "S1"
        return {i: target for i in order}
    known = {s.id for s in live}
    enrolled = {s.id for s in heard if s.enrolled}
    mapped = {i: i for i in order if i in enrolled and i in known}
    if parties is not None and parties[0] in mapped.values():
        return {i: mapped.get(i, parties[1]) for i in order}
    anonymous = 0
    result: Dict[str, str] = {}
    for i in order:
        if i in mapped:
            result[i] = mapped[i]
        else:
            anonymous += 1
            result[i] = f"{RECOVERED_PREFIX}{anonymous}"
    return result


@dataclass(frozen=True)
class RecoveredPart:
    """One batch transcript of recovered audio.

    ``offset``: the stream sample its audio starts at; ``names``: its
    speakers mapped to the session's (:func:`speaker_map`).
    """

    transcript: Transcript
    offset: int
    names: Mapping[str, str]


@dataclass
class Merged:
    """The merged transcript and what the merge did (counts only)."""

    transcript: Transcript
    added: int = 0
    dropped: int = 0
    seconds: float = 0.0
    """Seconds of speech the added sentences cover (what was recovered
    and merged, not the length of the gaps)."""
    speakers: List[str] = field(default_factory=list)
    unidentified: bool = False
    """Some added sentences are of voices that could not be told
    (``R1``…): they are in the transcript but feed no indicator."""


def merge_recovered(
    live: Transcript,
    parts: Sequence[RecoveredPart],
    spans: Sequence[Span],
    roles: Mapping[str, str],
) -> Merged:
    """Put the recovered sentences into ``live`` in place of its gaps.

    ``spans``: the recovered spans (:func:`recovered_spans`);
    ``roles``: the session's speaker id → role, for voices the live
    transcript did not list. The result is a valid ``transcript.v1``.
    """
    spans = merge_spans(spans)
    kept = [u for u in live.utterances if not _inside(spans, _middle(u))]
    dropped = len(live.utterances) - len(kept)
    added: List[Utterance] = []
    for part in parts:
        shift = part.offset / SAMPLE_RATE
        for u in part.transcript.utterances:
            start, end = u.start + shift, max(u.end, u.start) + shift
            if not _inside(spans, (start + end) / 2):
                continue
            added.append(u.model_copy(update={
                "start": round(start, 3), "end": round(end, 3),
                "speaker": part.names.get(u.speaker, u.speaker),
                "alternative_speaker": None, "words": [],
                "previous_speaker": None, "previous_utterance": None,
            }))
    tagged = [(u.start, u.end, 0, n, u) for n, u in enumerate(kept)] + \
        [(u.start, u.end, 1, n, u) for n, u in enumerate(added)]
    ordered = [t[4] for t in sorted(tagged, key=lambda t: t[:4])]
    utterances = regroup_utterances(ordered)
    listed = {s.id: s for s in live.speakers}
    heard = []
    for u in utterances:
        if u.speaker not in heard:
            heard.append(u.speaker)
    speakers = [listed[i] for i in heard if i in listed]
    for i in heard:
        if i not in listed:
            role = roles.get(i)
            speakers.append(Speaker(
                id=i, enrolled=False,
                role=cast(SpeakerRole, role) if role in ROLES else None,
            ))
    duration = max([live.session.duration]
                   + [u.end for u in utterances])
    speech = seconds_of([(round(u.start * SAMPLE_RATE),
                          round(u.end * SAMPLE_RATE)) for u in added])
    transcript = Transcript(
        session=live.session.model_copy(update={"duration": duration}),
        speakers=speakers, utterances=utterances, pipeline=live.pipeline,
    )
    return Merged(
        transcript, added=len(added), dropped=dropped, seconds=speech,
        speakers=sorted({u.speaker for u in added}),
        unidentified=any(u.speaker not in listed and u.speaker not in roles
                         for u in added),
    )


@dataclass
class RecoveryJob:
    """The persisted state of one session's recovery (no audio, no words).

    ``chunks``: the uploaded chunks by number, each ``[offset,
    samples]``; ``spans``: the recovered spans once known; ``tries``:
    tries of the current step; ``next_try``: when the step may run
    again (ISO time, ``None``: now); ``progress``: 0..1 while
    transcribing; ``seconds``: seconds of audio recovered.
    """

    session_id: str
    state: str
    created_at: str
    expires_at: str
    channels: int = 1
    chunks: Dict[str, List[int]] = field(default_factory=dict)
    bytes: int = 0
    tries: int = 0
    next_try: Optional[str] = None
    progress: float = 0.0
    seconds: float = 0.0
    added: int = 0
    failed_step: Optional[str] = None
    error: Optional[str] = None
    audio_deleted: bool = False
    unidentified: bool = False
    restarts: int = 0
    """Times a restart lost the batch job (not a failed try)."""
    meta: Dict[str, Any] = field(default_factory=dict)
    updated_at: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        """The job as JSON-ready data."""
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "RecoveryJob":
        """Read a job written by :meth:`to_dict` (unknown keys dropped)."""
        known = set(cls.__dataclass_fields__)
        values = {k: v for k, v in data.items() if k in known}
        if values.get("state") not in STATES:
            raise ValueError("not a recovery job state")
        return cls(**values)  # type: ignore[arg-type]

    @property
    def final(self) -> bool:
        """Tell whether the job has ended."""
        return self.state in FINAL_STATES

    def uploaded(self) -> List[Span]:
        """The spans the uploaded chunks cover."""
        return merge_spans([(int(o), int(o) + int(n))
                            for o, n in self.chunks.values()])

    def move(self, state: str, now: datetime) -> None:
        """Go to ``state`` (a new step: its tries start again)."""
        if state not in STATES:
            raise ValueError(f"unknown recovery state {state!r}")
        self.state = state
        self.tries = 0
        self.next_try = None
        self.updated_at = now.isoformat()

    def failed_try(self, error: str, now: datetime) -> bool:
        """Count a failed try of the current step; ``True`` if it ends.

        After :data:`MAX_TRIES` the job fails; before, the next try
        waits :data:`BACKOFF_SECONDS`.
        """
        self.tries += 1
        self.error = error[:200]
        self.updated_at = now.isoformat()
        if self.tries >= MAX_TRIES:
            self.failed_step = self.state
            self.state = "failed"
            self.next_try = None
            return True
        wait = BACKOFF_SECONDS[min(self.tries - 1, len(BACKOFF_SECONDS) - 1)]
        self.next_try = (now + timedelta(seconds=wait)).isoformat()
        return False

    def due(self, now: datetime) -> bool:
        """Tell whether the current step may run now."""
        if self.final or self.state == "uploading":
            return False
        return self.next_try is None \
            or datetime.fromisoformat(self.next_try) <= now

    def expired(self, now: datetime) -> bool:
        """Tell whether the job's audio is past its hard expiry."""
        return datetime.fromisoformat(self.expires_at) <= now

    def view(self) -> Dict[str, object]:
        """What the page shows (no audio, no words, no internals)."""
        return {
            "session_id": self.session_id, "state": self.state,
            "progress": round(self.progress, 3),
            "seconds": round(self.seconds, 1),
            "minutes": round(self.seconds / 60, 1),
            "chunks": sorted(int(n) for n in self.chunks),
            "added": self.added, "failed_step": self.failed_step,
            "unidentified": self.unidentified,
            "error": self.error, "expires_at": self.expires_at,
            "audio_deleted": self.audio_deleted,
            "updated_at": self.updated_at,
        }


class RecoverySettings(BaseModel):
    """How the service keeps and treats recovered live audio (#265).

    ``ttl_hours``: the server's hard limit for keeping uploaded audio
    (a job's own expiry, from the page, may be sooner); the audio is
    deleted as soon as its transcription succeeds, and the sweep runs at
    start and every ``sweep_seconds``. ``chunk_max_bytes``,
    ``session_max_bytes`` and ``total_max_bytes`` bound what may be
    uploaded (one chunk, one session, every session together);
    ``max_chunks`` the chunks of one session. ``tick_seconds``: how
    often the jobs are moved on.
    """

    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    ttl_hours: float = Field(default=24.0, gt=0, le=24 * 7)
    chunk_max_bytes: int = Field(default=1024 * 1024, ge=3200)
    session_max_bytes: int = Field(default=512 * 1024 * 1024, ge=3200)
    total_max_bytes: int = Field(default=2048 * 1024 * 1024, ge=3200)
    max_chunks: int = Field(default=8192, ge=1)
    tick_seconds: float = Field(default=5.0, gt=0)
    sweep_seconds: float = Field(default=3600.0, gt=0)
