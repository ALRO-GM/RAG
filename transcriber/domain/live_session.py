"""A live session: what starts it and how its audio arrives (stage 2).

A live session (``docs/platform/ARCHITECTURE.md`` §2, §4) is opened by
a start request (:class:`LiveSessionRequest`), then fed raw PCM audio
over a WebSocket and stopped. This module holds the rules of those
messages; no I/O:

* :class:`LiveSessionRequest`: the session id, one or two channels, who
  speaks on each channel and with which role (:class:`ChannelSpeaker`,
  from the product config of the orchestrator) and the options of the
  run.
* :func:`channel_speakers`: the attribution of the stereo mode (LV-5):
  channel ``n`` is one known person, no model involved.
* :func:`decode_pcm`: one binary message of 16-bit little-endian PCM at
  16 kHz (two channels interleaved) into one float array per channel.
* :class:`AudioBacklog`: how much audio waits for the recogniser; the
  socket stops reading when it is full (backpressure).
"""

from typing import List, Literal, Optional

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

from transcriber.domain.errors import InvalidInputError
from transcriber.domain.live import (
    MAX_HANG_MS,
    MIN_HANG_MS,
    SAMPLE_RATE,
    LiveSpeaker,
)
from transcriber.domain.live_diarization import (
    MAX_SLOTS,
    MonoDiarizationSettings,
)
from transcriber.domain.models import SESSION_ID_PATTERN
from transcriber.domain.voice_check import VoiceCheckSettings

FloatArray = npt.NDArray[np.float32]

PCM_BYTES = 2
"""Bytes per sample of the live audio (signed 16-bit, little endian)."""

MAX_MESSAGE_SECONDS = 1.0
"""Longest audio one WebSocket message may carry."""


Role = Literal[
    "interviewer", "candidate", "agent", "customer", "user", "bot",
    "unknown",
]
"""The speaker roles of ``live.event.v1`` (and of the product configs)."""

SPEAKER_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"


class ChannelSpeaker(BaseModel):
    """Who speaks on one channel: an id, a role and maybe a name.

    The orchestrator reads them from the product config
    (``source.channels`` of ``config/products/*.yaml``: ``{0: {speaker:
    S1, role: interviewer}, ...}``) and sends them with the start
    request; the transcriber never reads that config itself.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    channel: int = Field(ge=0, le=1)
    id: str = Field(pattern=SPEAKER_ID_PATTERN)
    role: Role = "unknown"
    name: Optional[str] = Field(default=None, min_length=1, max_length=80)


class Handoff(BaseModel):
    """What happens after the session (``handoff`` of the product config).

    ``transcript_v1`` publishes the session's transcript to the RAG,
    as a batch session's; ``run_post_call`` then starts the RAG's
    post-call analysis of it (P1).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    transcript_v1: bool = False
    run_post_call: bool = False


class Parties(BaseModel):
    """The two people of a one-microphone session (#249).

    Every sentence is ``employee``'s or ``other``'s (ids of
    ``speakers``): the diarizer's extra voices never become a third
    person. With ``voice_check`` the session first listens to the
    employee for a few seconds (the session's own voice reference,
    kept in memory only), then tells each sentence's side by its voice.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    employee: str = Field(pattern=SPEAKER_ID_PATTERN)
    other: str = Field(pattern=SPEAKER_ID_PATTERN)
    voice_check: bool = True


class LiveSessionRequest(BaseModel):
    """What a client asks for when it starts a live session.

    ``channels`` is 1 (one voice, or a mono mix) or 2 (each channel its
    own voice: microphone and call). ``speakers`` names who speaks on
    each channel (default: ``S1`` on channel 0, ``S2`` on channel 1,
    role ``unknown``); when given it lists every channel once.
    ``product`` is the product config the session runs under (only
    recorded); ``handoff`` says what happens with the transcript at the
    end. ``language`` ``None`` lets the recogniser detect it.
    ``hang_ms`` and ``partial_every_ms`` tune the endpointing and the
    partials (§6).

    ``diarize`` (one channel only, LV-6) tells the voices of the one
    channel apart with the streaming diarizer: ``speakers`` (all on
    channel 0, at most four) then give the ids and roles of the voices
    in the order they are first heard, and ``enrolled`` lists the
    enrolled people whose voiceprints may name them (``None``: every
    enrolled person; empty: nobody). ``parties`` (diarized only) makes
    it a two-person session (:class:`Parties`).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: str = Field(pattern=SESSION_ID_PATTERN)
    channels: int = Field(default=1, ge=1, le=2)
    speakers: List[ChannelSpeaker] = Field(default_factory=list)
    product: Optional[str] = Field(default=None, min_length=1, max_length=64)
    handoff: Handoff = Handoff()
    language: Optional[str] = Field(default="es", min_length=2, max_length=8)
    hang_ms: int = Field(default=500, ge=MIN_HANG_MS, le=MAX_HANG_MS)
    partial_every_ms: int = Field(default=800, ge=100, le=5000)
    diarize: bool = False
    enrolled: Optional[List[str]] = Field(default=None, max_length=50)
    parties: Optional[Parties] = None

    @model_validator(mode="after")
    def _every_channel_once(self) -> "LiveSessionRequest":
        """Refuse speakers that do not cover each channel exactly once.

        A diarized session lists its voices on channel 0 instead.
        """
        if self.diarize and self.channels != 1:
            raise ValueError("only a one-channel session is diarized")
        if self.enrolled is not None and not self.diarize:
            raise ValueError("enrolled is only used with diarize")
        ids = [s.id for s in self.speakers]
        if len(set(ids)) != len(ids):
            raise ValueError("speaker ids must be unique")
        if self.parties is not None:
            if not self.diarize:
                raise ValueError("parties are only used with diarize")
            pair = {self.parties.employee, self.parties.other}
            if len(pair) != 2 or not pair <= set(ids):
                raise ValueError(
                    "parties name two different speakers of the session"
                )
        if self.diarize:
            if len(self.speakers) > MAX_SLOTS or any(
                s.channel != 0 for s in self.speakers
            ):
                raise ValueError(
                    f"a diarized session lists up to {MAX_SLOTS} speakers,"
                    " all on channel 0"
                )
            return self
        if not self.speakers:
            return self
        channels = sorted(s.channel for s in self.speakers)
        if channels != list(range(self.channels)):
            raise ValueError(
                f"speakers must name channels 0-{self.channels - 1}"
                " once each"
            )
        return self


def channel_speakers(request: LiveSessionRequest) -> List[LiveSpeaker]:
    """Return the speaker of each channel, in channel order (LV-5).

    Stereo: channel ``n`` is ``speakers[n]``, and the events name the
    channel. One channel: a single speaker, no channel named (as
    ``live-file`` does). Without ``speakers``: ``S<n + 1>``, role
    ``unknown``. Diarized: the speakers given, in order (the voices
    heard first, second...), with no channel.
    """
    if request.diarize:
        return [
            LiveSpeaker(id=s.id, role=s.role, name=s.name)
            for s in request.speakers
        ]
    stereo = request.channels == 2
    given = {s.channel: s for s in request.speakers}
    out: List[LiveSpeaker] = []
    for channel in range(request.channels):
        speaker = given.get(channel)
        out.append(LiveSpeaker(
            id=speaker.id if speaker else f"S{channel + 1}",
            channel=channel if stereo else None,
            role=speaker.role if speaker else "unknown",
            name=speaker.name if speaker else None,
        ))
    return out


def decode_pcm(data: bytes, channels: int) -> List[FloatArray]:
    """Return the samples of one PCM message, one array per channel.

    ``data`` is signed 16-bit little-endian PCM at 16 kHz with the
    channels interleaved (frame ``n`` is ``ch0[n], ch1[n]``). Raises
    :class:`InvalidInputError` when it does not hold whole frames or is
    longer than :data:`MAX_MESSAGE_SECONDS`.
    """
    frame = PCM_BYTES * channels
    if len(data) % frame:
        raise InvalidInputError(
            f"PCM message of {len(data)} bytes is not a whole number of"
            f" {channels}-channel 16-bit frames"
        )
    if len(data) > frame * SAMPLE_RATE * MAX_MESSAGE_SECONDS:
        raise InvalidInputError(
            f"PCM message longer than {MAX_MESSAGE_SECONDS:g} s"
        )
    samples = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    if channels == 1:
        return [samples]
    interleaved = samples.reshape(-1, channels)
    return [
        np.ascontiguousarray(interleaved[:, n]) for n in range(channels)
    ]


class AudioBacklog:
    """Seconds of audio received but not yet read by the recogniser.

    ``limit`` seconds may wait; above it the source must stop reading
    (the socket then pushes back on the browser). ``behind`` seconds is
    the level at which the session warns that it runs behind.
    """

    def __init__(self, limit: float, behind: float) -> None:
        """Start empty."""
        self.limit = limit
        self.behind = behind
        self.seconds = 0.0
        self._warned = False

    def add(self, samples: int) -> None:
        """Count ``samples`` (per channel) received."""
        self.seconds += samples / SAMPLE_RATE

    def take(self, samples: int) -> None:
        """Count ``samples`` (per channel) read by the recogniser."""
        self.seconds = max(self.seconds - samples / SAMPLE_RATE, 0.0)

    def full(self) -> bool:
        """Tell whether the source must wait before sending more."""
        return self.seconds >= self.limit

    def warn(self) -> bool:
        """Tell, once per episode, that the recogniser fell behind.

        True the first time the backlog reaches ``behind``; again only
        after it has dropped below half of it.
        """
        if self.seconds >= self.behind and not self._warned:
            self._warned = True
            return True
        if self.seconds < self.behind / 2:
            self._warned = False
        return False


GpuChoice = Literal["wait", "gpu", "cpu"]
"""What a live session does after asking for the GPU (§7 steps 3-4)."""


def gpu_choice(
    waited: float, acked: bool, free_mib: Optional[int], need_mib: int,
    grace: float, timeout: float,
) -> GpuChoice:
    """Decide whether to start on the GPU, on the CPU, or wait more.

    The lease is written; the image worker answers it with an ack once
    it has freed the card. The session takes the GPU as soon as the
    ack is there and ``free_mib`` reaches ``need_mib``. Without an ack
    after ``grace`` seconds (no worker running, or one without the
    lease change) enough free memory is enough. After ``timeout``
    seconds with too little memory it degrades to the CPU; with no
    GPU memory reading at all (no GPU) it goes to the CPU at once.
    """
    if free_mib is None:
        return "cpu"
    if free_mib >= need_mib and (acked or waited >= grace):
        return "gpu"
    if waited >= timeout:
        return "cpu"
    return "wait"


class LiveServiceSettings(BaseModel):
    """How the service runs live sessions.

    ``model`` and ``compute_type`` choose the streaming recogniser on
    the GPU, ``cpu_model`` and ``cpu_compute_type`` the one of the
    degraded mode. ``need_mib`` is the GPU memory a session asks for
    (§5: about 2.2 GB live, stereo); ``ack_timeout``, ``ack_grace`` and
    ``poll_seconds`` time the wait for it (:func:`gpu_choice`).
    ``queue_seconds`` of audio may wait for the recogniser
    before the socket pushes back; ``behind_seconds`` waiting gives an
    ``asr_behind`` warning. ``max_sessions`` run at once (one GPU);
    ``kept_sessions`` ended ones stay readable. ``feed_timeout`` is how
    long a full queue may block a socket before the audio is refused;
    a session without audio for ``idle_seconds`` stops by itself.
    A diarized mono session asks for ``diarize_need_mib`` more GPU
    memory (Sortformer and the voice encoder) and names its voices
    with ``diarization``; a two-party one checks the employee's voice
    with ``voice_check`` (#249).
    """

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    model: str = "large-v3-turbo"
    compute_type: str = "int8"
    cpu_model: str = "small"
    cpu_compute_type: str = "int8"
    need_mib: int = Field(default=2200, ge=0)
    ack_timeout: float = Field(default=20.0, ge=0)
    ack_grace: float = Field(default=4.0, ge=0)
    poll_seconds: float = Field(default=0.5, gt=0)
    queue_seconds: float = Field(default=10.0, gt=0)
    behind_seconds: float = Field(default=3.0, gt=0)
    max_sessions: int = Field(default=1, ge=1)
    kept_sessions: int = Field(default=8, ge=0)
    feed_timeout: float = Field(default=30.0, gt=0)
    idle_seconds: float = Field(default=120.0, gt=0)
    diarize_need_mib: int = Field(default=1500, ge=0)
    diarization: MonoDiarizationSettings = MonoDiarizationSettings()
    voice_check: VoiceCheckSettings = VoiceCheckSettings()
