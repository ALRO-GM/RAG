"""The service's own models as lab tools, and the speaker mapping.

:class:`InProcessTools` runs the ``project`` family of the catalogue in
the service's process: the built-in diarization and the project's
Whisper recognisers. Both run the real transcription use case
(:class:`~transcriber.application.transcribe.TranscribeSession`) on a
scratch, in-memory session holding the lab's audio, so what the lab
measures is exactly what a transcription does (seams, words over
silence, identification with the session's expected speakers and
count, sentences kept whole). The built-in diarization's segments are
the turns of the attributed words; a recogniser keeps the words and
drops their speakers.

:func:`map_speakers` names the speakers of any diarizer: each label's
non-overlapped speech is embedded with the service's voice model and
matched one to one to the enrolled voiceprints; labels left over may
take a person from the built-in diarization by overlap in time.
"""

import contextlib
import time
from typing import (
    Callable,
    ContextManager,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
from scipy.optimize import linear_sum_assignment

from transcriber.application.lab_ports import ToolRequest
from transcriber.application.ports import (
    Audio,
    PlaybackAudio,
    RecordedAudio,
    SpeechRecogniser,
    VoiceEncoder,
    VoiceprintRepository,
)
from transcriber.application.transcribe import (
    TranscribeSession,
    TranscriptionOptions,
)
from transcriber.domain.comparison import frame_count, to_frames
from transcriber.domain.diarization import (
    FRAME,
    MAX_LABEL_WINDOWS,
    exclusive_regions,
    spread,
)
from transcriber.domain.errors import (
    LabStepError,
    NotFoundError,
    TranscriberError,
)
from transcriber.domain.lab import (
    BUILTIN_DIARIZER,
    Segment,
    SpeakerMapping,
    StageOutcome,
)
from transcriber.domain.models import (
    ConsentRecord,
    Session,
    SessionState,
    Transcript,
    Voiceprint,
    Word,
)
from transcriber.domain.parameters import TranscriptionParameters
from transcriber.domain.speakers import analysis_windows, turns_from_words
from transcriber.domain.voices import match_voices, unit_rows

SCRATCH_ID = "lab-scratch"
MAX_MAPPING_WINDOWS = MAX_LABEL_WINDOWS
"""Windows embedded per speaker at most (evenly spread)."""


class _ScratchSessions:
    """A session repository holding one session in memory."""

    def __init__(self, session: Session) -> None:
        """Hold ``session``; no audio bytes are kept (the decoder has it)."""
        self._session = session
        self._words: List[Word] = []
        self._transcript: Optional[Transcript] = None

    def get(self, session_id: str) -> Optional[Session]:
        """Return the session if the id is the scratch one."""
        return self._session if session_id == self._session.id else None

    def ids(self) -> List[str]:
        """Return the one id."""
        return [self._session.id]

    def create(self, session: Session) -> None:
        """Not used by a transcription."""
        raise NotImplementedError

    def save(self, session: Session) -> None:
        """Replace the session."""
        self._session = session

    def delete(self, session_id: str) -> None:
        """Not used by a transcription."""
        raise NotImplementedError

    def append_chunk(self, session_id: str, number: int, data: bytes) -> None:
        """Not used by a transcription."""
        raise NotImplementedError

    def store_recording(
        self, session_id: str, recording: RecordedAudio
    ) -> None:
        """Not used by a transcription."""
        raise NotImplementedError

    def recording(self, session_id: str) -> RecordedAudio:
        """Return an empty WAV recording: the decoder holds the audio."""
        return RecordedAudio(b"", "wav")

    def playback(self, session_id: str) -> PlaybackAudio:
        """Not used by a transcription."""
        raise NotFoundError("No playback in the lab's scratch session.")

    def save_words(self, session_id: str, words: Sequence[Word]) -> None:
        """Keep the attributed words."""
        self._words = list(words)

    def words(self, session_id: str) -> List[Word]:
        """Return the attributed words."""
        return list(self._words)

    def save_transcript(self, transcript: Transcript) -> None:
        """Keep the transcript."""
        self._transcript = transcript

    def transcript(self, session_id: str) -> Optional[Transcript]:
        """Return the transcript, once made."""
        return self._transcript


class _GivenAudio:
    """A decoder that returns audio already decoded."""

    def __init__(self, audio: Audio) -> None:
        """Hold ``audio``."""
        self._audio = audio

    def decode(self, recording: RecordedAudio) -> Audio:
        """Return the held audio whatever the bytes."""
        return self._audio


class _NoVoiceprints:
    """No one enrolled: what a corpus session is compared with."""

    def save(self, voiceprint: Voiceprint, consent: ConsentRecord) -> None:
        """Not used by a transcription."""
        raise NotImplementedError

    def get(self, speaker_id: str) -> Optional[Voiceprint]:
        """Nobody is enrolled."""
        return None

    def consent(self, speaker_id: str) -> Optional[ConsentRecord]:
        """Nobody gave consent here."""
        return None

    def all(self) -> List[Voiceprint]:
        """Return no voiceprint."""
        return []

    def delete(self, speaker_id: str) -> bool:
        """Nothing to delete."""
        return False


class _NoLocks:
    """Session locks for a session nobody else can see."""

    def hold(self, session_id: str) -> ContextManager[None]:
        """Return a context that holds nothing."""
        return contextlib.nullcontext()


class InProcessTools:
    """The ``project`` family: built-in diarization, project recognisers.

    ``recogniser_for`` returns the recogniser of a Hugging Face model
    name (a recogniser entry's first weight; ``""`` for the service's
    own, which the built-in diarization uses); ``release`` unloads every
    model this tool used. Parameters it reads: ``language`` (a code or
    ``auto``), ``num_speakers`` and ``expected_speakers``.
    """

    def __init__(
        self,
        recogniser_for: Callable[[str], SpeechRecogniser],
        encoder: VoiceEncoder,
        voiceprints: VoiceprintRepository,
        parameters: TranscriptionParameters = TranscriptionParameters(),
        release: Callable[[], None] = lambda: None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        """Keep the models; nothing is loaded yet."""
        self._recogniser_for = recogniser_for
        self._encoder = encoder
        self._voiceprints = voiceprints
        self._parameters = parameters
        self._release = release
        self._clock = clock

    def release(self) -> None:
        """Unload the models used."""
        self._release()

    def run(
        self, request: ToolRequest, cancelled: Callable[[], bool]
    ) -> StageOutcome:
        """Run the built-in diarization or a project recogniser."""
        entry = request.entry
        if cancelled():
            raise LabStepError("cancelled", "The run was cancelled.")
        if request.stage == "diarize" and entry.id == BUILTIN_DIARIZER:
            model = self._recogniser_for("")
        elif request.stage == "recognise" and entry.weights:
            model = self._recogniser_for(entry.weights[0])
        else:
            raise LabStepError(
                "unknown_model",
                f"{entry.id} is not a {request.stage} tool of the service.",
            )
        audio = request.load()
        started = self._clock()
        words, language = self._transcribe(model, audio, request)
        seconds = self._clock() - started
        if request.stage == "recognise":
            return StageOutcome(
                stage="recognise", model=entry.id,
                model_version=model.model_name, seconds=seconds,
                audio_seconds=audio.duration,
                words=[w.model_copy(update={"speaker": None})
                       for w in words],
                language=language,
                notes=["The service's transcription: seams resolved,"
                       " words over silence dropped, words scored."],
            )
        return StageOutcome(
            stage="diarize", model=entry.id,
            model_version=model.model_name, seconds=seconds,
            audio_seconds=audio.duration,
            segments=turn_segments(words, self._parameters.turn_gap),
            notes=["Turns of the service's attributed words (recognition"
                   f" with {model.model_name} included in the time)."],
        )

    def _transcribe(
        self, model: SpeechRecogniser, audio: Audio, request: ToolRequest
    ) -> Tuple[List[Word], Optional[str]]:
        """Transcribe ``audio`` as a session would be; return its words."""
        params = request.params
        language = str(params.get("language") or "auto")
        expected = params.get("expected_speakers")
        number = params.get("num_speakers")
        session = Session(
            id=SCRATCH_ID, language=language, state=SessionState.QUEUED,
        )
        scratch = _ScratchSessions(session)
        voiceprints: VoiceprintRepository = (
            _NoVoiceprints() if request.anonymous else self._voiceprints
        )
        transcribe = TranscribeSession(
            scratch, voiceprints, _GivenAudio(audio), model,
            self._encoder, _NoLocks(), self._parameters,
        )
        options = TranscriptionOptions(
            num_speakers=int(number) if number else None,
            expected_speakers=list(expected) if expected else None,
        )
        try:
            transcribe(SCRATCH_ID, options)
        except LabStepError:
            raise
        except TranscriberError as exc:
            raise LabStepError("failed", str(exc)) from exc
        return scratch.words(SCRATCH_ID), (
            None if language == "auto" else language
        )


def turn_segments(words: Sequence[Word], gap: float) -> List[Segment]:
    """Return one segment per turn of the attributed ``words``."""
    turns = turns_from_words(
        [w for w in words if w.speaker], gap
    ) if words else []
    return [
        Segment(start=t.start, end=t.end, speaker=t.speaker) for t in turns
    ]


def map_speakers(
    segments: Sequence[Segment],
    audio: Audio,
    encoder: VoiceEncoder,
    candidates: Sequence[Voiceprint],
    threshold: float,
    parameters: TranscriptionParameters,
    builtin: Optional[Sequence[Segment]] = None,
) -> List[SpeakerMapping]:
    """Name the labels of a diarization (see the module).

    ``candidates`` are the voiceprints that may be given (the
    session's expected people, or every enrolled person), accepted at
    ``threshold`` cosine or more, one person per label. A label left
    without a person takes, one to one, a person of the ``builtin``
    segments it shares the most time with (none already given).
    """
    labels = sorted({s.speaker for s in segments})
    regions = exclusive_regions(segments, audio.duration)
    speech = {
        label: round(sum(b - a for a, b in regions.get(label, [])), 3)
        for label in labels
    }
    windows: List[Tuple[float, float]] = []
    owners: List[str] = []
    for label in labels:
        own = spread(
            analysis_windows(regions.get(label, []), *parameters.fine()),
            MAX_MAPPING_WINDOWS,
        )
        windows.extend(own)
        owners.extend([label] * len(own))
    named: Dict[str, Tuple[str, float]] = {}
    if windows and candidates:
        vectors = unit_rows(encoder.embed(audio, windows))
        heard = [label for label in labels if label in owners]
        centroids = np.stack([
            vectors[[n for n, o in enumerate(owners) if o == label]]
            .mean(axis=0) for label in heard
        ])
        matched = match_voices(
            centroids, candidates, threshold, parameters.normaliser()
        )
        for label, attribution in zip(heard, matched):
            if attribution.enrolled and attribution.similarity is not None:
                named[label] = (attribution.speaker, attribution.similarity)
    by_overlap = _overlap_names(
        segments, builtin or [], [lb for lb in labels if lb not in named],
        {person for person, _ in named.values()}, audio.duration,
    )
    out: List[SpeakerMapping] = []
    for label in labels:
        if label in named:
            person, similarity = named[label]
            out.append(SpeakerMapping(
                label=label, speaker=person, method="voiceprint",
                similarity=round(similarity, 4),
                speech_seconds=speech[label],
            ))
        elif label in by_overlap:
            person, seconds = by_overlap[label]
            out.append(SpeakerMapping(
                label=label, speaker=person, method="builtin_overlap",
                overlap_seconds=round(seconds, 3),
                speech_seconds=speech[label],
            ))
        else:
            out.append(SpeakerMapping(
                label=label, speech_seconds=speech[label]
            ))
    return out


def _overlap_names(
    segments: Sequence[Segment],
    builtin: Sequence[Segment],
    labels: Sequence[str],
    given: set[str],
    duration: float,
) -> Dict[str, Tuple[str, float]]:
    """Pair ``labels`` one to one with the built-in's people by overlap.

    Only enrolled people of the built-in (not anonymous, not in
    ``given``) are candidates; a pair needs some shared time.
    """
    people = sorted({
        s.speaker for s in builtin
        if not s.speaker.startswith("S") or not s.speaker[1:].isdigit()
    } - given)
    if not labels or not people:
        return {}
    frames = frame_count(duration, FRAME)
    mine = to_frames(segments, frames, FRAME, labels).active
    theirs = to_frames(builtin, frames, FRAME, people).active
    shared = mine.astype(np.int64).T @ theirs.astype(np.int64)
    rows, cols = linear_sum_assignment(shared, maximize=True)
    return {
        labels[int(r)]: (people[int(c)], float(shared[r, c]) * FRAME)
        for r, c in zip(rows, cols) if shared[r, c] > 0
    }


def builtin_mapping(segments: Sequence[Segment]) -> List[SpeakerMapping]:
    """Return the mapping of the built-in diarization: it names itself.

    Its anonymous voices (``S1``...) stay unnamed.
    """
    out = []
    seconds: Dict[str, float] = {}
    for s in segments:
        seconds[s.speaker] = seconds.get(s.speaker, 0.0) + s.end - s.start
    for label in sorted(seconds):
        anonymous = label.startswith("S") and label[1:].isdigit()
        out.append(SpeakerMapping(
            label=label, speaker=None if anonymous else label,
            method="none" if anonymous else "pipeline",
            speech_seconds=round(seconds[label], 3),
        ))
    return out
