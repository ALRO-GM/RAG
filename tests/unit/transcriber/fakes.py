"""In-memory fakes of the ports in :mod:`transcriber.application.ports`.

No model, no file, no network: the use cases are tested quickly and
deterministically. A scripted :class:`Scene` says who speaks when; the
fake recogniser hears its words and the fake voice encoder returns, for
each window, the voice of the person speaking at the window's middle.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from transcriber.application.ports import (
    PCM_FORMAT,
    Audio,
    Job,
    JobStatus,
    RecognisedSpeech,
    RecordedAudio,
)
from transcriber.domain.errors import InvalidInputError
from transcriber.domain.models import (
    ConsentRecord,
    Session,
    SessionState,
    Transcript,
    Voiceprint,
    Word,
)
from transcriber.domain.voices import FloatArray

NOW = datetime(2026, 11, 14, 10, 0, tzinfo=timezone.utc)
RATE = 100  # samples per second of the fake audio: enough to cut windows
DIM = 8


def voice(index: int) -> FloatArray:
    """A distinct voice: one axis of its own plus a small shared part."""
    vector = np.zeros(DIM, dtype=np.float32)
    vector[index] = 1.0
    vector[DIM - 1] = 0.3
    return vector


@dataclass(frozen=True)
class Line:
    """``speaker`` says ``text`` from ``start`` to ``end`` (seconds)."""

    speaker: str
    start: float
    end: float
    text: str


@dataclass
class Scene:
    """Who speaks when, and what each person's voice sounds like."""

    lines: List[Line]
    voices: Dict[str, FloatArray]
    duration: float = 0.0

    def __post_init__(self) -> None:
        """Default the duration to the end of the last line plus 1 s."""
        if not self.duration:
            self.duration = max(line.end for line in self.lines) + 1.0

    def words(self) -> List[Word]:
        """The words of every line, spread evenly over its time."""
        out: List[Word] = []
        for line in self.lines:
            tokens = line.text.split() or [line.text]
            size = (line.end - line.start) / len(tokens)
            for n, token in enumerate(tokens):
                out.append(Word(
                    text=token, start=round(line.start + n * size, 3),
                    end=round(line.start + (n + 1) * size, 3),
                    probability=0.9,
                ))
        return out

    def speaker_at(self, time: float) -> Optional[str]:
        """The person speaking at ``time``, or the nearest line's."""
        if not self.lines:
            return None
        return min(
            self.lines,
            key=lambda line: 0.0 if line.start <= time <= line.end
            else min(abs(time - line.start), abs(time - line.end)),
        ).speaker


def meeting() -> Scene:
    """Ana, then Luis, then Ana again: two voices, three turns."""
    return Scene(
        lines=[
            Line("ana", 0.0, 12.0,
                 "Hola a todos. Hoy repartimos las tareas del proyecto"
                 " nuevo. Empiezo yo con el diseño de la base de datos."),
            Line("luis", 13.0, 25.0,
                 "Vale. Yo me encargo de la interfaz y de las pruebas"
                 " del sistema. Necesito dos semanas."),
            Line("ana", 26.0, 36.0,
                 "Perfecto. Lo revisamos el lunes en la reunión."),
        ],
        voices={"ana": voice(0), "luis": voice(1)},
    )


def reading(speaker: str = "ana", seconds: float = 30.0) -> Scene:
    """One person reading the enrolment text for ``seconds``."""
    text = " ".join(["palabra"] * int(seconds * 2))
    return Scene(
        lines=[Line(speaker, 0.5, seconds - 0.5, text)],
        voices={speaker: voice(0)}, duration=seconds,
    )


class FakeDecoder:
    """Decode any recording into silence as long as the scene."""

    def __init__(self, scene: Scene) -> None:
        """Remember the scene whose duration the audio has."""
        self.scene = scene
        self.decoded: List[RecordedAudio] = []
        self.failures: List[Exception] = []

    def decode(self, recording: RecordedAudio) -> Audio:
        """Return silent samples of the scene's duration."""
        self.decoded.append(recording)
        if self.failures:
            raise self.failures.pop(0)
        samples = np.zeros(int(self.scene.duration * RATE), dtype=np.float32)
        return Audio(samples=samples, sample_rate=RATE)


class FakeRecogniser:
    """Hear the words of the scene; can be told to fail."""

    model_name = "fake-whisper"

    def __init__(self, scene: Scene) -> None:
        """Remember the scene to hear."""
        self.scene = scene
        self.failures: List[Exception] = []
        self.languages: List[Optional[str]] = []

    def recognise(
        self, audio: Audio, language: Optional[str] = None
    ) -> RecognisedSpeech:
        """Return the scene's words, or raise the next queued failure."""
        self.languages.append(language)
        if self.failures:
            raise self.failures.pop(0)
        return RecognisedSpeech(words=self.scene.words(), language=language)


class FakeVoiceEncoder:
    """Embed each window as the voice heard at its middle, plus noise."""

    def __init__(self, scene: Scene, model_name: str = "fake-voice") -> None:
        """Remember the scene and the model name to report."""
        self.scene = scene
        self.model_name = model_name
        self.calls: List[List[Tuple[float, float]]] = []

    def embed(
        self, audio: Audio, windows: Sequence[Tuple[float, float]]
    ) -> FloatArray:
        """One vector per window; deterministic small noise."""
        self.calls.append(list(windows))
        rng = np.random.default_rng(len(windows))
        rows = []
        for start, end in windows:
            speaker = self.scene.speaker_at((start + end) / 2)
            base = (self.scene.voices[speaker] if speaker is not None
                    else np.zeros(DIM, dtype=np.float32))
            rows.append(base + rng.normal(0.0, 0.01, DIM))
        out: FloatArray = np.array(rows, dtype=np.float32).reshape(-1, DIM)
        return out


@dataclass
class FakeSessionRepository:
    """Sessions, audio, words and transcripts in dictionaries."""

    sessions: Dict[str, Session] = field(default_factory=dict)
    chunks: Dict[str, List[bytes]] = field(default_factory=dict)
    recordings: Dict[str, RecordedAudio] = field(default_factory=dict)
    stored_words: Dict[str, List[Word]] = field(default_factory=dict)
    transcripts: Dict[str, Transcript] = field(default_factory=dict)
    saved_states: List[Tuple[str, SessionState]] = field(default_factory=list)

    def get(self, session_id: str) -> Optional[Session]:
        """Return the session, if stored."""
        return self.sessions.get(session_id)

    def create(self, session: Session) -> None:
        """Store a new session."""
        if session.id in self.sessions:
            raise InvalidInputError(f"Session {session.id} exists.")
        self.save(session)

    def save(self, session: Session) -> None:
        """Replace the session and remember the state it was saved in."""
        self.sessions[session.id] = session
        self.saved_states.append((session.id, session.state))

    def delete(self, session_id: str) -> None:
        """Forget everything about the session."""
        self.sessions.pop(session_id, None)
        self.chunks.pop(session_id, None)
        self.recordings.pop(session_id, None)
        self.stored_words.pop(session_id, None)
        self.transcripts.pop(session_id, None)

    def append_chunk(self, session_id: str, number: int, data: bytes) -> None:
        """Append a chunk; the number must be the next one."""
        stored = self.chunks.setdefault(session_id, [])
        assert number == len(stored)
        stored.append(data)

    def store_recording(
        self, session_id: str, recording: RecordedAudio
    ) -> None:
        """Keep the imported recording."""
        self.recordings[session_id] = recording

    def recording(self, session_id: str) -> RecordedAudio:
        """Return the imported recording or the joined chunks."""
        if session_id in self.recordings:
            return self.recordings[session_id]
        return RecordedAudio(b"".join(self.chunks.get(session_id, [])),
                             PCM_FORMAT)

    def save_words(self, session_id: str, words: Sequence[Word]) -> None:
        """Keep the words."""
        self.stored_words[session_id] = list(words)

    def words(self, session_id: str) -> List[Word]:
        """Return the stored words."""
        return list(self.stored_words.get(session_id, []))

    def save_transcript(self, transcript: Transcript) -> None:
        """Keep the transcript."""
        self.transcripts[transcript.session.id] = transcript

    def transcript(self, session_id: str) -> Optional[Transcript]:
        """Return the transcript, if any."""
        return self.transcripts.get(session_id)


@dataclass
class FakeVoiceprintRepository:
    """Voiceprints and consent records in dictionaries."""

    voiceprints: Dict[str, Voiceprint] = field(default_factory=dict)
    consents: Dict[str, ConsentRecord] = field(default_factory=dict)

    def save(self, voiceprint: Voiceprint, consent: ConsentRecord) -> None:
        """Store (or replace) both."""
        self.voiceprints[voiceprint.speaker_id] = voiceprint
        self.consents[consent.speaker_id] = consent

    def get(self, speaker_id: str) -> Optional[Voiceprint]:
        """Return the voiceprint, if any."""
        return self.voiceprints.get(speaker_id)

    def consent(self, speaker_id: str) -> Optional[ConsentRecord]:
        """Return the consent record, if any."""
        return self.consents.get(speaker_id)

    def all(self) -> List[Voiceprint]:
        """Return every voiceprint, in enrolment order."""
        return list(self.voiceprints.values())

    def delete(self, speaker_id: str) -> bool:
        """Remove both; tell whether they existed."""
        self.consents.pop(speaker_id, None)
        return self.voiceprints.pop(speaker_id, None) is not None

    def enrol(
        self, speaker_id: str, vector: FloatArray, name: Optional[str] = None,
        model: str = "fake-voice",
    ) -> Voiceprint:
        """Store a voiceprint directly (a test shortcut)."""
        unit = vector / np.linalg.norm(vector)
        voiceprint = Voiceprint(
            speaker_id=speaker_id, model=model,
            vector=[float(x) for x in unit], consent_version="consent-1",
            enrolled_at=NOW,
        )
        self.save(voiceprint, ConsentRecord(
            speaker_id=speaker_id, name=name, consent_version="consent-1",
            text_version="es-1", given_at=NOW,
        ))
        return voiceprint


class RecordingPublisher:
    """Keep every published document."""

    def __init__(self) -> None:
        """Start with nothing published."""
        self.documents: List[Mapping[str, Any]] = []

    def publish(self, document: Mapping[str, Any]) -> None:
        """Keep the document."""
        self.documents.append(document)


class RecordingProgress:
    """Keep every report."""

    def __init__(self) -> None:
        """Start with no report."""
        self.reports: List[Tuple[str, float]] = []

    def report(self, stage: str, fraction: float) -> None:
        """Keep the report."""
        self.reports.append((stage, fraction))


class _JobProgress:
    """Progress that updates the fake runner's status of one session."""

    def __init__(self, runner: "FakeJobRunner", session_id: str) -> None:
        """Remember where to write."""
        self.runner = runner
        self.session_id = session_id

    def report(self, stage: str, fraction: float) -> None:
        """Record the stage as the session's running status."""
        self.runner.statuses[self.session_id] = JobStatus(
            session_id=self.session_id, state=SessionState.RUNNING,
            stage=stage, fraction=fraction,
        )


class FakeJobRunner:
    """Queue jobs and run them only when the test says so."""

    def __init__(self) -> None:
        """Start with an empty queue."""
        self.pending: List[Tuple[str, Job]] = []
        self.statuses: Dict[str, JobStatus] = {}
        self.errors: Dict[str, Exception] = {}
        self.refuse: Optional[Exception] = None

    def submit(self, session_id: str, job: Job) -> None:
        """Queue the job, or raise ``refuse`` if set."""
        if self.refuse is not None:
            raise self.refuse
        self.pending.append((session_id, job))
        self.statuses[session_id] = JobStatus(
            session_id=session_id, state=SessionState.QUEUED,
        )

    def status(self, session_id: str) -> Optional[JobStatus]:
        """Return the last known status of the session's job."""
        return self.statuses.get(session_id)

    def run_all(self) -> None:
        """Run the queued jobs in order, recording how each ended."""
        while self.pending:
            session_id, job = self.pending.pop(0)
            try:
                job(_JobProgress(self, session_id))
            except Exception as exc:  # the runner keeps going
                self.errors[session_id] = exc
                self.statuses[session_id] = JobStatus(
                    session_id=session_id, state=SessionState.FAILED,
                )
            else:
                self.statuses[session_id] = JobStatus(
                    session_id=session_id, state=SessionState.DONE,
                    stage="done", fraction=1.0,
                )


class FixedClock:
    """A clock stopped at ``when``."""

    def __init__(self, when: datetime = NOW) -> None:
        """Stop the clock at ``when``."""
        self.when = when

    def now(self) -> datetime:
        """Return the stopped time."""
        return self.when
