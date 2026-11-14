"""In-memory fakes of the ports in :mod:`transcriber.application.ports`.

No model, no file, no network: the use cases are tested quickly and
deterministically. A scripted :class:`Scene` says who speaks when; the
fake decoder gives audio that is loud while someone speaks and silent
otherwise, the fake recogniser hears the scene's words and the fake
voice encoder returns, for each window, the voice of the person speaking
at the window's middle.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

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
LOUD = 0.1  # sample level while someone speaks (silence is 0)


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
    """Decode any recording into the scene's audio: loud while speaking.

    ``silent`` lists spans (seconds) forced to silence, so a test can
    have the recogniser hear words where the audio has none.
    """

    def __init__(self, scene: Scene) -> None:
        """Remember the scene whose duration and lines the audio has."""
        self.scene = scene
        self.decoded: List[RecordedAudio] = []
        self.failures: List[Exception] = []
        self.silent: List[Tuple[float, float]] = []

    def decode(self, recording: RecordedAudio) -> Audio:
        """Return samples of the scene: :data:`LOUD` during each line."""
        self.decoded.append(recording)
        if self.failures:
            raise self.failures.pop(0)
        samples = np.zeros(int(self.scene.duration * RATE), dtype=np.float32)
        for line in self.scene.lines:
            samples[int(line.start * RATE):int(line.end * RATE)] = LOUD
        for start, end in self.silent:
            samples[int(start * RATE):int(end * RATE)] = 0.0
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


class FakeSessionLocks:
    """Session locks that record who held what (one thread only)."""

    def __init__(self) -> None:
        """Start with nothing held."""
        self.held: Dict[str, int] = {}
        self.taken: List[str] = []

    @contextmanager
    def hold(self, session_id: str) -> Iterator[None]:
        """Count the lock of ``session_id`` as held in the block."""
        self.taken.append(session_id)
        self.held[session_id] = self.held.get(session_id, 0) + 1
        try:
            yield
        finally:
            self.held[session_id] -= 1
            if not self.held[session_id]:
                del self.held[session_id]

    def holds(self, session_id: str) -> bool:
        """Tell whether the lock of ``session_id`` is held now."""
        return session_id in self.held


@dataclass
class FakeSessionRepository:
    """Sessions, audio, words and transcripts in dictionaries.

    With ``locks``, every write made without holding the session's
    lock is recorded in ``unlocked_writes`` as ``(operation, id)``.
    """

    sessions: Dict[str, Session] = field(default_factory=dict)
    chunks: Dict[str, List[bytes]] = field(default_factory=dict)
    recordings: Dict[str, RecordedAudio] = field(default_factory=dict)
    stored_words: Dict[str, List[Word]] = field(default_factory=dict)
    transcripts: Dict[str, Transcript] = field(default_factory=dict)
    saved_states: List[Tuple[str, SessionState]] = field(default_factory=list)
    locks: Optional[FakeSessionLocks] = None
    unlocked_writes: List[Tuple[str, str]] = field(default_factory=list)

    def _write(self, operation: str, session_id: str) -> None:
        """Record a write made without the session's lock."""
        if self.locks is not None and not self.locks.holds(session_id):
            self.unlocked_writes.append((operation, session_id))

    def get(self, session_id: str) -> Optional[Session]:
        """Return the session, if stored."""
        return self.sessions.get(session_id)

    def ids(self) -> List[str]:
        """Return the stored session ids."""
        return list(self.sessions)

    def create(self, session: Session) -> None:
        """Store a new session."""
        self._write("create", session.id)
        if session.id in self.sessions:
            raise InvalidInputError(f"Session {session.id} exists.")
        self.sessions[session.id] = session
        self.saved_states.append((session.id, session.state))

    def save(self, session: Session) -> None:
        """Replace the session and remember the state it was saved in."""
        self._write("save", session.id)
        self.sessions[session.id] = session
        self.saved_states.append((session.id, session.state))

    def delete(self, session_id: str) -> None:
        """Forget everything about the session."""
        self._write("delete", session_id)
        self.sessions.pop(session_id, None)
        self.chunks.pop(session_id, None)
        self.recordings.pop(session_id, None)
        self.stored_words.pop(session_id, None)
        self.transcripts.pop(session_id, None)

    def append_chunk(self, session_id: str, number: int, data: bytes) -> None:
        """Store chunk ``number``, dropping it and any later one first.

        Idempotent per chunk number, as the port requires; a gap is an
        :class:`InvalidInputError`.
        """
        self._write("append_chunk", session_id)
        stored = self.chunks.setdefault(session_id, [])
        if number < 0 or number > len(stored):
            raise InvalidInputError(f"Chunk {number} leaves a gap.")
        del stored[number:]
        stored.append(data)

    def store_recording(
        self, session_id: str, recording: RecordedAudio
    ) -> None:
        """Keep the imported recording."""
        self._write("store_recording", session_id)
        self.recordings[session_id] = recording

    def recording(self, session_id: str) -> RecordedAudio:
        """Return the imported recording or the joined chunks."""
        if session_id in self.recordings:
            return self.recordings[session_id]
        return RecordedAudio(b"".join(self.chunks.get(session_id, [])),
                             PCM_FORMAT)

    def save_words(self, session_id: str, words: Sequence[Word]) -> None:
        """Keep the words."""
        self._write("save_words", session_id)
        self.stored_words[session_id] = list(words)

    def words(self, session_id: str) -> List[Word]:
        """Return the stored words."""
        return list(self.stored_words.get(session_id, []))

    def save_transcript(self, transcript: Transcript) -> None:
        """Keep the transcript."""
        self._write("save_transcript", transcript.session.id)
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
    """Keep every report and every note."""

    def __init__(self) -> None:
        """Start with no report."""
        self.reports: List[Tuple[str, float]] = []
        self.notes: List[str] = []

    def report(self, stage: str, fraction: float) -> None:
        """Keep the report."""
        self.reports.append((stage, fraction))

    def note(self, message: str) -> None:
        """Keep the note."""
        self.notes.append(message)


class _JobProgress:
    """Progress that updates the fake runner's status of one session."""

    def __init__(self, runner: "FakeJobRunner", session_id: str) -> None:
        """Remember where to write."""
        self.runner = runner
        self.session_id = session_id

    def report(self, stage: str, fraction: float) -> None:
        """Record the stage as the session's running status (the last
        note is kept)."""
        previous = self.runner.statuses.get(self.session_id)
        self.runner.statuses[self.session_id] = JobStatus(
            session_id=self.session_id, state=SessionState.RUNNING,
            stage=stage, fraction=fraction,
            message=previous.message if previous is not None else None,
        )

    def note(self, message: str) -> None:
        """Attach the note to the session's status."""
        current = self.runner.statuses[self.session_id]
        self.runner.statuses[self.session_id] = current.model_copy(
            update={"message": message}
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
                    message=self.statuses[session_id].message,
                )


class FixedClock:
    """A clock stopped at ``when``."""

    def __init__(self, when: datetime = NOW) -> None:
        """Stop the clock at ``when``."""
        self.when = when

    def now(self) -> datetime:
        """Return the stopped time."""
        return self.when
