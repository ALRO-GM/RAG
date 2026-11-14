"""Transcription use cases: queue a session, transcribe it, report on it.

:class:`TranscribeSession` is the pipeline of ``docs/transcriber/
ARCHITECTURE.md`` §3: decode → recognise (timed words) → speech regions
and analysis windows → voice embeddings → voice groups (two passes, in
the domain) → identification against the enrolled voiceprints →
speakers on the words, smoothing, turns → sentences → transcript. Each
stage is reported to :class:`~transcriber.application.ports.Progress`.
A stage that fails leaves the session ``failed`` with the stage and the
reason; queueing it again (:class:`QueueTranscription`) retries.

Each queueing draws a new ``run_id`` for the session and the job
carries it. The job reads and writes the session under the session's
lock and, before each save, checks that the stored session is still the
one it runs for (same ``run_id``, still ``running``); if the session
was deleted, re-created or queued again it stops quietly, so a stale
job never overwrites or resurrects a session.
"""

import hashlib
import uuid
from typing import (
    AbstractSet,
    Callable,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from transcriber import __version__
from transcriber.application.common import (
    check_speaker_id,
    require_session,
)
from transcriber.application.ports import (
    Audio,
    AudioDecoder,
    JobRunner,
    JobStatus,
    Progress,
    RecordedAudio,
    SessionLocks,
    SessionRepository,
    SpeechRecogniser,
    VoiceEncoder,
    VoiceprintRepository,
)
from transcriber.domain.activity import heard_words, speech_activity
from transcriber.domain.errors import (
    InvalidInputError,
    NotFoundError,
    VoiceModelMismatchError,
)
from transcriber.domain.models import (
    Pipeline,
    Session,
    SessionState,
    Speaker,
    Transcript,
    Voiceprint,
    Word,
)
from transcriber.domain.parameters import TranscriptionParameters
from transcriber.domain.sentences import build_utterances
from transcriber.domain.sessions import move, require_state
from transcriber.domain.speakers import (
    analysis_windows,
    assign_speakers,
    smooth_speakers,
    spans_from_windows,
    speaker_label,
    speech_regions,
    turns_from_words,
)
from transcriber.domain.voices import (
    Attribution,
    FloatArray,
    group_voices,
    match_voices,
)

# Stages reported to Progress, with the share of the job done when each
# starts. The shares only order the progress bar; they are not timings.
STAGES: Tuple[Tuple[str, float], ...] = (
    ("decode", 0.0),
    ("recognise", 0.05),
    ("voices", 0.5),
    ("identify", 0.8),
    ("sentences", 0.85),
    ("save", 0.95),
)
DONE_STAGE = "done"


class TranscriptionOptions(BaseModel):
    """What the person starting a transcription knows about it.

    ``num_speakers``: how many people speak, when known (it makes the
    grouping far more reliable). It is taken as a fact: a number larger
    than the voices really present splits a real voice into several
    groups (one person then appears as two speakers), so give it only
    when sure. ``expected_speakers``: the enrolled ids of the people
    present; only their voiceprints are considered. ``None`` considers
    every enrolled person. ``language``: the language spoken, when it
    differs from the one the session was opened with; it replaces the
    session's language when the session is queued.
    """

    model_config = ConfigDict(frozen=True)

    language: Optional[str] = None
    num_speakers: Optional[int] = Field(default=None, ge=1)
    expected_speakers: Optional[List[str]] = None


class _Silent:
    """A :class:`Progress` that ignores every report."""

    def report(self, stage: str, fraction: float) -> None:
        """Ignore the report."""

    def note(self, message: str) -> None:
        """Ignore the note."""


class Candidates(NamedTuple):
    """The voiceprints a session is compared with.

    ``voiceprints`` are of the voice model in use; ``skipped`` counts
    the enrolled voiceprints left out because another model made them.
    """

    voiceprints: List[Voiceprint]
    skipped: int


def expected_voiceprints(
    voiceprints: VoiceprintRepository,
    options: TranscriptionOptions,
    model: str,
) -> Candidates:
    """Return the voiceprints of the people expected in the session.

    Without ``expected_speakers``, every enrolled voiceprint of voice
    model ``model``; those of another model cannot be compared and are
    counted as skipped. With ``expected_speakers``, those people's,
    each of whom must be enrolled (:class:`NotFoundError`) with
    ``model`` (:class:`VoiceModelMismatchError`: they must enrol
    again). An invalid id is an :class:`InvalidInputError`.
    """
    if options.expected_speakers is None:
        everyone = voiceprints.all()
        chosen = [v for v in everyone if v.model == model]
        return Candidates(chosen, len(everyone) - len(chosen))
    chosen = []
    for speaker_id in dict.fromkeys(options.expected_speakers):
        voiceprint = voiceprints.get(check_speaker_id(speaker_id))
        if voiceprint is None:
            raise NotFoundError(f"No enrolled speaker {speaker_id}.")
        if voiceprint.model != model:
            raise VoiceModelMismatchError(
                f"Speaker {speaker_id} was enrolled with voice model"
                f" {voiceprint.model}, not {model}; they must enrol again."
            )
        chosen.append(voiceprint)
    return Candidates(chosen, 0)


def skipped_note(count: int) -> str:
    """Return the note telling that ``count`` voiceprints were skipped."""
    return (
        f"{count} enrolled voiceprint(s) of another voice model were not"
        " compared; those people must enrol again to be recognised."
    )


STOPPED_NOTE = (
    "The session was deleted or queued again; this run stopped."
)


class _Result(NamedTuple):
    """What a run produced, before it is saved."""

    transcript: Transcript
    words: List[Word]


class TranscribeSession:
    """Turn the audio of a queued session into its transcript."""

    def __init__(
        self,
        sessions: SessionRepository,
        voiceprints: VoiceprintRepository,
        decoder: AudioDecoder,
        recogniser: SpeechRecogniser,
        encoder: VoiceEncoder,
        locks: SessionLocks,
        parameters: TranscriptionParameters = TranscriptionParameters(),
    ) -> None:
        """Wire the ports and take the thresholds from the settings."""
        self._sessions = sessions
        self._voiceprints = voiceprints
        self._decoder = decoder
        self._recogniser = recogniser
        self._encoder = encoder
        self._locks = locks
        self._parameters = parameters

    @property
    def voice_model(self) -> str:
        """Return the name of the voice model sessions are compared in."""
        return self._encoder.model_name

    def __call__(
        self,
        session_id: str,
        options: TranscriptionOptions = TranscriptionOptions(),
        progress: Optional[Progress] = None,
        run_id: Optional[str] = None,
    ) -> Optional[Transcript]:
        """Transcribe session ``session_id``, which must be ``queued``.

        The session is ``running`` while the stages run and ``done``
        once the transcript and the words are saved. Any error in a
        stage, or an interruption (any exception, ``KeyboardInterrupt``
        included), saves the session as ``failed`` with
        ``"<stage>: <why>"`` and is raised again.

        ``run_id`` is the run the job was queued for. If the stored
        session is gone or has another ``run_id`` when the job starts,
        or is no longer this run when a result is to be saved, nothing
        is saved and ``None`` is returned. Without ``run_id`` (a direct
        call) the session must exist.
        """
        report = progress or _Silent()
        with self._locks.hold(session_id):
            if run_id is None:
                session = require_session(self._sessions, session_id)
            else:
                found = self._sessions.get(session_id)
                if found is None or found.run_id != run_id:
                    report.note(STOPPED_NOTE)
                    return None
                session = found
            require_state(session, SessionState.QUEUED)
            running = move(session, SessionState.RUNNING)
            self._sessions.save(running)
        stages = iter(STAGES)
        stage = ""

        def advance() -> None:
            """Start the next stage and report it."""
            nonlocal stage
            stage, fraction = next(stages)
            report.report(stage, fraction)

        try:
            result = self._run(running, options, advance, report)
            saved = self._commit(running, result)
        except BaseException as exc:
            reason = str(exc).strip() or type(exc).__name__
            self._fail(running, f"{stage}: {reason}")
            raise
        if not saved:
            report.note(STOPPED_NOTE)
            return None
        report.report(DONE_STAGE, 1.0)
        return result.transcript

    def _current(self, running: Session) -> bool:
        """Tell whether the stored session is still the run ``running``.

        The caller holds the session's lock.
        """
        stored = self._sessions.get(running.id)
        return (
            stored is not None and stored.run_id == running.run_id
            and stored.state is SessionState.RUNNING
        )

    def _commit(self, running: Session, result: _Result) -> bool:
        """Save the words, transcript and session if still this run."""
        with self._locks.hold(running.id):
            if not self._current(running):
                return False
            self._sessions.save_words(running.id, result.words)
            self._sessions.save_transcript(result.transcript)
            self._sessions.save(result.transcript.session)
            return True

    def _fail(self, running: Session, reason: str) -> None:
        """Save the session ``failed`` with ``reason`` if still this run."""
        with self._locks.hold(running.id):
            if self._current(running):
                self._sessions.save(
                    move(running, SessionState.FAILED, reason)
                )

    def _run(
        self,
        running: Session,
        options: TranscriptionOptions,
        advance: Callable[[], None],
        report: Progress,
    ) -> _Result:
        """Run the stages; return the transcript and words to save."""
        p = self._parameters
        advance()  # decode
        recording = self._sessions.recording(running.id)
        audio = self._decoder.decode(recording)
        advance()  # recognise
        speech = self._recogniser.recognise(audio, running.language)
        # Recognisers invent words over silence: keep only the words
        # said where the audio has some energy.
        activity = speech_activity(
            audio.samples, audio.sample_rate, p.activity_frame,
            p.activity_threshold,
        )
        words = heard_words(
            [w for w in speech.words if w.text.strip()], activity
        )
        if not words:
            raise InvalidInputError("No speech recognised.")
        advance()  # voices
        regions = speech_regions(words, p.region_gap)
        coarse_windows = analysis_windows(regions, *p.coarse())
        fine_windows = analysis_windows(regions, *p.fine())
        groups = group_voices(
            self._embed(audio, coarse_windows),
            self._embed(audio, fine_windows),
            options.num_speakers, p.group_threshold, p.max_speakers,
            p.centre_windows, p.single_voice_threshold,
        )
        if fine_windows:
            spans = spans_from_windows(fine_windows, groups.fine)
        else:
            spans = spans_from_windows(coarse_windows, groups.coarse)
        words = assign_speakers(words, spans)
        if p.smoothing_run:
            words = smooth_speakers(words, p.smoothing_run)
        advance()  # identify
        # Only voiceprints of this voice model can be compared.
        candidates = expected_voiceprints(
            self._voiceprints, options, self.voice_model
        )
        if candidates.skipped:
            report.note(skipped_note(candidates.skipped))
        # Every enrolled id, compared or not, is kept off anonymous
        # labels.
        taken = {v.speaker_id for v in self._voiceprints.all()}
        names = self._identify(
            words, groups.vectors, candidates.voiceprints, taken
        )
        words = [
            w.model_copy(update={"speaker": names[w.speaker or ""].speaker})
            for w in words
        ]
        advance()  # sentences
        confidence = {a.speaker: a.similarity for a in names.values()}
        turns = [
            t.model_copy(update={"speaker_confidence": confidence[t.speaker]})
            for t in turns_from_words(words, p.turn_gap)
        ]
        utterances = build_utterances(turns)
        advance()  # save
        done = move(running, SessionState.DONE).model_copy(update={
            "duration": audio.duration,
            "audio_sha256": _sha256(recording),
        })
        transcript = Transcript(
            session=done,
            speakers=self._speakers(
                [u.speaker for u in utterances], names.values()
            ),
            utterances=utterances,
            pipeline=Pipeline(
                recogniser=self._recogniser.model_name,
                voice_model=self._encoder.model_name,
                transcriber=__version__,
            ),
        )
        return _Result(transcript, words)

    def _embed(
        self, audio: Audio, windows: Sequence[Tuple[float, float]]
    ) -> FloatArray:
        """Embed ``windows`` of ``audio``; no windows give zero rows."""
        if not windows:
            return np.zeros((0, 0), dtype=np.float32)
        return self._encoder.embed(audio, windows)

    def _identify(
        self,
        words: Sequence[Word],
        vectors: FloatArray,
        candidates: Sequence[Voiceprint],
        taken: AbstractSet[str],
    ) -> Dict[str, Attribution]:
        """Return the attribution of each diarizer label of ``words``.

        Groups are passed to the matching in order of first appearance
        in the final words, so ``S1`` is the first unknown voice heard;
        groups that kept no word are left out and take no voiceprint.
        Without any voice window every word has one unknown label,
        which becomes one anonymous speaker. Anonymous labels skip the
        ids in ``taken``.
        """
        order = list(dict.fromkeys(w.speaker or "" for w in words))
        if vectors.shape[0] == 0:
            matched = match_voices(
                np.zeros((len(order), 1), dtype=np.float32), [],
                taken=taken,
            )
        else:
            rows = {speaker_label(n): n for n in range(vectors.shape[0])}
            matched = match_voices(
                vectors[[rows[label] for label in order]], candidates,
                self._parameters.match_threshold,
                self._parameters.normaliser(), taken=taken,
            )
        return dict(zip(order, matched))

    def _speakers(
        self, heard: Sequence[str], attributions: Iterable[Attribution]
    ) -> List[Speaker]:
        """Return the speakers of ``heard`` in order of first appearance.

        Enrolled people carry the name of their consent record.
        """
        enrolled = {a.speaker for a in attributions if a.enrolled}
        speakers: List[Speaker] = []
        for speaker_id in dict.fromkeys(heard):
            name = None
            if speaker_id in enrolled:
                consent = self._voiceprints.consent(speaker_id)
                name = consent.name if consent is not None else None
            speakers.append(Speaker(
                id=speaker_id, name=name, enrolled=speaker_id in enrolled,
            ))
        return speakers


def _sha256(recording: RecordedAudio) -> str:
    """Return the SHA-256 of the recording's bytes."""
    return hashlib.sha256(recording.data).hexdigest()


def new_run_id() -> str:
    """Return a fresh random run id (32 hex characters)."""
    return uuid.uuid4().hex


class QueueTranscription:
    """Put a session in the job queue to be transcribed."""

    def __init__(
        self,
        sessions: SessionRepository,
        voiceprints: VoiceprintRepository,
        runner: JobRunner,
        transcribe: TranscribeSession,
        locks: SessionLocks,
        run_ids: Callable[[], str] = new_run_id,
    ) -> None:
        """Wire the ports, the use case the job runs and the run ids."""
        self._sessions = sessions
        self._voiceprints = voiceprints
        self._runner = runner
        self._transcribe = transcribe
        self._locks = locks
        self._run_ids = run_ids

    def __call__(
        self,
        session_id: str,
        options: TranscriptionOptions = TranscriptionOptions(),
    ) -> Session:
        """Queue an ``uploaded`` session, or a ``failed`` or ``done`` one.

        A ``failed`` session is retried; a ``done`` one is transcribed
        again (after enrolling someone or changing a threshold) and
        keeps its previous transcript until the new one is saved. The
        session gets a new ``run_id``, which the job carries. The
        expected speakers are checked now, so a typo or a voiceprint of
        another voice model is reported at once rather than as a failed
        job. ``options.language``, when given, replaces the session's
        language. If the runner refuses the job the session is saved as
        ``failed``. ``num_speakers`` and ``expected_speakers`` left out
        of ``options`` are taken from the session (what was chosen when
        it was started), so it is queued the same from any device.
        """
        with self._locks.hold(session_id):
            session = require_session(self._sessions, session_id)
            require_state(
                session, SessionState.UPLOADED, SessionState.FAILED,
                SessionState.DONE,
            )
            options = with_session_options(options, session)
            expected_voiceprints(
                self._voiceprints, options, self._transcribe.voice_model
            )
            if options.language is not None:
                session = _with_language(session, options.language)
            run_id = self._run_ids()
            queued = move(session, SessionState.QUEUED).model_copy(
                update={"run_id": run_id}
            )
            self._sessions.save(queued)

            def job(progress: Progress) -> None:
                """Transcribe this run of the session."""
                self._transcribe(session_id, options, progress, run_id)

            try:
                self._runner.submit(session_id, job)
            except Exception as exc:
                reason = str(exc).strip() or type(exc).__name__
                self._sessions.save(
                    move(queued, SessionState.FAILED, f"queue: {reason}")
                )
                raise
            return queued


def with_session_options(
    options: TranscriptionOptions, session: Session
) -> TranscriptionOptions:
    """Return ``options`` completed with the options kept on ``session``.

    A field given in ``options`` wins; one left out (``None``) takes the
    session's value.
    """
    expected = options.expected_speakers
    if expected is None and session.expected_speakers is not None:
        expected = list(session.expected_speakers)
    speakers = options.num_speakers
    if speakers is None:
        speakers = session.num_speakers
    return options.model_copy(update={
        "expected_speakers": expected, "num_speakers": speakers,
    })


def _with_language(session: Session, language: str) -> Session:
    """Return ``session`` with another ``language`` (validated)."""
    try:
        return Session.model_validate(
            {**session.model_dump(), "language": language}
        )
    except ValidationError as exc:
        raise InvalidInputError(f"Invalid language {language!r}.") from exc


class TranscriptionStatus:
    """Tell where a session is: its state and, while it runs, its stage."""

    def __init__(
        self, sessions: SessionRepository, runner: JobRunner
    ) -> None:
        """Wire the ports."""
        self._sessions = sessions
        self._runner = runner

    def __call__(self, session_id: str) -> JobStatus:
        """Return the status of ``session_id``.

        The state is the session's own; the stage and fraction come
        from the job runner while the session is queued or running. The
        last note of the session's job (voiceprints skipped, a stopped
        run) is kept in ``message``.
        """
        return self.of(require_session(self._sessions, session_id))

    def of(self, session: Session) -> JobStatus:
        """Return the status of a session already read (no second read)."""
        session_id = session.id
        job = self._runner.status(session_id)
        active = (SessionState.QUEUED, SessionState.RUNNING)
        if job is not None and session.state in active:
            return job.model_copy(update={"state": session.state})
        done = session.state is SessionState.DONE
        return JobStatus(
            session_id=session_id, state=session.state,
            stage=DONE_STAGE if done else None,
            fraction=1.0 if done else 0.0,
            message=job.message if job is not None else None,
        )
