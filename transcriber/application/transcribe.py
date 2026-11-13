"""Transcription use cases: queue a session, transcribe it, report on it.

:class:`TranscribeSession` is the pipeline of ``docs/transcriber/
ARCHITECTURE.md`` §3: decode → recognise (timed words) → speech regions
and analysis windows → voice embeddings → voice groups (two passes, in
the domain) → identification against the enrolled voiceprints →
speakers on the words, smoothing, turns → sentences → transcript. Each
stage is reported to :class:`~transcriber.application.ports.Progress`.
A stage that fails leaves the session ``failed`` with the stage and the
reason; queueing it again (:class:`QueueTranscription`) retries.
"""

import hashlib
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from transcriber import __version__
from transcriber.application.common import require_session
from transcriber.application.ports import (
    Audio,
    AudioDecoder,
    JobRunner,
    JobStatus,
    Progress,
    SessionRepository,
    SpeechRecogniser,
    VoiceEncoder,
    VoiceprintRepository,
)
from transcriber.domain.errors import InvalidInputError, NotFoundError
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
    grouping far more reliable). ``expected_speakers``: the enrolled
    ids of the people present; only their voiceprints are considered.
    ``None`` considers every enrolled person.
    """

    model_config = ConfigDict(frozen=True)

    num_speakers: Optional[int] = Field(default=None, ge=1)
    expected_speakers: Optional[List[str]] = None


class _Silent:
    """A :class:`Progress` that ignores every report."""

    def report(self, stage: str, fraction: float) -> None:
        """Ignore the report."""


def expected_voiceprints(
    voiceprints: VoiceprintRepository, options: TranscriptionOptions
) -> List[Voiceprint]:
    """Return the voiceprints of the people expected in the session.

    Every enrolled voiceprint without ``expected_speakers``; otherwise
    those people's, each of whom must be enrolled
    (:class:`NotFoundError` otherwise).
    """
    if options.expected_speakers is None:
        return voiceprints.all()
    chosen = []
    for speaker_id in dict.fromkeys(options.expected_speakers):
        voiceprint = voiceprints.get(speaker_id)
        if voiceprint is None:
            raise NotFoundError(f"No enrolled speaker {speaker_id}.")
        chosen.append(voiceprint)
    return chosen


class TranscribeSession:
    """Turn the audio of a queued session into its transcript."""

    def __init__(
        self,
        sessions: SessionRepository,
        voiceprints: VoiceprintRepository,
        decoder: AudioDecoder,
        recogniser: SpeechRecogniser,
        encoder: VoiceEncoder,
        parameters: TranscriptionParameters = TranscriptionParameters(),
    ) -> None:
        """Wire the ports and take the thresholds from the settings."""
        self._sessions = sessions
        self._voiceprints = voiceprints
        self._decoder = decoder
        self._recogniser = recogniser
        self._encoder = encoder
        self._parameters = parameters

    def __call__(
        self,
        session_id: str,
        options: TranscriptionOptions = TranscriptionOptions(),
        progress: Optional[Progress] = None,
    ) -> Transcript:
        """Transcribe session ``session_id``, which must be ``queued``.

        The session is ``running`` while the stages run and ``done``
        once the transcript and the words are saved. Any error in a
        stage saves the session as ``failed`` with ``"<stage>: <why>"``
        and is raised again.
        """
        report = progress or _Silent()
        session = require_session(self._sessions, session_id)
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
            transcript = self._run(running, options, advance)
            self._sessions.save(transcript.session)
        except Exception as exc:
            reason = str(exc).strip() or type(exc).__name__
            self._sessions.save(
                move(running, SessionState.FAILED, f"{stage}: {reason}")
            )
            raise
        report.report(DONE_STAGE, 1.0)
        return transcript

    def _run(
        self,
        running: Session,
        options: TranscriptionOptions,
        advance: Callable[[], None],
    ) -> Transcript:
        """Run the stages; save the words and the transcript."""
        p = self._parameters
        advance()  # decode
        recording = self._sessions.recording(running.id)
        audio = self._decoder.decode(recording)
        advance()  # recognise
        speech = self._recogniser.recognise(audio, running.language)
        words = [w for w in speech.words if w.text.strip()]
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
            p.centre_windows,
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
        model = self._encoder.model_name
        candidates = [
            v for v in expected_voiceprints(self._voiceprints, options)
            if v.model == model
        ]
        names = self._identify(words, groups.vectors, candidates)
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
            "audio_sha256": hashlib.sha256(recording.data).hexdigest(),
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
        self._sessions.save_words(running.id, words)
        self._sessions.save_transcript(transcript)
        return transcript

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
    ) -> Dict[str, Attribution]:
        """Return the attribution of each diarizer label of ``words``.

        Groups are passed to the matching in order of first appearance
        in the final words, so ``S1`` is the first unknown voice heard;
        groups that kept no word are left out and take no voiceprint.
        Without any voice window every word has one unknown label,
        which becomes one anonymous speaker.
        """
        order = list(dict.fromkeys(w.speaker or "" for w in words))
        if vectors.shape[0] == 0:
            matched = match_voices(
                np.zeros((len(order), 1), dtype=np.float32), []
            )
        else:
            rows = {speaker_label(n): n for n in range(vectors.shape[0])}
            matched = match_voices(
                vectors[[rows[label] for label in order]], candidates,
                self._parameters.match_threshold,
                self._parameters.normaliser(),
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


class QueueTranscription:
    """Put a session in the job queue to be transcribed."""

    def __init__(
        self,
        sessions: SessionRepository,
        voiceprints: VoiceprintRepository,
        runner: JobRunner,
        transcribe: TranscribeSession,
    ) -> None:
        """Wire the ports and the use case the job runs."""
        self._sessions = sessions
        self._voiceprints = voiceprints
        self._runner = runner
        self._transcribe = transcribe

    def __call__(
        self,
        session_id: str,
        options: TranscriptionOptions = TranscriptionOptions(),
    ) -> Session:
        """Queue an ``uploaded`` session, or retry a ``failed`` one.

        The expected speakers are checked now, so a typo is reported at
        once rather than as a failed job. If the runner refuses the job
        the session is saved as ``failed``.
        """
        session = require_session(self._sessions, session_id)
        require_state(session, SessionState.UPLOADED, SessionState.FAILED)
        expected_voiceprints(self._voiceprints, options)
        queued = move(session, SessionState.QUEUED)
        self._sessions.save(queued)

        def job(progress: Progress) -> None:
            """Transcribe the session, reporting to ``progress``."""
            self._transcribe(session_id, options, progress)

        try:
            self._runner.submit(session_id, job)
        except Exception as exc:
            reason = str(exc).strip() or type(exc).__name__
            self._sessions.save(
                move(queued, SessionState.FAILED, f"queue: {reason}")
            )
            raise
        return queued


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
        from the job runner while the session is queued or running.
        """
        session = require_session(self._sessions, session_id)
        job = self._runner.status(session_id)
        active = (SessionState.QUEUED, SessionState.RUNNING)
        if job is not None and session.state in active:
            return job.model_copy(update={"state": session.state})
        done = session.state is SessionState.DONE
        return JobStatus(
            session_id=session_id, state=session.state,
            stage=DONE_STAGE if done else None,
            fraction=1.0 if done else 0.0,
        )
