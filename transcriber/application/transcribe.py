"""Transcription use cases: queue a session, transcribe it, report on it.

:class:`TranscribeSession` is the pipeline of ``docs/transcriber/
ARCHITECTURE.md`` §3: decode → recognise (timed words) → seams resolved,
words over silence dropped → words scored (and the language of each
window detected) → speech regions and analysis windows → voice
embeddings → voice groups (two passes, in the domain; or, with a
diarizer of the lab, its segments: :meth:`TranscribeSession._external`)
→ identification
against the enrolled voiceprints (seeded at the match threshold, or,
when the session lists exactly who is present, clustered into that
many voices and assigned one to one above ``expected_floor``) →
speakers on the words, smoothing, sentences kept whole (weak changes of
speaker undone) → turns → sentences → certainty of each sentence's
speaker, overlaps (voice windows and, if configured, the overlap
detector) and language → transcript. Each
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
import logging
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
    Diarizer,
    JobRunner,
    JobStatus,
    LanguageSpan,
    OverlapDetector,
    Progress,
    RecordedAudio,
    SessionLocks,
    SessionRepository,
    SpeechRecogniser,
    VoiceEncoder,
    VoiceprintRepository,
)
from transcriber.domain.activity import heard_words, speech_activity
from transcriber.domain.certainty import (
    VoiceEvidence,
    add_detected_overlaps,
    assess_utterances,
    keep_sentences_whole,
)
from transcriber.domain.diarization import (
    BUILTIN_RULE,
    Diarization,
    DiarizerRule,
    label_order,
    label_windows,
    name_labels,
    segment_spans,
    segment_voices,
)
from transcriber.domain.errors import (
    InvalidInputError,
    LabStepError,
    NotFoundError,
    VoiceModelMismatchError,
)
from transcriber.domain.lab import BUILTIN_DIARIZER
from transcriber.domain.languages import (
    check_language,
    recognition_language,
    tag_languages,
)
from transcriber.domain.models import (
    Identification,
    Pipeline,
    Session,
    SessionState,
    Speaker,
    Transcript,
    Utterance,
    Voiceprint,
    Word,
)
from transcriber.domain.parameters import TranscriptionParameters
from transcriber.domain.seams import resolve_seams
from transcriber.domain.sentences import build_utterances
from transcriber.domain.sessions import carry_description, move, require_state
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
    SeededVoices,
    group_voices,
    is_single_voice,
    match_voices,
    name_seeded,
    seed_voices,
)

LOG = logging.getLogger(__name__)

SCORING_FAILED_NOTE = (
    "The recogniser could not score its words; the transcript has no"
    " recognition confidence."
)
DIARIZER_FALLBACK_NOTE = (
    "The diarizer {model} could not be used ({reason}); the voices were"
    " separated by the built-in grouping."
)
OVERLAP_FAILED_NOTE = (
    "The overlap detector failed; overlaps are flagged from the voice"
    " windows only."
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


def expected_rule(
    options: TranscriptionOptions, listed: Sequence[Voiceprint]
) -> bool:
    """Tell whether the expected-speakers rule applies.

    The session lists who is present (``expected_speakers``, each
    enrolled with the voice model in use: ``listed``) and the number of
    speakers given equals the length of that list.
    """
    expected = options.expected_speakers
    return (
        expected is not None and bool(listed)
        and options.num_speakers == len(set(expected)) == len(listed)
    )


def skipped_note(count: int) -> str:
    """Return the note telling that ``count`` voiceprints were skipped."""
    return (
        f"{count} enrolled voiceprint(s) of another voice model were not"
        " compared; those people must enrol again to be recognised."
    )


STOPPED_NOTE = (
    "The session was deleted or queued again; this run stopped."
)


class _Stopped(Exception):
    """The run is no longer the session's: stop quietly, save nothing."""


class _Voices(NamedTuple):
    """The voices of a run: labelled words and what identification needs.

    ``seeded`` is set when the voiceprints seeded the grouping;
    ``vectors`` holds one vector per voice label otherwise. ``heard``,
    when set, holds the label numbers that have a voice to match (a
    lab diarizer's label with no window has none and stays anonymous).
    """

    words: List[Word]
    vectors: FloatArray
    evidence: Optional[VoiceEvidence]
    seeded: Optional[SeededVoices]
    heard: Optional[AbstractSet[int]] = None


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
        overlaps: Optional[OverlapDetector] = None,
        diarizer: Optional[Diarizer] = None,
        diarizer_rule: DiarizerRule = BUILTIN_RULE,
    ) -> None:
        """Wire the ports and take the thresholds from the settings.

        ``overlaps`` is the optional overlap detector; without it
        overlaps are flagged from the voice windows only. ``diarizer``
        runs the lab's diarizers; ``diarizer_rule`` says which one a
        session gets (``builtin``, or no ``diarizer``: the built-in
        grouping of voice windows).
        """
        self._sessions = sessions
        self._voiceprints = voiceprints
        self._decoder = decoder
        self._recogniser = recogniser
        self._encoder = encoder
        self._locks = locks
        self._parameters = parameters
        self._overlaps = overlaps
        self._diarizer = diarizer
        self._rule = diarizer_rule

    @property
    def voice_model(self) -> str:
        """Return the name of the voice model sessions are compared in."""
        return self._encoder.model_name

    def check_language(self, language: str) -> str:
        """Return ``language`` if the recogniser can transcribe it.

        ``auto`` (detect) is always accepted; another code must be one
        the recogniser knows (:class:`InvalidInputError` otherwise).
        """
        return check_language(
            language, self._recogniser.supported_languages
        )

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
        except _Stopped:
            report.note(STOPPED_NOTE)
            return None
        except BaseException as exc:
            reason = str(exc).strip() or type(exc).__name__
            self._fail(running, f"{stage}: {reason}")
            raise
        if saved is None:
            report.note(STOPPED_NOTE)
            return None
        report.report(DONE_STAGE, 1.0)
        return saved

    def _current(self, running: Session) -> Optional[Session]:
        """Return the stored session if it is still the run ``running``.

        ``None`` when it is gone or another run. The caller holds the
        session's lock.
        """
        stored = self._sessions.get(running.id)
        if (
            stored is not None and stored.run_id == running.run_id
            and stored.state is SessionState.RUNNING
        ):
            return stored
        return None

    def _commit(
        self, running: Session, result: _Result
    ) -> Optional[Transcript]:
        """Save the words, transcript and session if still this run.

        The conversation type and goal are the stored ones (they may
        have been edited while the job ran). Returns the saved
        transcript, or ``None`` when the run is no longer current.
        """
        with self._locks.hold(running.id):
            stored = self._current(running)
            if stored is None:
                return None
            transcript = result.transcript.model_copy(update={
                "session": carry_description(
                    result.transcript.session, stored
                ),
            })
            self._sessions.save_words(running.id, result.words)
            self._sessions.save_transcript(transcript)
            self._sessions.save(transcript.session)
            return transcript

    def _fail(self, running: Session, reason: str) -> None:
        """Save the session ``failed`` with ``reason`` if still this run."""
        with self._locks.hold(running.id):
            stored = self._current(running)
            if stored is not None:
                self._sessions.save(carry_description(
                    move(running, SessionState.FAILED, reason), stored
                ))

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
        language = recognition_language(running.language)
        speech = self._recogniser.recognise(audio, language)
        # Recognisers invent words over silence: keep only the words
        # said where the audio has some energy. Passages returned twice
        # at the seams of the recogniser's windows go first.
        activity = speech_activity(
            audio.samples, audio.sample_rate, p.activity_frame,
            p.activity_threshold,
        )
        words = heard_words(
            [w for w in resolve_seams(speech.words) if w.text.strip()],
            activity,
        )
        if not words:
            raise InvalidInputError("No speech recognised.")
        words, languages = self._scored(audio, words, language, report)
        advance()  # voices
        # Only voiceprints of this voice model can be compared.
        candidates = expected_voiceprints(
            self._voiceprints, options, self.voice_model
        )
        listed = expected_rule(options, candidates.voiceprints)
        external = self._external(running, audio, options, report)
        if external is not None:
            voices = self._segment_voices(audio, words, external)
        else:
            voices = self._grouped_voices(
                audio, words, options, candidates.voiceprints, listed
            )
        words, vectors, evidence, seeded, heard = voices
        if p.smoothing_run:
            words = smooth_speakers(words, p.smoothing_run)
        moved: List[Tuple[float, float]] = []
        if p.whole_sentences:
            words, moved = keep_sentences_whole(
                words, evidence, p.whole_sentence_margin, p.turn_gap,
                p.whole_sentence_max_move, p.whole_sentence_max_moved,
            )
        advance()  # identify
        if candidates.skipped:
            report.note(skipped_note(candidates.skipped))
        # Every enrolled id, compared or not, is kept off anonymous
        # labels.
        taken = {v.speaker_id for v in self._voiceprints.all()}
        if seeded is not None:
            names = self._name_seeded(
                words, seeded, candidates.voiceprints, taken
            )
        else:
            names = self._identify(
                words, vectors, candidates.voiceprints, taken,
                p.expected_floor if listed else p.match_threshold, heard,
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
        if evidence is not None:
            evidence = evidence._replace(voices=[
                names[v].speaker if v is not None and v in names else None
                for v in evidence.voices
            ])
        utterances = assess_utterances(
            utterances, evidence, p.doubt_margin, moved
        )
        if self._overlaps is not None or external is not None:
            utterances = self._detected(
                audio, utterances, report,
                external.overlaps if external is not None else [],
            )
        if languages:
            utterances = tag_languages(
                utterances, [(s.start, s.end, s.language) for s in languages],
                running.language,
            )
        advance()  # save
        done = move(running, SessionState.DONE).model_copy(update={
            "duration": audio.duration,
            "audio_sha256": _sha256(recording),
        })
        transcript = Transcript(
            session=done,
            speakers=self._speakers(
                [u.speaker for u in utterances], names.values(), listed
            ),
            utterances=utterances,
            pipeline=Pipeline(
                recogniser=self._recogniser.model_name,
                voice_model=self._encoder.model_name,
                transcriber=__version__,
                diarizer=(
                    external.model if external is not None
                    else BUILTIN_DIARIZER
                ),
                diarizer_licence=(
                    external.licence if external is not None else None
                ),
                diarizer_commercial=(
                    external.commercial if external is not None else None
                ),
            ),
        )
        return _Result(transcript, words)

    def _grouped_voices(
        self, audio: Audio, words: List[Word],
        options: TranscriptionOptions, candidates: Sequence[Voiceprint],
        listed: bool,
    ) -> _Voices:
        """Find the voices with the built-in grouping of voice windows.

        The voiceprints seed the grouping (``identification`` ``seeded``)
        unless the expected-speakers rule applies (``listed``).
        """
        p = self._parameters
        regions = speech_regions(words, p.region_gap)
        coarse_windows = analysis_windows(regions, *p.coarse())
        fine_windows = analysis_windows(regions, *p.fine())
        coarse = self._embed(audio, coarse_windows)
        fine = self._embed(audio, fine_windows)
        seeded: Optional[SeededVoices] = None
        if (
            p.identification == "seeded" and candidates
            and coarse.shape[0] and not listed
        ):
            seeded = seed_voices(
                coarse, fine, candidates, options.num_speakers,
                p.match_threshold, p.normaliser(), p.group_threshold,
                p.max_speakers, self._one_voice,
            )
            coarse_labels, fine_labels = seeded.coarse, seeded.fine
            vectors, scores = seeded.vectors, seeded.fine_scores
        else:
            groups = group_voices(
                coarse, fine, options.num_speakers, p.group_threshold,
                p.max_speakers, p.centre_windows, p.single_voice_threshold,
                p.single_voice_quantile,
            )
            coarse_labels, fine_labels = groups.coarse, groups.fine
            vectors, scores = groups.vectors, groups.fine_scores
        evidence: Optional[VoiceEvidence] = None
        if fine_windows:
            spans = spans_from_windows(fine_windows, fine_labels)
            if scores is not None:
                evidence = VoiceEvidence(fine_windows, scores, [
                    speaker_label(n) for n in range(scores.shape[1])
                ])
        else:
            spans = spans_from_windows(coarse_windows, coarse_labels)
        return _Voices(assign_speakers(words, spans), vectors, evidence,
                       seeded)

    def _stale(self, running: Session) -> bool:
        """Tell whether the stored session is no longer the run ``running``.

        Read without the lock: it is only polled to stop a diarizer early
        (the session was deleted or queued again); saving still checks
        under the lock.
        """
        stored = self._sessions.get(running.id)
        return (
            stored is None or stored.run_id != running.run_id
            or stored.state is not SessionState.RUNNING
        )

    def _external(
        self, running: Session, audio: Audio, options: TranscriptionOptions,
        report: Progress,
    ) -> Optional[Diarization]:
        """Run the lab diarizer the rule picks; ``None``: use the built-in.

        The built-in grouping is used when there is no diarizer, the
        rule picks ``builtin``, the model cannot run here (its
        environment is missing), it fails, or it hears no one; each case
        but the first two is logged as a warning and noted on the job.
        A diarizer stopped because the session was deleted or queued
        again stops the run quietly (nothing is saved); one stopped
        because the service is stopping fails the job. Neither falls
        back.
        """
        if self._diarizer is None:
            return None
        model = self._rule.model_for(options.num_speakers)
        if model == BUILTIN_DIARIZER:
            return None
        problem = self._diarizer.problem(model)
        if problem is None:
            try:
                found = self._diarizer.diarize(
                    model, audio, options.num_speakers,
                    lambda: self._stale(running),
                )
            except LabStepError as exc:
                if exc.code == "cancelled":
                    if self._stale(running):
                        raise _Stopped() from exc
                    raise
                LOG.warning("The diarizer %s failed.", model, exc_info=True)
                problem = str(exc).strip() or exc.code
            except Exception as exc:
                LOG.warning("The diarizer %s failed.", model, exc_info=True)
                problem = str(exc).strip() or type(exc).__name__
            else:
                if any(s.end > s.start for s in found.segments):
                    return found
                problem = "it heard no speech"
        LOG.warning(
            "Diarizer %s not used (%s); falling back to the built-in"
            " grouping.", model, problem,
        )
        report.note(DIARIZER_FALLBACK_NOTE.format(
            model=model, reason=problem[:300]
        ))
        return None

    def _segment_voices(
        self, audio: Audio, words: List[Word], found: Diarization,
    ) -> _Voices:
        """Find the voices from an external diarizer's segments.

        Each word takes the segment it overlaps most; each label's voice
        is embedded over the stretches it speaks alone and compared with
        the voiceprints afterwards (no seeding: the diarizer already
        grouped the voices). The short windows over the recognised
        speech give each sentence's certainty, as with the built-in
        grouping.
        """
        p = self._parameters
        order = label_order(found.segments)
        windows, owners = label_windows(
            found.segments, order, audio.duration, *p.fine()
        )
        label_vectors = self._embed(audio, windows)
        fine_windows = analysis_windows(
            speech_regions(words, p.region_gap), *p.fine()
        )
        fine = self._embed(audio, fine_windows)
        if label_vectors.shape[0] == 0 and fine.shape[0]:
            label_vectors = np.zeros((0, fine.shape[1]), dtype=np.float32)
        voices = segment_voices(
            label_vectors, owners, len(order), fine, p.centre_windows
        )
        evidence: Optional[VoiceEvidence] = None
        if fine_windows and voices.fine_scores is not None:
            evidence = VoiceEvidence(fine_windows, voices.fine_scores, [
                speaker_label(n) for n in range(len(order))
            ])
        words = assign_speakers(words, segment_spans(found.segments, order))
        vectors = voices.vectors
        if vectors.shape[1] == 0:
            vectors = np.zeros((0, 0), dtype=np.float32)
        return _Voices(words, vectors, evidence, None,
                       frozenset(voices.heard))

    def _scored(
        self, audio: Audio, words: List[Word], language: Optional[str],
        report: Progress,
    ) -> Tuple[List[Word], List[LanguageSpan]]:
        """Return ``words`` with the recogniser's probability of each.

        Also the stretches whose language the recogniser detected while
        scoring. The scores are an extra: if the scoring fails (any
        ``Exception``) the words are returned without probability and
        no language, the failure is logged and noted, and the job goes
        on.
        """
        unscored = [w.model_copy(update={"probability": None}) for w in words]
        try:
            scored = self._recogniser.score_words(audio, words, language)
        except Exception:
            LOG.warning("Scoring the recognised words failed.", exc_info=True)
            report.note(SCORING_FAILED_NOTE)
            return unscored, []
        if len(scored.words) != len(words):
            LOG.warning("The recogniser scored %d of %d words.",
                        len(scored.words), len(words))
            report.note(SCORING_FAILED_NOTE)
            return unscored, []
        return [
            w.model_copy(update={"probability": s.probability})
            for w, s in zip(words, scored.words)
        ], list(scored.languages)

    def _detected(
        self, audio: Audio, utterances: List[Utterance], report: Progress,
        heard: Sequence[Tuple[float, float]] = (),
    ) -> List[Utterance]:
        """Flag the utterances heard with two voices at once.

        ``heard`` are the overlaps an external diarizer marked; the
        overlap detector's, when one is configured, are added. A failure
        of the detector (any ``Exception``) is logged and noted; the
        other flags stay and the job goes on.
        """
        detected = list(heard)
        if self._overlaps is not None:
            try:
                detected += self._overlaps.overlaps(audio)
            except Exception:
                LOG.warning("The overlap detector failed.", exc_info=True)
                report.note(OVERLAP_FAILED_NOTE)
                if not heard:
                    return utterances
        return add_detected_overlaps(
            utterances, detected, self._parameters.overlap_seconds
        )

    def _embed(
        self, audio: Audio, windows: Sequence[Tuple[float, float]]
    ) -> FloatArray:
        """Embed ``windows`` of ``audio``; no windows give zero rows."""
        if not windows:
            return np.zeros((0, 0), dtype=np.float32)
        return self._encoder.embed(audio, windows)

    def _one_voice(self, vectors: FloatArray) -> bool:
        """The single-voice rule of the parameters (raw long windows)."""
        p = self._parameters
        return is_single_voice(
            vectors, p.single_voice_threshold, p.single_voice_quantile
        )

    def _name_seeded(
        self,
        words: Sequence[Word],
        seeded: SeededVoices,
        candidates: Sequence[Voiceprint],
        taken: AbstractSet[str],
    ) -> Dict[str, Attribution]:
        """Return the attribution of each label of ``words`` (seeded).

        As in :meth:`_identify`, anonymous labels are numbered in order
        of first appearance in the final words.
        """
        order = list(dict.fromkeys(w.speaker or "" for w in words))
        group = {
            speaker_label(n): n for n in range(seeded.vectors.shape[0])
        }
        named = name_seeded(
            [group[label] for label in order], seeded, candidates,
            self._parameters.normaliser(), taken,
        )
        return dict(zip(order, named))

    def _identify(
        self,
        words: Sequence[Word],
        vectors: FloatArray,
        candidates: Sequence[Voiceprint],
        taken: AbstractSet[str],
        threshold: float,
        heard: Optional[AbstractSet[int]] = None,
    ) -> Dict[str, Attribution]:
        """Return the attribution of each diarizer label of ``words``.

        A group takes a voiceprint at ``threshold`` or more (the match
        threshold, or the floor of the expected-speakers rule). With
        ``heard``, only those group numbers may take one; the others
        stay anonymous (:func:`~transcriber.domain.diarization.
        name_labels`).

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
            picked = vectors[[rows[label] for label in order]]
            if heard is None:
                matched = match_voices(
                    picked, candidates, threshold,
                    self._parameters.normaliser(), taken=taken,
                )
            else:
                matched = name_labels(
                    picked, [rows[label] in heard for label in order],
                    candidates, threshold, self._parameters.normaliser(),
                    taken,
                )
        return dict(zip(order, matched))

    def _speakers(
        self, heard: Sequence[str], attributions: Iterable[Attribution],
        listed: bool = False,
    ) -> List[Speaker]:
        """Return the speakers of ``heard`` in order of first appearance.

        Enrolled people carry the name of their consent record and how
        they were identified: ``expected_floor`` when the
        expected-speakers rule named them (``listed``) with a similarity
        below the match threshold, else ``threshold``.
        """
        similarity = {
            a.speaker: a.similarity for a in attributions if a.enrolled
        }
        speakers: List[Speaker] = []
        for speaker_id in dict.fromkeys(heard):
            name = None
            identification: Optional[Identification] = None
            if speaker_id in similarity:
                consent = self._voiceprints.consent(speaker_id)
                name = consent.name if consent is not None else None
                value = similarity[speaker_id]
                identification = (
                    "expected_floor" if listed and value is not None
                    and value < self._parameters.match_threshold
                    else "threshold"
                )
            speakers.append(Speaker(
                id=speaker_id, name=name, enrolled=speaker_id in similarity,
                identification=identification,
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
            self._transcribe.check_language(session.language)
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
