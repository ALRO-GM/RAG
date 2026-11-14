"""Import public labelled corpora into the lab and summarise the scores.

:class:`ImportReference` turns each recording of a corpus
(:data:`~transcriber.domain.lab_reference.CORPORA`) into a session of
its own (``<corpus>-<recording>``, state ``uploaded``, no enrolled
people) holding the corpus's audio, plus the session's
:class:`~transcriber.domain.lab_reference.SessionReference` in the
lab's storage: speaker turns (overlap kept, anonymous corpus ids), the
reference words when the corpus has them, and the corpus, licence and
source the recording came under. Lab runs on such a session are then
measured against the reference as well (``reference`` of the results,
:func:`~transcriber.application.lab_compare.compare_run`).

Before a recording becomes a session its decoded audio goes through the
speech check (:mod:`~transcriber.domain.speech_check`): one that holds
steady noise rather than speech is skipped, as is one whose reference
the corpus gives broken (a turn that ends before it starts, say:
``invalid reference (...)``), and ``limit`` counts only the recordings
that can be used. :class:`CheckReferenceSpeech` runs the
same check on sessions imported before and stores its figures in their
reference, so that the summary and the overview leave out the ones that
fail.

:class:`ReferenceSummary` gathers the newest scored run of every
imported session and gives, per model and per pipeline, the mean and
spread of each metric over the recordings (``docs/transcriber/LAB.md``
§13).
"""

from typing import Any, Callable, Dict, List, Mapping, Optional

from pydantic import BaseModel, ConfigDict

from transcriber.application.lab_ports import (
    CorpusRecording,
    LabRepository,
    ReferenceCorpus,
)
from transcriber.application.ports import (
    AudioDecoder,
    Clock,
    SessionLocks,
    SessionRepository,
)
from transcriber.application.record import ImportRecording
from transcriber.domain.errors import InvalidInputError, StorageError
from transcriber.domain.models import Session
from transcriber.domain.speech_check import SpeechCheck, check_speech
from transcriber.domain.lab_reference import (
    CorpusSpec,
    MetricSpread,
    SessionReference,
    check_split,
    corpus_spec,
    make_reference,
    recording_session_id,
    reference_segments,
    share,
    spread,
)

SCORED_STATES = ("done", "partial")


class ImportedRecording(BaseModel):
    """What the import did with one recording.

    ``state``: ``imported``; ``exists`` when the session and its
    reference were already there (left as they were); ``repaired`` when
    an earlier import stopped before storing the reference; ``skipped``
    when the audio failed the speech check or the reference is invalid
    (no session is made; ``speech`` is ``None`` for the latter);
    ``checked`` when :class:`CheckReferenceSpeech` checked a session
    again. The numbers describe the reference (aggregates only, never
    text): ``speaker_seconds`` sums the turns (two people at once count
    twice). ``speech`` holds the speech check's figures when known and
    ``reason`` says why the recording is not used (``None``: it is).
    """

    model_config = ConfigDict(frozen=True)

    session_id: str
    recording: str
    state: str
    duration: Optional[float] = None
    speakers: int = 0
    turns: int = 0
    speaker_seconds: Optional[float] = None
    words: Optional[int] = None
    speech: Optional[SpeechCheck] = None
    reason: Optional[str] = None

    @property
    def usable(self) -> bool:
        """Tell whether the recording is a session that can be scored."""
        return self.state != "skipped" and self.reason is None


class ImportReference:
    """Create one lab session per recording of a public corpus."""

    def __init__(
        self,
        sessions: SessionRepository,
        repository: LabRepository,
        corpus: ReferenceCorpus,
        decoder: AudioDecoder,
        importer: ImportRecording,
        locks: SessionLocks,
        clock: Clock,
    ) -> None:
        """Wire the ports; ``importer`` stores a new session's audio."""
        self._sessions = sessions
        self._repository = repository
        self._corpus = corpus
        self._decoder = decoder
        self._importer = importer
        self._locks = locks
        self._clock = clock

    def __call__(
        self,
        corpus: str,
        split: Optional[str] = None,
        limit: Optional[int] = None,
        report: Callable[[ImportedRecording], None] = lambda _: None,
    ) -> List[ImportedRecording]:
        """Import up to ``limit`` usable recordings of ``split``.

        A recording whose session already exists is left as it is, so an
        interrupted import can be run again; one whose reference is
        invalid or whose audio fails the speech check is skipped and the
        corpus is read on until
        ``limit`` recordings can be used (already imported ones count
        unless they failed the check). ``report`` hears about each
        recording as soon as it is done.
        """
        if limit is not None and limit < 1:
            raise InvalidInputError("--limit must be 1 or more.")
        spec = corpus_spec(corpus)
        chosen = check_split(spec, split)
        done: List[ImportedRecording] = []
        usable = 0
        for item in self._corpus.recordings(spec, chosen, None):
            if limit is not None and usable >= limit:
                break
            result = self._one(spec, chosen, item)
            done.append(result)
            usable += result.usable
            report(result)
        return done

    def _one(
        self, spec: CorpusSpec, split: str, item: CorpusRecording
    ) -> ImportedRecording:
        """Import one recording (or find it imported, or repair it).

        A session of that id with its reference is left as it is
        (``exists``). One the import created but whose reference was
        never stored (the process stopped in between) gets it now
        (``repaired``); a session of that id made otherwise is refused.
        A new recording whose reference is invalid, or whose audio fails
        the speech check, is not imported (``skipped``); a repaired one
        keeps the speech figures in its reference, which leaves it out
        of the summary (its reference was valid when it was made).
        """
        session_id = recording_session_id(spec.id, item.recording)
        title = f"{spec.display_name} {item.recording}"
        found = self._sessions.get(session_id)
        stored = (
            None if found is None else self._repository.reference(session_id)
        )
        if stored is not None:
            return ImportedRecording(
                session_id=session_id, recording=item.recording,
                state="exists", speech=stored.speech,
                reason=stored.excluded(),
            )
        if found is not None and found.title != title:
            raise InvalidInputError(
                f"Session {session_id} exists and was not imported from"
                f" {spec.display_name}."
            )
        audio = self._decoder.decode(item.audio)
        duration = audio.duration
        if duration <= 0:
            raise InvalidInputError(
                f"Recording {item.recording!r} has no audio."
            )
        try:
            reference = make_reference(
                spec, split, item.recording, duration,
                reference_segments(item.starts, item.ends, item.speakers,
                                   duration),
                item.words, self._clock.now(),
            )
        except InvalidInputError as exc:
            if found is not None:
                raise
            return ImportedRecording(
                session_id=session_id, recording=item.recording,
                state="skipped", duration=round(duration, 3),
                reason=invalid_reference(exc),
            )
        speech = check_speech(audio.samples, audio.sample_rate)
        del audio
        if found is None and not speech.speech:
            return ImportedRecording(
                session_id=session_id, recording=item.recording,
                state="skipped", duration=round(duration, 3),
                speech=speech, reason=speech.reason(),
            )
        reference = reference.model_copy(update={"speech": speech})
        with self._locks.hold(session_id):
            if found is not None:
                self._store(found, duration, reference)
                return _described(session_id, reference, "repaired")
            session = self._importer(
                session_id, spec.language, item.audio, title=title,
            )
            try:
                self._store(session, duration, reference)
            except BaseException:
                self._repository.delete_session(session_id)
                self._sessions.delete(session_id)
                raise
        return _described(session_id, reference)

    def _store(
        self, session: Session, duration: float,
        reference: SessionReference,
    ) -> None:
        """Save the session's duration, then its reference."""
        self._sessions.save(session.model_copy(
            update={"duration": round(duration, 3)}
        ))
        self._repository.save_reference(session.id, reference)


INVALID_REFERENCE = "invalid reference"
"""The start of the reason a recording with an unusable reference has."""


def invalid_reference(error: InvalidInputError) -> str:
    """Return the skip reason of a recording whose reference is refused.

    ``invalid reference (<what is wrong>)``: a turn that ends before it
    starts, columns of different lengths, a turn without a speaker, or
    no turn at all.
    """
    return f"{INVALID_REFERENCE} ({str(error).strip().rstrip('.')})"


def _described(
    session_id: str, reference: SessionReference, state: str = "imported",
) -> ImportedRecording:
    """Return the import line of a new session."""
    seconds = sum(s.end - s.start for s in reference.segments)
    return ImportedRecording(
        session_id=session_id, recording=reference.recording,
        state=state, duration=round(reference.duration, 3),
        speakers=len(reference.speakers()), turns=len(reference.segments),
        speaker_seconds=round(seconds, 3),
        words=None if reference.words is None else len(reference.words),
        speech=reference.speech, reason=reference.excluded(),
    )


class CheckReferenceSpeech:
    """Run the speech check on sessions imported before it existed."""

    def __init__(
        self,
        sessions: SessionRepository,
        repository: LabRepository,
        decoder: AudioDecoder,
        locks: SessionLocks,
    ) -> None:
        """Wire the ports."""
        self._sessions = sessions
        self._repository = repository
        self._decoder = decoder
        self._locks = locks

    def __call__(
        self,
        corpus: str,
        report: Callable[[ImportedRecording], None] = lambda _: None,
    ) -> List[ImportedRecording]:
        """Check every imported session of ``corpus``; store the figures.

        Each session's audio is decoded and checked again and the
        figures go into its reference (state ``checked``; ``reason``
        set when it fails). Nothing is deleted: a failing session stays,
        left out of the summary and the overview.
        """
        spec = corpus_spec(corpus)
        done: List[ImportedRecording] = []
        for session_id in sorted(self._sessions.ids()):
            reference = _readable(self._repository, session_id)
            if reference is None or reference.corpus != spec.id:
                continue
            with self._locks.hold(session_id):
                audio = self._decoder.decode(
                    self._sessions.recording(session_id)
                )
                speech = check_speech(audio.samples, audio.sample_rate)
                del audio
                self._repository.save_reference(
                    session_id,
                    reference.model_copy(update={"speech": speech}),
                )
            result = ImportedRecording(
                session_id=session_id, recording=reference.recording,
                state="checked", duration=round(reference.duration, 3),
                speech=speech, reason=speech.reason(),
            )
            done.append(result)
            report(result)
        return done


def _readable(
    repository: LabRepository, session_id: str
) -> Optional[SessionReference]:
    """Return the session's reference; unreadable ones are skipped."""
    try:
        return repository.reference(session_id)
    except (StorageError, InvalidInputError):
        return None


TRACK_METRICS = (
    "der", "missed_rate", "false_alarm_rate", "confusion_rate", "jer",
    "overlap_precision", "overlap_recall", "speaker_count_error",
    "speaker_count_abs_error",
)
"""What the summary aggregates per diarization track."""


def _track_metrics(row: Mapping[str, Any]) -> Dict[str, Optional[float]]:
    """Return the summary's metrics of one stored reference track row."""
    speech = float(row["reference_speech"])
    count = row["speaker_count_error"]
    return {
        "der": row["der"],
        "missed_rate": share(float(row["missed"]), speech),
        "false_alarm_rate": share(float(row["false_alarm"]), speech),
        "confusion_rate": share(float(row["confusion"]), speech),
        "jer": row["jer"],
        "overlap_precision": row["overlap_precision"],
        "overlap_recall": row["overlap_recall"],
        "speaker_count_error": float(count),
        "speaker_count_abs_error": float(abs(count)),
    }


class ModelSummary(BaseModel):
    """One model (or pipeline) over the recordings it was scored on.

    ``kind``: ``diarization`` (a track: denoiser + diarizer),
    ``recognition`` (a transcript: denoiser + recogniser) or
    ``pipeline``. ``metrics`` maps each metric to its spread.
    """

    model_config = ConfigDict(frozen=True)

    kind: str
    id: str
    recordings: int
    metrics: Dict[str, MetricSpread]


class SummarisedRecording(BaseModel):
    """A recording the summary used, and the run it took the scores of."""

    model_config = ConfigDict(frozen=True)

    session_id: str
    corpus: str
    recording: str
    licence: str
    licence_commercial: bool
    run_id: Optional[str] = None
    duration: float


class ExcludedRecording(BaseModel):
    """An imported session the summary leaves out, and why."""

    model_config = ConfigDict(frozen=True)

    session_id: str
    corpus: str
    recording: str
    reason: str
    speech: Optional[SpeechCheck] = None


class CorpusSummary(BaseModel):
    """Mean and spread of every metric per model over the recordings.

    ``non_commercial`` is true when any recording comes from a
    non-commercial corpus: then the figures are for internal
    evaluation only. ``unscored`` lists imported sessions with no
    scored run yet; ``excluded`` the imported sessions left out (their
    audio failed the speech check), which count nowhere else.
    """

    model_config = ConfigDict(frozen=True)

    corpora: List[str]
    non_commercial: bool
    recordings: List[SummarisedRecording]
    unscored: List[str]
    models: List[ModelSummary]
    excluded: List[ExcludedRecording] = []


class ReferenceSummary:
    """Summarise the reference scores of every imported session."""

    def __init__(
        self, sessions: SessionRepository, repository: LabRepository
    ) -> None:
        """Wire the ports."""
        self._sessions = sessions
        self._repository = repository

    def __call__(self, corpus: Optional[str] = None) -> CorpusSummary:
        """Return the summary over the imported sessions (one corpus).

        Each session counts with its newest ``done`` or ``partial``
        run that has reference scores; one whose audio failed the
        speech check is listed in ``excluded`` instead.
        """
        if corpus is not None:
            corpus_spec(corpus)
        recordings: List[SummarisedRecording] = []
        unscored: List[str] = []
        excluded: List[ExcludedRecording] = []
        values: Dict[tuple[str, str], Dict[str, List[Optional[float]]]] = {}
        for session_id in sorted(self._sessions.ids()):
            reference = self._reference(session_id)
            if reference is None or (
                corpus is not None and reference.corpus != corpus
            ):
                continue
            reason = reference.excluded()
            if reason is not None:
                excluded.append(ExcludedRecording(
                    session_id=session_id, corpus=reference.corpus,
                    recording=reference.recording, reason=reason,
                    speech=reference.speech,
                ))
                continue
            run_id, scored = self._scored(session_id)
            recordings.append(SummarisedRecording(
                session_id=session_id, corpus=reference.corpus,
                recording=reference.recording, licence=reference.licence,
                licence_commercial=reference.licence_commercial,
                run_id=run_id, duration=round(reference.duration, 3),
            ))
            if scored is None:
                unscored.append(session_id)
                continue
            _collect(values, scored)
        models = []
        for (kind, ident), metrics in sorted(values.items()):
            spreads = {name: spread(v) for name, v in metrics.items()}
            models.append(ModelSummary(
                kind=kind, id=ident,
                recordings=max((s.n for s in spreads.values()), default=0),
                metrics=spreads,
            ))
        return CorpusSummary(
            corpora=sorted({r.corpus for r in recordings}),
            non_commercial=any(not r.licence_commercial for r in recordings),
            recordings=recordings, unscored=unscored, models=models,
            excluded=excluded,
        )

    def _reference(self, session_id: str) -> Optional[SessionReference]:
        """Return the session's reference; unreadable ones are skipped."""
        return _readable(self._repository, session_id)

    def _scored(
        self, session_id: str
    ) -> tuple[Optional[str], Optional[Mapping[str, Any]]]:
        """Return the newest scored run's id and its reference block."""
        for run in self._repository.runs(session_id):
            if run.state not in SCORED_STATES:
                continue
            results = self._repository.results(session_id, run.id)
            block = (results or {}).get("reference")
            if block:
                return run.id, block
        return None, None


def _collect(
    values: Dict[tuple[str, str], Dict[str, List[Optional[float]]]],
    scored: Mapping[str, Any],
) -> None:
    """Add one recording's reference rows to ``values``."""
    for row in scored.get("tracks", []):
        bucket = values.setdefault(("diarization", row["track"]), {})
        for name, value in _track_metrics(row).items():
            bucket.setdefault(name, []).append(value)
    for row in scored.get("transcripts", []):
        bucket = values.setdefault(("recognition", row["hypothesis"]), {})
        bucket.setdefault("wer", []).append(row["wer"])
    for row in scored.get("pipelines", []):
        bucket = values.setdefault(("pipeline", row["id"]), {})
        for name in ("der", "jer", "wer"):
            bucket.setdefault(name, []).append(row[name])
