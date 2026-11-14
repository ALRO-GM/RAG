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
    an earlier import stopped before storing the reference. The numbers
    describe the reference (aggregates only, never text):
    ``speaker_seconds`` sums the turns (two people at once count twice).
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
        """Import up to ``limit`` recordings of ``split`` of ``corpus``.

        A recording whose session already exists is skipped, so an
        interrupted import can be run again. ``report`` hears about
        each recording as soon as it is done.
        """
        if limit is not None and limit < 1:
            raise InvalidInputError("--limit must be 1 or more.")
        spec = corpus_spec(corpus)
        chosen = check_split(spec, split)
        done: List[ImportedRecording] = []
        for item in self._corpus.recordings(spec, chosen, limit):
            result = self._one(spec, chosen, item)
            done.append(result)
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
        """
        session_id = recording_session_id(spec.id, item.recording)
        title = f"{spec.display_name} {item.recording}"
        found = self._sessions.get(session_id)
        if found is not None and (
            self._repository.reference(session_id) is not None
        ):
            return ImportedRecording(
                session_id=session_id, recording=item.recording,
                state="exists",
            )
        if found is not None and found.title != title:
            raise InvalidInputError(
                f"Session {session_id} exists and was not imported from"
                f" {spec.display_name}."
            )
        duration = self._decoder.decode(item.audio).duration
        if duration <= 0:
            raise InvalidInputError(
                f"Recording {item.recording!r} has no audio."
            )
        reference = make_reference(
            spec, split, item.recording, duration,
            reference_segments(item.starts, item.ends, item.speakers,
                               duration),
            item.words, self._clock.now(),
        )
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
    )


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


class CorpusSummary(BaseModel):
    """Mean and spread of every metric per model over the recordings.

    ``non_commercial`` is true when any recording comes from a
    non-commercial corpus: then the figures are for internal
    evaluation only. ``unscored`` lists imported sessions with no
    scored run yet.
    """

    model_config = ConfigDict(frozen=True)

    corpora: List[str]
    non_commercial: bool
    recordings: List[SummarisedRecording]
    unscored: List[str]
    models: List[ModelSummary]


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
        run that has reference scores.
        """
        if corpus is not None:
            corpus_spec(corpus)
        recordings: List[SummarisedRecording] = []
        unscored: List[str] = []
        values: Dict[tuple[str, str], Dict[str, List[Optional[float]]]] = {}
        for session_id in sorted(self._sessions.ids()):
            reference = self._reference(session_id)
            if reference is None or (
                corpus is not None and reference.corpus != corpus
            ):
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
        )

    def _reference(self, session_id: str) -> Optional[SessionReference]:
        """Return the session's reference; unreadable ones are skipped."""
        try:
            return self._repository.reference(session_id)
        except (StorageError, InvalidInputError):
            return None

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
