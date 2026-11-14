"""Public reference corpora for the model lab, and scoring against them.

A lab session imported from a labelled public corpus (AMI, CALLHOME:
``docs/transcriber/LAB.md`` §13) carries a :class:`SessionReference`:
who spoke when according to the corpus (speaker turns, overlap kept,
speakers named by their anonymous corpus ids) and, when the corpus has
it, the words said with their times. The lab's comparison
(:mod:`transcriber.domain.comparison`) then measures every pipeline
against it as against one more track, so its numbers are real error
rates rather than agreement between models:

* :func:`score_track`: DER (missed, false alarm, confusion, with the
  comparison's collar and the best one-to-one mapping of the labels,
  :func:`~transcriber.domain.comparison.compare_tracks`), JER, overlap
  detection precision and recall and the speaker-count error;
* WER against the reference words is
  :func:`~transcriber.domain.comparison.compare_transcripts` with the
  reference as the reference transcript (the lab's normalisation);
* :func:`spread`: mean and spread of one metric over recordings.

This module holds plain data and arithmetic, with no I/O.
"""

import math
import re
import statistics
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from transcriber.domain.comparison import (
    ComparisonParameters,
    TranscriptComparison,
    Track,
    collar_mask,
    compare_tracks,
    frame_count,
    jaccard_error,
    local_names,
    summarise_track,
    to_frames,
)
from transcriber.domain.errors import InvalidInputError
from transcriber.domain.lab import Segment
from transcriber.domain.models import SESSION_ID_PATTERN, Word
from transcriber.domain.speech_check import SpeechCheck

REFERENCE_SCHEMA = "lab.reference.v1"
REFERENCE_TRACK = "reference"
"""Id of the reference in comparisons (no track id has no ``+``)."""

SPEAKER_SECONDS = 1.0
"""A speaker counts for the speaker count with this much speech."""


class CorpusSpec(BaseModel):
    """A public corpus the lab can import (one config of a dataset).

    ``repo`` and ``config`` locate it on the Hugging Face Hub (the
    shards are ``<config>/<split>-*.parquet``); ``licence`` and
    ``licence_commercial`` travel with every session imported from it;
    a ``gated`` corpus needs its terms accepted at ``source_url`` by
    the account whose token is used.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    display_name: str
    repo: str
    config: str
    splits: Tuple[str, ...]
    default_split: str
    licence: str
    licence_commercial: bool
    source_url: str
    language: str
    gated: bool = False


CORPORA: Dict[str, CorpusSpec] = {
    "ami": CorpusSpec(
        id="ami", display_name="AMI (single distant microphone)",
        repo="diarizers-community/ami", config="sdm",
        splits=("train", "validation", "test"), default_split="test",
        licence="CC BY 4.0", licence_commercial=True,
        source_url="https://huggingface.co/datasets/diarizers-community/ami",
        language="en",
    ),
    "callhome": CorpusSpec(
        id="callhome", display_name="CALLHOME Spanish",
        repo="diarizers-community/callhome", config="spa",
        splits=("data",), default_split="data",
        licence="CC BY-NC-SA 4.0", licence_commercial=False,
        source_url=(
            "https://huggingface.co/datasets/diarizers-community/callhome"
        ),
        language="es", gated=True,
    ),
}
"""The corpora the importer knows (read from the dataset cards)."""


def corpus_spec(name: str) -> CorpusSpec:
    """Return the corpus called ``name`` or raise ``InvalidInputError``."""
    spec = CORPORA.get(name)
    if spec is None:
        raise InvalidInputError(
            f"Unknown corpus {name!r}; known: {', '.join(sorted(CORPORA))}."
        )
    return spec


def check_split(spec: CorpusSpec, split: Optional[str]) -> str:
    """Return ``split`` (the corpus default if ``None``) if it exists."""
    chosen = split or spec.default_split
    if chosen not in spec.splits:
        raise InvalidInputError(
            f"{spec.id} has no split {chosen!r}; it has"
            f" {', '.join(spec.splits)}."
        )
    return chosen


_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def recording_session_id(corpus: str, recording: str) -> str:
    """Return the session id of a corpus recording: ``<corpus>-<id>``.

    Characters a session id may not hold become ``-``; the result is
    cut to 64 characters and checked.
    """
    ident = _UNSAFE.sub("-", f"{corpus}-{recording}").strip("-.")[:64]
    if not re.fullmatch(SESSION_ID_PATTERN, ident):
        raise InvalidInputError(f"No session id for recording {recording!r}.")
    return ident


def recording_name(path: Optional[str], fallback: str) -> str:
    """Return a recording's id from its file name (``ES2004a.Array1.wav``).

    The id is the name before its first dot; ``fallback`` when there is
    no name.
    """
    name = (path or "").replace("\\", "/").rsplit("/", 1)[-1]
    stem = name.split(".", 1)[0].strip()
    return stem or fallback


def reference_segments(
    starts: Sequence[float],
    ends: Sequence[float],
    speakers: Sequence[str],
    duration: float,
) -> List[Segment]:
    """Return a corpus's speaker turns as segments, overlap kept.

    The three columns must have one value per turn. A turn that starts
    at or after the end of the audio is dropped and one that runs past
    it is cut (annotations may outlast the file by a few frames); empty
    turns are dropped. Sorted by start, then speaker.
    """
    if not len(starts) == len(ends) == len(speakers):
        raise InvalidInputError(
            "The reference has columns of different lengths"
            f" ({len(starts)} starts, {len(ends)} ends,"
            f" {len(speakers)} speakers)."
        )
    out: List[Segment] = []
    for start, end, speaker in zip(starts, ends, speakers):
        a, b = float(start), float(end)
        if not (math.isfinite(a) and math.isfinite(b)) or a < 0 or b < a:
            raise InvalidInputError(
                f"Invalid reference turn from {a:g} s to {b:g} s."
            )
        name = str(speaker).strip()
        if not name:
            raise InvalidInputError("A reference turn has no speaker.")
        b = min(b, duration)
        if a >= b:
            continue
        out.append(Segment(start=round(a, 3), end=round(b, 3), speaker=name))
    out.sort(key=lambda s: (s.start, s.speaker, s.end))
    return out


class SessionReference(BaseModel):
    """The human reference of an imported session (see the module).

    ``segments`` are the corpus's speaker turns (overlap kept);
    ``words`` the reference transcript with times, ``None`` when the
    corpus has none. ``licence``, ``licence_commercial`` and
    ``source_url`` say under what terms the recording may be used.
    ``speech`` holds the figures of the speech check
    (:mod:`~transcriber.domain.speech_check`), ``None`` when the
    session was imported before the check existed and not checked
    since; a recording that failed it is not used (:meth:`excluded`).
    """

    model_config = ConfigDict(frozen=True)

    schema_name: str = REFERENCE_SCHEMA
    corpus: str = Field(min_length=1)
    recording: str = Field(min_length=1)
    split: str = Field(min_length=1)
    licence: str = Field(min_length=1)
    licence_commercial: bool
    source_url: str = Field(min_length=1)
    language: str = Field(min_length=2)
    duration: float = Field(gt=0, allow_inf_nan=False)
    segments: List[Segment]
    words: Optional[List[Word]] = None
    imported_at: Optional[AwareDatetime] = None
    speech: Optional[SpeechCheck] = None

    def excluded(self) -> Optional[str]:
        """Return why the recording is not used (``None``: it is)."""
        return None if self.speech is None else self.speech.reason()

    def speakers(self) -> List[str]:
        """Return the corpus ids of the people who speak, sorted."""
        return sorted({s.speaker for s in self.segments})

    def track(self) -> Track:
        """Return the reference as a comparison track."""
        return Track(REFERENCE_TRACK, self.segments, REFERENCE_TRACK)


def make_reference(
    spec: CorpusSpec,
    split: str,
    recording: str,
    duration: float,
    segments: Sequence[Segment],
    words: Optional[Sequence[Word]],
    imported_at: Optional[datetime] = None,
    speech: Optional[SpeechCheck] = None,
) -> SessionReference:
    """Return the reference of one recording of ``spec``.

    Words past the end of the audio are dropped; a recording with no
    speaker turn is refused (there would be nothing to score).
    """
    if not segments:
        raise InvalidInputError(
            f"Recording {recording!r} has no speaker turn to score."
        )
    kept = None if words is None else [
        w for w in words if w.start < duration
    ]
    return SessionReference(
        corpus=spec.id, recording=recording, split=split,
        licence=spec.licence, licence_commercial=spec.licence_commercial,
        source_url=spec.source_url, language=spec.language,
        duration=duration, segments=list(segments), words=kept,
        imported_at=imported_at, speech=speech,
    )


class ReferenceTrackScore(BaseModel):
    """One diarization measured against the reference (seconds).

    ``reference_speech`` is speaker time outside the collar of the
    reference's boundaries (a frame of two people counts twice);
    ``missed``, ``false_alarm`` and ``confusion`` are measured there
    after the best one-to-one mapping of the labels, and ``der`` is
    their sum over ``reference_speech``. ``jer``: Jaccard error rate on
    the same frames. Overlap (two or more speakers) is compared on every
    frame: ``overlap_precision`` = time both mark over the time the
    track marks, ``overlap_recall`` = over the time the reference
    marks. ``speaker_count_error`` = the track's speakers minus the
    reference's (each with at least one second of speech).
    """

    model_config = ConfigDict(frozen=True)

    track: str
    reference_speech: float = 0.0
    missed: float = 0.0
    false_alarm: float = 0.0
    confusion: float = 0.0
    der: Optional[float] = None
    jer: Optional[float] = None
    overlap_reference_seconds: float = 0.0
    overlap_hypothesis_seconds: float = 0.0
    overlap_both_seconds: float = 0.0
    overlap_precision: Optional[float] = None
    overlap_recall: Optional[float] = None
    reference_speakers: int = 0
    hypothesis_speakers: int = 0
    speaker_count_error: int = 0


def _round(value: Optional[float], digits: int = 4) -> Optional[float]:
    """Round a number that may be missing."""
    return None if value is None else round(value, digits)


def score_track(
    reference: SessionReference, hypothesis: Track,
    p: ComparisonParameters = ComparisonParameters(),
) -> ReferenceTrackScore:
    """Measure ``hypothesis`` against the reference (see the class)."""
    duration = reference.duration
    truth = reference.track()
    compared = compare_tracks(truth, hypothesis, duration, p)
    frames = frame_count(duration, p.frame)
    ref = to_frames(local_names(truth), frames, p.frame)
    hyp = to_frames(local_names(hypothesis), frames, p.frame)
    scored = ~collar_mask(truth.segments, frames, p.frame, p.collar)
    jer = jaccard_error(ref.active, hyp.active, scored)
    overlap = compared.overlap
    heard = summarise_track(truth, duration, p)
    found = summarise_track(hypothesis, duration, p)
    precision = (
        overlap.both_seconds / overlap.hypothesis_seconds
        if overlap.hypothesis_seconds > 0 else None
    )
    recall = (
        overlap.both_seconds / overlap.reference_seconds
        if overlap.reference_seconds > 0 else None
    )
    return ReferenceTrackScore(
        track=hypothesis.id,
        reference_speech=round(compared.reference_speech, 3),
        missed=round(compared.missed, 3),
        false_alarm=round(compared.false_alarm, 3),
        confusion=round(compared.confusion, 3),
        der=_round(compared.error_rate), jer=_round(jer),
        overlap_reference_seconds=round(overlap.reference_seconds, 3),
        overlap_hypothesis_seconds=round(overlap.hypothesis_seconds, 3),
        overlap_both_seconds=round(overlap.both_seconds, 3),
        overlap_precision=_round(precision), overlap_recall=_round(recall),
        reference_speakers=len(heard.speakers),
        hypothesis_speakers=len(found.speakers),
        speaker_count_error=len(found.speakers) - len(heard.speakers),
    )


class ReferenceTrackRow(ReferenceTrackScore):
    """A track's score with the models that made the track."""

    denoiser: str
    diarizer: str


class ReferenceTranscriptRow(TranscriptComparison):
    """A transcript's WER against the reference words, with its models."""

    denoiser: str
    recogniser: str


class ReferencePipelineRow(BaseModel):
    """A pipeline's main reference figures (its track's and transcript's).

    ``licence_commercial`` is false when one of its models or the
    corpus is non-commercial.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    track: Optional[str] = None
    transcript: Optional[str] = None
    der: Optional[float] = None
    jer: Optional[float] = None
    overlap_precision: Optional[float] = None
    overlap_recall: Optional[float] = None
    speaker_count_error: Optional[int] = None
    wer: Optional[float] = None
    licence_commercial: bool = True


class ReferenceResults(BaseModel):
    """Every pipeline of a run measured against the session's reference.

    The corpus fields repeat the session's reference;
    ``licence_commercial`` false flags a non-commercial corpus
    (internal evaluation only). ``collar`` is the comparison's.
    ``reference_tokens`` is ``None`` when the corpus has no transcript
    (then no WER is given).
    """

    model_config = ConfigDict(frozen=True)

    corpus: str
    recording: str
    split: str
    licence: str
    licence_commercial: bool
    source_url: str
    collar: float
    speakers: List[str]
    speech_seconds: float
    overlap_seconds: float
    reference_tokens: Optional[int] = None
    tracks: List[ReferenceTrackRow]
    transcripts: List[ReferenceTranscriptRow]
    pipelines: List[ReferencePipelineRow]


class MetricSpread(BaseModel):
    """One metric over ``n`` recordings: mean, sample SD, range.

    ``sd`` needs two recordings; every field but ``n`` is ``None``
    when no recording gave a value.
    """

    model_config = ConfigDict(frozen=True)

    n: int = 0
    mean: Optional[float] = None
    sd: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None


def spread(values: Sequence[Optional[float]]) -> MetricSpread:
    """Return the mean and spread of the values given (``None`` skipped)."""
    present = [float(v) for v in values if v is not None]
    if not present:
        return MetricSpread()
    return MetricSpread(
        n=len(present),
        mean=round(statistics.fmean(present), 4),
        sd=round(statistics.stdev(present), 4) if len(present) > 1 else None,
        min=round(min(present), 4), max=round(max(present), 4),
    )


def share(part: float, whole: float) -> Optional[float]:
    """Return ``part / whole`` (``None`` when ``whole`` is not positive)."""
    return part / whole if whole > 0 else None
