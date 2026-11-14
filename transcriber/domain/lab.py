"""Model lab: the catalogue, stage results, runs and reference labels.

The lab (``docs/transcriber/LAB.md``) runs a recording through every
chosen combination of a denoiser, a diarizer and a recogniser (a
*pipeline*) and compares what they produce. This module holds the
plain data of it, with no I/O:

* :class:`CatalogueEntry` / :class:`Catalogue`: the candidate models of
  the three stages, with licence, gating and the environment
  (``family``) that runs them;
* :class:`StageOutcome`: what one model returned for one audio (a
  denoised WAV, speaker segments or timed words) and what it cost;
* :class:`LabRun`: one request (session × models) with the state of
  each step, so a run can be followed, cancelled and resumed;
* :class:`SpeakerMapping`: how a diarizer's labels were given the
  enrolled people's ids, and by which method;
* :class:`ReferenceLabel`: who really spoke in a stretch, set by hand.

:func:`stage_key` names a stage result by the hash of what it depends
on, so a result is computed once and reused by every run that needs it.
"""

import hashlib
import json
import re
from datetime import datetime
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from transcriber.domain.errors import InvalidInputError
from transcriber.domain.models import SPEAKER_ID_PATTERN, Word

CATALOGUE_SCHEMA = "lab.catalogue.v1"
RESULT_SCHEMA = "lab.result.v1"

MODEL_ID_PATTERN = r"^[a-z0-9][a-z0-9._-]{0,47}$"
"""A catalogue id: also the ``--model`` value of a runner."""

FAMILY_PATTERN = r"^[a-z0-9][a-z0-9-]{0,31}$"
"""A family (one environment, one runner script)."""

LAB_RUN_ID_PATTERN = r"^[0-9a-f]{32}$"
STAGE_KEY_PATTERN = r"^[0-9a-f]{40}$"

PROJECT_FAMILY = "project"
"""The family run in the service's own process (no environment)."""

NO_DENOISER = "none"
"""The denoiser that returns the audio as it is."""

BUILTIN_DIARIZER = "builtin"
"""The service's own diarization (voice windows + grouping)."""

PIPELINE_SEPARATOR = "+"
"""Joins model ids into track, transcript and pipeline ids."""

ANONYMOUS_PREFIX = "~"
"""Prefix of a speaker no enrolled person was found for.

Enrolled ids cannot start with it, so an anonymous label never equals
a person's id.
"""

Stage = Literal["denoise", "diarize", "recognise"]
STAGES: Tuple[Stage, ...] = ("denoise", "diarize", "recognise")

ErrorCode = Literal[
    "failed", "unknown_model", "bad_input", "needs_token", "out_of_memory",
    "unsupported_language", "not_installed", "timeout", "cancelled",
]
"""Why a step failed; the runners' codes plus the service's own."""

StepState = Literal["pending", "running", "done", "failed", "cancelled"]
RunState = Literal[
    "queued", "running", "done", "partial", "failed", "cancelled",
    "interrupted",
]
"""``partial``: finished, some steps failed; ``interrupted``: the
process running it stopped (resume it)."""

MappingMethod = Literal["pipeline", "voiceprint", "builtin_overlap", "none"]
"""How a diarizer's speaker got its id (:class:`SpeakerMapping`)."""

LabelKind = Literal["person", "both", "nobody", "unsure"]

MAX_MODELS_PER_STAGE = 16
MAX_LABELS = 5000
MAX_LABEL_TEXT = 2000


def _seconds(**kwargs: Any) -> Any:
    """A finite, non-negative number of seconds."""
    return Field(ge=0, allow_inf_nan=False, **kwargs)


class CatalogueEntry(BaseModel):
    """One candidate model of one stage.

    ``family`` is the environment and runner script that run it
    (:data:`PROJECT_FAMILY` for the service's own process).
    ``licence_commercial`` is false for weights licensed for
    non-commercial use only: they may be evaluated, and every result
    shows the flag. ``gated`` models need the owner's Hugging Face
    token after accepting their conditions at ``token_url``.
    ``parameters`` are the defaults a run may override. ``lineage``
    groups near-duplicates (same method, other weights) for the
    consensus vote; it defaults to the id. ``revision`` is raised when
    the runner's output for this model changes (a fix): from 2 on it is
    part of the cache key, so results of the old runner are not reused.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    id: str = Field(pattern=MODEL_ID_PATTERN)
    stage: Stage
    family: str = Field(pattern=FAMILY_PATTERN)
    display_name: str = Field(min_length=1)
    source_url: Optional[str] = None
    licence: str = ""
    licence_commercial: bool = True
    gated: bool = False
    token_url: Optional[str] = None
    languages: List[str] = Field(default_factory=list)
    vram_gb: Optional[float] = Field(default=None, ge=0)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    weights: List[str] = Field(default_factory=list)
    weights_mb: Optional[float] = Field(default=None, ge=0)
    notes: str = ""
    lineage: Optional[str] = None
    revision: int = Field(default=1, ge=1)

    @property
    def vote_group(self) -> str:
        """Return the group whose tracks share one consensus vote."""
        return self.lineage or self.id


class Catalogue(BaseModel):
    """Every candidate model, by id."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    schema_: str = Field(default=CATALOGUE_SCHEMA, alias="schema")
    entries: List[CatalogueEntry]

    @model_validator(mode="after")
    def _unique(self) -> "Catalogue":
        """Reject a schema other than v1 and ids given twice."""
        if self.schema_ != CATALOGUE_SCHEMA:
            raise ValueError(f"expected schema {CATALOGUE_SCHEMA}")
        ids = [e.id for e in self.entries]
        if len(set(ids)) != len(ids):
            raise ValueError("catalogue ids must be unique")
        return self

    def get(self, model_id: str) -> Optional[CatalogueEntry]:
        """Return the entry ``model_id``, if there is one."""
        for entry in self.entries:
            if entry.id == model_id:
                return entry
        return None

    def require(self, model_id: str, stage: Stage) -> CatalogueEntry:
        """Return the entry ``model_id`` of ``stage``, or raise."""
        entry = self.get(model_id)
        if entry is None or entry.stage != stage:
            raise InvalidInputError(
                f"No {stage} model {model_id!r} in the catalogue."
            )
        return entry


class Segment(BaseModel):
    """A stretch in which one speaker talks (others may talk too)."""

    model_config = ConfigDict(frozen=True)

    start: float = _seconds()
    end: float = _seconds()
    speaker: str = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def _ordered(self) -> "Segment":
        """Reject a segment that ends before it starts."""
        if self.end < self.start:
            raise ValueError("a segment cannot end before it starts")
        return self


class StageError(BaseModel):
    """Why a step failed: a stable code and a message for the owner."""

    model_config = ConfigDict(frozen=True)

    code: ErrorCode
    message: str = ""


class StageOutcome(BaseModel):
    """What one model produced from one audio, and what it cost.

    Denoise gives ``audio_key`` (the stored audio it wrote); diarize
    gives ``segments`` (overlapping segments of different speakers
    allowed) and, when the model reports it natively, ``overlap``;
    recognise gives ``words`` and ``language``. ``seconds`` is the
    processing time without loading, ``peak_gpu_mb`` the most GPU
    memory the runner's process reserved.
    """

    model_config = ConfigDict(frozen=True)

    stage: Stage
    model: str = Field(pattern=MODEL_ID_PATTERN)
    model_version: Optional[str] = None
    device: Optional[str] = None
    seconds: Optional[float] = Field(default=None, ge=0)
    load_seconds: Optional[float] = Field(default=None, ge=0)
    peak_gpu_mb: Optional[float] = Field(default=None, ge=0)
    audio_seconds: Optional[float] = Field(default=None, ge=0)
    notes: List[str] = Field(default_factory=list)
    audio_key: Optional[str] = None
    segments: List[Segment] = Field(default_factory=list)
    overlap: List[Tuple[float, float]] = Field(default_factory=list)
    words: List[Word] = Field(default_factory=list)
    language: Optional[str] = None


class SpeakerMapping(BaseModel):
    """Who one speaker label of a diarizer is.

    ``speaker`` is an enrolled id, or ``None`` when nobody was found.
    ``method``: ``pipeline`` (the built-in diarization named it
    itself), ``voiceprint`` (its non-overlapped speech embedded with
    the service's voice model and matched to the voiceprints;
    ``similarity`` is the cosine), ``builtin_overlap`` (one-to-one
    overlap in time with the built-in pipeline's people;
    ``overlap_seconds``) or ``none``.
    """

    model_config = ConfigDict(frozen=True)

    label: str = Field(min_length=1, max_length=64)
    speaker: Optional[str] = None
    method: MappingMethod = "none"
    similarity: Optional[float] = None
    overlap_seconds: Optional[float] = None
    speech_seconds: float = _seconds(default=0.0)

    def name(self) -> str:
        """Return the id compared across pipelines.

        An unmatched label becomes ``~<label>``, which never equals an
        enrolled id.
        """
        return self.speaker or f"{ANONYMOUS_PREFIX}{self.label}"


def named_segments(
    segments: Sequence[Segment], mapping: Sequence[SpeakerMapping]
) -> List[Segment]:
    """Return ``segments`` with each label replaced by its mapped name.

    A label missing from ``mapping`` is treated as unmatched.
    """
    names = {m.label: m.name() for m in mapping}
    return [
        s.model_copy(update={
            "speaker": names.get(s.speaker, ANONYMOUS_PREFIX + s.speaker)
        })
        for s in segments
    ]


class Step(BaseModel):
    """One stage of one model on one audio, inside a run.

    ``denoiser`` names the audio the step reads (the output of that
    denoiser; ``none`` is the original). ``key`` is the stage key of
    the result once known; ``cached`` tells it was found, not computed.
    """

    model_config = ConfigDict(frozen=True)

    stage: Stage
    model: str = Field(pattern=MODEL_ID_PATTERN)
    denoiser: str = Field(pattern=MODEL_ID_PATTERN)
    state: StepState = "pending"
    key: Optional[str] = Field(default=None, pattern=STAGE_KEY_PATTERN)
    cached: bool = False
    error: Optional[StageError] = None
    seconds: Optional[float] = Field(default=None, ge=0)

    @property
    def name(self) -> str:
        """Return ``stage:denoiser+model`` (``stage:model`` for denoise)."""
        if self.stage == "denoise":
            return f"denoise:{self.model}"
        return f"{self.stage}:{self.denoiser}{PIPELINE_SEPARATOR}{self.model}"


class LabRequest(BaseModel):
    """The models a run combines, and the options it passes them.

    ``params`` overrides catalogue defaults per model id. ``language``
    (for the recognisers) defaults to the session's; ``num_speakers``
    (for the diarizers) to the session's.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    denoisers: List[str] = Field(
        default_factory=lambda: [NO_DENOISER],
        min_length=1, max_length=MAX_MODELS_PER_STAGE,
    )
    diarizers: List[str] = Field(
        default_factory=lambda: [BUILTIN_DIARIZER],
        max_length=MAX_MODELS_PER_STAGE,
    )
    recognisers: List[str] = Field(
        default_factory=list, max_length=MAX_MODELS_PER_STAGE
    )
    params: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    language: Optional[str] = Field(default=None, max_length=16)
    num_speakers: Optional[int] = Field(default=None, ge=1, le=16)

    @model_validator(mode="after")
    def _distinct(self) -> "LabRequest":
        """Reject a model chosen twice, an invalid id or an empty run."""
        for kind in (self.denoisers, self.diarizers, self.recognisers):
            if len(set(kind)) != len(kind):
                raise ValueError("a model is chosen twice")
            for model_id in kind:
                if not re.fullmatch(MODEL_ID_PATTERN, model_id):
                    raise ValueError(f"invalid model id {model_id!r}")
        if not self.diarizers and not self.recognisers:
            raise ValueError("choose at least one diarizer or recogniser")
        for model_id, values in self.params.items():
            if not re.fullmatch(MODEL_ID_PATTERN, model_id):
                raise ValueError(f"invalid model id {model_id!r}")
            if len(json.dumps(values, default=str)) > 2000:
                raise ValueError("parameters too long")
        return self


class LabRun(BaseModel):
    """One run of the lab on a session, with the state of each step.

    ``worker_pid`` is the process that queued or runs it: a run left
    ``queued`` or ``running`` by a process that is gone was interrupted.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(pattern=LAB_RUN_ID_PATTERN)
    session_id: str
    created_at: AwareDatetime
    finished_at: Optional[AwareDatetime] = None
    request: LabRequest
    state: RunState = "queued"
    steps: List[Step] = Field(default_factory=list)
    message: Optional[str] = None
    worker_pid: Optional[int] = Field(default=None, ge=0)

    def step(self, name: str) -> Optional[Step]:
        """Return the step called ``name`` (see :attr:`Step.name`)."""
        for step in self.steps:
            if step.name == name:
                return step
        return None

    def with_step(self, step: Step) -> "LabRun":
        """Return the run with the step of the same name replaced."""
        return self.model_copy(update={"steps": [
            step if s.name == step.name else s for s in self.steps
        ]})

    def pipelines(self) -> List["PipelineRef"]:
        """Return every denoiser × diarizer × recogniser of the run.

        Without diarizers a pipeline has no speakers (``diarizer``
        ``None``); without recognisers it has no words.
        """
        r = self.request
        diarizers: List[Optional[str]] = list(r.diarizers) or [None]
        recognisers: List[Optional[str]] = list(r.recognisers) or [None]
        return [
            PipelineRef(denoiser=d, diarizer=z, recogniser=w)
            for d in r.denoisers for z in diarizers for w in recognisers
        ]


class PipelineRef(BaseModel):
    """One combination of a run: what it is called and what it reads."""

    model_config = ConfigDict(frozen=True)

    denoiser: str
    diarizer: Optional[str] = None
    recogniser: Optional[str] = None

    @property
    def id(self) -> str:
        """Return ``denoiser+diarizer+recogniser`` (``-`` for none)."""
        return PIPELINE_SEPARATOR.join(
            (self.denoiser, self.diarizer or "-", self.recogniser or "-")
        )

    @property
    def track(self) -> Optional[str]:
        """Return the diarization track id, ``denoiser+diarizer``."""
        if self.diarizer is None:
            return None
        return track_id(self.denoiser, self.diarizer)

    @property
    def transcript(self) -> Optional[str]:
        """Return the transcript id, ``denoiser+recogniser``."""
        if self.recogniser is None:
            return None
        return track_id(self.denoiser, self.recogniser)


def track_id(denoiser: str, model: str) -> str:
    """Return the id of what ``model`` made of ``denoiser``'s audio."""
    return f"{denoiser}{PIPELINE_SEPARATOR}{model}"


def stage_key(
    audio_key: str,
    stage: Stage,
    model: str,
    params: Mapping[str, Any],
    extra: Optional[Mapping[str, Any]] = None,
) -> str:
    """Return the cache key of a stage result (40 hex characters).

    It hashes the audio read (``audio_key``: the original's SHA-256 or
    the key of the denoise result that wrote it), the stage, the model,
    its parameters and anything else the result depends on
    (``extra``, e.g. the voiceprints a built-in diarization names
    people with), so equal inputs give the same key in every run.
    """
    document = json.dumps(
        {
            "contract": RESULT_SCHEMA, "audio": audio_key, "stage": stage,
            "model": model, "params": dict(params),
            "extra": dict(extra or {}),
        },
        sort_keys=True, separators=(",", ":"), default=str,
    )
    return hashlib.sha1(document.encode("utf-8")).hexdigest()


def merged_params(
    entry: CatalogueEntry, overrides: Mapping[str, Any],
    standard: Mapping[str, Any],
) -> Dict[str, Any]:
    """Return the parameters a model runs with.

    The catalogue defaults, then the standard values the service
    passes (``language``, ``num_speakers``) where the model declares
    them, then the run's overrides of declared parameters. An override
    of an undeclared parameter is an :class:`InvalidInputError`.
    """
    unknown = sorted(set(overrides) - set(entry.parameters))
    if unknown:
        raise InvalidInputError(
            f"Model {entry.id} has no parameter {', '.join(unknown)}."
        )
    values = dict(entry.parameters)
    for key, value in standard.items():
        if key in values and value is not None:
            values[key] = value
    values.update(overrides)
    return values


class ReferenceLabel(BaseModel):
    """Who really spoke from ``start`` to ``end``, set by the owner.

    ``kind``: ``person`` (one enrolled id in ``speakers``), ``both``
    (two or more ids speaking at once), ``nobody`` or ``unsure``
    (ignored by the scoring). ``text`` is the correct text, when given.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    start: float = _seconds()
    end: float = _seconds()
    kind: LabelKind
    speakers: List[str] = Field(default_factory=list, max_length=8)
    text: Optional[str] = Field(default=None, max_length=MAX_LABEL_TEXT)
    updated_at: Optional[datetime] = None

    @model_validator(mode="after")
    def _consistent(self) -> "ReferenceLabel":
        """Check the times and that the speakers fit the kind."""
        if self.end <= self.start:
            raise ValueError("a label must end after it starts")
        for speaker in self.speakers:
            if not re.fullmatch(SPEAKER_ID_PATTERN, speaker):
                raise ValueError(f"invalid speaker id {speaker!r}")
        if len(set(self.speakers)) != len(self.speakers):
            raise ValueError("a speaker is listed twice")
        wanted = {"person": (1, 1), "both": (2, 8), "nobody": (0, 0),
                  "unsure": (0, 8)}[self.kind]
        if not wanted[0] <= len(self.speakers) <= wanted[1]:
            raise ValueError(
                f"a {self.kind!r} label needs {wanted[0]} to {wanted[1]}"
                " speakers"
            )
        return self

    def reference(self) -> Optional[Tuple[str, ...]]:
        """Return the people speaking, or ``None`` when unsure."""
        if self.kind == "unsure":
            return None
        return tuple(sorted(self.speakers))


def check_labels(
    labels: Sequence[ReferenceLabel], duration: float,
    enrolled: Sequence[str],
) -> List[ReferenceLabel]:
    """Return ``labels`` in time order after checking them as a set.

    Labels may not overlap one another, must lie within the recording
    (plus half a second) and may name only ``enrolled`` people. At most
    :data:`MAX_LABELS`. Raises :class:`InvalidInputError`.
    """
    if len(labels) > MAX_LABELS:
        raise InvalidInputError(f"At most {MAX_LABELS} labels.")
    ordered = sorted(labels, key=lambda label: label.start)
    known = set(enrolled)
    for before, after in zip(ordered, ordered[1:]):
        if after.start < before.end - 1e-6:
            raise InvalidInputError(
                f"Labels overlap at {after.start:.2f} s."
            )
    for label in ordered:
        if label.end > duration + 0.5:
            raise InvalidInputError(
                f"A label ends at {label.end:.2f} s, after the recording."
            )
        strangers = sorted(set(label.speakers) - known)
        if strangers:
            raise InvalidInputError(
                f"Not enrolled: {', '.join(strangers)}."
            )
    return ordered
