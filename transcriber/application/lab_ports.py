"""Ports of the model lab's use cases (``docs/transcriber/LAB.md``).

The lab runs models the service does not import: each family of
models lives in its own environment and is started as a subprocess by
an outbound adapter (:class:`LabTool`); the service's own models (the
built-in diarization, the project's Whisper) run in process behind the
same port. Stage results, runs, speaker mappings, comparison results
and reference labels are stored by a :class:`LabRepository` under the
data directory, never in the repository. A :class:`ReferenceCorpus`
gives the recordings of a public labelled corpus to import.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
)

from transcriber.application.ports import Audio, PlaybackAudio, RecordedAudio
from transcriber.domain.lab import (
    Catalogue,
    CatalogueEntry,
    LabRun,
    ReferenceLabel,
    SpeakerMapping,
    Stage,
    StageOutcome,
)
from transcriber.domain.lab_reference import CorpusSpec, SessionReference
from transcriber.domain.models import Word


@dataclass(frozen=True)
class ToolRequest:
    """One stage of one model on one audio.

    ``audio`` is the WAV to read (16 kHz mono); ``load`` decodes it for
    an in-process tool; ``workspace`` is an empty directory the tool
    writes into (a denoiser writes ``audio.wav`` there). ``inputs``
    holds results the stage depends on (the built-in diarization reads
    the project recogniser's words, ``inputs["words"]``). ``anonymous``
    asks an in-process tool to compare the voices with no enrolled
    person (a session imported from a public corpus).
    """

    entry: CatalogueEntry
    stage: Stage
    params: Mapping[str, Any]
    audio: Path
    workspace: Path
    load: Callable[[], Audio]
    inputs: Mapping[str, StageOutcome] = field(default_factory=dict)
    anonymous: bool = False


class LabTool(Protocol):
    """Run one stage of one model; raise ``LabStepError`` on failure."""

    def run(
        self, request: ToolRequest, cancelled: Callable[[], bool]
    ) -> StageOutcome:
        """Return what the model produced.

        ``cancelled`` is polled while the stage runs; once it is true
        the stage stops and ``LabStepError("cancelled")`` is raised.
        """
        ...

    def release(self) -> None:
        """Free any model held in this process (GPU memory)."""
        ...


class CatalogueSource(Protocol):
    """Where the catalogue of candidate models is read from."""

    def load(self) -> Catalogue:
        """Return the catalogue."""
        ...


class LabEnvironment(Protocol):
    """What is installed on this machine for the lab."""

    def family_problem(self, family: str) -> Optional[str]:
        """Return why ``family``'s environment cannot run, or ``None``."""
        ...

    def token_present(self) -> bool:
        """Tell whether a Hugging Face token is available."""
        ...

    def weights_cached(self, weight: str) -> Optional[bool]:
        """Tell whether ``weight`` (a repo id) is downloaded.

        ``None`` when it cannot be told (a URL, another cache).
        """
        ...


class ProcessTable(Protocol):
    """Whether the process that runs a lab run is still alive."""

    def current(self) -> int:
        """Return this process's id."""
        ...

    def alive(self, pid: int) -> bool:
        """Tell whether process ``pid`` exists."""
        ...


class LabRepository(Protocol):
    """Stage results, runs, mappings, results and labels of a session.

    Every session id and key is checked by the adapter before a path
    is made from it.
    """

    def store_original(self, session_id: str, audio: Audio) -> str:
        """Store the session's audio as 16 kHz mono WAV; return its key.

        The key is the SHA-256 of the samples written; writing the same
        audio again keeps the file.
        """
        ...

    def original_key(self, session_id: str) -> Optional[str]:
        """Return the key of the stored original audio, if stored."""
        ...

    def audio_path(self, session_id: str, audio_key: str) -> Path:
        """Return the WAV of ``audio_key`` (NotFoundError if absent)."""
        ...

    def load_audio(self, session_id: str, audio_key: str) -> Audio:
        """Return the samples of ``audio_key``."""
        ...

    def playback(self, session_id: str, audio_key: str) -> PlaybackAudio:
        """Return ``audio_key``'s WAV for the player, by byte ranges."""
        ...

    def new_workspace(self, session_id: str) -> Path:
        """Return a new, empty directory for one stage to write into."""
        ...

    def discard(self, session_id: str, workspace: Path) -> None:
        """Remove a workspace that is not kept."""
        ...

    def keep(
        self, session_id: str, key: str, outcome: StageOutcome,
        workspace: Path,
    ) -> StageOutcome:
        """Store ``outcome`` (and the workspace's files) under ``key``.

        A denoise outcome's ``audio.wav`` becomes the audio of ``key``;
        the outcome returned carries ``audio_key``.
        """
        ...

    def outcome(self, session_id: str, key: str) -> Optional[StageOutcome]:
        """Return the stored outcome of ``key``, if any."""
        ...

    def mapping(
        self, session_id: str, key: str
    ) -> Optional[List[SpeakerMapping]]:
        """Return the stored speaker mapping ``key``, if any."""
        ...

    def save_mapping(
        self, session_id: str, key: str, mapping: List[SpeakerMapping]
    ) -> None:
        """Store a speaker mapping."""
        ...

    def save_run(self, run: LabRun) -> None:
        """Store (create or replace) a run."""
        ...

    def run(self, session_id: str, run_id: str) -> Optional[LabRun]:
        """Return the run, if it exists."""
        ...

    def runs(self, session_id: str) -> List[LabRun]:
        """Return the session's runs, newest first."""
        ...

    def request_cancel(self, session_id: str, run_id: str) -> None:
        """Ask the process running ``run_id`` to stop."""
        ...

    def cancel_requested(self, session_id: str, run_id: str) -> bool:
        """Tell whether the run was asked to stop."""
        ...

    def clear_cancel(self, session_id: str, run_id: str) -> None:
        """Forget a request to stop (before the run is resumed)."""
        ...

    def save_results(
        self, session_id: str, run_id: str, results: Dict[str, Any]
    ) -> None:
        """Store the comparison of a run (JSON-ready)."""
        ...

    def results(
        self, session_id: str, run_id: str
    ) -> Optional[Dict[str, Any]]:
        """Return the stored comparison of a run, if any."""
        ...

    def labels(self, session_id: str) -> List[ReferenceLabel]:
        """Return the session's reference labels (empty if none)."""
        ...

    def save_labels(
        self, session_id: str, labels: List[ReferenceLabel]
    ) -> None:
        """Replace the session's reference labels."""
        ...

    def reference(self, session_id: str) -> Optional[SessionReference]:
        """Return the session's corpus reference, if it was imported."""
        ...

    def save_reference(
        self, session_id: str, reference: SessionReference
    ) -> None:
        """Store the corpus reference of an imported session."""
        ...

    def delete_session(self, session_id: str) -> None:
        """Remove everything stored for the session (nothing: no error)."""
        ...


@dataclass(frozen=True)
class CorpusRecording:
    """One recording of a public corpus, as the corpus gives it.

    ``recording`` is the corpus's id of it; ``audio`` its file (any
    format the decoder reads, at the corpus's rate); ``starts``,
    ``ends`` and ``speakers`` the reference's speaker turns, one value
    per turn; ``words`` the reference transcript with times, ``None``
    when the corpus has none.
    """

    recording: str
    audio: RecordedAudio
    starts: Sequence[float]
    ends: Sequence[float]
    speakers: Sequence[str]
    words: Optional[Sequence[Word]] = None


class ReferenceCorpus(Protocol):
    """Where the recordings of a public corpus come from.

    The adapter downloads only the shards it reads. A gated corpus
    whose terms the account has not accepted raises
    :class:`~transcriber.domain.errors.CorpusAccessError`.
    """

    def recordings(
        self, spec: CorpusSpec, split: str, limit: Optional[int]
    ) -> Iterator[CorpusRecording]:
        """Yield the recordings of ``split`` in corpus order.

        At most ``limit`` (every recording when ``None``).
        """
        ...
