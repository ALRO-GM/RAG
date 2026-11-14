"""In-memory fakes of the lab ports (application.lab_ports).

:class:`FakeLabRepository` keeps everything in dictionaries;
:class:`ScriptedTool` answers each model with a function of the
request, counting its calls; :class:`FakeEnvironment` and
:class:`FakeProcesses` say what a test wants about the machine.
"""

import hashlib
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Set,
)

from tests.unit.transcriber.fakes import BytesPlayback
from transcriber.application.lab_ports import ToolRequest
from transcriber.application.ports import Audio
from transcriber.domain.errors import (
    InvalidInputError,
    LabStepError,
    NotFoundError,
)
from transcriber.domain.lab import (
    Catalogue,
    CatalogueEntry,
    LabRun,
    ReferenceLabel,
    Segment,
    SpeakerMapping,
    StageOutcome,
)
from transcriber.domain.lab_reference import SessionReference
from transcriber.domain.models import Word


class FakeLabRepository:
    """The lab's storage in dictionaries."""

    def __init__(self) -> None:
        """Start empty."""
        self.audio: Dict[str, Audio] = {}
        self.originals: Dict[str, str] = {}
        self.outcomes: Dict[str, StageOutcome] = {}
        self.mappings: Dict[str, List[SpeakerMapping]] = {}
        self.stored_runs: Dict[str, LabRun] = {}
        self.cancels: Set[str] = set()
        self.stored_results: Dict[str, Dict[str, Any]] = {}
        self.stored_labels: Dict[str, List[ReferenceLabel]] = {}
        self.references: Dict[str, SessionReference] = {}
        self.written: Dict[Path, Audio] = {}
        self.discarded: List[Path] = []
        self.workspaces = 0

    def store_original(self, session_id: str, audio: Audio) -> str:
        """Keep the audio under the hash of its samples."""
        key = hashlib.sha256(audio.samples.tobytes()).hexdigest()
        self.audio[key] = audio
        self.originals[session_id] = key
        return key

    def original_key(self, session_id: str) -> Optional[str]:
        """Return the original's key, if stored."""
        return self.originals.get(session_id)

    def audio_path(self, session_id: str, audio_key: str) -> Path:
        """Return a made-up path of a stored audio."""
        if audio_key not in self.audio:
            raise NotFoundError("No such lab audio.")
        return Path(f"/lab/{audio_key}.wav")

    def load_audio(self, session_id: str, audio_key: str) -> Audio:
        """Return a stored audio."""
        return self.audio[audio_key]

    def playback(self, session_id: str, audio_key: str) -> BytesPlayback:
        """Return the audio's samples as bytes."""
        if audio_key not in self.audio:
            raise NotFoundError("No such lab audio.")
        return BytesPlayback(self.audio[audio_key].samples.tobytes())

    def excerpt(
        self, session_id: str, audio_key: str, start: float, seconds: float
    ) -> BytesPlayback:
        """Return the excerpt's samples as bytes (no real WAV header)."""
        if audio_key not in self.audio:
            raise NotFoundError("No such lab audio.")
        audio = self.audio[audio_key]
        first = int(round(start * audio.sample_rate))
        if first >= len(audio.samples):
            raise InvalidInputError(
                "The excerpt starts after the end of the audio."
            )
        last = first + max(1, int(round(seconds * audio.sample_rate)))
        return BytesPlayback(audio.samples[first:last].tobytes())

    def new_workspace(self, session_id: str) -> Path:
        """Return a new made-up directory."""
        self.workspaces += 1
        return Path(f"/work/{self.workspaces}")

    def discard(self, session_id: str, workspace: Path) -> None:
        """Remember the discarded workspace."""
        self.discarded.append(workspace)

    def keep(
        self, session_id: str, key: str, outcome: StageOutcome,
        workspace: Path,
    ) -> StageOutcome:
        """Keep the outcome; a denoiser's audio becomes audio ``key``."""
        if outcome.stage == "denoise":
            self.audio[key] = self.written.pop(workspace)
            outcome = outcome.model_copy(update={"audio_key": key})
        self.outcomes[key] = outcome
        return outcome

    def outcome(self, session_id: str, key: str) -> Optional[StageOutcome]:
        """Return a kept outcome."""
        return self.outcomes.get(key)

    def mapping(
        self, session_id: str, key: str
    ) -> Optional[List[SpeakerMapping]]:
        """Return a kept mapping."""
        return self.mappings.get(key)

    def save_mapping(
        self, session_id: str, key: str, mapping: List[SpeakerMapping]
    ) -> None:
        """Keep a mapping."""
        self.mappings[key] = mapping

    def save_run(self, run: LabRun) -> None:
        """Keep a run."""
        self.stored_runs[run.id] = run

    def run(self, session_id: str, run_id: str) -> Optional[LabRun]:
        """Return a run of the session."""
        run = self.stored_runs.get(run_id)
        return run if run is not None and run.session_id == session_id \
            else None

    def runs(self, session_id: str) -> List[LabRun]:
        """Return the session's runs, newest first."""
        return sorted(
            (r for r in self.stored_runs.values()
             if r.session_id == session_id),
            key=lambda r: r.created_at, reverse=True,
        )

    def request_cancel(self, session_id: str, run_id: str) -> None:
        """Mark the run to stop."""
        self.cancels.add(run_id)

    def cancel_requested(self, session_id: str, run_id: str) -> bool:
        """Tell whether the run was marked."""
        return run_id in self.cancels

    def clear_cancel(self, session_id: str, run_id: str) -> None:
        """Unmark the run."""
        self.cancels.discard(run_id)

    def save_results(
        self, session_id: str, run_id: str, results: Dict[str, Any]
    ) -> None:
        """Keep results."""
        self.stored_results[run_id] = results

    def results(
        self, session_id: str, run_id: str
    ) -> Optional[Dict[str, Any]]:
        """Return kept results."""
        return self.stored_results.get(run_id)

    def labels(self, session_id: str) -> List[ReferenceLabel]:
        """Return the labels."""
        return list(self.stored_labels.get(session_id, []))

    def save_labels(
        self, session_id: str, labels: List[ReferenceLabel]
    ) -> None:
        """Keep the labels."""
        self.stored_labels[session_id] = list(labels)

    def reference(self, session_id: str) -> Optional[SessionReference]:
        """Return the session's corpus reference, if any."""
        return self.references.get(session_id)

    def save_reference(
        self, session_id: str, reference: SessionReference
    ) -> None:
        """Keep the session's corpus reference."""
        self.references[session_id] = reference

    def delete_session(self, session_id: str) -> None:
        """Forget the session's original, runs, results and labels.

        Stage outcomes, mappings and audio are keyed by content here,
        not by session, so they stay (the fake shares them).
        """
        self.originals.pop(session_id, None)
        self.stored_labels.pop(session_id, None)
        self.references.pop(session_id, None)
        for run_id in [r.id for r in self.runs(session_id)]:
            del self.stored_runs[run_id]
            self.stored_results.pop(run_id, None)
            self.cancels.discard(run_id)


Answer = Callable[[ToolRequest], StageOutcome]


class ScriptedTool:
    """A lab tool answering each model id with a function.

    A model without an answer raises ``LabStepError(unknown_model)``.
    A denoiser's answer leaves the input audio, halved, as its output.
    ``before`` runs at every call (to cancel a run mid-way).
    """

    def __init__(
        self, repository: FakeLabRepository,
        answers: Optional[Mapping[str, Answer]] = None,
    ) -> None:
        """Answer with ``answers`` (model id → function)."""
        self.repository = repository
        self.answers: Dict[str, Answer] = dict(answers or {})
        self.calls: List[str] = []
        self.released = 0
        self.before: Callable[[], None] = lambda: None

    def run(
        self, request: ToolRequest, cancelled: Callable[[], bool]
    ) -> StageOutcome:
        """Answer the request, recording it."""
        self.before()
        name = f"{request.stage}:{request.entry.id}"
        self.calls.append(name)
        if cancelled():
            raise LabStepError("cancelled", "cancelled")
        answer = self.answers.get(request.entry.id)
        if answer is None:
            raise LabStepError("unknown_model", request.entry.id)
        outcome = answer(request)
        if request.stage == "denoise":
            audio = request.load()
            self.repository.written[request.workspace] = Audio(
                audio.samples * 0.5, audio.sample_rate
            )
        return outcome

    def release(self) -> None:
        """Count the releases."""
        self.released += 1


def halves(labels: List[str]) -> Answer:
    """A diarizer giving the first half to one label, the rest to another."""
    def answer(request: ToolRequest) -> StageOutcome:
        """Split the audio's time in two."""
        duration = request.load().duration
        return StageOutcome(
            stage="diarize", model=request.entry.id, seconds=1.0,
            peak_gpu_mb=500.0, device="cuda",
            segments=[
                Segment(start=0, end=duration / 2, speaker=labels[0]),
                Segment(start=duration / 2, end=duration,
                        speaker=labels[1]),
            ],
        )
    return answer


def scene_turns(times: List[tuple[float, float, str]]) -> Answer:
    """A diarizer answering fixed ``(start, end, label)`` segments."""
    def answer(request: ToolRequest) -> StageOutcome:
        """Return the fixed segments."""
        return StageOutcome(
            stage="diarize", model=request.entry.id, seconds=2.0,
            segments=[Segment(start=s, end=e, speaker=x)
                      for s, e, x in times],
        )
    return answer


def hears(words: List[Word]) -> Answer:
    """A recogniser hearing ``words``."""
    def answer(request: ToolRequest) -> StageOutcome:
        """Return the words."""
        return StageOutcome(
            stage="recognise", model=request.entry.id, seconds=3.0,
            peak_gpu_mb=900.0, words=words,
            language=request.params.get("language"),
        )
    return answer


def denoises(request: ToolRequest) -> StageOutcome:
    """A denoiser's outcome."""
    return StageOutcome(stage="denoise", model=request.entry.id,
                        seconds=0.5, peak_gpu_mb=100.0)


def fails(code: str) -> Answer:
    """A model that fails with ``code``."""
    def answer(request: ToolRequest) -> StageOutcome:
        """Raise."""
        raise LabStepError(code, f"{request.entry.id} failed: {code}")
    return answer


def crashes(request: ToolRequest) -> StageOutcome:
    """A model that raises an unexpected error."""
    raise RuntimeError("boom")


class StaticCatalogue:
    """A catalogue source over fixed entries."""

    def __init__(self, entries: List[Dict[str, Any]]) -> None:
        """Validate the entries once."""
        self.catalogue = Catalogue.model_validate(
            {"schema": "lab.catalogue.v1", "entries": entries}
        )

    def load(self) -> Catalogue:
        """Return the catalogue."""
        return self.catalogue

    def entry(self, model_id: str) -> CatalogueEntry:
        """Return one entry."""
        found = self.catalogue.get(model_id)
        assert found is not None
        return found


class FakeEnvironment:
    """A machine whose families, token and cache the test sets."""

    def __init__(self) -> None:
        """Every family ready, no token, nothing cached."""
        self.problems: Dict[str, str] = {}
        self.token = False
        self.cached: Dict[str, Optional[bool]] = {}

    def family_problem(self, family: str) -> Optional[str]:
        """Return the problem set for ``family``."""
        return self.problems.get(family)

    def token_present(self) -> bool:
        """Return the token flag."""
        return self.token

    def weights_cached(self, weight: str) -> Optional[bool]:
        """Return what the test set (unknown by default)."""
        return self.cached.get(weight)


class FakeProcesses:
    """Process ids: this one is 100; the others alive are listed."""

    def __init__(self) -> None:
        """Only this process is alive."""
        self.pid = 100
        self.others: Set[int] = set()

    def current(self) -> int:
        """Return this process's id."""
        return self.pid

    def alive(self, pid: int) -> bool:
        """Tell whether ``pid`` is this process or a listed one."""
        return pid == self.pid or pid in self.others
