"""Ports of the soft-skills use cases."""

from typing import (
    Any, Callable, ContextManager, Dict, List, Mapping, NamedTuple,
    Optional, Protocol, Sequence, Tuple,
)

from src.domain.skills.act_tagging import ActInput, Probabilities
from src.domain.skills.analysis_run import AnalysisRun
from src.domain.skills.assessment import SessionAssessment
from src.domain.skills.judging import JudgeState
from src.domain.skills.rubric import RubricCatalogue
from src.domain.skills.speech import Sentence
from src.domain.skills.transcript import Transcript


class ChatModel(Protocol):
    """A language model answering one system + user message."""

    @property
    def model_name(self) -> str:
        """Return the identifier of the language model."""
        ...

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        """Return the model's reply."""
        ...


class JudgeStateStore(Protocol):
    """Keep the judge chosen from the web and what is known of the quota."""

    def load(self) -> JudgeState:
        """Return the stored state (an empty one when nothing is stored)."""
        ...

    def save(self, state: JudgeState) -> None:
        """Replace the stored state."""
        ...

    def update(self, change: Callable[[JudgeState], JudgeState]) -> JudgeState:
        """Apply ``change`` to the stored state atomically; return it.

        Concurrent updates (the web choosing a judge while a job
        records the quota) must not lose each other's fields.
        """
        ...


class TranscriptRepository(Protocol):
    """Read diarized transcripts."""

    def read(self, path: str) -> Transcript:
        """Return the transcript stored at ``path``."""
        ...

    def find(self, session_id: str) -> Transcript:
        """Return the transcript of a session by its identifier."""
        ...

    def sessions(self) -> List[str]:
        """Return the identifiers of the stored sessions."""
        ...


class TranscriptStore(Protocol):
    """Keep ``transcript.v1`` documents received from the transcriber."""

    def put(self, session_id: str, document: Mapping[str, Any]) -> bool:
        """Store ``document`` as session ``session_id``, replacing any copy.

        Return whether an earlier copy was replaced.
        """
        ...

    def get(self, session_id: str) -> Dict[str, Any]:
        """Return the stored document of ``session_id``.

        Raises :class:`~src.domain.errors.NotFoundError` if there is none.
        """
        ...

    def delete(self, session_id: str) -> bool:
        """Remove the document of ``session_id``; tell whether it existed."""
        ...


class CatalogueRepository(Protocol):
    """Read rubric catalogues."""

    def load(self, name: str, language: str) -> RubricCatalogue:
        """Return the catalogue ``name`` written in ``language``."""
        ...

    def available(self) -> List[Tuple[str, str]]:
        """Return the (name, language) pairs that can be loaded."""
        ...


class AssessmentStore(Protocol):
    """Keep the assessment of every session (``skills assess`` output)."""

    def all(self) -> List[SessionAssessment]:
        """Return every stored assessment."""
        ...

    def get(self, reference: str) -> SessionAssessment:
        """Return one assessment by session id or file path.

        Raises :class:`~src.domain.errors.NotFoundError` if there is none.
        """
        ...


class ActTagger(Protocol):
    """An automatic dialogue-act tagger (#82).

    Returns, for each sentence, a probability per category (the 13 of
    :data:`~src.domain.skills.dialogue_acts.CATEGORIES`), calibrated by
    whatever the approach uses (temperature, outcome tables).
    """

    @property
    def name(self) -> str:
        """Return the approach's name (``rules``, ``llm``, ``encoder``...)."""
        ...

    @property
    def version(self) -> str:
        """Return what identifies the weights, rules or model used."""
        ...

    def predict(self, items: Sequence[ActInput]) -> List[Probabilities]:
        """Return one distribution per item, in order."""
        ...


class StepOutput(NamedTuple):
    """What an optional analysis step produced.

    ``data`` is written as the step's JSON file; ``counts`` go to the
    step's record; ``model`` names who produced it and ``note`` how
    (which approach answered, thresholds).
    """

    data: Any
    counts: Dict[str, int]
    model: str
    note: str = ""


class DialogueActTagger(Protocol):
    """Label each sentence with its dialogue acts (step d, #82)."""

    def tag(
        self, transcript: Transcript, sentences: Sequence[Sentence]
    ) -> StepOutput:
        """Return the acts of ``sentences`` (JSON-ready ``data``)."""
        ...


class EventExtractor(Protocol):
    """Derive decisions, action items, open questions (step e, #83)."""

    def extract(
        self,
        transcript: Transcript,
        sentences: Sequence[Sentence],
        acts: Optional[Any],
    ) -> StepOutput:
        """Return the events; ``acts`` is step d's data, if it ran."""
        ...


class AnalysisStore(Protocol):
    """Keep the steps and results of each session's analysis.

    ``name`` is a plain file name chosen by the use case (the step
    outputs, ``report-<n>.html``); implementations refuse anything
    else, so neither a session id nor a name can reach outside.
    """

    def load(self, session_id: str) -> Optional[AnalysisRun]:
        """Return the stored run of ``session_id``, if any."""
        ...

    def save(self, run: AnalysisRun) -> None:
        """Store ``run`` (replacing the previous record)."""
        ...

    def write(self, session_id: str, name: str, text: str) -> str:
        """Store one result file; return where it went."""
        ...

    def read(self, session_id: str, name: str) -> str:
        """Return one result file (``NotFoundError`` when missing)."""
        ...

    def clear(self, session_id: str) -> None:
        """Delete the previous run's record and results (a restart)."""
        ...

    def erase(self, session_id: str) -> None:
        """Delete everything kept of the session (it was deleted)."""
        ...


class AnalysisLauncher(Protocol):
    """Run ``skills analyze`` on a session apart from the caller."""

    def start(
        self,
        session_id: str,
        restart: bool,
        synthetic: bool,
        steps: Sequence[str] = (),
    ) -> None:
        """Start the analysis of ``session_id`` and return at once.

        ``steps``, when not empty, are the only steps run again.
        """
        ...

    def running(self, session_id: str) -> bool:
        """Tell whether the analysis of ``session_id`` is running now."""
        ...

    def stop(self, session_id: str) -> bool:
        """Stop the analysis of ``session_id``; tell whether one ran."""
        ...


class AnalysisLock(Protocol):
    """One analysis per session at a time, across processes.

    Held for the whole run; released when the run ends or its process
    dies, so a crashed run never blocks the next one.
    """

    def acquire(self, session_id: str) -> Optional[int]:
        """Take the session; return this worker's id, ``None`` if taken."""
        ...

    def release(self, session_id: str) -> None:
        """Give the session back."""
        ...

    def discard(self, session_id: str) -> None:
        """Give the session back, leaving no trace of the lock.

        For a session found to have no transcript: what taking the lock
        created is removed when the session holds nothing else.
        """
        ...

    def holder(self, session_id: str) -> Optional[int]:
        """Return the worker id holding the session now, if any."""
        ...


class StartLog(Protocol):
    """The analyses started through the API, kept across server restarts.

    Each start is ``(time in seconds since the epoch, session id)``;
    the start limits are counted on them.
    """

    def load(self) -> List[Tuple[float, str]]:
        """Return the starts kept (empty when none)."""
        ...

    def save(self, starts: Sequence[Tuple[float, str]]) -> None:
        """Keep ``starts`` (replacing the previous ones)."""
        ...


class SessionEraser(Protocol):
    """Delete what other modules derived from a session (retention)."""

    def erase(self, session_id: str) -> None:
        """Delete it; nothing when there is none."""
        ...


class SessionGuard(Protocol):
    """Hold a session while its transcript is deleted (#121).

    No analysis may start, read the transcript and write its files
    while the transcript is being deleted; the analysis is deleted
    last, still holding the session.
    """

    def erasing(self, session_id: str) -> ContextManager[None]:
        """Hold the session for the block, then delete its analysis.

        Raises :class:`~src.domain.errors.BusyError`, before the block
        runs, when the session cannot be held (its analysis runs
        elsewhere); nothing is deleted then.
        """
        ...
