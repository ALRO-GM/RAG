"""Ports of the soft-skills use cases."""

from typing import (
    Any, Callable, Dict, List, Mapping, Protocol, Sequence, Tuple,
)

from src.domain.skills.act_tagging import ActInput, Probabilities
from src.domain.skills.assessment import SessionAssessment
from src.domain.skills.judging import JudgeState
from src.domain.skills.rubric import RubricCatalogue
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
