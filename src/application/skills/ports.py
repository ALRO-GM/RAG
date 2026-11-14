"""Ports of the soft-skills use cases."""

from typing import Any, Dict, List, Mapping, Protocol, Tuple

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


class CatalogueRepository(Protocol):
    """Read rubric catalogues."""

    def load(self, name: str, language: str) -> RubricCatalogue:
        """Return the catalogue ``name`` written in ``language``."""
        ...

    def available(self) -> List[Tuple[str, str]]:
        """Return the (name, language) pairs that can be loaded."""
        ...
