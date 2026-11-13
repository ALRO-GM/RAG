"""Ports of the soft-skills use cases."""

from typing import List, Protocol, Tuple

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


class CatalogueRepository(Protocol):
    """Read rubric catalogues."""

    def load(self, name: str, language: str) -> RubricCatalogue:
        """Return the catalogue ``name`` written in ``language``."""
        ...

    def available(self) -> List[Tuple[str, str]]:
        """Return the (name, language) pairs that can be loaded."""
        ...
