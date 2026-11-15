"""Assessment store adapter: ``<session>.assessment.json`` files."""

from pathlib import Path
from typing import List

from src.adapters.outbound.json_dataset_repository import read_model
from src.domain.errors import NotFoundError
from src.domain.skills.assessment import SessionAssessment

SUFFIX = ".assessment.json"
"""Name ending of a file written by ``skills assess``."""


class FileSystemAssessmentStore:
    """Read the session assessments kept in one directory.

    ``skills assess`` writes ``<session>.assessment.json`` there; a
    reference is either such a session id or the path of a file.
    """

    def __init__(self, directory: str = "data/output/skills"):
        """Look assessments up under ``directory``."""
        self.directory = Path(directory)

    def all(self) -> List[SessionAssessment]:
        """Return every assessment of the directory, by file name."""
        if not self.directory.is_dir():
            return []
        return [
            read_model(str(file), SessionAssessment)
            for file in sorted(self.directory.glob(f"*{SUFFIX}"))
            if file.is_file()
        ]

    def get(self, reference: str) -> SessionAssessment:
        """Return the assessment at path ``reference`` or of that session."""
        file = Path(str(reference))
        if file.is_file():
            return read_model(str(file), SessionAssessment)
        candidate = self.directory / f"{reference}{SUFFIX}"
        if candidate.is_file():
            return read_model(str(candidate), SessionAssessment)
        raise NotFoundError(
            f"No assessment {reference!r}: neither a file nor a session "
            f"in {self.directory}."
        )
