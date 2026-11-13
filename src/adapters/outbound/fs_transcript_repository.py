"""Transcript repository adapter: transcripts as files on disk."""

import json
from pathlib import Path
from typing import List

from src.domain.errors import CorruptedDataError, NotFoundError, RagError
from src.domain.skills.speech import transcript_document
from src.domain.skills.transcript import Transcript, parse_transcript

EXTENSIONS = (".json", ".txt", ".md")


class FileSystemTranscriptRepository:
    """Read transcripts from JSON or ``Speaker: text`` files.

    The session identifier of a file is its name without extension,
    unless the JSON document names one.
    """

    def __init__(self, directory: str = "data/transcripts"):
        """Look sessions up by identifier under ``directory``."""
        self.directory = Path(directory)

    def read(self, path: str) -> Transcript:
        """Return the transcript stored at ``path``."""
        file = Path(str(path))
        if not file.is_file():
            raise NotFoundError(f"Transcript not found: {file}")
        try:
            text = file.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            raise CorruptedDataError(f"{file} is not a UTF-8 text file.")
        except OSError as exc:
            raise RagError(f"Cannot read {file}: {exc}")
        return parse_transcript(file.stem, text)

    def find(self, session_id: str) -> Transcript:
        """Return the transcript named ``session_id`` in the directory.

        The file name is the identifier here, whatever the document
        declares, so that references by identifier always resolve.
        """
        for extension in EXTENSIONS:
            candidate = self.directory / f"{session_id}{extension}"
            if candidate.is_file():
                found = self.read(str(candidate))
                return found.model_copy(update={"session_id": session_id})
        raise NotFoundError(
            f"No transcript for session {session_id!r} in {self.directory}."
        )

    def sessions(self) -> List[str]:
        """Return the session identifiers found in the directory."""
        if not self.directory.is_dir():
            return []
        return sorted({
            file.stem for file in self.directory.iterdir()
            if file.is_file() and file.suffix in EXTENSIONS
        })

    def save(self, transcript: Transcript, path: str = "") -> str:
        """Write ``transcript`` as JSON and return the file path.

        Without ``path`` the file is ``<directory>/<session_id>.json``,
        where :meth:`find` looks it up.
        """
        target = Path(path) if path else (
            self.directory / f"{transcript.session_id}.json"
        )
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps(
                    transcript_document(transcript), ensure_ascii=False,
                    indent=1,
                ),
                encoding="utf-8",
            )
        except OSError as exc:
            raise RagError(f"Cannot write {target}: {exc}")
        return str(target)
