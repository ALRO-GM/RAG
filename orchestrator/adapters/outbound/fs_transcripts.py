"""Recorded conversations for replays: ``*.transcript.v1.json`` files.

The transcript id is the file name without ``.transcript.v1.json``.
Several folders can be searched (the published transcripts, the
synthetic demo calls of ``examples/platform``); the first one holding
an id wins. Only names made of safe characters are looked up, so an id
can never reach outside the folders.
"""

import re
from pathlib import Path
from typing import Dict, List, Sequence

from orchestrator.adapters.outbound.files import ConfigError, load_transcript
from orchestrator.application.ports import Document
from orchestrator.domain.events import Transcript

SUFFIX = ".transcript.v1.json"
TRANSCRIPT_ID = r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}"


class DirTranscripts:
    """Transcripts kept as files in ``folders``."""

    def __init__(self, folders: Sequence[Path]):
        """Keep the folders (missing ones are skipped)."""
        self.folders = list(folders)

    def _files(self) -> Dict[str, Path]:
        found: Dict[str, Path] = {}
        for folder in self.folders:
            if not folder.is_dir():
                continue
            for path in sorted(folder.glob(f"*{SUFFIX}")):
                found.setdefault(path.name[:-len(SUFFIX)], path)
        return found

    def list(self) -> List[Document]:
        """Every transcript, with its title and size."""
        out: List[Document] = []
        for transcript_id, path in sorted(self._files().items()):
            try:
                transcript = load_transcript(path)
            except ConfigError:
                continue
            out.append({
                "id": transcript_id,
                "title": transcript.session.title or transcript_id,
                "type": transcript.session.type,
                "utterances": len(transcript.utterances),
                "speakers": len(transcript.speakers),
                "seconds": round(
                    transcript.utterances[-1].end, 1
                ) if transcript.utterances else 0.0,
            })
        return out

    def load(self, transcript_id: str) -> Transcript:
        """Return one transcript; ``KeyError`` when there is none."""
        if not re.fullmatch(TRANSCRIPT_ID, transcript_id or ""):
            raise KeyError(transcript_id)
        for folder in self.folders:
            path = folder / f"{transcript_id}{SUFFIX}"
            if path.is_file():
                try:
                    return load_transcript(path)
                except ConfigError as exc:
                    raise KeyError(transcript_id) from exc
        raise KeyError(transcript_id)
