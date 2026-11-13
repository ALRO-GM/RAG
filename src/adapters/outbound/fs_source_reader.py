"""Source reader adapter: resolve source locations to file text."""

from pathlib import Path
from typing import Dict, List, Optional, Sequence

from src.domain.models import Chunk, MinimalSource


class FileSystemSourceReader:
    """Read the text of source locations from the corpus on disk.

    Paths are resolved relative to the working directory, as written
    in the search results. Unreadable files and empty ranges are
    skipped, so one bad source does not prevent answering from the rest.
    """

    def read(self, sources: Sequence[MinimalSource]) -> List[Chunk]:
        """Return the readable sources as chunks, in order."""
        texts: Dict[str, Optional[str]] = {}
        chunks: List[Chunk] = []
        for src in sources:
            if src.file_path not in texts:
                try:
                    texts[src.file_path] = Path(src.file_path).read_text(
                        encoding="utf-8", errors="replace"
                    )
                except OSError:
                    texts[src.file_path] = None
            text = texts[src.file_path]
            if text is None:
                continue
            start = max(0, src.first_character_index)
            end = min(len(text), src.last_character_index)
            if end <= start:
                continue
            chunks.append(Chunk(
                file_path=src.file_path,
                first_character_index=start,
                last_character_index=end,
                kind="source",
                text=text[start:end],
            ))
        return chunks
