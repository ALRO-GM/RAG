"""Corpus adapter: indexable files under a directory on disk."""

import hashlib
import os
from pathlib import Path
from typing import List, Sequence

from src.domain.errors import NotFoundError

DEFAULT_EXTENSIONS = (".py", ".md", ".txt", ".rst")


class FileSystemCorpus:
    """List and hash files below ``root`` with the given extensions.

    Paths are returned relative to the current working directory (the
    project root), e.g. ``data/raw/vllm-0.10.1/docs/index.md``, which is
    exactly the form the evaluator compares against.
    """

    def __init__(
        self, root: str, extensions: Sequence[str] = DEFAULT_EXTENSIONS
    ):
        """Index files under ``root`` ending with one of ``extensions``."""
        self.root = Path(root)
        self.extensions = tuple(e.lower() for e in extensions)

    def list_files(self) -> List[str]:
        """Return the indexable files in a deterministic walk order.

        Directories and file names are visited in sorted order; hidden
        directories are skipped.
        """
        if not self.root.is_dir():
            raise NotFoundError(f"Corpus directory not found: {self.root}")
        found: List[str] = []
        for current, dirs, files in os.walk(self.root):
            dirs[:] = sorted(d for d in dirs if not d.startswith("."))
            for name in sorted(files):
                if name.lower().endswith(self.extensions):
                    path = os.path.relpath(os.path.join(current, name))
                    found.append(Path(path).as_posix())
        return found

    def file_hash(self, path: str) -> str:
        """Return the SHA-256 digest of the file's bytes."""
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()
