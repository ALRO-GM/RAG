"""The access code of ``transcriber serve``: made fresh, or kept on disk.

A fresh code is random and URL-safe (:func:`new_access_code`). With
``serve --keep-code`` it is kept in ``<data_dir>/access_code`` (mode
600) so a printed QR code keeps working across restarts
(:func:`kept_access_code`). A kept file that others may read, or that
does not hold a valid code, is refused rather than trusted or replaced
silently.
"""

import os
import re
import secrets
import stat
from pathlib import Path

from transcriber.domain.errors import AccessCodeError

__all__ = [
    "AccessCodeError", "CODE_FILE", "code_file", "kept_access_code",
    "new_access_code",
]

CODE_BYTES = 32
"""Random bytes in a code (43 URL-safe characters)."""

MIN_KEPT_LENGTH = 32
"""Shortest code accepted from the kept file."""

CODE_FILE = "access_code"
CODE_PATTERN = re.compile(r"[A-Za-z0-9_-]+")


def new_access_code() -> str:
    """Return a fresh random URL-safe access code (43 characters)."""
    return secrets.token_urlsafe(CODE_BYTES)


def code_file(data_dir: Path) -> Path:
    """Return where ``serve --keep-code`` keeps the code."""
    return data_dir / CODE_FILE


def kept_access_code(path: Path) -> str:
    """Return the code kept in ``path``, creating it (mode 600) if absent.

    Raises :class:`AccessCodeError` when the file is readable by others
    or does not hold a URL-safe code of at least
    :data:`MIN_KEPT_LENGTH` characters.
    """
    if path.exists():
        mode = stat.S_IMODE(path.stat().st_mode)
        if mode & 0o077:
            raise AccessCodeError(
                f"{path} is readable by others (mode {mode:o}); delete it"
                " to make a new code."
            )
        code = path.read_text(encoding="utf-8").strip()
        if len(code) < MIN_KEPT_LENGTH or not CODE_PATTERN.fullmatch(code):
            raise AccessCodeError(
                f"{path} does not hold a valid code; delete it to make a"
                " new one."
            )
        return code
    path.parent.mkdir(parents=True, exist_ok=True)
    code = new_access_code()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(code + "\n")
    return code
