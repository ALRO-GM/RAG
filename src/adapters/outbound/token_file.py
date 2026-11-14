"""Secret adapter: read a bearer token from a private file.

On the workstation the ingest token of ``POST /transcripts`` lives in a
file outside the repository (``make rag-token`` writes
``~/.config/ragputin/rag-token`` with mode 600), so it never appears on
a command line, in a log or in a setting. The file is refused unless it
is a regular file of this user that nobody else may read or write, and
no message ever contains its content.

``python3 -m src.adapters.outbound.token_file <path>`` (what ``make
rag-token`` runs, with the standard library only) makes the file when it
is missing and checks it when it exists; the token is never printed.
"""

import os
import re
import secrets
import stat
import sys
from pathlib import Path
from typing import List, Sequence

from src.domain.errors import InvalidInputError

TOKEN_PATTERN = r"^[A-Za-z0-9_.~+/=-]{32,256}$"
"""What ``make rag-token`` writes (URL-safe base64), with some slack."""
TOKEN_BYTES = 32
"""Random bytes of a new token (43 URL-safe characters)."""


def read_token_file(path: Path) -> str:
    """Return the token kept in ``path`` (``~`` is expanded).

    Raises :class:`~src.domain.errors.InvalidInputError` when the file
    is missing, is not a regular file of this user, may be read or
    written by others, or does not hold a token.
    """
    path = Path(path).expanduser()
    try:
        info = path.lstat()
    except FileNotFoundError:
        raise InvalidInputError(
            f"{path} does not exist: run `make rag-token`."
        ) from None
    except OSError as exc:
        raise InvalidInputError(f"Cannot read {path}: {exc.strerror}.")
    if not stat.S_ISREG(info.st_mode):
        raise InvalidInputError(f"{path} is not a regular file.")
    if info.st_uid != os.getuid():
        raise InvalidInputError(f"{path} belongs to another user.")
    mode = stat.S_IMODE(info.st_mode)
    if mode & 0o077:
        raise InvalidInputError(
            f"{path} may be read by others (mode {mode:o}): run"
            f" `chmod 600 {path}`."
        )
    try:
        token = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        raise InvalidInputError(f"Cannot read {path}.") from None
    if not re.fullmatch(TOKEN_PATTERN, token):
        raise InvalidInputError(f"{path} does not hold a token.")
    return token


def make_token_file(path: Path) -> bool:
    """Make ``path`` with a new token unless it exists; tell if made.

    The file is created with mode 600 (its directory with 700 when it
    is missing), never overwritten. An existing file must pass
    :func:`read_token_file`.
    """
    path = Path(path).expanduser()
    if os.path.lexists(path):
        read_token_file(path)
        return False
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    token = secrets.token_urlsafe(TOKEN_BYTES)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(token + "\n")
    return True


def main(argv: Sequence[str]) -> int:
    """Make or check the token file named by ``argv``; return the exit code."""
    arguments: List[str] = list(argv)
    if len(arguments) != 1:
        print("usage: python3 -m src.adapters.outbound.token_file <path>",
              file=sys.stderr)
        return 2
    try:
        made = make_token_file(Path(arguments[0]))
    except (InvalidInputError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    where = Path(arguments[0]).expanduser()
    print(f"{'Made' if made else 'Kept'} the RAG token in {where} (mode 600).")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
