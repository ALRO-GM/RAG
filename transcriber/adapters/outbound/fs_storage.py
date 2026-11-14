"""File-system helpers shared by the storage adapters.

Every write goes to a temporary file in the target's directory, is
flushed to disk and then renamed over the target (:func:`atomic_write`),
so a process killed mid-write leaves either the old file or the new
one, never half of either. Every id is checked against its domain
pattern, and every path against the storage root, before a file is
touched (:func:`child_path`).
"""

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from transcriber.domain.errors import InvalidInputError, StorageError


def atomic_write(path: Path, data: bytes) -> None:
    """Replace ``path`` with ``data`` in one step.

    The bytes go to a hidden temporary file next to ``path`` (same file
    system, so the rename is atomic), are synced to disk and renamed
    over ``path``; the directory is synced too, so the rename survives
    a power cut. The temporary file is removed if anything fails.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    sync_directory(path.parent)


def sync_directory(directory: Path) -> None:
    """Flush a directory entry (a rename or a new file) to disk."""
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:  # pragma: no cover - platforms without directory fds
        return
    try:
        os.fsync(descriptor)
    except OSError:  # pragma: no cover - file systems that refuse it
        pass
    finally:
        os.close(descriptor)


def write_json(path: Path, value: Any) -> None:
    """Write ``value`` as UTF-8 JSON to ``path`` atomically."""
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    atomic_write(path, (text + "\n").encode("utf-8"))


def read_json(path: Path) -> Any:
    """Read the JSON file at ``path``.

    Raises :class:`StorageError` when the file is not valid JSON or
    cannot be read (a directory in its place, no permission), and lets
    :class:`FileNotFoundError` through for the caller to decide. The
    error never names more of the path than the file name.
    """
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StorageError(f"Cannot read {path.name}: {exc}") from exc
    except OSError as exc:
        raise StorageError(
            f"Cannot read {path.name}: {exc.strerror or type(exc).__name__}"
        ) from exc


def child_path(
    root: Path, name: str, pattern: str, what: str, suffix: str = ""
) -> Path:
    """Return ``root / (name + suffix)`` after checking the id and path.

    ``name`` must match ``pattern`` (the domain's id pattern, which has
    no separators and cannot start with a dot), and the path must not
    be a symbolic link: a link would make one id an alias of another
    record (deleting the alias would delete the record) or of a place
    outside ``root``. Raises :class:`InvalidInputError` otherwise.
    """
    if not isinstance(name, str) or not re.fullmatch(pattern, name):
        raise InvalidInputError(f"Invalid {what} id {name!r}.")
    base = root.resolve()
    path = base / f"{name}{suffix}"
    if path.is_symlink() or path.resolve().parent != base:
        raise InvalidInputError(f"Invalid {what} id {name!r}.")
    return path
