"""The login password, kept only as a salted scrypt hash on disk.

``<data_dir>/password.json`` (mode 600) holds the scrypt parameters, a
random salt and the hash; the password itself is never stored or logged.
:func:`set_password` writes it (replacing any previous one) and
:class:`PasswordStore` checks a given password in constant time.
"""

import hashlib
import hmac
import json
import os
import secrets
import stat
from pathlib import Path
from typing import Any, Dict, Optional

from transcriber.domain.errors import PasswordError

__all__ = [
    "MIN_PASSWORD_LENGTH", "PASSWORD_FILE", "PasswordStore", "password_file",
    "set_password",
]

PASSWORD_FILE = "password.json"
MIN_PASSWORD_LENGTH = 4
MAX_PASSWORD_LENGTH = 256
SALT_BYTES = 16
SCRYPT_N = 2 ** 15
SCRYPT_R = 8
SCRYPT_P = 1
MAX_MEMORY = 128 * 1024 * 1024
KEY_BYTES = 32


def password_file(data_dir: Path) -> Path:
    """Return where the password hash is kept."""
    return data_dir / PASSWORD_FILE


def _derive(
    password: str, salt: bytes, n: int, r: int, p: int
) -> bytes:
    """Return the scrypt key of ``password``."""
    return hashlib.scrypt(
        password.encode("utf-8", "surrogateescape"), salt=salt, n=n, r=r,
        p=p, maxmem=MAX_MEMORY, dklen=KEY_BYTES,
    )


def set_password(path: Path, password: str) -> None:
    """Keep the hash of ``password`` in ``path`` (mode 600), replacing it.

    Raises :class:`PasswordError` when the password is too short or too
    long.
    """
    if not MIN_PASSWORD_LENGTH <= len(password) <= MAX_PASSWORD_LENGTH:
        raise PasswordError(
            f"The password needs {MIN_PASSWORD_LENGTH} to"
            f" {MAX_PASSWORD_LENGTH} characters."
        )
    salt = secrets.token_bytes(SALT_BYTES)
    record = {
        "scheme": "scrypt", "n": SCRYPT_N, "r": SCRYPT_R, "p": SCRYPT_P,
        "salt": salt.hex(),
        "hash": _derive(password, salt, SCRYPT_N, SCRYPT_R, SCRYPT_P).hex(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.unlink(missing_ok=True)
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(record, handle)
        handle.write("\n")
    os.replace(temporary, path)


class PasswordStore:
    """Check passwords against the hash kept in a file."""

    def __init__(self, path: Path) -> None:
        """Read the hash from ``path`` on every check (it may change)."""
        self._path = path

    def is_set(self) -> bool:
        """Tell whether a password has been set."""
        return self._path.is_file()

    def _record(self) -> Optional[Dict[str, Any]]:
        """Return the stored record, or ``None`` if there is none."""
        if not self._path.is_file():
            return None
        if stat.S_IMODE(self._path.stat().st_mode) & 0o077:
            raise PasswordError(
                f"{self._path} is readable by others; set the password"
                " again."
            )
        try:
            record = json.loads(self._path.read_text(encoding="utf-8"))
            if record["scheme"] != "scrypt":
                raise ValueError("scheme")
            return {
                "n": int(record["n"]), "r": int(record["r"]),
                "p": int(record["p"]),
                "salt": bytes.fromhex(record["salt"]),
                "hash": bytes.fromhex(record["hash"]),
            }
        except (ValueError, KeyError, TypeError) as exc:
            raise PasswordError(
                f"{self._path} does not hold a valid password hash."
            ) from exc

    def verify(self, password: str) -> bool:
        """Tell whether ``password`` is the one set, in constant time.

        ``False`` when none is set or the stored hash is unusable.
        """
        try:
            record = self._record()
        except PasswordError:
            return False
        if record is None or len(password) > MAX_PASSWORD_LENGTH:
            return False
        try:
            key = _derive(
                password, record["salt"], record["n"], record["r"],
                record["p"],
            )
        except (ValueError, MemoryError):
            return False
        return hmac.compare_digest(key, record["hash"])
