"""Workstation set-up of the tunnel over HTTPS (stdlib only).

Two commands, both local; nothing is uploaded:

``client``
    Download the pinned chisel release for this machine's system and
    processor from GitHub, check the archive's SHA-256 against
    ``infra/chisel-<version>.sha256`` (taken from the release's own
    checksums file and committed), and install the binary with mode 755
    (``make tunnel-client``). A mismatch installs nothing.

``secret``
    Make the workstation's credential (``user:password``, a random
    43-character password) in a file of mode 600 if it is missing, and
    write the matching chisel users file for the server, which allows
    that user only the one reverse remote ``R:0.0.0.0:<port>``
    (``make tunnel-secret``). It prints where both are, never the
    password. chisel 1.12.0 checks that list on the remotes declared at
    connection and again on every channel asked for later; the docker
    proof (``infra/tests/docker_proof.py``) shows a hand-made client
    with the credential refused SOCKS and every channel to another
    destination (``web:80``, chisel itself, the host, an outside
    address).
"""

import argparse
import gzip
import hashlib
import json
import os
import platform
import re
import secrets
import stat
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

VERSION = "1.12.0"
HERE = Path(__file__).resolve().parent
CHECKSUMS = HERE / f"chisel-{VERSION}.sha256"
RELEASES = "https://github.com/jpillora/chisel/releases/download"
USER = "workstation"
PASSWORD_BYTES = 32
"""Random bytes of the password (43 URL-safe characters)."""

DEFAULT_CHISEL = Path("~/.local/share/ragputin/bin/chisel")
DEFAULT_CREDENTIAL = Path("~/.config/ragputin/tunnel-credential")
DEFAULT_USERS = Path("~/.config/ragputin/tunnel-users.json")
DEFAULT_REMOTE_PORT = 9000

Fetch = Callable[[str], bytes]
"""Download a URL; return its bytes."""


class SetupError(Exception):
    """Something the user must fix; the message says what."""


def platform_name(system: str, machine: str) -> str:
    """Return chisel's ``<os>_<arch>`` for this machine.

    Raises :class:`SetupError` for a system or processor without a
    pinned checksum.
    """
    systems = {"linux": "linux", "darwin": "darwin"}
    machines = {"x86_64": "amd64", "amd64": "amd64",
                "aarch64": "arm64", "arm64": "arm64"}
    os_name = systems.get(system.lower())
    arch = machines.get(machine.lower())
    if os_name is None or arch is None:
        raise SetupError(f"No pinned chisel build for {system}/{machine}.")
    return f"{os_name}_{arch}"


def pinned_checksums(path: Path = CHECKSUMS) -> Dict[str, str]:
    """Return ``{archive name: sha256}`` from the committed file."""
    found: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (\S+)", line.strip())
        if match:
            found[match.group(2)] = match.group(1)
    return found


def fetch(url: str) -> bytes:
    """Download ``url`` over HTTPS (the real fetcher)."""
    if not url.startswith("https://"):
        raise SetupError(f"Refusing a download that is not https: {url}")
    with urllib.request.urlopen(url, timeout=120) as answer:
        data: bytes = answer.read()
    return data


def install_client(
    target: Path,
    name: str,
    checksums: Dict[str, str],
    download: Fetch = fetch,
) -> str:
    """Download, check and install chisel for ``name`` (``linux_amd64``).

    Returns the archive's SHA-256. Raises :class:`SetupError` when
    there is no pinned checksum for it or the download does not match
    (nothing is installed then).
    """
    archive = f"chisel_{VERSION}_{name}.gz"
    expected = checksums.get(archive)
    if expected is None:
        raise SetupError(f"No pinned checksum for {archive}.")
    data = download(f"{RELEASES}/v{VERSION}/{archive}")
    digest = hashlib.sha256(data).hexdigest()
    if digest != expected:
        raise SetupError(
            f"{archive}: SHA-256 {digest} is not the pinned {expected};"
            " nothing installed."
        )
    binary = gzip.decompress(data)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=target.parent,
                                             prefix=".chisel-")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(binary)
        os.chmod(temporary, 0o755)
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return digest


def _private_dir(path: Path) -> None:
    """Make ``path``'s parent directory, mode 700 when it is new."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)


def _write_private(path: Path, text: str, exclusive: bool) -> None:
    """Write ``text`` to ``path`` with mode 600."""
    flags = os.O_WRONLY | os.O_CREAT | (os.O_EXCL if exclusive
                                        else os.O_TRUNC)
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.chmod(path, 0o600)


def credential(path: Path) -> Tuple[str, bool]:
    """Return the credential kept in ``path``, making it if missing.

    The second value tells whether it was made now. Raises
    :class:`SetupError` for an existing file others may read or one
    without a credential.
    """
    if path.exists():
        mode = stat.S_IMODE(path.stat().st_mode)
        if mode & 0o077:
            raise SetupError(
                f"{path} may be read by others (mode {mode:o}): run"
                f" `chmod 600 {path}`."
            )
        kept = path.read_text(encoding="utf-8").strip()
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}:[A-Za-z0-9_.~+/=-]{32,256}",
                            kept):
            raise SetupError(f"{path} does not hold a user:password.")
        return kept, False
    _private_dir(path)
    made = f"{USER}:{secrets.token_urlsafe(PASSWORD_BYTES)}"
    _write_private(path, made + "\n", exclusive=True)
    return made, True


def users_file(made: str, remote_port: int) -> str:
    """Return the chisel users file allowing ``made`` one reverse port."""
    allowed = rf"^R:0\.0\.0\.0:{remote_port}$"
    return json.dumps({made: [allowed]}, indent=2) + "\n"


def write_users(path: Path, made: str, remote_port: int) -> None:
    """Write the users file for the server to ``path`` (mode 600)."""
    _private_dir(path)
    _write_private(path, users_file(made, remote_port), exclusive=False)


def main(
    argv: Optional[List[str]] = None,
    out: Callable[[str], None] = print,
    download: Fetch = fetch,
) -> int:
    """Run ``client`` or ``secret``; return the exit code."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    client = commands.add_parser("client", help="install the chisel client")
    client.add_argument("--out", type=Path, default=DEFAULT_CHISEL)
    secret = commands.add_parser("secret", help="make the credential")
    secret.add_argument("--credential", type=Path,
                        default=DEFAULT_CREDENTIAL)
    secret.add_argument("--users", type=Path, default=DEFAULT_USERS)
    secret.add_argument("--remote-port", type=int,
                        default=DEFAULT_REMOTE_PORT)
    args = parser.parse_args(argv)
    try:
        if args.command == "client":
            target = args.out.expanduser()
            name = platform_name(platform.system(), platform.machine())
            digest = install_client(target, name, pinned_checksums(),
                                    download)
            out(f"chisel {VERSION} ({name}, archive SHA-256 {digest})"
                f" installed at {target}")
            return 0
        if not 1024 <= args.remote_port <= 65535:
            raise SetupError("--remote-port must be in 1024-65535.")
        path = args.credential.expanduser()
        made, new = credential(path)
        users = args.users.expanduser()
        write_users(users, made, args.remote_port)
        out(f"Credential {'made' if new else 'kept'}: {path} (mode 600)")
        out(f"Users file for the server: {users} (mode 600);"
            " `make infra-up` uploads it")
        return 0
    except SetupError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
