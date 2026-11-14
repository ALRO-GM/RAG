"""A tiny fake ``chisel`` executable for tunnel tests (never the real one).

:func:`install_fake_chisel` writes a shell script named ``chisel`` into a
directory. It appends its arguments, and whether ``AUTH`` was set, to
the log file named at install time (``<log>.env``); the first client
refuses like a wrong credential and exits 1; every later client writes
chisel's ``Connected`` line and runs until it gets SIGTERM, which it
records in ``<log>.term``, or until the process that started it is
gone: a ``serve`` killed by a failing test does not leave it running
forever. The log's path is written into the script
because the tunnel passes the client only a few environment variables.
"""

from pathlib import Path

SCRIPT = """#!/bin/sh
FAKE_CHISEL_LOG='%s'
echo "$*" >> "$FAKE_CHISEL_LOG"
if [ -n "$AUTH" ]; then echo "AUTH=set" >> "$FAKE_CHISEL_LOG.env"; fi
runs=$(grep -c -- '^client ' "$FAKE_CHISEL_LOG")
if [ "$runs" -le 1 ]; then
  echo "2026/11/14 08:22:08 client: Connection error: ssh: handshake failed:\
 ssh: unable to authenticate" >&2
  echo "2026/11/14 08:22:08 connection attempts exhausted" >&2
  exit 1
fi
trap 'echo terminated >> "$FAKE_CHISEL_LOG.term"; exit 0' TERM
echo "2026/11/14 08:22:15 client: Connected (Latency 171us)" >&2
parent=$PPID
while kill -0 "$parent" 2>/dev/null; do sleep 0.05; done
"""


def install_fake_chisel(directory: Path, log: Path) -> Path:
    """Write the fake ``chisel`` logging to ``log``; return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "chisel"
    path.write_text(SCRIPT % log, encoding="utf-8")
    path.chmod(0o755)
    return path


def write_credential(path: Path) -> str:
    """Write a credential file of mode 600; return the credential."""
    credential = "workstation:" + "s3cret-" * 6
    path.write_text(credential + "\n", encoding="utf-8")
    path.chmod(0o600)
    return credential
