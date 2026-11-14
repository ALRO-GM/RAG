"""A tiny fake ``ssh`` executable for tunnel tests (never the real one).

:func:`install_fake_ssh` writes a shell script named ``ssh`` into a
directory. It appends its arguments to ``$FAKE_SSH_LOG``; a clean-up
(``... true``) exits 0; the first forward (``-N``) writes an error and
exits 255; every later forward runs until it gets SIGTERM, which it
records in ``$FAKE_SSH_LOG.term``.
"""

from pathlib import Path

SCRIPT = """#!/bin/sh
echo "$*" >> "$FAKE_SSH_LOG"
last=""
for arg in "$@"; do last="$arg"; done
if [ "$last" = "true" ]; then exit 0; fi
forwards=$(grep -c -- '^-N ' "$FAKE_SSH_LOG")
if [ "$forwards" -le 1 ]; then
  echo "Error: remote port forwarding failed for listen path" >&2
  exit 255
fi
trap 'echo terminated >> "$FAKE_SSH_LOG.term"; exit 0' TERM
while true; do sleep 0.05; done
"""


def install_fake_ssh(directory: Path) -> Path:
    """Write the fake ``ssh`` into ``directory``; return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "ssh"
    path.write_text(SCRIPT, encoding="utf-8")
    path.chmod(0o755)
    return path
