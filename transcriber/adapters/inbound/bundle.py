"""Build the static front end that the server serves from ``/srv``.

``python -m transcriber.adapters.inbound.bundle --out dist/site --commit
<sha>`` copies the pages of ``transcriber/adapters/inbound/web`` (not
their ``package.json``, which only tells node the files are ES modules
for the tests) to ``--out`` and writes ``version.json`` with the commit
and the build time, which the deploy job's smoke test reads back from
https://rag.peer4u.app/version.json.

Standard library only: CI runs it without installing the project.
"""

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional, Sequence

WEB_DIR = Path(__file__).resolve().parent / "web"
"""The pages, as written (no build step)."""

LEFT_OUT = ("package.json", "node_modules", "*.test.*", ".*")
"""Names never copied to the bundle."""

VERSION_FILE = "version.json"


def build_bundle(
    out: Path,
    commit: str,
    source: Path = WEB_DIR,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> List[str]:
    """Write the bundle to ``out`` (replaced if it exists).

    Return the relative paths written, sorted. ``out`` may not be
    ``source``, contain it or lie inside it.
    """
    out = out.resolve()
    source = source.resolve()
    if out == source or out in source.parents or source in out.parents:
        raise ValueError(f"{out} would overwrite the pages it copies.")
    if not commit.strip():
        raise ValueError("The commit is required.")
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(source, out, ignore=shutil.ignore_patterns(*LEFT_OUT))
    version = {
        "commit": commit.strip(),
        "built_at": now().astimezone(timezone.utc).isoformat(
            timespec="seconds"
        ),
    }
    (out / VERSION_FILE).write_text(
        json.dumps(version) + "\n", encoding="utf-8"
    )
    return sorted(
        p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Build the bundle from the command line; return the exit code."""
    parser = argparse.ArgumentParser(
        prog="python -m transcriber.adapters.inbound.bundle",
        description="Build the transcriber's static front end.",
    )
    parser.add_argument("--out", type=Path, default=Path("dist/site"))
    parser.add_argument(
        "--commit", default="local", help="commit written to version.json"
    )
    args = parser.parse_args(argv)
    try:
        files = build_bundle(args.out, args.commit)
    except (OSError, ValueError) as exc:
        print(f"bundle: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {len(files)} files to {args.out} (commit {args.commit}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
