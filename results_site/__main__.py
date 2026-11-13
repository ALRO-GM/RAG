"""Command line: ``python -m results_site build [--out DIR] [--root DIR]``.

``--commit`` overrides the commit shown on the site (CI passes the one
it is building, so the deploy can check that it went live).
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from results_site.builder import SiteBuilder
from results_site.summaries import SiteError


def git_commit(root: Path) -> str:
    """Short commit of ``root``, or ``""`` outside a git checkout."""
    try:
        done = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=False, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return done.stdout.strip() if done.returncode == 0 else ""


def main(argv: Optional[List[str]] = None) -> int:
    """Run the command; return the process exit code."""
    parser = argparse.ArgumentParser(
        prog="python -m results_site",
        description="Build the static results site.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="write the site to --out")
    build.add_argument("--root", type=Path, default=Path("."))
    build.add_argument("--out", type=Path, default=Path("dist/site"))
    build.add_argument(
        "--commit", default=None,
        help="commit shown on the site (default: git HEAD of --root)",
    )
    args = parser.parse_args(argv)
    commit = args.commit if args.commit is not None else git_commit(args.root)
    try:
        manifest = SiteBuilder(args.root, args.out).build(commit=commit)
    except (SiteError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    docs = sum(len(g["items"]) for g in manifest["docs"])
    ready = ", ".join(k for k, v in manifest["sections"].items() if v)
    print(f"Site written to {args.out} ({docs} documents; "
          f"sections: {ready or 'none'}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
