"""Entry point: ``uv run --no-sync python -m transcriber <command> ...``."""

import sys


def main() -> None:
    """Run the command line and exit with its code."""
    from transcriber.adapters.inbound.cli import main as run

    sys.exit(run())


if __name__ == "__main__":
    main()
