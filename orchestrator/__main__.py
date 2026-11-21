"""Entry point: ``python -m orchestrator <command>``."""

from orchestrator.adapters.inbound.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
