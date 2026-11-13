"""Entry point: ``uv run python -m src <command> [options]``."""

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")


def main() -> None:
    """Run the Fire CLI."""
    import fire

    from src.adapters.inbound.cli import RagCLI

    fire.Fire(RagCLI)


if __name__ == "__main__":
    main()
