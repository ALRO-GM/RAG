"""Entry point: ``uv run python -m transcriber <command> [options]``."""


def main() -> None:
    """Run the Fire CLI."""
    import fire

    from transcriber.adapters.inbound.cli import TranscriberCLI

    fire.Fire(TranscriberCLI(), name="transcriber")


if __name__ == "__main__":
    main()
