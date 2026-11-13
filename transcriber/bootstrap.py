"""Composition root: builds the transcriber from its settings.

Later tickets wire the adapters and use cases here; for now the
container only carries the settings.
"""

from dataclasses import dataclass
from typing import Optional

from transcriber.config import TranscriberSettings


@dataclass(frozen=True)
class TranscriberContainer:
    """Everything the inbound adapters need, built once."""

    settings: TranscriberSettings


def build_container(
    settings: Optional[TranscriberSettings] = None,
) -> TranscriberContainer:
    """Return the container for ``settings`` (the defaults if omitted)."""
    return TranscriberContainer(settings=settings or TranscriberSettings())
