"""Progress adapter: tqdm progress bars on stderr."""

from typing import Iterable, Iterator, TypeVar

from tqdm import tqdm

T = TypeVar("T")


class TqdmProgress:
    """Display a tqdm bar for each tracked loop."""

    def __init__(self, enabled: bool = True):
        """Show bars only when ``enabled``."""
        self.enabled = enabled

    def track(
        self, items: Iterable[T], total: int, description: str
    ) -> Iterator[T]:
        """Yield ``items`` while updating a progress bar."""
        yield from tqdm(
            items, total=total, desc=description, disable=not self.enabled
        )
