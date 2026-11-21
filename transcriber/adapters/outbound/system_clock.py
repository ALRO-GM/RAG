"""Clock adapters: the current time (timezone-aware) and a monotonic timer."""

import time
from datetime import datetime, timezone


class SystemClock:
    """The current time of this machine, in UTC."""

    def now(self) -> datetime:
        """Return the current time with its UTC offset."""
        return datetime.now(timezone.utc)


class SystemTimer:
    """This machine's monotonic clock and :func:`time.sleep`."""

    def monotonic(self) -> float:
        """Return :func:`time.monotonic`."""
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        """Wait ``seconds`` (nothing for zero or less)."""
        if seconds > 0:
            time.sleep(seconds)
