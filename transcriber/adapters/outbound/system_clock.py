"""Clock adapter: the system's current time, timezone-aware."""

from datetime import datetime, timezone


class SystemClock:
    """The current time of this machine, in UTC."""

    def now(self) -> datetime:
        """Return the current time with its UTC offset."""
        return datetime.now(timezone.utc)
