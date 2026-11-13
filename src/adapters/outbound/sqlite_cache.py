"""Cache adapter: a key-value store in SQLite."""

import sqlite3
from pathlib import Path
from typing import Optional

from src.domain.errors import RagError

CACHE_FILE = "query_cache.sqlite"


class SqliteCache:
    """A small persistent key-value store.

    Keys are opaque: the use cases derive them (including the index
    fingerprint where results depend on the index). A cache failure never
    breaks a query: errors simply turn into cache misses.
    """

    def __init__(self, directory: str, enabled: bool = True):
        """Open (or create) the cache file inside ``directory``."""
        self.enabled = enabled
        self._conn: Optional[sqlite3.Connection] = None
        if not enabled:
            return
        if not Path(directory).is_dir():
            # No index directory yet: never create one as a side effect.
            return
        try:
            self._conn = sqlite3.connect(
                str(Path(directory) / CACHE_FILE), check_same_thread=False
            )
            with self._conn:
                self._conn.execute(
                    "CREATE TABLE IF NOT EXISTS cache "
                    "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
                )
        except (sqlite3.Error, OSError):
            self._conn = None

    def get(self, key: str) -> Optional[str]:
        """Return the cached value for ``key`` or ``None``."""
        if self._conn is None:
            return None
        try:
            row = self._conn.execute(
                "SELECT value FROM cache WHERE key = ?", (key,)
            ).fetchone()
        except sqlite3.Error:
            return None
        return str(row[0]) if row else None

    def set(self, key: str, value: str) -> None:
        """Store ``value`` under ``key``."""
        if self._conn is None:
            return
        try:
            with self._conn:
                self._conn.execute(
                    "INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)",
                    (key, value),
                )
        except sqlite3.Error:
            pass

    def clear(self) -> int:
        """Delete every entry and return how many were removed."""
        if self._conn is None:
            raise RagError("Cache is disabled or unavailable.")
        with self._conn:
            count = self._conn.execute("DELETE FROM cache").rowcount
        return int(count)

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
