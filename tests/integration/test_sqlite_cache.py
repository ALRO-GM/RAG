"""SQLite cache adapter."""

from pathlib import Path

import pytest

from src.adapters.outbound.sqlite_cache import CACHE_FILE, SqliteCache
from src.domain.errors import RagError


def test_round_trip_and_persistence(tmp_path: Path) -> None:
    """Values survive closing and reopening the cache."""
    (tmp_path / "idx").mkdir()
    cache = SqliteCache(str(tmp_path / "idx"))
    assert cache.get("k") is None
    cache.set("k", "v1")
    cache.set("k", "v2")
    cache.set("other", "é ✓")
    assert cache.get("k") == "v2"
    cache.close()
    cache.close()
    assert (tmp_path / "idx" / CACHE_FILE).is_file()
    reopened = SqliteCache(str(tmp_path / "idx"))
    assert reopened.get("k") == "v2"
    assert reopened.get("other") == "é ✓"
    assert reopened.clear() == 2
    assert reopened.get("k") is None
    assert reopened.clear() == 0


def test_disabled_cache(tmp_path: Path) -> None:
    """A disabled cache stores nothing and creates no file."""
    cache = SqliteCache(str(tmp_path / "idx"), enabled=False)
    cache.set("k", "v")
    assert cache.get("k") is None
    assert not (tmp_path / "idx").exists()
    with pytest.raises(RagError, match="disabled"):
        cache.clear()


def test_closed_cache_degrades_to_misses(tmp_path: Path) -> None:
    """After closing, reads miss and writes are ignored."""
    cache = SqliteCache(str(tmp_path))
    cache.set("k", "v")
    cache.close()
    cache.set("k", "w")
    assert cache.get("k") is None


def test_unusable_location_degrades(tmp_path: Path) -> None:
    """A cache that cannot be created never breaks queries."""
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    try:
        cache = SqliteCache(str(blocker / "sub"))
    except OSError:
        pytest.fail("cache creation must not raise")
    cache.set("k", "v")
    assert cache.get("k") is None


def test_corrupted_database_degrades(tmp_path: Path) -> None:
    """A garbage cache file turns into cache misses."""
    (tmp_path / CACHE_FILE).write_bytes(b"this is not sqlite" * 100)
    cache = SqliteCache(str(tmp_path))
    cache.set("k", "v")
    assert cache.get("k") is None


def test_missing_directory_is_not_created(tmp_path: Path) -> None:
    """A cache pointed at a missing index dir creates nothing."""
    target = tmp_path / "nowhere"
    cache = SqliteCache(str(target))
    cache.set("k", "v")
    assert cache.get("k") is None
    assert not target.exists()
