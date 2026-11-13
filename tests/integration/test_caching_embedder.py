"""Persistent embedding cache wrapped around a fake embedder."""

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest

from src.adapters.outbound.caching_embedder import CachingEmbedder
from tests.fakes import FakeEmbedder

TEXTS = ["load a lora adapter", "kv cache blocks", "fp8 quantization"]


def cached(
    tmp_path: Path, namespace: str = "model|128"
) -> Tuple[CachingEmbedder, FakeEmbedder]:
    """Return a caching embedder under ``tmp_path`` and its inner fake."""
    inner = FakeEmbedder("inner-model")
    return CachingEmbedder(inner, str(tmp_path / "emb"), namespace), inner


def test_first_call_embeds_and_matches_the_inner_model(
    tmp_path: Path
) -> None:
    """Uncached texts go to the inner embedder; vectors are unchanged."""
    embedder, inner = cached(tmp_path)
    vectors = embedder.encode(TEXTS)
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(vectors, FakeEmbedder().encode(TEXTS))
    assert inner.encoded == TEXTS
    assert (embedder.hits, embedder.misses) == (0, 3)
    assert embedder.model_name == "inner-model"


def test_second_call_hits_the_cache(tmp_path: Path) -> None:
    """Known texts are served without calling the inner embedder."""
    embedder, inner = cached(tmp_path)
    first = embedder.encode(TEXTS)
    second = embedder.encode(TEXTS)
    assert inner.calls == 1
    np.testing.assert_array_equal(first, second)
    assert (embedder.hits, embedder.misses) == (3, 3)


def test_only_new_texts_are_embedded_and_order_is_kept(
    tmp_path: Path
) -> None:
    """Hits and misses are merged back in the order of the request."""
    embedder, inner = cached(tmp_path)
    embedder.encode(TEXTS[:2])
    mixed = ["draft model", TEXTS[1], TEXTS[0], TEXTS[2]]
    vectors = embedder.encode(mixed)
    assert inner.encoded[2:] == ["draft model", TEXTS[2]]
    np.testing.assert_allclose(vectors, FakeEmbedder().encode(mixed))
    assert (embedder.hits, embedder.misses) == (2, 4)


def test_duplicates_in_one_call_are_embedded_once(tmp_path: Path) -> None:
    """A text repeated in a batch reaches the inner embedder once."""
    embedder, inner = cached(tmp_path)
    vectors = embedder.encode([TEXTS[0], TEXTS[1], TEXTS[0], TEXTS[0]])
    assert inner.encoded == TEXTS[:2]
    assert vectors.shape == (4, inner.dim)
    np.testing.assert_array_equal(vectors[0], vectors[2])
    np.testing.assert_array_equal(vectors[0], vectors[3])
    assert not np.array_equal(vectors[0], vectors[1])
    assert embedder.hits + embedder.misses == 4


def test_cache_persists_across_instances(tmp_path: Path) -> None:
    """A new embedder over the same directory reuses the vectors."""
    first, _ = cached(tmp_path)
    expected = first.encode(TEXTS)
    second, inner = cached(tmp_path)
    np.testing.assert_array_equal(second.encode(TEXTS), expected)
    assert inner.calls == 0
    assert (second.hits, second.misses) == (3, 0)
    assert (second.directory / "keys.json").is_file()
    assert (second.directory / "vectors.npy").is_file()
    second.encode(["new text"])
    third, inner = cached(tmp_path)
    third.encode([*TEXTS, "new text"])
    assert inner.calls == 0


def test_namespaces_do_not_share_vectors(tmp_path: Path) -> None:
    """Another model configuration has its own cache directory."""
    first, _ = cached(tmp_path, "model-a|128")
    first.encode(TEXTS)
    other, inner = cached(tmp_path, "model-a|256")
    other.encode(TEXTS)
    assert inner.encoded == TEXTS
    assert other.directory != first.directory
    assert other.directory.parent == first.directory.parent


def test_empty_request(tmp_path: Path) -> None:
    """No text gives no row, with an empty or a filled cache."""
    embedder, _ = cached(tmp_path)
    assert embedder.encode([]).shape[0] == 0
    embedder.encode(TEXTS)
    assert embedder.encode([]).shape == (0, 32)
    assert (embedder.hits, embedder.misses) == (0, 3)


def corrupt(directory: Path, kind: str) -> None:
    """Damage the cache files in ``directory``."""
    if kind == "keys":
        (directory / "keys.json").write_text("{not json")
    elif kind == "keys-not-a-list":
        (directory / "keys.json").write_text('{"a": 1}')
    elif kind == "vectors":
        (directory / "vectors.npy").write_bytes(b"junk, not numpy")
    elif kind == "vectors-missing":
        (directory / "vectors.npy").unlink()
    elif kind == "vectors-empty":
        (directory / "vectors.npy").write_bytes(b"")
    else:
        keys = json.loads((directory / "keys.json").read_text())
        (directory / "keys.json").write_text(json.dumps(keys + ["extra"]))


@pytest.mark.parametrize("kind", [
    "keys", "keys-not-a-list", "vectors", "vectors-missing", "length",
    "vectors-empty",
])
def test_corrupt_cache_is_ignored_and_rebuilt(
    tmp_path: Path, kind: str
) -> None:
    """A broken cache only costs a re-embedding; it is then rewritten."""
    first, _ = cached(tmp_path)
    expected = first.encode(TEXTS)
    corrupt(first.directory, kind)
    second, inner = cached(tmp_path)
    np.testing.assert_array_equal(second.encode(TEXTS), expected)
    assert inner.encoded == TEXTS
    assert (second.hits, second.misses) == (0, 3)
    third, inner = cached(tmp_path)
    third.encode(TEXTS)
    assert inner.calls == 0
