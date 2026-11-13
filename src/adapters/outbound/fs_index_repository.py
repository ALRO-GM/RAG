"""Index repository adapter: the index as files in a directory.

Layout of ``index_dir``::

    manifest.json           IndexManifest (written last = commit marker)
    chunks.jsonl            one Chunk per line
    bm25_weights.npz        precomputed BM25 weights (CSC sparse matrix)
    bm25_vocabulary.json    term -> column
    embeddings.npy          float32 (chunks x dim), memory-mapped on load
    records.pkl             per-file records for incremental indexing
"""

import json
import pickle
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
from pydantic import ValidationError
from scipy import sparse

from src.adapters.outbound.bm25_index import BM25Index
from src.application.indexing import INDEX_FORMAT_VERSION
from src.application.ports import FileRecord, IndexSnapshot
from src.domain.errors import CorruptedDataError, NotFoundError
from src.domain.models import Chunk, IndexManifest

MANIFEST_FILE = "manifest.json"
CHUNKS_FILE = "chunks.jsonl"
BM25_FILE = "bm25_weights.npz"
VOCAB_FILE = "bm25_vocabulary.json"
EMBEDDINGS_FILE = "embeddings.npy"
RECORDS_FILE = "records.pkl"


class FileSystemIndexRepository:
    """Persist :class:`IndexSnapshot` objects in a directory.

    Loading is cached in-process per (directory, fingerprint): a server
    or a batch run loads the files once, and a re-index (new
    fingerprint) is picked up automatically.
    """

    def __init__(self, index_dir: str):
        """Store the index in ``index_dir``."""
        self.index_dir = Path(index_dir)

    def load_records(
        self, params: Mapping[str, object]
    ) -> Dict[str, FileRecord]:
        """Return cached file records if built with the same ``params``."""
        path = self.index_dir / RECORDS_FILE
        if not path.is_file():
            return {}
        try:
            with open(path, "rb") as handle:
                saved = pickle.load(handle)
        except (OSError, pickle.UnpicklingError, EOFError, AttributeError,
                ImportError):
            return {}
        if not isinstance(saved, dict) or saved.get("params") != dict(params):
            return {}
        records = saved.get("records")
        return records if isinstance(records, dict) else {}

    def save(
        self,
        snapshot: IndexSnapshot,
        records: Mapping[str, FileRecord],
        params: Mapping[str, object],
    ) -> None:
        """Write every artefact; the manifest goes last."""
        if not isinstance(snapshot.lexical, BM25Index):
            raise TypeError("FileSystemIndexRepository stores BM25 indexes.")
        directory = self.index_dir
        directory.mkdir(parents=True, exist_ok=True)
        weights, vocabulary = snapshot.lexical.state()
        sparse.save_npz(directory / BM25_FILE, weights.tocsc())
        with open(directory / VOCAB_FILE, "w", encoding="utf-8") as handle:
            json.dump(vocabulary, handle)
        with open(directory / CHUNKS_FILE, "w", encoding="utf-8") as handle:
            for chunk in snapshot.chunks:
                handle.write(chunk.model_dump_json() + "\n")
        emb_path = directory / EMBEDDINGS_FILE
        if snapshot.embeddings is not None:
            np.save(emb_path, snapshot.embeddings)
        elif emb_path.exists():
            emb_path.unlink()
        with open(directory / RECORDS_FILE, "wb") as handle:
            pickle.dump(
                {"params": dict(params), "records": dict(records)},
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        with open(directory / MANIFEST_FILE, "w", encoding="utf-8") as handle:
            handle.write(snapshot.manifest.model_dump_json(indent=2))

    def load(self) -> IndexSnapshot:
        """Return the persisted index (cached while unchanged)."""
        manifest = read_manifest(self.index_dir)
        return _load_cached(str(self.index_dir.resolve()),
                            manifest.fingerprint)


def read_manifest(path: Path) -> IndexManifest:
    """Read and validate the manifest of the index at ``path``."""
    manifest_path = path / MANIFEST_FILE
    if not manifest_path.is_file():
        raise NotFoundError(
            f"No index found in {path}. Run the 'index' command first."
        )
    try:
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = IndexManifest.model_validate_json(handle.read())
    except (OSError, ValidationError) as exc:
        raise CorruptedDataError(f"Corrupted index manifest: {exc}")
    if manifest.version != INDEX_FORMAT_VERSION:
        raise CorruptedDataError(
            "The index was built by another version. Re-run "
            "'index --force'."
        )
    return manifest


@lru_cache(maxsize=2)
def _load_cached(directory: str, fingerprint: str) -> IndexSnapshot:
    """Load the index files; cached per (directory, fingerprint)."""
    path = Path(directory)
    manifest = read_manifest(path)
    try:
        weights: Any = sparse.load_npz(path / BM25_FILE).tocsc()
        with open(path / VOCAB_FILE, encoding="utf-8") as handle:
            vocabulary: Dict[str, int] = json.load(handle)
        chunks = _read_chunks(path / CHUNKS_FILE)
        embeddings = None
        if manifest.semantic:
            embeddings = np.load(path / EMBEDDINGS_FILE, mmap_mode="r")
    except (OSError, ValueError, ValidationError,
            zipfile.BadZipFile) as exc:
        raise CorruptedDataError(
            f"Corrupted index in {path} ({exc}). Re-run 'index --force'."
        )
    if len(chunks) != weights.shape[0] or (
        embeddings is not None and embeddings.shape[0] != len(chunks)
    ):
        raise CorruptedDataError(
            "Index files are inconsistent. Re-run 'index --force'."
        )
    return IndexSnapshot(
        manifest, chunks, BM25Index(weights, vocabulary), embeddings
    )


def _read_chunks(path: Path) -> List[Chunk]:
    """Read the chunk metadata (one JSON object per line)."""
    with open(path, encoding="utf-8") as handle:
        return [
            Chunk.model_validate_json(line) for line in handle if line.strip()
        ]
