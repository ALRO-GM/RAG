"""Recovery store adapter: recovered live audio on disk, temporarily (#265).

Implements :class:`~transcriber.application.ports.RecoveryStore` under
``<data_dir>/live-recovery/<session>/`` (the data directory is ignored
by git), never under «Sesiones»:

``recovery.json``
    the job's state: offsets and sizes of the chunks, the step, tries,
    counts and times; no audio and no words.
``audio/<n>.pcm``
    the uploaded chunks (16-bit PCM, channels interleaved), deleted as
    soon as their transcription succeeds, when the job ends or at its
    expiry (:meth:`~transcriber.application.live_recovery.LiveRecovery.
    sweep`).
``text/<name>.json``
    the job's working documents (the live transcript before the merge,
    the recovered parts' transcripts), deleted when the job ends.

Directories are private (700) and files are written atomically through
a private temporary file (600). Session ids are checked against the id
pattern before any path is built, and chunk numbers are integers.
"""

import io
import json
import re
import shutil
import wave
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from transcriber.adapters.outbound.fs_storage import (
    atomic_write,
    child_path,
    read_json,
    write_json,
)
from transcriber.domain.errors import InvalidInputError
from transcriber.domain.live_recovery import SAMPLE_RATE, RecoveryJob
from transcriber.domain.models import SESSION_ID_PATTERN

RECOVERY_DIR = "live-recovery"
JOB_FILE = "recovery.json"
AUDIO_DIR = "audio"
TEXT_DIR = "text"
TEXT_NAME = r"^[a-z][a-z0-9-]{0,31}$"


class FileSystemRecoveryStore:
    """The recovered live audio of a data directory."""

    def __init__(self, data_dir: Path) -> None:
        """Keep the root; nothing is created until the first write."""
        self.root = Path(data_dir) / RECOVERY_DIR

    def _folder(self, session_id: str, create: bool = False) -> Path:
        """The session's folder, after checking the id."""
        if create:
            self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        folder = child_path(self.root, session_id, SESSION_ID_PATTERN,
                            "session") if self.root.exists() \
            else self.root / _checked(session_id)
        if create:
            folder.mkdir(mode=0o700, exist_ok=True)
        return folder

    def _sub(self, session_id: str, name: str) -> Path:
        """A private sub-folder of the session's, created."""
        folder = self._folder(session_id, create=True) / name
        folder.mkdir(mode=0o700, exist_ok=True)
        return folder

    def ids(self) -> List[str]:
        """Return the sessions with a folder here."""
        if not self.root.is_dir():
            return []
        return sorted(p.name for p in self.root.iterdir()
                      if p.is_dir() and not p.is_symlink()
                      and not p.name.startswith("."))

    def load(self, session_id: str) -> Optional[RecoveryJob]:
        """Return the session's job, or ``None``."""
        path = self._folder(session_id) / JOB_FILE
        try:
            data = read_json(path)
        except FileNotFoundError:
            return None
        if not isinstance(data, dict):
            raise ValueError(f"{JOB_FILE} is not an object")
        return RecoveryJob.from_dict(data)

    def save(self, job: RecoveryJob) -> None:
        """Write the job's state atomically."""
        write_json(self._folder(job.session_id, create=True) / JOB_FILE,
                   job.to_dict())

    def put_chunk(self, session_id: str, number: int, data: bytes) -> None:
        """Keep chunk ``number`` (replacing it)."""
        atomic_write(self._sub(session_id, AUDIO_DIR) / f"{int(number)}.pcm",
                     data)

    def _chunk(self, session_id: str, number: int) -> bytes:
        """Return one chunk's bytes (empty when it is gone)."""
        path = self._folder(session_id) / AUDIO_DIR / f"{int(number)}.pcm"
        try:
            return path.read_bytes()
        except FileNotFoundError:
            return b""

    def wav(
        self, session_id: str, chunks: Mapping[str, Sequence[int]],
        part: Tuple[int, int], spans: Sequence[Tuple[int, int]],
        channels: int, channel: Optional[int],
    ) -> bytes:
        """A mono 16 kHz WAV of ``part`` (see the port).

        Samples outside ``spans``, and time no chunk covers, are
        silence. Only the chunks that overlap the part are read; the
        caller bounds the part's length (one recovered group).
        """
        start, stop = part
        length = max(stop - start, 0)
        out = np.zeros(length, dtype=np.int16)
        keep = np.zeros(length, dtype=bool)
        for a, b in spans:
            lo, hi = max(a, start) - start, min(b, stop) - start
            if hi > lo:
                keep[lo:hi] = True
        for number, (offset, samples) in chunks.items():
            begin = int(offset) - start
            if begin >= length or begin + int(samples) <= 0:
                continue
            data = self._chunk(session_id, int(number))
            pcm = np.frombuffer(data[:len(data) - len(data) % (2 * channels)],
                                dtype="<i2").reshape(-1, channels)
            mono = pcm[:, channel] if channel is not None and channels == 2 \
                else pcm.mean(axis=1).astype(np.int16) if channels == 2 \
                else pcm[:, 0]
            lo, hi = max(begin, 0), min(begin + len(mono), length)
            if hi > lo:
                out[lo:hi] = mono[lo - begin:hi - begin]
        out[~keep] = 0
        return _wav(out)

    def missing(self, session_id: str, numbers: Sequence[str]) -> List[str]:
        """The chunk numbers among ``numbers`` whose file is gone."""
        folder = self._folder(session_id) / AUDIO_DIR
        return [n for n in numbers
                if not (folder / f"{int(n)}.pcm").is_file()]

    def audio_bytes(self) -> int:
        """Return the bytes of every chunk kept."""
        total = 0
        for session_id in self.ids():
            folder = self.root / session_id / AUDIO_DIR
            if folder.is_dir():
                total += sum(p.stat().st_size for p in folder.glob("*.pcm"))
        return total

    def delete_audio(self, session_id: str) -> int:
        """Delete the session's chunks; return how many files went."""
        folder = self._folder(session_id) / AUDIO_DIR
        if not folder.is_dir():
            return 0
        count = len(list(folder.iterdir()))
        shutil.rmtree(folder)
        return count

    def save_text(
        self, session_id: str, name: str, document: Mapping[str, Any],
    ) -> None:
        """Keep a working document (atomically)."""
        atomic_write(
            self._sub(session_id, TEXT_DIR) / f"{_name(name)}.json",
            json.dumps(document, ensure_ascii=False).encode("utf-8"))

    def read_text(
        self, session_id: str, name: str,
    ) -> Optional[Dict[str, Any]]:
        """Return a working document, or ``None``."""
        path = self._folder(session_id) / TEXT_DIR / f"{_name(name)}.json"
        try:
            data = read_json(path)
        except FileNotFoundError:
            return None
        return data if isinstance(data, dict) else None

    def delete_texts(self, session_id: str) -> None:
        """Delete the working documents."""
        shutil.rmtree(self._folder(session_id) / TEXT_DIR,
                      ignore_errors=True)

    def delete(self, session_id: str) -> None:
        """Delete the session's folder and everything in it."""
        shutil.rmtree(self._folder(session_id), ignore_errors=True)


def _checked(session_id: str) -> str:
    """``session_id`` checked against the pattern (no root yet)."""
    if not isinstance(session_id, str) \
            or not re.fullmatch(SESSION_ID_PATTERN, session_id):
        raise InvalidInputError(f"Invalid session id {session_id!r}.")
    return session_id


def _name(name: str) -> str:
    """A working document's name, checked."""
    if not re.fullmatch(TEXT_NAME, name):
        raise ValueError(f"invalid document name {name!r}")
    return name


def _wav(samples: "np.ndarray[Any, np.dtype[np.int16]]") -> bytes:
    """``samples`` (mono, 16 kHz) as WAV bytes."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        out.writeframes(samples.astype("<i2").tobytes())
    return buffer.getvalue()
