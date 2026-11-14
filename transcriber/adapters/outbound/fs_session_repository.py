"""Session repository: one directory per session on the local disk.

Layout under ``<root>/sessions/<id>/``:

``session.json``
    the :class:`~transcriber.domain.models.Session`.
``audio.pcm`` and ``chunks.json``
    while recording: the raw :data:`PCM_FORMAT` chunks appended in
    order, and the byte offset at which each chunk ends.
``audio.flac``
    once a recorded session is closed (saved in any state other than
    ``recording``), the PCM is encoded losslessly and the two files
    above are removed.
``audio.<ext>``
    an imported recording, kept as it came.
``words.json`` and ``transcript.json``
    the attributed words and the transcript (the transcriber's own
    format, words included; the contract is made from it on publish).

JSON and audio files are written atomically (:func:`atomic_write`).
Chunk appends are made safe by ``chunks.json``: appending chunk ``n``
first cuts ``audio.pcm`` back to where chunk ``n - 1`` ended, so bytes
left by a process killed mid-append are never kept twice.
"""

import io
import os
import re
import shutil
import threading
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, TypeVar

from pydantic import TypeAdapter, ValidationError

from transcriber.adapters.outbound.av_audio_decoder import (
    PCM_RATE,
    pcm_samples,
)
from transcriber.adapters.outbound.fs_storage import (
    atomic_write,
    child_path,
    read_json,
    write_json,
)
from transcriber.application.ports import PCM_FORMAT, RecordedAudio
from transcriber.domain.errors import (
    InvalidInputError,
    NotFoundError,
    StorageError,
)
from transcriber.domain.models import (
    SESSION_ID_PATTERN,
    Session,
    SessionState,
    Transcript,
    Word,
)

SESSION_FILE = "session.json"
PCM_FILE = "audio.pcm"
CHUNKS_FILE = "chunks.json"
CLOSED_FORMAT = "flac"
WORDS_FILE = "words.json"
TRANSCRIPT_FILE = "transcript.json"
AUDIO_STEM = "audio"
FORMAT_PATTERN = r"^[a-z0-9]{1,10}$"

_WORDS = TypeAdapter(List[Word])
T = TypeVar("T")


class FileSystemSessionRepository:
    """Sessions, their audio, words and transcripts under ``root``."""

    def __init__(self, root: Path) -> None:
        """Keep sessions under ``root / "sessions"`` (created on write)."""
        self.root = Path(root) / "sessions"
        self._lock = threading.RLock()

    def _directory(self, session_id: str) -> Path:
        """Return the directory of ``session_id`` after checking the id."""
        return child_path(
            self.root, session_id, SESSION_ID_PATTERN, "session"
        )

    def _existing(self, session_id: str) -> Path:
        """Return the directory of a stored session or raise NotFound."""
        directory = self._directory(session_id)
        if not (directory / SESSION_FILE).is_file():
            raise NotFoundError(f"No session {session_id}.")
        return directory

    def get(self, session_id: str) -> Optional[Session]:
        """Return the session, or ``None`` if it does not exist."""
        path = self._directory(session_id) / SESSION_FILE
        try:
            data = read_json(path)
        except FileNotFoundError:
            return None
        return _parse(Session.model_validate, data, path)

    def ids(self) -> List[str]:
        """Return the id of every stored session, sorted.

        A stored session is a directory with a valid id holding a
        ``session.json``; anything else under the root is ignored.
        """
        if not self.root.is_dir():
            return []
        return sorted(
            entry.name for entry in self.root.iterdir()
            if re.fullmatch(SESSION_ID_PATTERN, entry.name)
            and entry.is_dir() and not entry.is_symlink()
            and (entry / SESSION_FILE).is_file()
        )

    def create(self, session: Session) -> None:
        """Store a new session; :class:`InvalidInputError` if it exists."""
        with self._lock:
            directory = self._directory(session.id)
            if (directory / SESSION_FILE).exists():
                raise InvalidInputError(
                    f"Session {session.id} already exists."
                )
            directory.mkdir(parents=True, exist_ok=True)
            write_json(
                directory / SESSION_FILE, session.model_dump(mode="json")
            )

    def save(self, session: Session) -> None:
        """Replace the stored session with ``session``.

        A session saved in any state but ``recording`` whose audio is
        still raw PCM has its audio encoded as ``audio.flac`` first.
        """
        with self._lock:
            directory = self._existing(session.id)
            if session.state is not SessionState.RECORDING:
                self._close_audio(directory)
            write_json(
                directory / SESSION_FILE, session.model_dump(mode="json")
            )
            if session.state is not SessionState.RECORDING:
                for name in (PCM_FILE, CHUNKS_FILE):
                    (directory / name).unlink(missing_ok=True)

    def delete(self, session_id: str) -> None:
        """Remove the session with its audio, words and transcript."""
        with self._lock:
            directory = self._directory(session_id)
            if directory.is_dir():
                shutil.rmtree(directory)

    def append_chunk(self, session_id: str, number: int, data: bytes) -> None:
        """Store chunk ``number`` (from 0) of :data:`PCM_FORMAT` audio.

        Idempotent per chunk number, as the port requires: chunk
        ``number`` replaces whatever followed chunk ``number - 1`` (a
        chunk half-written by a killed process, or the same chunk
        stored before), and later chunks are dropped; a gap in the
        numbering is an :class:`InvalidInputError`.
        """
        with self._lock:
            directory = self._existing(session_id)
            ends = self._chunk_ends(directory)
            if number < 0 or number > len(ends):
                raise InvalidInputError(
                    f"Session {session_id} expects chunk {len(ends)},"
                    f" got {number}."
                )
            start = ends[number - 1] if number else 0
            pcm = directory / PCM_FILE
            size = pcm.stat().st_size if pcm.exists() else 0
            if size < start:
                raise StorageError(
                    f"{PCM_FILE} of {session_id} is shorter than its"
                    f" {CHUNKS_FILE} says."
                )
            # Append mode writes at the end, which the cut moves back to
            # where chunk ``number - 1`` ended.
            with pcm.open("ab") as stream:
                stream.truncate(start)
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            write_json(
                directory / CHUNKS_FILE, ends[:number] + [start + len(data)]
            )

    def _chunk_ends(self, directory: Path) -> List[int]:
        """Return the end offsets of the chunks accepted so far."""
        try:
            ends = read_json(directory / CHUNKS_FILE)
        except FileNotFoundError:
            return []
        if not isinstance(ends, list) or not all(
            isinstance(e, int) for e in ends
        ):
            raise StorageError(
                f"Damaged {CHUNKS_FILE} in {directory.name}."
            )
        return list(ends)

    def _close_audio(self, directory: Path) -> None:
        """Encode the PCM of a finished recording as FLAC."""
        pcm = directory / PCM_FILE
        if not pcm.is_file() or self._stored_audio(directory) is not None:
            return
        import soundfile

        ends = self._chunk_ends(directory)
        data = pcm.read_bytes()[:ends[-1] if ends else 0]
        buffer = io.BytesIO()
        soundfile.write(
            buffer, pcm_samples(data), PCM_RATE, format="FLAC",
            subtype="PCM_16",
        )
        atomic_write(
            directory / f"{AUDIO_STEM}.{CLOSED_FORMAT}", buffer.getvalue()
        )

    def _stored_audio(self, directory: Path) -> Optional[Path]:
        """Return the encoded or imported audio file, if there is one."""
        found = sorted(
            p for p in directory.glob(f"{AUDIO_STEM}.*")
            if p.name != PCM_FILE and p.is_file()
        )
        return found[0] if found else None

    def store_recording(
        self, session_id: str, recording: RecordedAudio
    ) -> None:
        """Store a whole recording (an imported file) as ``audio.<ext>``."""
        kind = recording.format.lower().lstrip(".")
        if kind == PCM_FORMAT or not re.fullmatch(FORMAT_PATTERN, kind):
            raise InvalidInputError(
                f"Unsupported recording format {recording.format!r}."
            )
        with self._lock:
            directory = self._existing(session_id)
            for old in directory.glob(f"{AUDIO_STEM}.*"):
                old.unlink()
            (directory / CHUNKS_FILE).unlink(missing_ok=True)
            atomic_write(directory / f"{AUDIO_STEM}.{kind}", recording.data)

    def recording(self, session_id: str) -> RecordedAudio:
        """Return the stored audio of the session.

        The encoded or imported file when there is one, else the PCM
        recorded so far. :class:`NotFoundError` without any audio.
        """
        directory = self._existing(session_id)
        stored = self._stored_audio(directory)
        if stored is not None:
            return RecordedAudio(stored.read_bytes(), stored.suffix[1:])
        pcm = directory / PCM_FILE
        if pcm.is_file():
            ends = self._chunk_ends(directory)
            data = pcm.read_bytes()[:ends[-1] if ends else 0]
            if data:
                return RecordedAudio(data, PCM_FORMAT)
        raise NotFoundError(f"Session {session_id} has no audio.")

    def save_words(self, session_id: str, words: Sequence[Word]) -> None:
        """Store the attributed words of the session (for the player)."""
        with self._lock:
            directory = self._existing(session_id)
            write_json(
                directory / WORDS_FILE,
                [w.model_dump(mode="json") for w in words],
            )

    def words(self, session_id: str) -> List[Word]:
        """Return the stored words of the session (empty if none)."""
        path = self._existing(session_id) / WORDS_FILE
        try:
            data = read_json(path)
        except FileNotFoundError:
            return []
        return _parse(_WORDS.validate_python, data, path)

    def save_transcript(self, transcript: Transcript) -> None:
        """Store the transcript of ``transcript.session``."""
        with self._lock:
            directory = self._existing(transcript.session.id)
            write_json(
                directory / TRANSCRIPT_FILE,
                transcript.model_dump(mode="json"),
            )

    def transcript(self, session_id: str) -> Optional[Transcript]:
        """Return the transcript of the session, if there is one."""
        path = self._existing(session_id) / TRANSCRIPT_FILE
        try:
            data = read_json(path)
        except FileNotFoundError:
            return None
        return _parse(Transcript.model_validate, data, path)


def _parse(read: Callable[[Any], T], data: Any, path: Path) -> T:
    """Validate stored ``data``; a mismatch is a :class:`StorageError`."""
    try:
        return read(data)
    except ValidationError as exc:
        raise StorageError(
            f"Damaged {path.name} in {path.parent.name}: {exc.errors()[0]}"
        ) from exc
