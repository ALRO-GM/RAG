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
    above are removed, in this order: the FLAC, then ``session.json``,
    then the PCM. While the session is ``recording`` the PCM is the
    recording; a FLAC found beside it was left by a close that a killed
    process did not finish, and it is replaced when the session is
    closed again.
``imported.<ext>``
    an imported recording, kept as it came. Its name never collides
    with the files the recorder writes (``audio.pcm``, ``audio.flac``);
    imports stored by earlier versions as ``audio.<ext>`` are still read.
``playback-<key>.wav``
    the WAV rendering the player reads (:mod:`~transcriber.adapters.
    outbound.wav_playback`), made on the first request, written block by
    block (bounded memory) and atomically. ``<key>`` identifies the
    stored audio file it was made from (name, inode, size, modification
    time), so a new import or a close makes a new rendering and the old
    one is removed; deleting the session removes it too.
``words.json`` and ``transcript.json``
    the attributed words and the transcript (the transcriber's own
    format, words included; the contract is made from it on publish).

JSON and audio files are written atomically (:func:`atomic_write`).
Chunk appends are made safe by ``chunks.json``: appending chunk ``n``
first cuts ``audio.pcm`` back to where chunk ``n - 1`` ended, so bytes
left by a process killed mid-append are never kept twice.

Ids are checked against their pattern and refused when their path is a
symbolic link (:func:`child_path`). Reading a stored file that is not
what it should be raises :class:`StorageError`, never a raw
:class:`OSError`.
"""

import hashlib
import io
import os
import re
import shutil
import threading
from pathlib import Path
from typing import (
    Any, Callable, Dict, Iterator, List, Optional, Sequence, TypeVar,
)

from pydantic import TypeAdapter, ValidationError

from transcriber.adapters.outbound.av_audio_decoder import (
    PCM_RATE,
    pcm16_blocks,
    pcm_samples,
)
from transcriber.adapters.outbound.fs_storage import (
    atomic_file,
    atomic_write,
    child_path,
    read_json,
    sync_directory,
    write_json,
)
from transcriber.adapters.outbound.wav_playback import (
    FilePlayback,
    LivePlayback,
    write_wav,
)
from transcriber.application.ports import (
    PCM_FORMAT,
    PlaybackAudio,
    RecordedAudio,
)
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
IMPORTED_STEM = "imported"
PLAYBACK_PREFIX = "playback-"
RECORDER_FORMATS = frozenset({"pcm", PCM_FORMAT})
"""Formats an import may not claim: the recorder's raw PCM."""
FORMAT_PATTERN = r"^[a-z0-9]{1,10}$"

Blocks = Callable[[Path, str], Iterator[bytes]]
"""Decode a stored file (path, format) into mono 16-bit PCM blocks."""

_WORDS = TypeAdapter(List[Word])
T = TypeVar("T")


class FileSystemSessionRepository:
    """Sessions, their audio, words and transcripts under ``root``."""

    def __init__(self, root: Path, blocks: Blocks = pcm16_blocks) -> None:
        """Keep sessions under ``root / "sessions"`` (created on write).

        ``blocks`` decodes stored audio for the player's rendering.
        """
        self.root = Path(root) / "sessions"
        self._blocks = blocks
        self._lock = threading.RLock()
        self._render_locks: Dict[str, threading.Lock] = {}

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
        """Return the session, or ``None`` if it does not exist.

        A ``session.json`` that cannot be read or is not a session is a
        :class:`StorageError`.
        """
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
        still raw PCM has the whole PCM encoded as ``audio.flac`` first
        (replacing a FLAC an interrupted close left); the PCM is removed
        only once the FLAC and ``session.json`` are on disk.
        """
        with self._lock:
            directory = self._existing(session.id)
            closing = session.state is not SessionState.RECORDING
            if closing:
                self._close_audio(directory)
            write_json(
                directory / SESSION_FILE, session.model_dump(mode="json")
            )
            if closing:
                for name in (PCM_FILE, CHUNKS_FILE):
                    (directory / name).unlink(missing_ok=True)
                sync_directory(directory)

    def delete(self, session_id: str) -> None:
        """Remove the session with its audio, words and transcript."""
        with self._lock:
            directory = self._directory(session_id)
            if directory.is_dir():
                # A rendering being written meanwhile may add a file
                # while the tree is removed: try once more.
                shutil.rmtree(directory, ignore_errors=True)
                if directory.exists():
                    shutil.rmtree(directory)

    def append_chunk(self, session_id: str, number: int, data: bytes) -> None:
        """Store chunk ``number`` (from 0) of :data:`PCM_FORMAT` audio.

        Only the next chunk or the last one stored again is accepted:
        chunk ``number`` replaces whatever followed chunk ``number - 1``
        (a chunk half-written by a killed process, or the same chunk
        stored before). A gap in the numbering, or a number below the
        last stored chunk (which would drop every chunk after it), is
        an :class:`InvalidInputError`.
        """
        with self._lock:
            directory = self._existing(session_id)
            ends = self._chunk_ends(directory)
            if number < max(len(ends) - 1, 0) or number > len(ends):
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
        """Encode the whole PCM of a finished recording as FLAC.

        Done whenever the PCM is still there, even if a FLAC exists:
        that one was written by a close that did not finish and may
        miss the chunks received since.
        """
        pcm = directory / PCM_FILE
        if not pcm.is_file():
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
            p for p in (
                *directory.glob(f"{IMPORTED_STEM}.*"),
                *directory.glob(f"{AUDIO_STEM}.*"),
            )
            if p.name != PCM_FILE and p.is_file() and not p.is_symlink()
            and not p.name.endswith(".tmp")
        )
        return found[0] if found else None

    def store_recording(
        self, session_id: str, recording: RecordedAudio
    ) -> None:
        """Store a whole recording (an imported file) as ``imported.<ext>``.

        Raw PCM (the recorder's own format) or a format that is not a
        short lowercase extension is an :class:`InvalidInputError`.
        """
        kind = recording.format.lower().lstrip(".")
        if kind in RECORDER_FORMATS or not re.fullmatch(
            FORMAT_PATTERN, kind
        ):
            raise InvalidInputError(
                f"Unsupported recording format {recording.format!r}."
            )
        with self._lock:
            directory = self._existing(session_id)
            for old in (
                *directory.glob(f"{AUDIO_STEM}.*"),
                *directory.glob(f"{IMPORTED_STEM}.*"),
            ):
                old.unlink()
            (directory / CHUNKS_FILE).unlink(missing_ok=True)
            atomic_write(
                directory / f"{IMPORTED_STEM}.{kind}", recording.data
            )

    def recording(self, session_id: str) -> RecordedAudio:
        """Return the stored audio of the session.

        While the session is ``recording``, the PCM received so far
        (a FLAC beside it is stale); otherwise the encoded or imported
        file, or the PCM if there is none. :class:`NotFoundError`
        without any audio.
        """
        directory = self._existing(session_id)
        current = self.get(session_id)
        recording = (
            current is not None and current.state is SessionState.RECORDING
        )
        stored = None if recording else self._stored_audio(directory)
        if stored is not None:
            return RecordedAudio(stored.read_bytes(), stored.suffix[1:])
        pcm = directory / PCM_FILE
        if pcm.is_file():
            ends = self._chunk_ends(directory)
            data = pcm.read_bytes()[:ends[-1] if ends else 0]
            if data:
                return RecordedAudio(data, PCM_FORMAT)
        raise NotFoundError(f"Session {session_id} has no audio.")

    def playback(self, session_id: str) -> PlaybackAudio:
        """Return the session's audio as WAV for the player.

        While the session is ``recording`` (or when only PCM is stored)
        it is the PCM accepted so far behind a WAV header. Otherwise it
        is ``playback-<key>.wav``, rendered from the stored file the
        first time and kept until that file changes; one session is
        rendered by one request at a time, and other sessions are not
        held up meanwhile. :class:`NotFoundError` without any audio.
        """
        directory = self._existing(session_id)
        current = self.get(session_id)
        recording = (
            current is not None and current.state is SessionState.RECORDING
        )
        stored = None if recording else self._stored_audio(directory)
        if stored is None:
            return self._live(session_id, directory)
        try:
            with self._render_lock(session_id):
                target = self._rendering(stored)
                if not target.is_file():
                    with atomic_file(target, make_parent=False) as stream:
                        write_wav(
                            stream, self._blocks(stored, stored.suffix[1:])
                        )
                for old in directory.glob(f"{PLAYBACK_PREFIX}*.wav"):
                    if old != target:
                        old.unlink(missing_ok=True)
                return FilePlayback(target)
        except FileNotFoundError as exc:
            # The session was deleted, or its audio replaced, meanwhile.
            raise NotFoundError(f"Session {session_id} has no audio.") from exc

    def _live(self, session_id: str, directory: Path) -> PlaybackAudio:
        """Return the PCM received so far as WAV, or raise NotFound."""
        ends = self._chunk_ends(directory)
        if ends and ends[-1] > 0:
            try:
                return LivePlayback(directory / PCM_FILE, ends[-1])
            except FileNotFoundError:
                pass
        raise NotFoundError(f"Session {session_id} has no audio.")

    def _render_lock(self, session_id: str) -> threading.Lock:
        """Return the lock that serialises renderings of one session."""
        with self._lock:
            return self._render_locks.setdefault(session_id, threading.Lock())

    @staticmethod
    def _rendering(stored: Path) -> Path:
        """Return the rendering's path for the stored file as it is now."""
        info = stored.stat()
        identity = (
            f"{stored.name}:{info.st_ino}:{info.st_size}:{info.st_mtime_ns}"
        )
        key = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
        return stored.parent / f"{PLAYBACK_PREFIX}{key}.wav"

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
