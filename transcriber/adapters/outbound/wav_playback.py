"""WAV files for the browser's player, read by byte ranges.

The player (``docs/transcriber/ARCHITECTURE.md`` §7) sets the audio to
a time and plays from there, so the browser must be able to seek to any
second without downloading the file. Stored audio is FLAC (recorded
sessions) or whatever was imported (m4a, mp3, webm...). Browsers seek
those by guessing a byte offset and then searching for a frame, which
is slow or imprecise for FLAC without a seek table, and some formats do
not play at all in Safari (Ogg, older WebM). So the player is given a
16-bit PCM WAV at 16 kHz mono: every browser plays it, and a time maps
to a byte offset exactly, which makes each seek one HTTP range request.
It costs 32 KB per second of audio (115 MB per hour) in transit; only
the ranges played are sent.

:func:`write_wav` renders a stream of PCM blocks to a file with bounded
memory; :class:`FilePlayback` serves a rendered file and
:class:`LivePlayback` the raw PCM of a session still recording behind a
WAV header made on the fly. Both read only the bytes asked for, from a
file descriptor opened once, so a file replaced or removed meanwhile
does not break a read in progress.
"""

import os
import struct
from pathlib import Path
from typing import BinaryIO, Iterable

from transcriber.domain.errors import InvalidInputError

WAV_HEADER_BYTES = 44
PCM_RATE = 16000
SAMPLE_WIDTH = 2
MAX_DATA_BYTES = 0xFFFFFFFF - (WAV_HEADER_BYTES - 8)
"""Largest PCM payload a RIFF size field can describe (about 37 hours)."""


def wav_header(data_bytes: int, rate: int = PCM_RATE) -> bytes:
    """Return the 44-byte header of a mono 16-bit PCM WAV file.

    ``data_bytes`` is the size of the samples that follow it.
    """
    if not 0 <= data_bytes <= MAX_DATA_BYTES:
        raise InvalidInputError("The audio is too long to play as WAV.")
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", WAV_HEADER_BYTES - 8 + data_bytes, b"WAVE",
        b"fmt ", 16, 1, 1, rate, rate * SAMPLE_WIDTH, SAMPLE_WIDTH,
        8 * SAMPLE_WIDTH, b"data", data_bytes,
    )


def write_wav(
    stream: BinaryIO,
    blocks: Iterable[bytes],
    rate: int = PCM_RATE,
    max_bytes: int = MAX_DATA_BYTES,
) -> int:
    """Write ``blocks`` of mono 16-bit PCM as a WAV file to ``stream``.

    Only one block is held at a time; the header is written first with
    a zero size and rewritten at the end (``stream`` must be seekable).
    Return the number of PCM bytes written. Raises
    :class:`InvalidInputError` for audio without any sample or longer
    than ``max_bytes`` (at most what a WAV file can hold).
    """
    start = stream.tell()
    stream.write(wav_header(0, rate))
    size = 0
    for block in blocks:
        size += len(block)
        if size > min(max_bytes, MAX_DATA_BYTES):
            raise InvalidInputError("The audio is too long to play as WAV.")
        stream.write(block)
    if size == 0:
        raise InvalidInputError("The recording contains no audio.")
    end = stream.tell()
    stream.seek(start)
    stream.write(wav_header(size, rate))
    stream.seek(end)
    return size


class FilePlayback:
    """A rendered WAV file, read by byte ranges."""

    def __init__(self, path: Path) -> None:
        """Open ``path`` (:class:`FileNotFoundError` if it is gone)."""
        self._fd = os.open(path, os.O_RDONLY)
        self.size = os.fstat(self._fd).st_size

    def read(self, offset: int, length: int) -> bytes:
        """Return up to ``length`` bytes from ``offset``."""
        length = max(min(length, self.size - offset), 0)
        return os.pread(self._fd, length, offset) if length else b""

    def close(self) -> None:
        """Close the file (reading afterwards is an error)."""
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1


class LivePlayback:
    """The PCM of a recording in progress, as a WAV file.

    ``pcm_bytes`` is how much of the PCM file had been accepted when the
    request came; later chunks are not part of this answer. A chunk
    retried meanwhile may cut the file back for an instant: bytes that
    are missing then are read as silence, so the answer keeps the
    length it announced.
    """

    def __init__(self, pcm: Path, pcm_bytes: int) -> None:
        """Open the PCM file (:class:`FileNotFoundError` if it is gone)."""
        self._header = wav_header(pcm_bytes)
        self._fd = os.open(pcm, os.O_RDONLY)
        self._pcm_bytes = pcm_bytes
        self.size = WAV_HEADER_BYTES + pcm_bytes

    def read(self, offset: int, length: int) -> bytes:
        """Return up to ``length`` bytes from ``offset``."""
        end = min(offset + length, self.size)
        if end <= offset:
            return b""
        head = self._header[offset:end] if offset < WAV_HEADER_BYTES else b""
        first = max(offset, WAV_HEADER_BYTES) - WAV_HEADER_BYTES
        wanted = end - WAV_HEADER_BYTES - first
        if wanted <= 0:
            return head
        body = os.pread(self._fd, wanted, first)
        return head + body + bytes(wanted - len(body))

    def close(self) -> None:
        """Close the file (reading afterwards is an error)."""
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1
