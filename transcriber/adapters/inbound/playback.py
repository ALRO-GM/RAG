"""Byte ranges of the audio the browser's player asks for.

The player (``docs/transcriber/ARCHITECTURE.md`` §7) seeks by asking
for one byte range of a WAV file in which a time is a byte offset (the
rendering is made and kept by the session repository, see
:mod:`transcriber.adapters.outbound.wav_playback`). This module reads
the ``Range`` header (:func:`parse_range`) and streams the bytes asked
for, a block at a time (:func:`byte_blocks`), so neither a seek nor a
whole download ever holds the recording in memory.
"""

from typing import Iterator, Optional, Tuple, Union

from transcriber.application.ports import PlaybackAudio

WAV_TYPE = "audio/wav"

MAX_RANGE_HEADER = 100
"""Longest ``Range`` header read; a longer one is ignored (RFC 9110)."""

MAX_DIGITS = 18
"""Longest number read in a range (more than any file has bytes)."""

READ_BLOCK = 256 * 1024
"""Bytes read from the rendering per block sent."""

DIGITS = frozenset("0123456789")


class Unsatisfiable:
    """A range request none of whose bytes exist (answered with 416)."""


UNSATISFIABLE = Unsatisfiable()
ByteRange = Tuple[int, int]
"""First and last byte, both included."""


def _number(text: str) -> Optional[int]:
    """Return ``text`` as a byte position, ``None`` if it is not one.

    An empty text is ``-1`` (the side of the range left open).
    """
    if not text:
        return -1
    if len(text) > MAX_DIGITS or not set(text) <= DIGITS:
        return None
    return int(text)


def parse_range(
    header: Optional[str], size: int
) -> Union[None, ByteRange, Unsatisfiable]:
    """Return the byte range ``header`` asks of a ``size``-byte body.

    ``None`` means "send the whole body": no header, a header this
    server does not understand (another unit, several ranges, a
    malformed or overlong one, first byte after last), which RFC 9110
    lets a server ignore. A range that starts past the end, or an empty
    suffix, is :data:`UNSATISFIABLE`. The last byte is clamped to the
    body. Parsing takes time linear in the header, which is read only
    up to :data:`MAX_RANGE_HEADER` characters.
    """
    if header is None or len(header) > MAX_RANGE_HEADER:
        return None
    unit, equals, spec = header.partition("=")
    if not equals or unit.strip().lower() != "bytes":
        return None
    first_text, dash, last_text = spec.partition("-")
    first = _number(first_text.strip())
    last = _number(last_text.strip())
    if not dash or first is None or last is None:
        return None
    if first < 0:
        if last < 0:
            return None
        if last == 0 or size == 0:
            return UNSATISFIABLE
        return max(size - last, 0), size - 1
    if last < 0:
        last = max(size - 1, first)
    if last < first:
        return None
    if first >= size:
        return UNSATISFIABLE
    return first, min(last, size - 1)


def byte_blocks(
    audio: PlaybackAudio, first: int, last: int, block: int = READ_BLOCK
) -> Iterator[bytes]:
    """Yield bytes ``first`` to ``last`` of ``audio``, then close it.

    Reads at most ``block`` bytes at a time. ``audio`` is closed when
    the bytes are sent, when reading fails, or when the generator is
    closed early (the client went away).
    """
    try:
        position = first
        while position <= last:
            data = audio.read(position, min(block, last - position + 1))
            if not data:
                break
            yield data
            position += len(data)
    finally:
        audio.close()
