"""The demo video's captions and frame pacing: pure logic, no I/O.

The recording keeps marks (``"home"``, ``"live"``, ``"result"``...):
the second, from the start of the video, at which each scene began.
Captions are written relative to a mark, so they follow the scene even
when the page took a little longer to load; :func:`resolve` turns them
into absolute times and checks them.

Chrome's screencast sends a frame only when the page changes, each with
its capture time. :func:`resample` picks, for every tick of a constant
frame rate, the last frame captured at or before it: the video plays in
real time at an even rate, and a still page simply repeats its frame.
"""

from bisect import bisect_right
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

MAX_DURATION_S = 120.0
"""The video must not be longer than two minutes."""
MAX_CAPTION_CHARS = 60
"""A caption is one short line."""
MIN_CAPTION_S = 2.0
"""Shortest time a caption stays on screen (time to read it)."""


@dataclass(frozen=True)
class Caption:
    """One line shown from ``start`` to ``end`` seconds after ``mark``."""

    mark: str
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Shown:
    """A caption at absolute video times."""

    start: float
    end: float
    text: str


CAPTIONS: Tuple[Caption, ...] = (
    Caption("home", 0.8, 5.0, "Tres servicios, cada uno por separado"),
    Caption("home", 5.4, 9.6, "Veamos una entrevista de ejemplo, en directo"),
    Caption("live", 1.0, 8.5, "Una entrevista ficticia, en tiempo real"),
    Caption("live", 13.4, 21.0,
            "Sugerencias en tiempo real mientras entrevistas"),
    Caption("live", 21.6, 27.0, "Qué preguntar después, con un ejemplo listo"),
    Caption("live", 28.0, 34.0,
            "Aviso: la edad no puede influir en la selección"),
    Caption("live", 34.6, 41.5,
            "Reformulado: se pregunta por la experiencia"),
    Caption("live", 42.2, 52.0, "Los temas del puesto se marcan solos"),
    Caption("live", 52.8, 64.5, "Y te recuerda lo que falta por preguntar"),
    Caption("live", 65.2, 68.8, "Cinco de cinco temas cubiertos"),
    Caption("ended", 0.3, 3.6,
            "Fin de la entrevista: el análisis ya está listo"),
    Caption("result", 0.6, 6.6,
            "Al terminar, un resumen claro de la candidata"),
    Caption("result", 7.2, 13.4, "Puntos fuertes y qué mejorar, con ejemplos"),
    Caption("result", 14.0, 19.4,
            "Cada competencia frente al nivel que pide el puesto"),
    Caption("result", 20.0, 25.6, "Y el detalle, solo si lo necesitas"),
)
"""What the video says, scene by scene (Spanish, like the page)."""


def resolve(captions: Sequence[Caption], marks: Dict[str, float],
            duration: float) -> List[Shown]:
    """Return the captions at absolute times, checked.

    Raises ``ValueError`` when a mark is missing, a caption is too long
    or too short, two captions overlap, or one ends after ``duration``.
    """
    shown: List[Shown] = []
    for caption in captions:
        if caption.mark not in marks:
            raise ValueError(f"no mark {caption.mark!r} for {caption.text!r}")
        if len(caption.text) > MAX_CAPTION_CHARS or "\n" in caption.text:
            raise ValueError(f"caption too long: {caption.text!r}")
        if caption.end - caption.start < MIN_CAPTION_S:
            raise ValueError(f"caption too short: {caption.text!r}")
        base = marks[caption.mark]
        shown.append(Shown(base + caption.start, base + caption.end,
                           caption.text))
    shown.sort(key=lambda item: item.start)
    for before, after in zip(shown, shown[1:]):
        if after.start < before.end:
            raise ValueError(
                f"captions overlap: {before.text!r} and {after.text!r}"
            )
    if shown and shown[-1].end > duration:
        raise ValueError(f"caption after the end: {shown[-1].text!r}")
    return shown


def resample(stamps: Sequence[float], start: float, end: float,
             fps: int) -> List[int]:
    """Return, for each tick of ``fps`` in ``[start, end)``, a frame index.

    ``stamps`` are the capture times of the frames, in order. Each tick
    takes the last frame captured at or before it (the first frame for
    ticks before it), so the output runs in real time at an even rate.
    """
    if not stamps:
        raise ValueError("no frames were captured")
    if fps <= 0 or end <= start:
        raise ValueError("need a positive rate and end > start")
    if any(b < a for a, b in zip(stamps, stamps[1:])):
        raise ValueError("frame times must not go backwards")
    count = int(round((end - start) * fps))
    indices: List[int] = []
    for tick in range(count):
        at = start + tick / fps
        indices.append(max(0, bisect_right(stamps, at) - 1))
    return indices


def longest_hold(indices: Sequence[int], fps: int) -> float:
    """Return the longest time, in seconds, one frame stays on screen."""
    longest = run = 0
    previous = None
    for index in indices:
        run = run + 1 if index == previous else 1
        previous = index
        longest = max(longest, run)
    return longest / fps


def holds(indices: Sequence[int], fps: int,
          minimum: float) -> List[Tuple[float, float]]:
    """Return ``(start, seconds)`` of each frame held ``minimum`` s or more.

    A still page is held on purpose; a hold while something moves would
    be a stutter, so the report lists them to be checked against the
    timeline.
    """
    found: List[Tuple[float, float]] = []
    start = 0
    for tick in range(1, len(indices) + 1):
        if tick == len(indices) or indices[tick] != indices[start]:
            length = (tick - start) / fps
            if length >= minimum:
                found.append((start / fps, length))
            start = tick
    return found


def fade_alpha(shown: Shown, fade: float) -> str:
    """Return an ffmpeg ``alpha`` expression fading ``shown`` in and out."""
    a, b = shown.start, shown.end
    return (f"if(lt(t,{a:.3f}),0,if(lt(t,{a + fade:.3f}),(t-{a:.3f})/{fade},"
            f"if(lt(t,{b - fade:.3f}),1,if(lt(t,{b:.3f}),"
            f"({b:.3f}-t)/{fade},0))))")


def jpeg_size(data: bytes) -> Tuple[int, int]:
    """Return ``(width, height)`` from a JPEG's start-of-frame marker."""
    index = 2
    while index + 9 <= len(data):
        if data[index] != 0xFF:
            index += 1
            continue
        marker = data[index + 1]
        length = int.from_bytes(data[index + 2:index + 4], "big")
        if marker in (0xC0, 0xC1, 0xC2):
            height = int.from_bytes(data[index + 5:index + 7], "big")
            width = int.from_bytes(data[index + 7:index + 9], "big")
            return width, height
        index += 2 + length
    raise ValueError("not a JPEG with a frame header")
