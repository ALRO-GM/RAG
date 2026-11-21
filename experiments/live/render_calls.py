"""Render the platform's simulated calls as STEREO and MONO wav (LV-12).

Stand-alone, run in the Piper virtual environment of
``experiments/skills/render_simulation.py`` (which it reuses for the
voices and the wav writer)::

    ~/.local/share/ragputin/tts-venv/bin/python \\
        experiments/live/render_calls.py \\
        examples/platform/sim-soporte-lumbre.transcript.v1.json \\
        ~/.local/share/ragputin/voices OUT_DIR

The first speaker of the transcript is channel 0, the second channel 1
(two Piper voices, one per channel). Writes ``<id>.stereo.wav``,
``<id>.mono.wav`` (the two channels summed) and ``<id>.reference.json``
(sentence, speaker, channel, start and end in seconds).
"""

import argparse
import json
import sys
import wave
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import numpy.typing as npt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "skills"))
import render_simulation as sim  # noqa: E402

Samples = npt.NDArray[np.float64]
"""Audio samples, one column per channel."""

CALL_VOICES = {
    "agente": ("es_ES-sharvard-medium", 1),
    "cliente": ("es_ES-davefx-medium", 0),
    "entrevistador": ("es_ES-sharvard-medium", 0),
    "candidata": ("es_ES-mls_10246-low", 0),
}
"""Speaker id of the calls → (Piper voice, speaker of the voice)."""


def render_channels(
    document: Dict[str, Any], voices: "sim.Voices",
) -> Tuple[Samples, List[Dict[str, Any]]]:
    """Return a (samples, channels) array and the reference."""
    order = [s["id"] for s in document["speakers"]]
    clock, previous = 0.5, None
    placed: List[Tuple[int, int, Samples]] = []
    reference: List[Dict[str, Any]] = []
    for u in document["utterances"]:
        audio = voices.say(u["speaker"], u["text"])
        if previous is None:
            start = clock
        else:
            start = clock + (0.25 if u["turn"] == previous else 0.45)
        end = start + audio.size / sim.RATE
        placed.append(
            (order.index(u["speaker"]), int(start * sim.RATE), audio))
        reference.append({
            "id": u["id"], "speaker": u["speaker"],
            "channel": order.index(u["speaker"]), "text": u["text"],
            "start": round(start, 3), "end": round(end, 3),
        })
        clock, previous = end, u["turn"]
    total = np.zeros((int((clock + 1.0) * sim.RATE), len(order)))
    for channel, offset, audio in placed:
        total[offset:offset + audio.size, channel] += audio
    return total, reference


def write(path: Path, audio: Samples) -> None:
    """Write 16-bit 16 kHz PCM, one column per channel."""
    peak = float(np.max(np.abs(audio)))
    scaled = audio / peak * 0.9 if peak > 0.9 else audio
    with wave.open(str(path), "wb") as out:
        out.setnchannels(scaled.shape[1])
        out.setsampwidth(2)
        out.setframerate(sim.RATE)
        out.writeframes((scaled * 32767).astype("<i2").tobytes())


def main() -> None:
    """Render one transcript to stereo and mono."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("transcript", type=Path)
    parser.add_argument("voices", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    document = json.loads(args.transcript.read_text(encoding="utf-8"))
    sim.VOICES.update(CALL_VOICES)
    audio, reference = render_channels(document, sim.Voices(args.voices))
    args.output.mkdir(parents=True, exist_ok=True)
    name = document["session"]["id"]
    write(args.output / f"{name}.stereo.wav", audio)
    write(args.output / f"{name}.mono.wav", audio.sum(axis=1, keepdims=True))
    (args.output / f"{name}.reference.json").write_text(json.dumps({
        "session": name, "seconds": round(audio.shape[0] / sim.RATE, 2),
        "speakers": [s["id"] for s in document["speakers"]],
        "utterances": reference,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{name}: {audio.shape[0] / sim.RATE:.1f} s,"
          f" {len(reference)} sentences")


if __name__ == "__main__":
    main()
