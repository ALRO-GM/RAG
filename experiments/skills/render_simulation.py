"""Render a simulated discussion to audio with Piper, one voice per person.

Stand-alone script, run in its own virtual environment (Piper is
GPL-3.0 and not a dependency of this project; its Spanish voices have
their own licences, ``docs/skills/SIMULATION.md``)::

    python3 -m venv ~/.local/share/ragputin/tts-venv
    ~/.local/share/ragputin/tts-venv/bin/pip install piper-tts==1.8.0
    ~/.local/share/ragputin/tts-venv/bin/python -m piper.download_voices \\
        --download-dir ~/.local/share/ragputin/voices \\
        es_ES-davefx-medium es_ES-sharvard-medium \\
        es_ES-mls_10246-low es_ES-mls_9972-low
    ~/.local/share/ragputin/tts-venv/bin/python \\
        experiments/skills/render_simulation.py \\
        examples/skills/simulated/sim-reunion-verano.transcript.v1.json \\
        ~/.local/share/ragputin/voices data/simulated

It reads the ``transcript.v1`` written by ``skills simulate`` (speaker,
order, overlap marks), synthesises every sentence with the voice of its
speaker and lays the sentences out one after another: a pause between
turns, a shorter one inside a turn, and a turn marked ``overlap``
starting 0.6 s before the previous sentence ends, mixed on top of it.
Writes ``<id>.wav`` (16 kHz, mono, 16-bit), ``<id>.reference.json``
(the rendered start and end of every sentence and who said it: the
reference to score the transcriber against) and one enrolment sample
per person (``enrol-<speaker>.wav``, the voice reading ``--enrol-text``).
Only numpy and Piper are imported: nothing of the project.
"""

import argparse
import json
import wave
from pathlib import Path
from typing import Any, Dict, List, Tuple


import numpy as np
import numpy.typing as npt
from piper import PiperVoice, SynthesisConfig  # type: ignore[import-not-found]

Audio = npt.NDArray[np.float64]
"""Mono audio samples in -1..1."""

RATE = 16000
VOICES: Dict[str, Tuple[str, int]] = {
    "lucia": ("es_ES-sharvard-medium", 1),
    "andres": ("es_ES-davefx-medium", 0),
    "nerea": ("es_ES-mls_10246-low", 0),
    "joaquin": ("es_ES-sharvard-medium", 0),
}
"""Speaker id → (Piper voice, speaker of a multi-speaker voice)."""

ENROL_TEXT = (
    "Hola, me llamo {name}. Estoy leyendo este texto para que el sistema "
    "reconozca mi voz en las reuniones. Hablo con mi ritmo de siempre, sin "
    "prisa y sin cambiar el tono, como cuando explico algo a un compañero. "
    "Después de esta lectura, la grabación se usará solo para identificarme "
    "y podré pedir que se borre cuando quiera. Por la mañana suelo llegar "
    "pronto, preparo un café y repaso los pedidos del día con calma. Si "
    "algo no cuadra, lo comento en la reunión y buscamos juntos una "
    "solución que funcione para todos. Los viernes, antes de irme, dejo "
    "anotado lo que queda pendiente para la semana siguiente."
)


def resample(audio: Audio, rate: int) -> Audio:
    """Return ``audio`` at :data:`RATE` by linear interpolation."""
    if rate == RATE or audio.size == 0:
        return audio
    length = int(round(audio.size * RATE / rate))
    positions = np.linspace(0, audio.size - 1, length)
    out: Audio = np.interp(positions, np.arange(audio.size), audio)
    return out


class Voices:
    """Load each Piper voice once and synthesise float audio at 16 kHz."""

    def __init__(self, directory: Path) -> None:
        """Voices are read from ``directory`` (``<name>.onnx``)."""
        self.directory = directory
        self.loaded: Dict[str, Any] = {}

    def say(self, speaker: str, text: str) -> Audio:
        """Return ``text`` said by ``speaker``'s voice."""
        name, number = VOICES[speaker]
        if name not in self.loaded:
            path = self.directory / f"{name}.onnx"
            self.loaded[name] = PiperVoice.load(path)
        voice = self.loaded[name]
        config = SynthesisConfig(speaker_id=number)
        parts = [
            chunk.audio_float_array for chunk in voice.synthesize(text, config)
        ]
        audio = np.concatenate(parts) if parts else np.zeros(0)
        return resample(audio.astype(np.float64), voice.config.sample_rate)


def write_wav(path: Path, audio: Audio) -> None:
    """Write mono 16-bit PCM at :data:`RATE`."""
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    scaled = audio / peak * 0.9 if peak > 0.9 else audio
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(RATE)
        out.writeframes((scaled * 32767).astype("<i2").tobytes())


def render(
    document: Dict[str, Any], voices: Voices, turn_pause: float,
    sentence_pause: float, overlap: float,
) -> Tuple[Audio, List[Dict[str, Any]]]:
    """Lay the sentences out in time; return the audio and the reference."""
    pieces: List[Tuple[int, Audio]] = []
    reference: List[Dict[str, Any]] = []
    clock = 0.5
    previous_turn = None
    for u in document["utterances"]:
        audio = voices.say(u["speaker"], u["text"])
        if u.get("overlap") and reference:
            start = max(clock - overlap, reference[-1]["start"] + 0.3)
        elif previous_turn is None:
            start = clock
        else:
            same = u["turn"] == previous_turn
            start = clock + (sentence_pause if same else turn_pause)
        end = start + audio.size / RATE
        pieces.append((int(start * RATE), audio))
        reference.append({
            "id": u["id"], "speaker": u["speaker"], "text": u["text"],
            "start": round(start, 3), "end": round(end, 3),
            "overlap": bool(u.get("overlap")),
        })
        clock = max(clock, end)
        previous_turn = u["turn"]
    total = np.zeros(int((clock + 1.0) * RATE))
    for offset, audio in pieces:
        total[offset:offset + audio.size] += audio
    return total, reference


def main() -> None:
    """Render the discussion, its reference and the enrolment samples."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("transcript", type=Path)
    parser.add_argument("voices", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--turn-pause", type=float, default=0.45)
    parser.add_argument("--sentence-pause", type=float, default=0.25)
    parser.add_argument("--overlap", type=float, default=0.6)
    args = parser.parse_args()
    document = json.loads(args.transcript.read_text(encoding="utf-8"))
    voices = Voices(args.voices)
    args.output.mkdir(parents=True, exist_ok=True)
    audio, reference = render(
        document, voices, args.turn_pause, args.sentence_pause, args.overlap
    )
    session = document["session"]["id"]
    write_wav(args.output / f"{session}.wav", audio)
    names = {s["id"]: s["name"] for s in document["speakers"]}
    (args.output / f"{session}.reference.json").write_text(json.dumps({
        "session": session, "seconds": round(audio.size / RATE, 2),
        "voices": {k: {"voice": VOICES[k][0], "speaker": VOICES[k][1],
                       "name": names[k]} for k in names},
        "utterances": reference,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    for speaker, name in names.items():
        sample = voices.say(speaker, ENROL_TEXT.format(name=name))
        write_wav(args.output / f"enrol-{speaker}.wav", sample)
    overlaps = sum(
        1 for a, b in zip(reference, reference[1:]) if b["start"] < a["end"]
    )
    print(f"{session}: {audio.size / RATE / 60:.1f} min, "
          f"{len(reference)} sentences, {overlaps} overlapping starts")


if __name__ == "__main__":
    main()
