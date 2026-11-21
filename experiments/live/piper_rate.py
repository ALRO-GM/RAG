"""Piper's own speaking rate (words per second) at length scale 1.

The voice bot can match an enrolled operator's speaking rate (#207,
``voicebot.match_rate``): the length scale is Piper's rate divided by
the operator's. This script measures Piper's rate for the two voices
of the «Ajustes» page on ten fixed Spanish sentences: words of the
text divided by the seconds of audio Piper writes (its leading and
trailing silence trimmed at -40 dBFS, as speech regions would be).

Run: ``.venv/bin/python experiments/live/piper_rate.py``; writes
``experiments/live/piper_rate_results.json``.
"""

import json
import sys
import wave
from io import BytesIO
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from orchestrator.adapters.outbound.piper_tts import PiperTts  # noqa: E402
from orchestrator.application.ports import CancelToken  # noqa: E402
from orchestrator.domain.speech import tts_request  # noqa: E402

SENTENCES = (
    "Hola, soy el asistente de voz de soporte y estoy aquí para ayudarte.",
    "Para cambiar la contraseña entra en tu perfil y pulsa en seguridad.",
    "El pedido sale del almacén en un plazo de dos días laborables.",
    "Si la factura no aparece, revisa la carpeta de documentos del mes.",
    "Puedo darte de alta en el servicio ahora mismo si me lo confirmas.",
    "La garantía cubre cualquier defecto de fabricación durante dos años.",
    "Te recomiendo reiniciar el router y esperar un par de minutos.",
    "El horario de atención es de nueve de la mañana a seis de la tarde.",
    "Ya he registrado tu incidencia y un técnico te llamará mañana.",
    "¿Hay algo más en lo que pueda ayudarte antes de terminar?",
)
VOICES = ("es_ES-davefx-medium", "es_ES-sharvard-medium")
OUT = ROOT / "experiments" / "live" / "piper_rate_results.json"


def voiced_seconds(wav: bytes) -> float:
    """Seconds between the first and last sample above -40 dBFS."""
    with wave.open(BytesIO(wav), "rb") as reader:
        rate = reader.getframerate()
        frames = reader.readframes(reader.getnframes())
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float64) / 32768
    loud = np.flatnonzero(np.abs(samples) > 10 ** (-40 / 20))
    if loud.size == 0:
        return 0.0
    return float(loud[-1] - loud[0] + 1) / rate


def main() -> None:
    """Measure and write the result file."""
    tts = PiperTts()
    results: Dict[str, Dict[str, object]] = {}
    try:
        for voice in VOICES:
            words = 0
            seconds = 0.0
            per: List[float] = []
            for i, text in enumerate(SENTENCES):
                request = tts_request(f"r{i}", text, voice, "es",
                                      stream=False)
                chunk = next(iter(tts.synthesize(request, CancelToken())))
                n = len(text.split())
                s = voiced_seconds(chunk.audio)
                words += n
                seconds += s
                per.append(round(n / s, 3))
            results[voice] = {
                "words": words, "voiced_seconds": round(seconds, 3),
                "words_per_second": round(words / seconds, 3),
                "per_sentence": per,
            }
    finally:
        tts.close()
    OUT.write_text(json.dumps({
        "length_scale": 1.0, "sentences": list(SENTENCES),
        "voices": results,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for voice, row in results.items():
        print(voice, row["words_per_second"])


if __name__ == "__main__":
    main()
