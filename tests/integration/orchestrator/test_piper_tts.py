"""PiperTts against a stand-in ``piper`` command (no Piper, no model).

The stand-in speaks Piper's protocol: it reads one line per utterance,
writes a tiny WAV per line into the ``-d`` directory and logs ``Wrote
<path>`` on its standard error. ``DIE`` makes it exit, ``SLOW`` makes
it wait, ``NOWAV`` logs a path it never wrote.
"""

import sys
import textwrap
import time
import wave
from io import BytesIO
from pathlib import Path
from typing import Iterator, List

import pytest

from orchestrator.adapters.outbound.fakes import FakeTts
from orchestrator.adapters.outbound.piper_tts import PiperTts
from orchestrator.application.ports import CancelToken, TtsUnavailable
from orchestrator.domain.speech import tts_request

FAKE_PIPER = textwrap.dedent('''
    import sys, time, wave
    from pathlib import Path
    args = sys.argv[1:]
    out = Path(args[args.index("-d") + 1])
    rate = 22050 if "-s" not in args else 16000
    print("loading voice", args[args.index("-m") + 1], file=sys.stderr,
          flush=True)
    n = 0
    for line in sys.stdin:
        line = line.strip()
        if line == "DIE.":
            sys.exit(3)
        if line == "SLOW.":
            time.sleep(5)
        n += 1
        path = out / f"{n}.wav"
        if line != "NOWAV.":
            with wave.open(str(path), "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
                w.writeframes(b"\\x01\\x00" * (100 * len(line)))
        print(f"INFO:piper:Wrote {path}", file=sys.stderr, flush=True)
''')


@pytest.fixture
def piper(tmp_path: Path) -> Iterator[PiperTts]:
    script = tmp_path / "fake_piper.py"
    script.write_text(FAKE_PIPER, encoding="utf-8")
    voices = tmp_path / "voices"
    voices.mkdir()
    (voices / "es_ES-test-medium.onnx").write_bytes(b"onnx")
    tts = PiperTts(voices_dir=voices, timeout=2.0,
                   command=[sys.executable, str(script)],
                   workdir=tmp_path / "out")
    (tmp_path / "out").mkdir()
    yield tts
    tts.close()


def request(text: str, **kwargs: object) -> dict[str, object]:
    doc = tts_request(str(kwargs.pop("request_id", "r1")), text,
                      "es_ES-test-medium", "es", **kwargs)  # type: ignore
    return doc


def test_each_sentence_is_one_wav_chunk(piper: PiperTts) -> None:
    chunks = list(piper.synthesize(
        request("Hola, soy el asistente. ¿En qué puedo ayudarte hoy?"),
        CancelToken()))
    assert [c.seq for c in chunks] == [0, 1]
    assert chunks[0].text == "Hola, soy el asistente."
    with wave.open(BytesIO(chunks[0].audio), "rb") as reader:
        assert reader.getframerate() == 22050
        assert reader.getnframes() == 100 * len(chunks[0].text)
    assert chunks[0].sample_rate == 22050 and chunks[0].format == "wav"
    assert chunks[0].synth_ms >= 0
    assert not any(piper.workdir.iterdir())


def test_pcm_format_and_speaker_and_no_stream(piper: PiperTts) -> None:
    doc = request("Una frase. Y otra frase más.", audio_format="pcm_s16le",
                  stream=False)
    doc["speaker"] = 1
    [chunk] = list(piper.synthesize(doc, CancelToken()))
    assert chunk.text == "Una frase. Y otra frase más."
    assert len(chunk.audio) == 2 * 100 * len(chunk.text)
    assert chunk.sample_rate == 16000


def test_cancel_stops_after_the_current_sentence(piper: PiperTts) -> None:
    cancel = CancelToken()
    got: List[str] = []
    for chunk in piper.synthesize(
        request("Primera frase larga. Segunda frase larga. Tercera."),
        cancel,
    ):
        got.append(chunk.text)
        cancel.cancel()
    assert got == ["Primera frase larga."]


def test_a_later_request_interrupts_the_one_it_names(
    piper: PiperTts,
) -> None:
    first = piper.synthesize(
        request("Primera frase larga. Segunda frase larga.",
                request_id="t1"), CancelToken())
    assert next(first).seq == 0
    second = list(piper.synthesize(
        request("Respuesta nueva aquí.", request_id="t2",
                interrupt_id="t1"), CancelToken()))
    assert [c.text for c in second] == ["Respuesta nueva aquí."]
    assert list(first) == []


def test_a_dead_process_is_reported_then_restarted(piper: PiperTts) -> None:
    with pytest.raises(TtsUnavailable):
        list(piper.synthesize(request("DIE."), CancelToken()))
    again = list(piper.synthesize(request("Vuelvo a hablar."),
                                  CancelToken()))
    assert again[0].text == "Vuelvo a hablar."


def test_slow_and_missing_audio_are_unavailable(piper: PiperTts) -> None:
    piper.timeout = 0.5
    with pytest.raises(TtsUnavailable, match="in time"):
        list(piper.synthesize(request("SLOW."), CancelToken()))
    with pytest.raises(TtsUnavailable, match="missing"):
        list(piper.synthesize(request("NOWAV."), CancelToken()))


def test_unknown_voice_and_bad_requests(piper: PiperTts) -> None:
    doc = request("Hola.")
    doc["voice"] = "es_ES-none-low"
    with pytest.raises(TtsUnavailable, match="no Piper voice"):
        list(piper.synthesize(doc, CancelToken()))
    doc["voice"] = "../x"
    with pytest.raises(ValueError):
        list(piper.synthesize(doc, CancelToken()))
    with pytest.raises(ValueError):
        list(piper.synthesize({"schema": "other"}, CancelToken()))


def test_a_missing_command_is_unavailable(tmp_path: Path) -> None:
    voices = tmp_path / "v"
    voices.mkdir()
    (voices / "x.onnx").write_bytes(b"")
    tts = PiperTts(voices_dir=voices,
                   command=[str(tmp_path / "no-such-python")])
    try:
        with pytest.raises(TtsUnavailable, match="cannot run"):
            tts.warm("x")
    finally:
        tts.close()


def test_warm_starts_the_voice(piper: PiperTts) -> None:
    started = time.monotonic()
    piper.warm("es_ES-test-medium")
    assert time.monotonic() - started < 2.0
    assert len(piper._processes) == 1


def test_fake_tts_gives_one_tone_per_sentence() -> None:
    slept: List[float] = []
    tts = FakeTts(delay=0.1, sleep=slept.append)
    doc = request("Una frase de prueba. Otra frase de prueba.")
    chunks = list(tts.synthesize(doc, CancelToken()))
    assert len(chunks) == 2 and slept == [0.1, 0.1]
    assert chunks[0].audio[:4] == b"RIFF"
    assert chunks[0].duration_ms == pytest.approx(200.0, abs=1)
    doc["format"] = "pcm_s16le"
    cancel = CancelToken()
    cancel.cancel()
    assert list(tts.synthesize(doc, cancel)) == []
