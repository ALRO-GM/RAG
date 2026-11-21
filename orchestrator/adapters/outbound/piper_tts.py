"""Text to speech with Piper, run as a separate process.

Piper (``piper-tts``, piper1-gpl) is GPL-3.0, so it is never imported:
it runs in its own virtual environment (``~/.local/share/ragputin/
tts-venv``, the one of ``experiments/skills/render_simulation.py``) as
its own command line, ``python -m piper -m <voice>.onnx -d <dir>``.
That command keeps the voice loaded and reads one line of text per
utterance from its standard input; for each line it writes a WAV file
to ``<dir>`` and logs ``Wrote <path>`` on its standard error, which is
what tells this adapter that the sentence is ready. One process is
kept per voice and controls (length scale, noise scale, noise_w,
sentence silence, speaker), started on first use and restarted if it
dies; one sentence at a time goes through it.

Sentence streaming: the request's text is cut into sentences
(:func:`~orchestrator.domain.speech.split_sentences`) and each one is
yielded as soon as its WAV is written; the caller's cancel token, or a
later request naming this one in ``interrupt_id``, stops the rest
(barge-in). The audio is the WAV file itself or, for ``pcm_s16le``,
its frames; ``sample_rate`` is the voice's (22050 Hz for the
``medium`` voices). The text is never logged.
"""

import queue
import re
import subprocess
import tempfile
import threading
import time
import wave
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, IO, Iterator, List, NamedTuple, Optional, Tuple

from orchestrator.application.ports import (
    CancelToken,
    Document,
    TtsUnavailable,
)
from orchestrator.domain.speech import (
    TTS_SCHEMA,
    VOICE_ID,
    SpeechChunk,
    split_sentences,
)

DEFAULT_PYTHON = Path("~/.local/share/ragputin/tts-venv/bin/python")
DEFAULT_VOICES = Path("~/.local/share/ragputin/voices")
WROTE = re.compile(r"Wrote (.+\.wav)\s*$")
"""Piper's log line once a sentence's WAV file is written."""


class _Key(NamedTuple):
    """What one Piper process is started with (internal)."""

    voice: str
    length_scale: float = 1.0
    speaker: Optional[int] = None
    noise_scale: Optional[float] = None
    noise_w: Optional[float] = None
    sentence_silence: Optional[float] = None

    @classmethod
    def of(cls, request: Dict[str, Any]) -> "_Key":
        """Return the key of a ``tts.request.v1``."""
        def number(name: str) -> Optional[float]:
            value = request.get(name)
            return float(value) if value is not None else None

        speaker = request.get("speaker")
        return cls(
            voice=str(request.get("voice") or ""),
            length_scale=float(request.get("length_scale") or 1.0),
            speaker=int(speaker) if speaker is not None else None,
            noise_scale=number("noise_scale"), noise_w=number("noise_w"),
            sentence_silence=number("sentence_silence"),
        )


class _Piper:
    """One running ``python -m piper`` process (internal)."""

    def __init__(self, command: List[str], outdir: Path):
        """Start the process and the reader of its standard error."""
        self.outdir = outdir
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE, text=True, encoding="utf-8",
            bufsize=1,
        )
        self.lines: "queue.Queue[str]" = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        """Copy every line of the process's standard error to the queue."""
        stream: Optional[IO[str]] = self.process.stderr
        if stream is None:  # pragma: no cover - always a pipe
            return
        for line in stream:
            self.lines.put(line)
        self.lines.put("")

    def alive(self) -> bool:
        """Whether the process still runs."""
        return self.process.poll() is None

    def say(self, text: str, timeout: float) -> bytes:
        """Synthesise one line; return the WAV file's bytes."""
        stdin = self.process.stdin
        if stdin is None or not self.alive():
            raise TtsUnavailable("piper is not running")
        try:
            stdin.write(text.replace("\n", " ") + "\n")
            stdin.flush()
        except OSError as exc:
            raise TtsUnavailable("piper stopped") from exc
        deadline = time.monotonic() + timeout
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TtsUnavailable("piper did not answer in time")
            try:
                line = self.lines.get(timeout=left)
            except queue.Empty:
                continue
            if line == "":
                raise TtsUnavailable("piper stopped")
            match = WROTE.search(line)
            if match is None:
                continue
            path = Path(match.group(1).strip())
            try:
                data = path.read_bytes()
                path.unlink()
            except OSError as exc:
                raise TtsUnavailable("piper's audio is missing") from exc
            return data

    def close(self) -> None:
        """Stop the process."""
        try:
            if self.process.stdin is not None:
                self.process.stdin.close()
        except OSError:
            pass
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()


class PiperTts:
    """The :class:`~orchestrator.application.ports.TextToSpeech` of Piper.

    Args:
        python: The Python of Piper's own virtual environment.
        voices_dir: Where ``<voice>.onnx`` (and its ``.onnx.json``) are.
        timeout: Seconds to wait for one sentence (the first one also
            loads the voice).
        command: The command prefix that runs Piper (default
            ``[python, "-m", "piper"]``; replaced in tests).
        workdir: Where the WAV files are written (a temporary directory
            by default).
    """

    def __init__(
        self,
        python: Path = DEFAULT_PYTHON,
        voices_dir: Path = DEFAULT_VOICES,
        timeout: float = 20.0,
        command: Optional[List[str]] = None,
        workdir: Optional[Path] = None,
    ):
        """Keep the settings; no process is started yet."""
        self.voices_dir = voices_dir.expanduser()
        self.timeout = timeout
        self.command = command or [str(python.expanduser()), "-m", "piper"]
        self._tmp: Optional[tempfile.TemporaryDirectory[str]] = None
        if workdir is None:
            self._tmp = tempfile.TemporaryDirectory(prefix="piper-")
            workdir = Path(self._tmp.name)
        self.workdir = workdir
        self._processes: Dict[_Key, _Piper] = {}
        self._lock = threading.Lock()
        self._active: Dict[str, CancelToken] = {}
        self._active_lock = threading.Lock()

    def warm(self, voice: str, length_scale: float = 1.0) -> None:
        """Start the voice's process and speak one word, off the hot path."""
        self._say(_Key(voice, length_scale), "Hola.")

    def synthesize(
        self, request: Document, cancel: CancelToken
    ) -> Iterator[SpeechChunk]:
        """Speak ``request`` sentence by sentence (see the module)."""
        if request.get("schema") != TTS_SCHEMA:
            raise ValueError("not a tts.request.v1")
        key = _Key.of(request)
        if not VOICE_ID.match(key.voice):
            raise ValueError(f"invalid voice id {key.voice!r}")
        audio_format = request.get("format") or "pcm_s16le"
        request_id = str(request["request_id"])
        text = str(request.get("text") or "")
        sentences = (split_sentences(text) if request.get("stream", True)
                     else [" ".join(text.split())])
        self._register(request_id, request.get("interrupt_id"), cancel)
        try:
            for seq, sentence in enumerate(s for s in sentences if s):
                if cancel.cancelled:
                    return
                started = time.perf_counter()
                wav = self._say(key, sentence)
                elapsed = (time.perf_counter() - started) * 1000.0
                if cancel.cancelled:
                    return
                audio, rate = _audio(wav, audio_format)
                yield SpeechChunk(
                    request_id=request_id, seq=seq, text=sentence,
                    audio=audio, format=audio_format, sample_rate=rate,
                    synth_ms=round(elapsed, 1),
                )
        finally:
            with self._active_lock:
                if self._active.get(request_id) is cancel:
                    del self._active[request_id]

    def close(self) -> None:
        """Stop every Piper process and remove the WAV directory."""
        with self._lock:
            for process in self._processes.values():
                process.close()
            self._processes.clear()
        if self._tmp is not None:
            self._tmp.cleanup()
            self._tmp = None

    def _register(
        self, request_id: str, interrupt_id: object, cancel: CancelToken
    ) -> None:
        """Remember the request and stop the one it replaces."""
        with self._active_lock:
            if isinstance(interrupt_id, str):
                replaced = self._active.pop(interrupt_id, None)
                if replaced is not None:
                    replaced.cancel()
            self._active[request_id] = cancel

    def _say(self, key: _Key, text: str) -> bytes:
        """Synthesise one sentence with the process of ``key``."""
        with self._lock:
            process = self._processes.get(key)
            if process is None or not process.alive():
                process = self._start(key)
                self._processes[key] = process
            try:
                return process.say(text, self.timeout)
            except TtsUnavailable:
                process.close()
                self._processes.pop(key, None)
                raise

    def _start(self, key: _Key) -> _Piper:
        """Start Piper for one voice."""
        model = self.voices_dir / f"{key.voice}.onnx"
        if not model.is_file():
            raise TtsUnavailable(f"no Piper voice {key.voice!r}")
        command = self.command + [
            "-m", str(model), "-d", str(self.workdir),
            "--length-scale", f"{key.length_scale:g}",
        ]
        if key.speaker is not None:
            command += ["-s", str(key.speaker)]
        for flag, value in (("--noise-scale", key.noise_scale),
                            ("--noise-w-scale", key.noise_w),
                            ("--sentence-silence", key.sentence_silence)):
            if value is not None:
                command += [flag, f"{value:g}"]
        try:
            return _Piper(command, self.workdir)
        except OSError as exc:
            raise TtsUnavailable(f"cannot run piper: {exc}") from exc


def _audio(wav: bytes, audio_format: str) -> Tuple[bytes, int]:
    """Return the audio in ``audio_format`` and its sample rate."""
    try:
        with wave.open(BytesIO(wav), "rb") as reader:
            rate = reader.getframerate()
            frames = reader.readframes(reader.getnframes())
    except (wave.Error, EOFError) as exc:
        raise TtsUnavailable("piper wrote an unreadable WAV") from exc
    return (wav if audio_format == "wav" else frames), rate
