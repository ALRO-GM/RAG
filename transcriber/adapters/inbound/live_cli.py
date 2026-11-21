"""``transcriber live-file``: run the live transcription over a WAV file.

The file stands in for a microphone: it is read 32 ms at a time, in
real time unless ``--fast``, and every ``live.event.v1`` event is
written as one JSON line (to standard output or ``--out``). Without
``--channels`` the channels are averaged into one stream; with it each
channel is its own stream and speaker (the stereo case: microphone on
one channel, the call on the other). A summary of the finals' timings
(p50 and p90 of ``endpoint_ms``, ``asr_ms``, ``lag_ms``) goes to the
error stream.
"""

import argparse
import json
import re
from pathlib import Path
from typing import IO, Callable, Dict, List, Optional

from transcriber.adapters.inbound.cli import Output, UsageError
from transcriber.application.live import (
    LiveOptions,
    LiveSettings,
    LiveSummary,
    LiveTranscription,
)
from transcriber.domain.live import (
    MAX_HANG_MS,
    MIN_HANG_MS,
    EndpointSettings,
    percentile,
)
from transcriber.domain.models import SESSION_ID_PATTERN
from transcriber.domain.voices import FloatArray

Builder = Callable[[LiveOptions], LiveTranscription]
Reader = Callable[[Path, bool], List[FloatArray]]


def add_live_parser(
    commands: "argparse._SubParsersAction[argparse.ArgumentParser]",
) -> None:
    """Add the ``live-file`` command."""
    live = commands.add_parser(
        "live-file", help="simulate live transcription from a WAV file"
    )
    live.add_argument("wav", type=Path, help="the audio file")
    live.add_argument("--channels", action="store_true",
                      help="one stream per channel (stereo)")
    live.add_argument("--out", type=Path,
                      help="write the events here (JSON lines)")
    live.add_argument("--session", default="live-file",
                      help="session id of the events")
    live.add_argument("--fast", action="store_true",
                      help="read as fast as possible, not in real time")
    live.add_argument("--hang-ms", type=int, default=500,
                      help=f"silence that ends a segment, {MIN_HANG_MS}"
                      f"-{MAX_HANG_MS} (default: 500)")
    live.add_argument("--partial-every-ms", type=int, default=800,
                      help="decode partials this often (default: 800)")
    live.add_argument("--language", default="es",
                      help="language, or 'auto' (default: es)")
    live.add_argument("--model", default="large-v3-turbo",
                      help="faster-whisper model (default: large-v3-turbo)")
    live.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                      help="where the recogniser runs (default: cuda)")
    live.add_argument("--compute-type", default="int8",
                      help="CTranslate2 compute type (default: int8)")


def summary_lines(summary: LiveSummary) -> List[str]:
    """Return the run's counts and the p50/p90 of the finals' timings."""
    lines = [f"{summary.finals} finals, {summary.partials} partials"]
    series: Dict[str, List[float]] = {
        "endpoint_ms": summary.endpoint_ms,
        "asr_ms": summary.asr_ms,
        "lag_ms": summary.lag_ms,
    }
    for name, values in series.items():
        p50, p90 = percentile(values, 0.5), percentile(values, 0.9)
        if p50 is not None and p90 is not None:
            lines.append(f"{name}: p50 {p50:.0f}, p90 {p90:.0f}")
    return lines


class LiveFileCommand:
    """Run ``live-file`` with a builder of the live transcription."""

    def __init__(self, build: Builder, read: Reader, output: Output) -> None:
        """Keep the builder (tests give fakes), the file reader, the output."""
        self._build = build
        self._read = read
        self._out = output

    def run(self, args: argparse.Namespace) -> int:
        """Transcribe the file; return the exit code."""
        if not re.match(SESSION_ID_PATTERN, args.session):
            raise UsageError(f"invalid session id: {args.session}")
        if not MIN_HANG_MS <= args.hang_ms <= MAX_HANG_MS:
            raise UsageError(
                f"--hang-ms must be {MIN_HANG_MS}-{MAX_HANG_MS}"
            )
        channels = self._read(args.wav, args.channels)
        options = LiveOptions(
            model=args.model, device=args.device,
            compute_type=args.compute_type,
            language=None if args.language == "auto" else args.language,
            settings=LiveSettings(
                endpoint=EndpointSettings(hang_ms=args.hang_ms),
                partial_every_ms=args.partial_every_ms,
                realtime=not args.fast,
            ),
        )
        live = self._build(options)
        sink: Optional[IO[str]] = None
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            sink = args.out.open("w", encoding="utf-8")
        try:
            def emit(event: Dict[str, object]) -> None:
                line = json.dumps(event, ensure_ascii=False)
                if sink is None:
                    self._out.line(line)
                else:
                    sink.write(line + "\n")
                    sink.flush()

            summary = live.run(args.session, channels, emit)
        finally:
            if sink is not None:
                sink.close()
        for line in summary_lines(summary):
            self._out.note(line)
        return 0
