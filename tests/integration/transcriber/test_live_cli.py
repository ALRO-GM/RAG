"""``transcriber live-file`` over generated WAV files, with fake models.

The command line runs in this process; ``build_live`` is replaced by a
builder of the live use case over the fakes (no model, no GPU), so the
WAV reading, the channel split, the JSON lines and the summary are the
real ones.
"""

import io
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import jsonschema
import numpy as np
import pytest
import soundfile

from tests.unit.transcriber.live_fakes import (
    LIVE_SCHEMA,
    LoudnessVad,
    ScriptedAsr,
    TimerClock,
    VirtualTimer,
    bursts,
)
from transcriber.adapters.inbound import live_cli
from transcriber.adapters.inbound.cli import main
from transcriber.adapters.outbound.av_audio_decoder import read_channels
from transcriber.application.live import LiveOptions, LiveTranscription

built: List[LiveOptions] = []


def fake_build(options: LiveOptions) -> LiveTranscription:
    built.append(options)
    timer = VirtualTimer()
    asr = ScriptedAsr(lambda seconds, final: "Hola a todos.", timer, 0.1)
    return LiveTranscription(
        LoudnessVad, asr, TimerClock(timer), timer, options.settings
    )


@pytest.fixture(autouse=True)
def fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    import transcriber.bootstrap

    built.clear()
    monkeypatch.setattr(transcriber.bootstrap, "build_live", fake_build)


def cli(tmp_path: Path, *args: str) -> Tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(
        list(args), environ={"TRANSCRIBER_DATA_DIR": str(tmp_path / "d")},
        out=out, err=err,
    )
    return code, out.getvalue(), err.getvalue()


def stereo_wav(path: Path, rate: int = 16000) -> Path:
    left = bursts([(0.5, 1.5)], 4.0)
    right = bursts([(2.5, 3.5)], 4.0)
    data = np.stack([left, right], axis=1)
    if rate != 16000:
        data = data[::2]
    soundfile.write(str(path), data, rate)
    return path


def test_stereo_file_gives_one_stream_per_channel(tmp_path: Path) -> None:
    wav = stereo_wav(tmp_path / "call.wav")
    out = tmp_path / "events" / "call.jsonl"
    code, stdout, stderr = cli(
        tmp_path, "live-file", str(wav), "--channels", "--out", str(out),
        "--session", "call-1", "--hang-ms", "300",
    )
    assert code == 0 and stdout == ""
    events = [json.loads(line) for line in out.read_text().splitlines()]
    for event in events:
        jsonschema.validate(event, LIVE_SCHEMA)
    finals = [e["utterance"] for e in events if e["type"] == "final"]
    assert [(u["speaker"], u["channel"]) for u in finals] == [
        ("S1", 0), ("S2", 1),
    ]
    assert {e["session_id"] for e in events} == {"call-1"}
    assert "2 finals" in stderr and "lag_ms: p50" in stderr
    (options,) = built
    assert options.settings.endpoint.hang_ms == 300
    assert options.settings.realtime and options.language == "es"
    assert (options.model, options.device, options.compute_type) == (
        "large-v3-turbo", "cuda", "int8",
    )


def test_mono_mix_to_stdout_resampled(tmp_path: Path) -> None:
    wav = stereo_wav(tmp_path / "call.wav", rate=8000)
    code, stdout, _ = cli(
        tmp_path, "live-file", str(wav), "--fast", "--language", "auto",
        "--device", "cpu",
    )
    assert code == 0
    events: List[Dict[str, Any]] = [
        json.loads(line) for line in stdout.splitlines()
    ]
    assert events[0]["source"]["channels"] == 1
    finals = [e for e in events if e["type"] == "final"]
    assert len(finals) == 2 and {
        f["utterance"]["speaker"] for f in finals
    } == {"S1"}
    assert built[0].language is None and not built[0].settings.realtime


@pytest.mark.parametrize("args", [
    ["--session", "bad id"], ["--hang-ms", "100"], ["--hang-ms", "800"],
])
def test_bad_options_are_usage_errors(tmp_path: Path, args: List[str]) -> None:
    wav = stereo_wav(tmp_path / "call.wav")
    code, _, stderr = cli(tmp_path, "live-file", str(wav), *args)
    assert code == 2 and "live-file" in stderr


def test_unreadable_and_too_many_channels(tmp_path: Path) -> None:
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not audio")
    assert cli(tmp_path, "live-file", str(bad))[0] == 1
    three = tmp_path / "three.wav"
    soundfile.write(str(three), np.zeros((1600, 3), dtype=np.float32), 16000)
    assert cli(tmp_path, "live-file", str(three), "--channels")[0] == 1
    assert len(read_channels(three, False)) == 1


def test_summary_lines_without_finals() -> None:
    from transcriber.application.live import LiveSummary

    assert live_cli.summary_lines(LiveSummary()) == ["0 finals, 0 partials"]
