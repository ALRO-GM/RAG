"""Mono diarized live sessions: attribution, speaker_update, naming (LV-6)."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import jsonschema
import numpy as np
import pytest

from tests.unit.transcriber.live_fakes import (
    LIVE_SCHEMA,
    LoudnessVad,
    ScriptedAsr,
    TimerClock,
    VirtualTimer,
)
from transcriber.adapters.outbound.fake_models import (
    FAKE_VOICE_MODEL,
    FakeVoiceEncoder,
)
from transcriber.application.live import (
    FloatArray,
    LiveSettings,
    LiveSummary,
    LiveTranscription,
)
from transcriber.application.live_diarization import (
    LiveNaming,
    MonoAttributor,
)
from transcriber.application.ports import Audio
from transcriber.domain.errors import DiarizerError, InvalidInputError
from transcriber.domain.live import (
    SAMPLE_RATE,
    EndpointSettings,
    LiveSpeaker,
    LiveUtterance,
)
from transcriber.domain.live_diarization import MonoDiarizationSettings
from transcriber.domain.live_transcript import last_finals, live_transcript
from transcriber.domain.models import Pipeline, Voiceprint

ENDPOINT = EndpointSettings(hang_ms=500, min_speech_ms=0, pre_roll_ms=0)
LOW, HIGH = 220.0, 880.0
SETTINGS = MonoDiarizationSettings(naming_seconds=2.0, wait_ms=100)


def tones(spans: Sequence[Tuple[float, float, float]],
          total: float) -> FloatArray:
    """Silence of ``total`` seconds with a tone of ``hz`` in each span."""
    out = np.zeros(int(total * SAMPLE_RATE), dtype=np.float32)
    for start, end, hz in spans:
        a, b = int(start * SAMPLE_RATE), int(end * SAMPLE_RATE)
        t = np.arange(b - a, dtype=np.float32) / SAMPLE_RATE
        out[a:b] = 0.5 * np.sin(2 * np.pi * hz * t)
    return out


def sentences(seconds: float, final: bool) -> str:
    """One word per half second; a final ends with a period."""
    words = " ".join(["palabra"] * max(int(seconds * 2), 1))
    return words + "." if final else words


class ScriptedDiarizer:
    """Frames by pitch, ``lag`` seconds behind the audio fed."""

    model_name = "scripted-diarizer"
    frame_seconds = 0.08
    slots = 4

    def __init__(
        self, lag: float = 1.0, fail_start: bool = False,
        fail_after: Optional[int] = None,
    ) -> None:
        self.lag = lag
        self.fail_start = fail_start
        self.fail_after = fail_after
        self.audio: List[FloatArray] = []
        self.fed = 0
        self.given = 0
        self.ended = False
        self.started = 0
        self.closed = 0
        self.waits: List[Tuple[float, float]] = []
        self.calls = 0

    def start(self) -> None:
        self.started += 1
        if self.fail_start:
            raise DiarizerError("not installed")

    def feed(self, samples: FloatArray) -> None:
        self.audio.append(samples)
        self.fed += len(samples)

    def _row(self, n: int, audio: FloatArray) -> List[float]:
        size = int(self.frame_seconds * SAMPLE_RATE)
        frame = audio[n * size:(n + 1) * size]
        row = [0.0] * self.slots
        if len(frame) and float(np.abs(frame).max()) > 0.1:
            crossings = int(np.sum(np.diff(np.sign(frame)) != 0))
            row[0 if crossings < 100 else 1] = 0.95
        return row

    def frames(
        self, until: float = 0.0, timeout: float = 0.0
    ) -> Tuple[int, List[List[float]]]:
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise DiarizerError("the diarizer exited (code 1)")
        self.waits.append((until, timeout))
        heard = self.fed / SAMPLE_RATE
        if not self.ended:
            heard -= self.lag
        ready = max(int(heard / self.frame_seconds), 0)
        audio: FloatArray = (
            np.concatenate(self.audio) if self.audio
            else np.zeros(0, dtype=np.float32)
        )
        first = self.given
        rows = [self._row(n, audio) for n in range(first, max(ready, first))]
        self.given = max(ready, first)
        return first, rows

    def finish(self) -> None:
        self.ended = True

    def close(self) -> None:
        self.closed += 1


def enrolled(ident: str, hz: float) -> Voiceprint:
    """A voiceprint of the fake encoder for a tone of ``hz``."""
    encoder = FakeVoiceEncoder()
    audio = Audio(samples=tones([(0.0, 3.0, hz)], 3.0),
                  sample_rate=SAMPLE_RATE)
    rows = encoder.embed(audio, [(0.0, 1.5), (1.5, 3.0)])
    vector = rows.mean(axis=0)
    vector /= np.linalg.norm(vector)
    return Voiceprint(
        speaker_id=ident, model=FAKE_VOICE_MODEL,
        vector=[float(x) for x in vector], consent_version="c1",
        enrolled_at=datetime(2026, 11, 21, tzinfo=timezone.utc),
    )


class Released:
    """Counts the releases of the encoder."""

    def __init__(self) -> None:
        self.count = 0

    def __call__(self) -> None:
        self.count += 1


def transcription(
    timer: VirtualTimer, diarizer: ScriptedDiarizer,
    naming: Optional[LiveNaming] = None,
    settings: MonoDiarizationSettings = SETTINGS,
    background: bool = True,
) -> LiveTranscription:
    def attributor(given: Sequence[LiveSpeaker]) -> MonoAttributor:
        return MonoAttributor(
            diarizer, timer, settings, naming, given, background
        )

    return LiveTranscription(
        LoudnessVad, ScriptedAsr(sentences, timer, cost=0.1),
        TimerClock(timer), timer,
        LiveSettings(endpoint=ENDPOINT, realtime=True), attributor,
    )


def run(
    audio: FloatArray, diarizer: ScriptedDiarizer,
    naming: Optional[LiveNaming] = None,
    settings: MonoDiarizationSettings = SETTINGS,
    background: bool = True,
) -> Tuple[List[Dict[str, Any]], LiveSummary]:
    timer = VirtualTimer()
    live = transcription(timer, diarizer, naming, settings, background)
    events: List[Dict[str, Any]] = []
    summary = live.run("mono-1", [audio], events.append, diarize=True)
    for event in events:
        jsonschema.validate(event, LIVE_SCHEMA)
    return events, summary


TWO_VOICES = tones([(1.0, 4.0, LOW), (5.5, 8.5, HIGH)], 10.0)


def of(events: List[Dict[str, Any]], kind: str) -> List[Dict[str, Any]]:
    return [e["utterance"] for e in events if e["type"] == kind]


def test_each_final_gets_its_voice_and_late_frames_revise_it() -> None:
    diarizer = ScriptedDiarizer(lag=1.0)
    events, summary = run(TWO_VOICES, diarizer)
    started = events[0]
    assert started["source"]["kind"] == "mono_diarized"
    assert [s["id"] for s in started["speakers"]] == ["S1", "S2", "S3", "S4"]
    finals = of(events, "final")
    assert [u["speaker"] for u in finals] == ["S1", "S2"]
    # The frames lag the audio: each final goes out provisional...
    assert all(u["speaker_provisional"] for u in finals)
    updates = of(events, "speaker_update")
    # ... and a speaker_update confirms it once its frames are there.
    assert {u["id"] for u in updates} == {u["id"] for u in finals}
    assert all(not u["speaker_provisional"] for u in updates)
    for update in updates:
        final = next(u for u in finals if u["id"] == update["id"])
        assert update["revision"] == final["revision"] + 1
        assert update["text"] == final["text"]
    assert summary.speaker_updates == len(updates)
    assert len(summary.attribution_ms) == 2
    assert diarizer.started == 1 and diarizer.closed == 1
    # The final waits a little for the frames of its last audio.
    assert 0.1 in [round(timeout, 3) for _, timeout in diarizer.waits]
    assert finals[1]["turn"] > finals[0]["turn"]


def test_a_wrong_provisional_speaker_is_corrected() -> None:
    # The second voice starts 0.6 s after the first stops: its final
    # goes out before any frame of it, with the last speaker known.
    audio = tones([(1.0, 3.0, LOW), (3.6, 5.6, HIGH)], 9.0)
    events, _ = run(audio, ScriptedDiarizer(lag=2.5))
    finals = {u["id"]: u for u in of(events, "final")}
    updates = of(events, "speaker_update")
    last = {u["id"]: u["speaker"] for u in updates}
    assert set(last.values()) == {"S1", "S2"}
    changed = [u for u in updates if finals[u["id"]]["speaker"]
               != u["speaker"]]
    assert changed and changed[0]["speaker"] == "S2"
    transcript = live_transcript(
        "mono-1", events, [LiveSpeaker("S1"), LiveSpeaker("S2")],
        Pipeline(recogniser="r", voice_model="none", transcriber="0",
                 diarizer="scripted-diarizer"),
    )
    assert [u.speaker for u in transcript.utterances][-1] == "S2"


def test_enrolled_voices_are_named_and_earlier_finals_renamed() -> None:
    released = Released()
    naming = LiveNaming(
        FakeVoiceEncoder(), [enrolled("ana", LOW), enrolled("luis", 2000)],
        {"ana": "Ana"}, release=released,
    )
    events, _ = run(TWO_VOICES, ScriptedDiarizer(lag=1.0), naming)
    roster = {s["id"]: s for s in events[0]["speakers"]}
    assert roster["ana"]["enrolled"] and roster["ana"]["name"] == "Ana"
    assert roster["luis"]["name"] is None
    last = {u["id"]: u["speaker"] for u in last_finals(events)}
    first, second = sorted(last)
    assert last[first] == "ana" and last[second] == "S2"
    renamed = [u for u in of(events, "speaker_update")
               if u["speaker"] == "ana"]
    assert renamed
    assert released.count == 1


def test_naming_in_the_callers_thread_names_the_same_voices() -> None:
    naming = LiveNaming(FakeVoiceEncoder(), [enrolled("ana", LOW)])
    events, _ = run(TWO_VOICES, ScriptedDiarizer(lag=1.0), naming,
                    background=False)
    last = {u["id"]: u["speaker"] for u in last_finals(events)}
    assert sorted(last.values()) == ["S2", "ana"]

    class Broken(FakeVoiceEncoder):
        def embed(self, audio: Audio, windows: Any) -> FloatArray:
            raise InvalidInputError("bad window")

    broken = LiveNaming(Broken(), [enrolled("ana", LOW)])
    events, _ = run(TWO_VOICES, ScriptedDiarizer(lag=1.0), broken,
                    background=False)
    assert "ana" not in {u["speaker"] for u in last_finals(events)}


def test_given_ids_and_roles_name_the_voices_in_order() -> None:
    timer = VirtualTimer()
    live = transcription(timer, ScriptedDiarizer(lag=0.0))
    events: List[Dict[str, Any]] = []
    engine = live.engine(
        "mono-2", [LiveSpeaker("agente", role="agent"),
                   LiveSpeaker("cliente", role="customer", name="Eva")],
        events.append, paced=True, diarize=True,
    )
    assert engine.kind == "mono_diarized"
    assert engine.diarizer == "scripted-diarizer"
    engine.start()
    engine.push([TWO_VOICES])
    engine.finish()
    roster = events[0]["speakers"]
    assert [(s["id"], s["role"]) for s in roster[:2]] == [
        ("agente", "agent"), ("cliente", "customer"),
    ]
    assert roster[1]["name"] == "Eva" and roster[2]["id"] == "S1"
    assert [u["speaker"] for u in of(events, "final")] == [
        "agente", "cliente",
    ]


def test_a_diarizer_that_cannot_start_only_stops_attribution() -> None:
    diarizer = ScriptedDiarizer(fail_start=True)
    events, summary = run(TWO_VOICES, diarizer)
    warnings = [e["warning"] for e in events if e["type"] == "warning"]
    assert len(warnings) == 1 and warnings[0]["code"] == "other"
    assert "not installed" in warnings[0]["message"]
    finals = of(events, "final")
    assert len(finals) == 2
    assert all(not u["speaker_provisional"] for u in finals)
    assert of(events, "speaker_update") == []
    assert diarizer.fed == 0 and diarizer.closed == 1


def test_a_diarizer_that_dies_warns_once_and_keeps_going() -> None:
    diarizer = ScriptedDiarizer(lag=0.0, fail_after=20)
    events, _ = run(TWO_VOICES, diarizer)
    warnings = [e for e in events if e["type"] == "warning"]
    assert len(warnings) == 1
    assert "exited" in warnings[0]["warning"]["message"]
    assert len(of(events, "final")) == 2


def test_diarization_needs_one_channel_and_a_diarizer() -> None:
    timer = VirtualTimer()
    live = transcription(timer, ScriptedDiarizer())
    assert live.diarizes
    with pytest.raises(InvalidInputError):
        live.engine("x", [LiveSpeaker("S1", 0), LiveSpeaker("S2", 1)],
                    lambda e: None, diarize=True)
    plain = LiveTranscription(
        LoudnessVad, ScriptedAsr(sentences), TimerClock(timer), timer,
    )
    assert not plain.diarizes
    with pytest.raises(InvalidInputError):
        plain.engine("x", [], lambda e: None, diarize=True)


def test_a_failed_warm_up_stops_the_diarizer() -> None:
    timer = VirtualTimer()
    diarizer = ScriptedDiarizer()

    class Broken(ScriptedAsr):
        def warm_up(self) -> None:
            raise RuntimeError("no model")

    live = LiveTranscription(
        LoudnessVad, Broken(sentences), TimerClock(timer), timer,
        LiveSettings(endpoint=ENDPOINT),
        lambda given: MonoAttributor(diarizer, timer, SETTINGS, None, given),
    )
    engine = live.engine("x", [], lambda e: None, diarize=True)
    with pytest.raises(RuntimeError):
        engine.start()
    assert diarizer.closed == 1


def test_attributor_guesses_partials_and_skips_unclean_naming() -> None:
    timer = VirtualTimer()
    diarizer = ScriptedDiarizer(lag=0.0)
    naming = LiveNaming(FakeVoiceEncoder(), [enrolled("ana", LOW)])
    mono = MonoAttributor(diarizer, timer, SETTINGS, naming)
    assert mono.model_name == "scripted-diarizer"
    mono.start()
    assert mono.guess(0.0, 1.0) == "S1"
    audio = tones([(0.0, 1.0, HIGH)], 1.0)
    for n in range(0, len(audio), 512):
        mono.feed(audio[n:n + 512])
    assert mono.poll() == []
    assert mono.guess(0.0, 1.0) == "S2"
    # Audio no longer kept cannot feed the naming.
    old = LiveUtterance(id="t0.s0", turn=0, index=0, speaker="S2",
                        start=0.0, end=60.0, text="x.", revision=0)
    [said] = mono.attribute([old])
    assert said.speaker == "S2"
    assert mono.attribute([]) == []
    assert mono.finish() == []
    assert mono.stats.attribution_ms


def test_naming_failures_are_ignored() -> None:
    class Broken(FakeVoiceEncoder):
        def embed(self, audio: Audio, windows: Any) -> FloatArray:
            raise InvalidInputError("bad window")

    naming = LiveNaming(Broken(), [enrolled("ana", LOW)])
    events, _ = run(TWO_VOICES, ScriptedDiarizer(lag=0.0), naming)
    assert {u["speaker"] for u in of(events, "final")} == {"S1", "S2"}
