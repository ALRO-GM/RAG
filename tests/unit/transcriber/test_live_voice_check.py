"""Two people on one microphone: the employee's voice check (#249).

The scripted diarizer gives slot 0 to the low tone and slot 1 to the
high one (``SplitDiarizer`` moves the low tone to slot 2 after a while,
as Sortformer may split one person); the fake voice encoder tells the
tones apart by their spectrum. No real model.
"""

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
from tests.unit.transcriber.test_live_mono_diarization import (
    ENDPOINT,
    HIGH,
    LOW,
    SETTINGS,
    ScriptedDiarizer,
    enrolled,
    sentences,
    tones,
)
from transcriber.adapters.outbound.fake_models import FakeVoiceEncoder
from transcriber.application.live import (
    FloatArray,
    LiveSettings,
    LiveTranscription,
)
from transcriber.application.live_diarization import (
    LiveNaming,
    MonoAttributor,
    PartyPlan,
)
from transcriber.domain.errors import TranscriberError
from transcriber.domain.live import LiveSpeaker
from transcriber.domain.live_transcript import last_finals
from transcriber.domain.voice_check import (
    PartySides,
    VoiceCheck,
    VoiceCheckSettings,
)

CHECK = VoiceCheckSettings(
    seconds=2.0, min_seconds=1.0, wait_seconds=6.0, slot_seconds=1.0,
    match=0.9, split=0.99,
)
EMPLOYEE = LiveSpeaker("ana", role="interviewer", name="Ana")
OTHER = LiveSpeaker("S2", role="candidate")


class SplitDiarizer(ScriptedDiarizer):
    """Like :class:`ScriptedDiarizer`, but after ``split`` seconds the
    low tone goes to slot 2 (one person in two slots)."""

    def __init__(self, split: float, lag: float = 1.0) -> None:
        super().__init__(lag=lag)
        self.split = split

    def _row(self, n: int, audio: FloatArray) -> List[float]:
        row = super()._row(n, audio)
        if n * self.frame_seconds >= self.split and row[0] > 0:
            row[0], row[2] = 0.0, 0.95
        return row


class Requests:
    """Sends the page's requests when the stream reaches a time."""

    def __init__(self) -> None:
        self.mono: Optional[MonoAttributor] = None
        self.at: List[Tuple[float, str, str, bool]] = []
        self.sent: List[str] = []

    def due(self, now: float) -> None:
        mono = self.mono
        assert mono is not None
        for item in list(self.at):
            at, action, ident, mine = item
            if now >= at:
                self.at.remove(item)
                self.sent.append(action)
                if action == "skip":
                    mono.skip()
                else:
                    mono.correct(ident, mine)


def run(
    audio: FloatArray, diarizer: ScriptedDiarizer,
    check: bool = True, prints: Sequence[Any] = (),
    requests: Optional[Requests] = None,
    settings: VoiceCheckSettings = CHECK,
) -> Tuple[List[Dict[str, Any]], MonoAttributor]:
    timer = VirtualTimer()
    made: List[MonoAttributor] = []

    def attributor(given: Sequence[LiveSpeaker]) -> MonoAttributor:
        mono = MonoAttributor(
            diarizer, timer, SETTINGS,
            LiveNaming(FakeVoiceEncoder(), list(prints)), given,
            background=False,
            parties=PartyPlan(EMPLOYEE, OTHER, check, settings),
        )
        made.append(mono)
        if requests is not None:
            requests.mono = mono
        return mono

    live = LiveTranscription(
        LoudnessVad, ScriptedAsr(sentences, timer, cost=0.1),
        TimerClock(timer), timer,
        LiveSettings(endpoint=ENDPOINT, realtime=True), attributor,
    )
    events: List[Dict[str, Any]] = []

    def emit(event: Dict[str, Any]) -> None:
        events.append(event)
        if requests is not None and requests.mono is not None:
            requests.due(float(event["stream_time"]))

    live.run("two-1", [audio], emit, diarize=True)
    for event in events:
        jsonschema.validate(event, LIVE_SCHEMA)
    return events, made[0]


def checks(events: List[Dict[str, Any]]) -> List[str]:
    return [e["voice_check"]["state"] for e in events
            if e["type"] == "voice_check"]


def speakers(events: List[Dict[str, Any]]) -> List[str]:
    return [u["speaker"] for u in sorted(
        last_finals(events), key=lambda u: u["start"])]


# The employee speaks first (the check), the other person answers, the
# employee again.
TALK = tones([(1.0, 4.0, LOW), (5.5, 8.5, HIGH), (10.0, 12.5, LOW),
              (14.0, 16.0, HIGH)], 18.0)


def test_the_check_makes_the_first_voice_the_employee() -> None:
    events, mono = run(TALK, ScriptedDiarizer(lag=1.0))
    started = events[0]
    assert started["source"]["kind"] == "mono_diarized"
    assert [(s["id"], s["role"]) for s in started["speakers"]] == [
        ("ana", "interviewer"), ("S2", "candidate")]
    states = checks(events)
    assert states[0] == "listening" and states[-1] == "recognised"
    assert speakers(events) == ["ana", "S2", "ana", "S2"]
    # Two people only: never an anonymous slot label.
    every = {e["utterance"]["speaker"] for e in events if "utterance" in e}
    assert every <= {"ana", "S2"}
    # The check's sentence is the employee's from its first emission.
    first = next(e["utterance"] for e in events if e["type"] == "final")
    assert first["speaker"] == "ana" and not first["speaker_provisional"]
    kinds = {r["event"] for r in mono.diagnosis()}
    assert {"final", "check", "reference", "slots", "sentence"} <= kinds
    assert all(isinstance(v, (int, float, str, bool, list, type(None)))
               for r in mono.diagnosis() for v in r.values())


def test_a_split_voice_is_found_by_its_voice() -> None:
    # After 9 s the diarizer puts the employee in a third slot: its voice
    # is the check's, so it is the employee, never a third person.
    events, mono = run(TALK, SplitDiarizer(split=9.0))
    assert speakers(events) == ["ana", "S2", "ana", "S2"]
    assert mono.sides is not None
    assert mono.sides.side(2) == "employee"
    assert {mono.sides.slots[k].source for k in (0, 2)} == {"voice", "split"}


def test_skipping_keeps_the_old_way_and_corrections_still_work() -> None:
    requests = Requests()
    requests.at = [(0.0, "skip", "", False)]
    events, mono = run(TALK, ScriptedDiarizer(lag=1.0), requests=requests)
    assert "skipped" in checks(events)
    assert "recognised" not in checks(events)
    # Nobody enrolled, no check: everyone is the other person.
    assert set(speakers(events)) == {"S2"}

    requests = Requests()
    requests.at = [(0.0, "skip", "", False), (14.5, "mine", "t0.s2", True)]
    events, _ = run(TALK, ScriptedDiarizer(lag=1.0), requests=requests)
    # «Esta soy yo» on the second low-tone sentence: that voice (slot 0)
    # is the employee's from then on, and its earlier sentence too.
    assert speakers(events) == ["ana", "S2", "ana", "S2"]


def test_this_is_not_me_moves_a_voice_to_the_other_side() -> None:
    requests = Requests()
    requests.at = [(14.5, "not_mine", "t2.s0", False)]
    events, mono = run(TALK, ScriptedDiarizer(lag=1.0), requests=requests)
    last = {u["id"]: u["speaker"] for u in last_finals(events)}
    assert last["t2.s0"] == "S2"
    # The check's own sentence stays the employee's (it was pinned).
    assert last["t0.s0"] == "ana"
    assert any(r["event"] == "correction" for r in mono.diagnosis())


def test_a_check_too_short_says_so() -> None:
    audio = tones([(1.0, 1.6, LOW), (9.0, 12.0, HIGH)], 14.0)
    events, _ = run(audio, ScriptedDiarizer(lag=1.0))
    assert checks(events)[-1] == "short"


def test_an_enrolment_still_names_the_employee_without_a_check() -> None:
    events, _ = run(TALK, ScriptedDiarizer(lag=1.0), check=False,
                    prints=[enrolled("ana", LOW)])
    assert checks(events) == []
    assert speakers(events)[2] == "ana"
    assert events[0]["speakers"][0]["enrolled"] is True


class FailingEncoder(FakeVoiceEncoder):
    """An encoder that cannot embed."""

    def embed(self, audio: Any, windows: Any) -> Any:
        raise TranscriberError("no model")


def test_a_check_that_cannot_be_embedded_fails_quietly() -> None:
    timer = VirtualTimer()
    mono = MonoAttributor(
        ScriptedDiarizer(), timer, SETTINGS,
        LiveNaming(FailingEncoder(), []), (), background=False,
        parties=PartyPlan(EMPLOYEE, OTHER, True, CHECK),
    )
    live = LiveTranscription(
        LoudnessVad, ScriptedAsr(sentences, timer, cost=0.1),
        TimerClock(timer), timer,
        LiveSettings(endpoint=ENDPOINT, realtime=True), lambda g: mono,
    )
    events: List[Dict[str, Any]] = []
    live.run("two-2", [TALK], events.append, diarize=True)
    assert checks(events)[-1] == "failed"
    assert mono.voice_check and mono.employee_id == "ana"


def test_the_engine_hands_out_its_voice_requests() -> None:
    timer = VirtualTimer()
    mono = MonoAttributor(
        ScriptedDiarizer(), timer, SETTINGS,
        LiveNaming(FakeVoiceEncoder(), []), (), background=False,
        parties=PartyPlan(EMPLOYEE, OTHER, True, CHECK),
    )
    live = LiveTranscription(
        LoudnessVad, ScriptedAsr(sentences, timer), TimerClock(timer),
        timer, LiveSettings(endpoint=ENDPOINT, realtime=True),
        lambda g: mono,
    )
    engine = live.engine("two-3", [], lambda e: None, diarize=True)
    assert engine.voice is mono
    plain = MonoAttributor(ScriptedDiarizer(), timer, SETTINGS)
    other = LiveTranscription(
        LoudnessVad, ScriptedAsr(sentences, timer), TimerClock(timer),
        timer, LiveSettings(endpoint=ENDPOINT), lambda g: plain,
    ).engine("one", [], lambda e: None, diarize=True)
    assert other.voice is None and other.diagnosis() == []
    assert plain.notices() == [] and plain.employee_id is None


# ---------------------------------------------------------------- domain


def test_voice_check_states() -> None:
    check = VoiceCheck(CHECK)
    assert check.listening and check.needed == 2.0
    assert check.settle(0.0) is None
    check.hear(1.0, 1.5)
    assert check.settle(2.5) is None
    check.hear(3.0, 0.6)
    assert check.settle(3.6) == "recognised"
    check.hear(5.0, 3.0)  # no longer listening
    assert check.heard == pytest.approx(2.1)
    assert not check.skip()

    late = VoiceCheck(CHECK)
    late.hear(1.0, 1.2)
    assert late.settle(7.5) == "recognised"  # min_seconds after the wait
    short = VoiceCheck(CHECK)
    short.hear(1.0, 0.5)
    assert short.settle(7.5) == "short"
    skipped = VoiceCheck(CHECK)
    assert skipped.skip() and skipped.state == "skipped"
    failed = VoiceCheck(CHECK)
    failed.fail()
    assert failed.state == "failed"


def unit(x: Sequence[float]) -> FloatArray:
    v = np.asarray(x, dtype=np.float32)
    out: FloatArray = (v / np.linalg.norm(v)).astype(np.float32)
    return out


def test_party_sides_one_slot_per_person() -> None:
    sides = PartySides(CHECK)
    assert sides.side(None) == "other" and sides.side(3) == "other"
    assert not sides.unsure(0)  # nothing known before the check
    sides.voices({1: (unit([0, 1]), 3.0)})
    assert sides.slots[1].score is None  # no reference yet
    sides.recognise(np.asarray([2.0, 0.0], dtype=np.float32), 0)
    # The check's slot stands in until a voice is compared.
    assert sides.side(0) == "employee" and sides.slots[0].source == "check"
    assert not sides.unsure(0) and sides.unsure(5)
    sides.voices({1: (unit([0, 1]), 3.0), 0: (unit([1, 0.1]), 0.5)})
    # Slot 1 is far from the check: nobody reaches the match yet.
    assert sides.side(1) == "other" and sides.side(0) == "employee"
    assert not sides.unsure(1) and not sides.unsure(0)
    sides.voices({2: (unit([1, 0.1]), 3.0), 3: (unit([1, 0.3]), 3.0)})
    # The nearest slot is the employee; the check's slot steps aside.
    assert sides.side(2) == "employee" and sides.slots[2].source == "voice"
    assert sides.side(0) == "other" and sides.side(3) == "other"
    sides.voices({0: (unit([1, 0.12]), 3.0)})
    # As close as the split threshold: one person in two slots.
    assert sides.side(0) == "employee" and sides.slots[0].source == "split"
    sides.pin(1, "employee")
    assert sides.side(1) == "employee" and not sides.unsure(1)
    sides.pin(2, "other")
    assert sides.side(2) == "other"
    assert sides.side(0) == "employee"  # now the nearest free slot
    sides.pin(0, "other")
    assert sides.check_slot is None
    sides.set_check_slot(3)
    assert sides.sides()[3] == "employee"
    rows = sides.diagnosis()
    assert [r["slot"] for r in rows] == [0, 1, 2, 3]
    assert rows[1]["pinned"] == "employee"


def test_no_slot_near_enough_keeps_the_check_slot() -> None:
    sides = PartySides(VoiceCheckSettings(match=0.99))
    sides.recognise(np.asarray([1.0, 0.0], dtype=np.float32), 1)
    sides.voices({0: (unit([1, 0.3]), 3.0)})
    assert sides.side(1) == "employee" and sides.side(0) == "other"


def test_a_check_without_voice_has_no_reference() -> None:
    sides = PartySides(CHECK)
    sides.recognise(np.zeros(2, dtype=np.float32), None)
    assert sides.reference is None and sides.check_slot is None
