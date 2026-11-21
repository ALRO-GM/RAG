"""VoiceBot with turn-taking: the interviewer waits for the candidate."""

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from orchestrator.adapters.outbound.fakes import (
    FakeKnowledge,
    FakeReasoner,
    FakeTts,
)
from orchestrator.adapters.outbound.runtime import InlineExecutor
from orchestrator.application.ports import CancelToken, Document
from orchestrator.application.voicebot import VoiceBot
from orchestrator.domain.records import TurnRecord
from orchestrator.domain.turn_taking import BotTurn, TurnConfig
from tests.unit.orchestrator.fixtures import product
from tests.unit.orchestrator.test_live_assist import ManualExecutor
from tests.unit.orchestrator.test_voicebot import CHUNKS, Output, final

ROLES = {"S1": "candidate"}


class Clock:
    """Time moves only when the test says so."""

    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds


@dataclass
class Plan:
    """A directed turn."""

    payload: Document
    query: str


@dataclass
class Director:
    """Records what the bot asked; asks a fixed question."""

    closed: bool = False
    target: Optional[str] = "s1"
    asked: List[Tuple[str, Optional[str], BotTurn]] = field(
        default_factory=list)
    heard_ids: List[str] = field(default_factory=list)
    spoken_turns: List[int] = field(default_factory=list)

    def heard(self, utterance: str, text: str, start: float,
              end: float) -> None:
        self.heard_ids.append(utterance)

    def plan_turn(self, answer: str, utterance: Optional[str],
                  cancel: CancelToken, kind: BotTurn = "question") -> Plan:
        self.asked.append((answer, utterance, kind))
        return Plan({"action": kind}, answer or "pregunta")

    def spoken(self, record: TurnRecord) -> None:
        self.spoken_turns.append(record.turn)

    def finish(self) -> Optional[Document]:
        return None


class Interviewer(FakeReasoner):
    """Says a fixed question whatever the step."""

    def reason(self, request: Document, cancel: CancelToken) -> Document:
        return self._answer(request, "Cuéntame tu experiencia.")

    reason_stream = None  # type: ignore[assignment]


def interviewer(
    config: TurnConfig = TurnConfig(backchannel="off"),
    executor: Any = None, tts: Any = None,
) -> Tuple[VoiceBot, Director, Output, Clock]:
    out, clock, director = Output(), Clock(), Director()
    b = VoiceBot(
        product("voicebot.interview"),
        FakeKnowledge(CHUNKS, everything=True), Interviewer(),
        tts or FakeTts(), out, clock, executor or InlineExecutor(), ROLES,
        director, config,
    )
    b.open("vb")
    return b, director, out, clock


def run(b: VoiceBot, clock: Clock, until: float, step: float = 0.1) -> None:
    while clock.t < until - 1e-9:
        clock.t = round(clock.t + step, 3)
        b.tick()


def said(uid: str, text: str, kind: str = "final") -> Any:
    event = final(text, 1, uid, kind=kind)
    utterance = event.utterance
    assert utterance is not None
    utterance.speaker = "S1"
    return event


def test_finals_add_up_until_the_candidate_stays_silent() -> None:
    b, director, out, clock = interviewer()
    assert len(out.records) == 1  # the opening
    asked_at = b.turn_taking.asked_at  # type: ignore[union-attr]
    assert asked_at is not None and asked_at > 0
    clock.t = 7.0
    b.on_event(said("u1", "Trabajé dos años en caja en un"))
    run(b, clock, 12.0)
    b.on_event(said("u2", "supermercado del barrio."))
    run(b, clock, 15.4)
    assert len(out.records) == 1
    run(b, clock, 15.6)
    assert len(out.records) == 2
    answer, utterance, kind = director.asked[-1]
    assert answer == ("Trabajé dos años en caja en un supermercado del "
                      "barrio.")
    assert (utterance, kind) == ("u2", "question")
    assert out.records[1].trigger.utterance == "u2"
    assert director.heard_ids == ["u1", "u2"]


def test_the_candidate_speaking_drops_a_turn_being_prepared() -> None:
    executor = ManualExecutor()
    b, director, out, clock = interviewer(executor=executor)
    executor.wait_idle()  # the opening speaks
    clock.t = 7.0
    b.on_event(said("u1", "Trabajé en caja."))
    run(b, clock, 10.6)
    assert len(executor.jobs) == 1  # a turn is being prepared
    b.on_event(said("u2", "y también", kind="partial"))
    assert out.of("interrupt")[-1]["by"] == "candidate"
    executor.wait_idle()
    assert out.records[-1].status == "cancelled"
    assert out.records[-1].spoken == []
    b.on_event(said("u2", "Y también reponía."))
    run(b, clock, 14.0)
    assert not executor.jobs
    run(b, clock, 14.2)
    executor.wait_idle()
    assert [r.status for r in out.records][-1] == "ok"
    assert director.asked[-1][0] == "Trabajé en caja. Y también reponía."


def test_the_opening_is_never_dropped() -> None:
    executor = ManualExecutor()
    b, _, out, clock = interviewer(executor=executor)
    b.on_event(said("u0", "Hola", kind="partial"))
    assert out.of("interrupt") == []
    executor.wait_idle()
    assert out.records[0].status == "ok"


def test_an_opening_stopped_by_the_candidate_still_starts_the_wait(
) -> None:
    executor = ManualExecutor()
    b, director, out, clock = interviewer(executor=executor)
    assert b.barge_in("vad") is True
    executor.wait_idle()
    assert out.records[0].status == "cancelled"
    assert b.turn_taking.asked_at == 0.0  # type: ignore[union-attr]
    run(b, clock, 10.05)
    executor.wait_idle()
    assert director.asked[-1] == ("", None, "reask")


def test_silence_re_asks_once_and_a_hint_keeps_the_answer() -> None:
    b, director, out, clock = interviewer()
    start = b.turn_taking.asked_at  # type: ignore[union-attr]
    assert start is not None
    run(b, clock, start + 10.05)
    assert director.asked[-1] == ("", None, "reask")
    assert len(out.records) == 2
    again = b.turn_taking.asked_at  # type: ignore[union-attr]
    assert again is not None
    clock.t = again + 7.0
    b.on_event(said("u1", "Pues en mi último trabajo, eh"))
    run(b, clock, clock.t + 7.05)
    assert director.asked[-1] == ("Pues en mi último trabajo, eh", None,
                                  "hint")
    assert b.turn_taking.answer == (  # type: ignore[union-attr]
        "Pues en mi último trabajo, eh")


def test_voice_activity_restarts_the_wait() -> None:
    b, director, out, clock = interviewer()
    start = b.turn_taking.asked_at  # type: ignore[union-attr]
    assert start is not None
    run(b, clock, start + 8.0)
    assert b.barge_in("vad") is False  # nothing to stop, but heard
    run(b, clock, start + 17.9)
    assert len(out.records) == 1
    run(b, clock, start + 18.1)
    assert director.asked[-1][2] == "reask"


def test_a_closed_interview_takes_no_more_turns() -> None:
    b, director, out, clock = interviewer()
    director.closed = True
    clock.t = 7.0
    b.on_event(said("u1", "Gracias a ti."))
    run(b, clock, 20.0)
    assert len(out.records) == 1
    assert b.skips == {"closed": 1}


def test_a_halted_session_does_not_tick() -> None:
    b, director, out, clock = interviewer()
    b.halted = "quota"
    run(b, clock, 30.0)
    assert len(out.records) == 1


def talk(b: VoiceBot, clock: Clock, start: float, seconds: int) -> None:
    """Speak for ``seconds`` with a 1.5 s pause every 3 s."""
    t = start
    i = 0
    while t < start + seconds:
        for k in range(4):
            clock.t = t + k
            b.on_event(said(f"u{i}", "y luego estuve", kind="partial"))
            b.tick()
        b.on_event(said(f"u{i}", "Estuve en el almacén y"))
        run(b, clock, t + 4.5)
        t += 4.5
        i += 1


def test_backchannels_are_spoken_quietly_and_not_as_the_bot() -> None:
    b, _, out, clock = interviewer(TurnConfig())
    b.warm_backchannels()
    talk(b, clock, 7.0, 30)
    sent = out.of("backchannel")
    assert len(sent) >= 2
    assert sent[0]["text"] == "Ajá." and sent[1]["text"] == "Vale."
    assert sent[0]["mode"] == "voice" and sent[0]["audio"]
    assert sent[0]["volume"] == 0.35 and sent[0]["format"] == "wav"
    assert len(out.records) == 1  # no turn while the answer goes on
    assert not b.playback.speaking(clock.t)


def test_visual_backchannels_carry_no_audio() -> None:
    b, _, out, clock = interviewer(TurnConfig(backchannel="visual"))
    b.warm_backchannels()
    talk(b, clock, 7.0, 20)
    sent = out.of("backchannel")
    assert sent and all(m["audio"] is None for m in sent)
    assert all(m["mode"] == "visual" for m in sent)


def test_a_failing_voice_leaves_the_backchannel_visible() -> None:
    from orchestrator.application.ports import TtsUnavailable

    class Broken(FakeTts):
        def synthesize(self, request: Document, cancel: CancelToken) -> Any:
            if request["text"] in ("Ajá.", "Vale.", "Entiendo.", "Mm-hm."):
                raise TtsUnavailable("down")
            return super().synthesize(request, cancel)

    b, _, out, clock = interviewer(TurnConfig(), tts=Broken())
    b.warm_backchannels()
    talk(b, clock, 7.0, 20)
    sent = out.of("backchannel")
    assert sent and sent[0]["audio"] is None


def test_without_warming_the_tokens_are_still_shown() -> None:
    b, _, out, clock = interviewer(TurnConfig())
    talk(b, clock, 7.0, 20)
    assert out.of("backchannel")[0]["audio"] is None


def test_backchannels_off_means_no_warming_and_none_sent() -> None:
    tts = FakeTts()
    b, _, out, clock = interviewer(TurnConfig(backchannel="off"), tts=tts)
    before = len(tts.requests)
    b.warm_backchannels()
    assert len(tts.requests) == before
    talk(b, clock, 7.0, 20)
    assert out.of("backchannel") == []


def test_the_support_bot_has_no_turn_taking() -> None:
    out = Output()
    b = VoiceBot(product("voicebot"), FakeKnowledge(CHUNKS, everything=True),
                 FakeReasoner(), FakeTts(), out, Clock(), InlineExecutor(),
                 {"S1": "user"})
    b.tick()
    b.warm_backchannels()
    b.on_event(final("¿Puedo devolver un producto?"))
    assert len(out.records) == 1
