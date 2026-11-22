"""Interview sessions of the voice bot, and the voice bot's topic (#208)."""

from typing import Any, Iterator, List, Optional, cast

import pytest

from orchestrator.adapters.outbound.fakes import (
    FakeKnowledge,
    FakeReasoner,
    FakeTts,
)
from orchestrator.adapters.outbound.runtime import InlineExecutor
from orchestrator.application.interview import (
    InterviewPlanner,
    InterviewWiring,
)
from orchestrator.application.ports import CancelToken, Document
from orchestrator.application.sessions import (
    LiveSessions,
    SessionError,
    turn_config,
)
from orchestrator.application.voicebot import VoiceBot
from orchestrator.domain.config import InterviewSection
from orchestrator.domain.live_choices import apply_live_choices
from tests.unit.orchestrator.fixtures import product, transcript
from tests.unit.orchestrator.test_interview_usecases import (
    OFFER,
    Handoff,
    Store,
)
from tests.unit.orchestrator.test_sessions import (
    Clock,
    FakeEvents,
    FakeTranscriber,
    InlineRunner,
    Transcripts,
    messages,
)
from tests.unit.orchestrator.test_voicebot_sessions import user_final


class PacedEvents(FakeEvents):
    """Events with ``gap_s`` seconds of ticking before each (and after).

    The session's clock moves only here, and the bot's turn-taking is
    ticked every half second, as the session's ticker would.
    """

    def __init__(self, documents: List[Document], gap_s: float = 8.0,
                 tail_s: float = 12.0) -> None:
        super().__init__(documents)
        self.gap_s = gap_s
        self.tail_s = tail_s
        self.sessions: Optional[LiveSessions] = None

    def _tick(self, session_id: str, seconds: float) -> None:
        assert self.sessions is not None
        bot = cast(VoiceBot, self.sessions._find(session_id).assist)
        for _ in range(int(seconds / 0.5)):
            self.sessions.clock.sleep(0.5)
            bot.tick()

    def follow(self, session_id: str, stop: CancelToken
               ) -> Iterator[Document]:
        assert self.sessions is not None
        for document in self.documents:
            self._tick(session_id, self.gap_s)
            now = self.sessions.clock.now()
            said = dict(document["utterance"], start=max(0.0, now - 1.0),
                        end=now)
            yield dict(document, utterance=said)
        self._tick(session_id, self.tail_s)


def live(events: List[Document], wired: bool = True,
         handoff: Any = None, ticker: Any = None) -> LiveSessions:
    store = Store()
    reasoner = FakeReasoner()
    knowledge = FakeKnowledge([{"id": "m1", "text": "Pregunta STAR"}],
                              everything=True)
    planner = InterviewPlanner(store, knowledge, reasoner, None,
                               lambda: InterviewSection(), new_id=lambda:
                               "plan-1")
    paced = PacedEvents(events)
    sessions = LiveSessions(
        {"voicebot": product("voicebot"),
         "voicebot.interview": product("voicebot.interview")},
        knowledge, reasoner, Transcripts({"t": transcript()}), Clock(),
        InlineRunner(), InlineExecutor, FakeTranscriber(),
        paced, None, lambda: "vb-1", tts=FakeTts(),
        interview=InterviewWiring(planner, store, reasoner,
                                  handoff or Handoff()) if wired else None,
        ticker=ticker,
    )
    paced.sessions = sessions
    return sessions


def test_the_interviewer_is_offered_only_with_its_wiring() -> None:
    names = {p["product"] for p in live([], wired=False).list_products()}
    assert names == {"voicebot"}
    shown = {p["product"]: p for p in live([]).list_products()}
    assert shown["voicebot.interview"]["interview"] is True
    assert shown["voicebot.interview"]["domain"] == "interview.generic"
    assert shown["voicebot"]["interview"] is False


def test_an_interview_session_speaks_first_and_hands_over() -> None:
    handoff = Handoff()
    sessions = live([
        user_final("t0.s0", "Estuve dos años en caja en un supermercado "
                   "del barrio atendiendo a clientes", 1),
        user_final("t1.s0", "No lo conozco.", 2),
        user_final("t2.s0", "No lo conozco.", 3),
        user_final("t3.s0", "No lo conozco.", 4),
        user_final("t4.s0", "No lo conozco.", 5),
        user_final("t5.s0", "Vale, perfecto entonces", 6),
    ], handoff=handoff)
    plan = sessions.interview.planner.create(  # type: ignore[union-attr]
        {"offer": OFFER})
    view = sessions.start_live("voicebot.interview", "mic", plan=plan["id"],
                               candidate="  Ana   García ")
    assert view["plan"] == "plan-1" and view["candidate"] == "Ana García"
    body = sessions.transcriber.bodies[0]  # type: ignore[union-attr]
    assert body["handoff"] == {"transcript_v1": False,
                               "run_post_call": False}
    assert body["speakers"] == [{"channel": 0, "id": "S1",
                                 "role": "candidate"}]
    found = messages(sessions, "vb-1")
    kinds = [m["type"] for m in found]
    turns = [m["record"] for m in found if m["type"] == "turn"]
    assert turns[0]["trigger"]["role"] == "bot"  # the opening
    assert kinds.count("coverage") == len(turns)
    assert "closed" in [m.get("state") for m in found
                        if m["type"] == "interview"]
    assert sessions.get("vb-1")["skipped"].get("closed", 0) >= 1
    result = [m for m in found if m["type"] == "interview_result"][0]
    assert result["sheet"]["candidate"] == "Ana García"
    assert result["sheet"]["closed_by"] == "covered"
    assert handoff.transcripts[0]["utterances"][0]["speaker"] == "BOT"
    assert all(r["stages"]["director_ms"] is not None for r in turns)
    rows = sessions.interview.planner.results(  # type: ignore[union-attr]
        "plan-1")["candidates"]
    assert rows[0]["candidate"] == "Ana García"


def test_interview_sessions_need_a_known_plan() -> None:
    sessions = live([])
    with pytest.raises(SessionError) as caught:
        sessions.start_live("voicebot.interview", "mic")
    assert caught.value.code == "invalid_request"
    with pytest.raises(SessionError) as caught:
        sessions.start_live("voicebot.interview", "mic", plan="plan-9")
    assert caught.value.code == "not_found"
    with pytest.raises(SessionError) as caught:
        sessions.start_live("voicebot", "mic", plan="plan-1")
    assert caught.value.code == "invalid_request"


def test_a_voicebot_session_takes_a_topic_domain() -> None:
    sessions = live([user_final("t0.s0", "¿Qué hago en la entrevista?", 1)])
    sessions.start_live("voicebot", "mic", domain="interview.generic")
    knowledge: FakeKnowledge = sessions.knowledge  # type: ignore
    assert knowledge.queries[-1]["domain"] == "interview.generic"
    for bad in ("Bad Domain", "../x"):
        with pytest.raises(SessionError):
            live([]).start_live("voicebot", "mic", domain=bad)
    with pytest.raises(SessionError):
        live([]).start_live("voicebot.interview", "mic", plan="plan-1",
                            domain="support.demo")


def test_saved_settings_tune_the_interviewer() -> None:
    config = product("voicebot.interview")
    tuned = apply_live_choices(config, {
        "interview.slot_filler": "heuristic", "interview.max_probes": "2",
        "interview.minutes": "30", "interview.post_call": "on",
        "voicebot.reasoner_model": "sonnet",
    })
    section = tuned.interview
    assert section is not None
    assert (section.filler, section.max_probes, section.minutes,
            section.post_call) == ("heuristic", 2, 30, True)
    assert tuned.reasoner is not None and tuned.reasoner.model == "sonnet"
    same = apply_live_choices(config, {"interview.max_probes": "9"})
    assert same.interview == config.interview


def test_the_interview_waits_for_the_end_of_each_answer() -> None:
    sessions = live([
        user_final("t0.s0", "Estuve dos años en caja en un", 1),
        user_final("t0.s1", "supermercado del barrio atendiendo a "
                   "clientes cada día.", 2),
    ])
    sessions.events.gap_s = 2.0  # type: ignore[union-attr]
    sessions.events.tail_s = 20.0  # type: ignore[union-attr]
    plan = sessions.interview.planner.create(  # type: ignore[union-attr]
        {"offer": OFFER})
    sessions.start_live("voicebot.interview", "mic", plan=plan["id"])
    found = messages(sessions, "vb-1")
    turns = [m["record"] for m in found if m["type"] == "turn"]
    assert len(turns) == 3  # the opening, the answer, then a re-ask
    answer = turns[1]["trigger"]
    assert answer["utterance"] == "t0.s1"
    assert answer["text"] == ("Estuve dos años en caja en un supermercado "
                              "del barrio atendiendo a clientes cada día.")
    assert [m["mode"] for m in found if m["type"] == "backchannel"] == []


def test_the_session_ticks_the_interviewer_while_it_runs() -> None:
    ticker = InlineRunner(hold=True)
    sessions = live([], ticker=ticker)
    plan = sessions.interview.planner.create(  # type: ignore[union-attr]
        {"offer": OFFER})
    sessions.start_live("voicebot.interview", "mic", plan=plan["id"])
    [loop] = ticker.jobs
    loop()  # the session has ended: the loop returns at once
    assert sessions.get("vb-1")["state"] == "ended"


def test_the_ticker_loop_ticks_until_the_session_stops() -> None:
    ticker = InlineRunner(hold=True)
    sessions = live([], ticker=ticker)
    plan = sessions.interview.planner.create(  # type: ignore[union-attr]
        {"offer": OFFER})
    sessions.events = PacedEvents([], gap_s=0, tail_s=0)
    sessions.runner = InlineRunner(hold=True)
    sessions.start_live("voicebot.interview", "mic", plan=plan["id"])
    session = sessions._find("vb-1")
    clock = sessions.clock
    ticks: List[float] = []

    class Counting:
        def sleep(self, seconds: float) -> None:
            clock.sleep(seconds)
            ticks.append(seconds)
            if len(ticks) >= 150:
                session.stop_token.cancel()

        def now(self) -> float:
            return clock.now()

    sessions.clock = Counting()
    [loop] = ticker.jobs
    loop()
    assert len(ticks) == 150 and ticks[0] == 0.1
    bot = cast(VoiceBot, session.assist)
    assert bot.turn_taking is not None and bot.turn_taking.reasked


def test_the_turn_taking_thresholds_come_from_the_settings() -> None:
    tuned = apply_live_choices(product("voicebot.interview"), {
        "interview.turn_silence_s": "4.5", "interview.think_s": "8",
        "interview.max_wait_s": "12", "interview.stuck_s": "9",
        "interview.backchannel": "visual",
    })
    assert tuned.interview is not None
    config = turn_config(tuned.interview)
    assert (config.turn_silence_s, config.think_s, config.max_wait_s,
            config.stuck_s, config.backchannel) == (4.5, 8.0, 12.0, 9.0,
                                                    "visual")
    for step, bad in (("interview.turn_silence_s", "1.5"),
                      ("interview.think_s", "16"),
                      ("interview.max_wait_s", "x"),
                      ("interview.backchannel", "loud")):
        same = apply_live_choices(product("voicebot.interview"),
                                  {step: bad})
        assert same.interview == product("voicebot.interview").interview


def test_the_backchannels_are_synthesised_after_the_opening() -> None:
    sessions = live([])
    plan = sessions.interview.planner.create(  # type: ignore[union-attr]
        {"offer": OFFER})
    sessions.runner = InlineRunner(hold=True)
    sessions.start_live("voicebot.interview", "mic", plan=plan["id"])
    bot = cast(VoiceBot, sessions._find("vb-1").assist)
    order: List[str] = []

    class Executor:
        def wait_idle(self) -> None:
            order.append("opening over")

    bot.executor = Executor()  # type: ignore[assignment]
    bot.warm_backchannels = (  # type: ignore[method-assign]
        lambda: order.append("backchannels"))
    backchannels = sessions.runner.jobs[0]
    backchannels()
    assert order == ["opening over", "backchannels"]
