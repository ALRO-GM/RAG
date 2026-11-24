"""Voice-bot sessions: the bot's feed, barge-in, audio dropped later."""

from typing import Any, Dict, List

import pytest

from orchestrator.adapters.outbound.fakes import (
    FakeKnowledge,
    FakeReasoner,
    FakeTts,
)
from orchestrator.adapters.outbound.runtime import InlineExecutor
from orchestrator.application.feed import Feed
from orchestrator.application.ports import Document, TurnSink
from orchestrator.application.sessions import LiveSessions, SessionError
from orchestrator.domain.records import TurnRecord
from tests.unit.orchestrator.fixtures import product, transcript
from tests.unit.orchestrator.test_sessions import (
    Clock,
    FakeEvents,
    FakeTranscriber,
    InlineRunner,
    Transcripts,
    messages,
)


class Turns:
    def __init__(self) -> None:
        self.records: List[TurnRecord] = []

    def write(self, record: TurnRecord) -> None:
        self.records.append(record)


def bot_sessions(events: List[Document], tts: Any = None,
                 turns: Any = None, runner: Any = None) -> LiveSessions:
    def turn_log(session_id: str, name: str) -> TurnSink:
        assert name == "voicebot"
        return turns if turns is not None else Turns()

    return LiveSessions(
        {"voicebot": product("voicebot"),
         "live_assist.support": product()},
        FakeKnowledge([{"id": "k1", "text": "Devoluciones en 30 días."}],
                      everything=True),
        FakeReasoner(), Transcripts({"t": transcript()}), Clock(),
        runner or InlineRunner(), InlineExecutor, FakeTranscriber(),
        FakeEvents(events), None, lambda: "vb-1",
        tts=tts, turn_log=turn_log if turns is not None else None,
    )


def user_final(uid: str, text: str, seq: int) -> Document:
    return {
        "schema": "live.event.v1", "session_id": "vb-1", "seq": seq,
        "type": "final", "at": "2026-11-21T10:00:00+01:00",
        "stream_time": float(seq),
        "utterance": {"id": uid, "turn": seq, "index": 0, "speaker": "S1",
                      "start": 0.0, "end": float(seq), "text": text,
                      "revision": 0},
        "timings": {"lag_ms": 600.0},
    }


def test_the_voicebot_is_offered_only_with_a_voice() -> None:
    assert {p["product"] for p in bot_sessions([]).list_products()} == {
        "live_assist.support"}
    shown = {p["product"]: p for p in
             bot_sessions([], tts=FakeTts()).list_products()}
    assert shown["voicebot"]["kind"] == "speaker"
    assert shown["live_assist.support"]["kind"] == "live_page"


def test_a_mic_session_is_answered_out_loud() -> None:
    turns = Turns()
    live = bot_sessions(
        [user_final("t0.s0", "¿Puedo devolver un producto?", 1),
         user_final("t1.s0", "¿Y cuánto tarda el reembolso?", 2)],
        tts=FakeTts(), turns=turns)
    view = live.start_live("voicebot", "mic")
    assert view["page"]["kind"] == "speaker"
    body = live.transcriber.bodies[0]  # type: ignore[union-attr]
    assert body["speakers"] == [{"channel": 0, "id": "S1", "role": "user"}]
    assert "parties" not in body and "diarize" not in body  # #249
    assert body["hang_ms"] == 400
    found = messages(live, "vb-1")
    kinds = [m["type"] for m in found]
    assert kinds.count("bot") == 2 and kinds.count("turn") == 2
    speech = [m for m in found if m["type"] == "speech"]
    assert {m["turn"] for m in speech} == {1, 2}
    assert all("audio" not in m for m in speech if m["turn"] == 1)
    assert all(m["audio"] for m in speech if m["turn"] == 2)
    assert live.get("vb-1")["calls"] == {"ok": 2}
    assert [r.turn for r in turns.records] == [1, 2]
    assert live.barge_in("vb-1") == {"stopped": False}  # ended


def test_barge_in_reaches_the_bot_of_a_running_session() -> None:
    runner = InlineRunner(hold=True)
    live = bot_sessions([], tts=FakeTts(), runner=runner)
    live.start_live("voicebot", "mic", "vb-2")
    assert live.barge_in("vb-2") == {"stopped": False}
    support = bot_sessions([], tts=FakeTts(), runner=InlineRunner())
    support.start_replay("demo-1", "live_assist.support", "t", None)
    with pytest.raises(SessionError) as caught:
        support.barge_in("demo-1")
    assert caught.value.code == "wrong_state"


def test_a_voicebot_replay_speaks_for_each_user_final() -> None:
    live = bot_sessions([], tts=FakeTts())
    live.start_replay("demo-vb", "voicebot", "t", None)
    found = messages(live, "demo-vb")
    assert any(m["type"] == "speech" for m in found)
    assert live.get("demo-vb")["state"] == "ended"


def test_feed_strip_removes_a_key_of_kept_messages() -> None:
    feed = Feed(limit=2)
    one = feed.publish({"type": "speech", "audio": "a"})
    two = feed.publish({"type": "speech", "audio": "b"})
    three = feed.publish({"type": "speech", "audio": "c"})
    feed.strip([one, two], "audio")
    found = dict(feed.read(0, 0.0)[0])
    assert one not in found and "audio" not in found[two]
    assert found[three]["audio"] == "c"


def test_turn_log_writes_jsonl(tmp_path: Any) -> None:
    from orchestrator.adapters.outbound.jsonl_sink import JsonlTurns
    from orchestrator.domain.records import TriggerInfo
    sink = JsonlTurns(tmp_path / "x" / "t.jsonl")
    sink.write(TurnRecord(session_id="s", turn=1, request_id="r",
                          trigger=TriggerInfo(utterance="u", speaker="S1",
                                              role="user", stream_time=0,
                                              text="hola"),
                          status="ok", spoken=["Hola."]))
    line: Dict[str, Any] = __import__("json").loads(
        (tmp_path / "x" / "t.jsonl").read_text("utf-8"))
    assert line["spoken"] == ["Hola."] and "audio" not in line
