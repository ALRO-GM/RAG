"""VoiceBot: knowledge, streamed reply, one sentence at a time to TTS."""

from typing import Any, Callable, Dict, Iterator, List, Optional

import base64

from orchestrator.adapters.outbound.fakes import (
    FakeKnowledge,
    FakeReasoner,
    FakeTts,
)
from orchestrator.adapters.outbound.runtime import InlineExecutor
from orchestrator.application.ports import (
    CancelToken,
    Document,
    KnowledgeUnavailable,
    ReasonerUnavailable,
    TtsUnavailable,
)
from orchestrator.application.voicebot import BOT_SPEAKER, VoiceBot
from orchestrator.domain.events import LiveEvent
from orchestrator.domain.records import TurnRecord
from orchestrator.domain.speech import SpeechChunk
from tests.unit.orchestrator.fixtures import product
from tests.unit.orchestrator.test_live_assist import ManualExecutor, StepClock

CHUNKS = [{"id": "kb:1", "text": "Las devoluciones se aceptan durante 30 "
           "días desde la entrega."}]
ROLES = {"S1": "user"}


class Output:
    """Collects what the bot publishes and records."""

    def __init__(self) -> None:
        self.messages: List[Document] = []
        self.records: List[TurnRecord] = []

    def publish(self, message: Document) -> None:
        self.messages.append(message)

    def record(self, record: TurnRecord) -> None:
        self.records.append(record)

    def of(self, kind: str) -> List[Document]:
        return [m for m in self.messages if m["type"] == kind]


def final(text: str, seq: int = 1, uid: str = "t0.s0",
          lag: Optional[float] = 650.0, kind: str = "final") -> LiveEvent:
    data: Dict[str, Any] = {
        "schema": "live.event.v1", "session_id": "vb", "seq": seq,
        "type": kind, "at": "2026-11-21T10:00:00+01:00",
        "stream_time": float(seq),
        "utterance": {"id": uid, "turn": 0, "index": 0, "speaker": "S1",
                      "start": 0.0, "end": float(seq), "text": text,
                      "revision": 0},
    }
    if lag is not None:
        data["timings"] = {"lag_ms": lag}
    return LiveEvent.model_validate(data)


def bot(reasoner: Any = None, tts: Any = None, knowledge: Any = None,
        executor: Any = None, output: Optional[Output] = None,
        config: Any = None) -> VoiceBot:
    return VoiceBot(
        config or product("voicebot"),
        knowledge or FakeKnowledge(CHUNKS, everything=True),
        reasoner or FakeReasoner(), tts or FakeTts(), output or Output(),
        StepClock(), executor or InlineExecutor(), ROLES,
    )


def test_a_user_final_is_answered_out_loud_sentence_by_sentence() -> None:
    out = Output()
    tts = FakeTts()
    b = bot(tts=tts, output=out)
    b.on_event(final("¿Puedo devolver un producto?", lag=640.0))
    [record] = out.records
    assert record.status == "ok" and record.cites == ["kb:1"]
    speech = out.of("speech")
    assert [s["text"] for s in speech] == [
        "Te cuento lo que sé.",
        "Las devoluciones se aceptan durante 30 días desde la entrega.",
    ]
    assert record.spoken == [s["text"] for s in speech]
    first = base64.b64decode(speech[0]["audio"])
    assert first[:4] == b"RIFF" and speech[0]["format"] == "wav"
    assert speech[0]["sample_rate"] == 16000
    assert [r["request_id"] for r in tts.requests] == ["vb-b1-s0",
                                                       "vb-b1-s1"]
    assert tts.requests[0]["voice"] == "es_ES-davefx-medium"
    stages = record.stages
    assert stages.event_lag_ms == 640.0
    assert stages.knowledge_ms is not None
    assert stages.first_text_ms is not None
    assert stages.first_sentence_ms is not None
    assert stages.first_audio_ms is not None and stages.total_ms is not None
    assert stages.first_audio_ms <= stages.total_ms
    assert record.audio_ms > 0
    assert out.messages[0] == {"type": "bot", "turn": 1,
                               "state": "thinking", "utterance": "t0.s0"}


def test_the_first_sentence_is_spoken_before_the_reply_ends() -> None:
    order: List[str] = []

    class Tracing(FakeReasoner):
        def reason_stream(self, request: Document, cancel: CancelToken,
                          on_text: Callable[[str], None]) -> Document:
            def traced(piece: str) -> None:
                order.append("piece")
                on_text(piece)
            response = super().reason_stream(request, cancel, traced)
            order.append("end")
            return response

    class Recording(FakeTts):
        def synthesize(self, request: Document,
                       cancel: CancelToken) -> Iterator[SpeechChunk]:
            order.append("tts")
            return super().synthesize(request, cancel)

    bot(reasoner=Tracing(), tts=Recording()).on_event(
        final("¿Puedo devolver algo?"))
    assert order.index("tts") < order.index("end")
    assert order.count("tts") == 2


def test_the_query_is_the_users_whole_turn() -> None:
    knowledge = FakeKnowledge(CHUNKS, everything=True)
    executor = ManualExecutor()
    b = bot(knowledge=knowledge, executor=executor)
    b.on_event(final("Hola, buenas tardes.", 1, "t0.s0"))
    b.on_event(final("Quiero devolver una lámpara.", 2, "t0.s1"))
    for job in executor.jobs:
        job()
    assert [q["text"] for q in knowledge.queries] == [
        "Hola, buenas tardes. Quiero devolver una lámpara.",
    ]
    assert b.skips == {}


def test_a_newer_final_cancels_the_turn_and_stops_playback() -> None:
    out = Output()
    executor = ManualExecutor()
    reasoner = FakeReasoner()
    b = bot(reasoner=reasoner, output=out, executor=executor)
    b.on_event(final("Hola, buenas tardes.", 1, "t0.s0"))
    b.on_event(final("Quiero devolver una lámpara.", 2, "t0.s1"))
    assert out.of("interrupt") == [{"type": "interrupt", "turn": 1,
                                    "by": "new_turn"}]
    for job in executor.jobs:
        job()
    assert [r.status for r in out.records] == ["cancelled", "ok"]
    assert out.records[0].interrupted_by == "new_turn"
    assert len(reasoner.requests) == 1
    assert {s["turn"] for s in out.of("speech")} == {2}


def test_without_cancel_stale_the_newer_final_is_skipped() -> None:
    config = product("voicebot")
    assert config.trigger is not None
    config = config.model_copy(update={"trigger": config.trigger.model_copy(
        update={"cancel_stale": False})})
    executor = ManualExecutor()
    b = bot(executor=executor, config=config)
    b.on_event(final("Hola, buenas tardes.", 1, "t0.s0"))
    b.on_event(final("Quiero devolver una lámpara.", 2, "t0.s1"))
    assert b.skips == {"in_flight": 1} and len(executor.jobs) == 1


def test_the_bots_words_join_the_window_of_the_next_turn() -> None:
    reasoner = FakeReasoner()
    b = bot(reasoner=reasoner)
    b.on_event(final("¿Puedo devolver algo?", 1, "t0.s0"))
    b.on_event(final("¿Y cuánto tarda el reembolso?", 2, "t1.s0"))
    window = reasoner.requests[1]["context"]["window"]
    assert [w["speaker"] for w in window] == ["S1", BOT_SPEAKER, "S1"]
    assert reasoner.requests[1]["context"]["speaker_roles"][BOT_SPEAKER] == \
        "bot"
    assert reasoner.requests[1]["template"] == "conversational.voicebot@1"
    assert reasoner.requests[1]["constraints"]["deadline_ms"] == 1800


def test_a_reasoner_that_does_not_stream_is_spoken_at_the_end() -> None:
    class Plain:
        def __init__(self) -> None:
            self.inner = FakeReasoner()

        def reason(self, request: Document, cancel: CancelToken) -> Document:
            return self.inner.reason(request, cancel)

    out = Output()
    bot(reasoner=Plain(), output=out).on_event(final("¿Puedo devolver?"))
    assert len(out.of("speech")) == 2 and out.records[0].status == "ok"
    assert out.records[0].stages.first_text_ms is None


def test_the_final_text_completes_what_the_stream_missed() -> None:
    class Truncated(FakeReasoner):
        def reason_stream(self, request: Document, cancel: CancelToken,
                          on_text: Callable[[str], None]) -> Document:
            on_text('{"text": "Primera frase completa. Seg')
            return {"schema": "reasoner.response.v1",
                    "request_id": request["request_id"],
                    "role": "conversational", "status": "ok",
                    "reply": {"text": "Primera frase completa. Segunda "
                              "frase al final.", "cites": []},
                    "model": "m", "backend": "local",
                    "latency": {"total_ms": 1.0}}

    out = Output()
    bot(reasoner=Truncated(), output=out).on_event(final("Hola, ¿qué tal?"))
    assert [s["text"] for s in out.of("speech")] == [
        "Primera frase completa.", "Segunda frase al final.",
    ]


def test_failures_say_the_fallback_and_quota_halts() -> None:
    out = Output()
    b = bot(reasoner=FakeReasoner(status="timeout"), output=out)
    b.on_event(final("¿Puedo devolver algo?"))
    assert out.records[0].status == "timeout"
    assert out.of("speech")[0]["text"].startswith("Perdona")
    out = Output()
    b = bot(reasoner=FakeReasoner(status="quota"), output=out)
    b.on_event(final("¿Puedo devolver algo?", 1, "t0.s0"))
    b.on_event(final("¿Me oyes ahora?", 2, "t1.s0"))
    assert b.halted == "quota" and b.skips == {"halted": 1}
    assert out.of("speech") == []


def test_unreachable_services() -> None:
    class Down:
        def query(self, query: Document) -> Document:
            raise KnowledgeUnavailable("down")

        def reason(self, request: Document, cancel: CancelToken) -> Document:
            raise ReasonerUnavailable("down")

    out = Output()
    bot(knowledge=Down(), reasoner=Down(), output=out).on_event(
        final("¿Puedo devolver algo?"))
    record = out.records[0]
    assert record.knowledge_status == "unavailable" and record.chunks == 0
    assert record.status == "error" and record.error is not None
    assert record.spoken and record.spoken[0].startswith("Perdona")


def test_a_tts_failure_still_shows_the_text() -> None:
    class Mute:
        def synthesize(self, request: Document,
                       cancel: CancelToken) -> Iterator[SpeechChunk]:
            raise TtsUnavailable("no voice")

    out = Output()
    bot(tts=Mute(), output=out).on_event(final("¿Puedo devolver algo?"))
    speech = out.of("speech")
    assert speech and all(s["audio"] is None for s in speech)
    assert out.records[0].error == {"kind": "tts", "message": "no voice"}


def test_short_or_foreign_finals_do_not_start_a_turn() -> None:
    out = Output()
    b = bot(output=out)
    b.on_event(final("Vale."))
    b.on_event(final("hola", kind="partial"))
    assert out.records == [] and b.skips == {"min_words": 1}


def test_the_product_must_be_the_voicebot() -> None:
    import pytest
    with pytest.raises(ValueError):
        bot(config=product("live_assist.support"))


def test_english_fallback() -> None:
    config = product("voicebot").model_copy(update={"language": "en"})
    out = Output()
    bot(reasoner=FakeReasoner(status="error"), output=out,
        config=config).on_event(final("Can I return it?"))
    assert out.of("speech")[0]["text"].startswith("Sorry")
