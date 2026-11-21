"""Barge-in and echo: the domain rules and the VoiceBot's use of them."""

from orchestrator.adapters.outbound.fakes import FakeTts
from orchestrator.domain.bargein import BargeInPolicy, Playback, is_echo
from tests.unit.orchestrator.test_live_assist import ManualExecutor
from tests.unit.orchestrator.test_voicebot import Output, bot, final

SAID = ["Las devoluciones se aceptan durante 30 días desde la entrega."]


def test_echo_is_mostly_the_bots_own_words() -> None:
    assert is_echo("las devoluciones se aceptan durante treinta dias", SAID)
    assert is_echo("Devoluciones, ¡aceptan!", SAID)
    assert not is_echo("¿Y si el paquete llegó roto?", SAID)
    assert not is_echo("devoluciones", SAID)
    assert not is_echo("las devoluciones", [])


def test_playback_end_is_estimated_from_the_chunks() -> None:
    p = Playback(margin_s=0.2)
    assert not p.speaking(0.0) and not p.recent(0.0, 5.0)
    p.add(1, 1000.0, now=10.0)
    p.add(1, 500.0, now=10.1)
    assert p.until == 11.5 and p.turn == 1
    assert p.speaking(11.6) and not p.speaking(11.8)
    assert p.recent(12.5, 1.5)
    p.stop()
    assert not p.speaking(10.5)
    p.add(2, 100.0, now=20.0)
    assert p.until == 20.1


def test_policy_needs_speaking_user_words_and_no_echo() -> None:
    policy = BargeInPolicy(["user"])
    speech = final("Espera, otra pregunta", kind="partial")
    assert policy.decide(speech, "user", True, SAID) == "speech"
    assert policy.decide(speech, "user", False, SAID) is None
    assert policy.decide(speech, "bot", True, SAID) is None
    assert policy.decide(final("eh", kind="partial"), "user", True,
                         SAID) is None
    echo = final("se aceptan durante 30 días", kind="partial")
    assert policy.decide(echo, "user", True, SAID) is None
    ended = speech.model_copy(update={"type": "session_ended"})
    assert policy.decide(ended, "user", True, SAID) is None


def test_user_speech_over_the_bot_interrupts_its_playback() -> None:
    out = Output()
    b = bot(output=out, tts=FakeTts(seconds_per_word=0.5))
    b.on_event(final("¿Puedo devolver un producto?", 1, "t0.s0"))
    assert b.playback.speaking(b.clock.now())
    b.on_event(final("Espera, una cosa más", 2, "t1.s0", kind="partial"))
    assert out.of("interrupt") == [{"type": "interrupt", "turn": 1,
                                    "by": "speech"}]
    assert out.records[0].interrupted_by == "speech"
    assert not b.playback.speaking(b.clock.now())


def test_echo_of_the_bot_is_neither_a_turn_nor_a_barge_in() -> None:
    out = Output()
    b = bot(output=out, tts=FakeTts(seconds_per_word=0.5))
    b.on_event(final("¿Puedo devolver un producto?", 1, "t0.s0"))
    b.on_event(final("las devoluciones se aceptan durante 30 días", 2,
                     "t1.s0", kind="partial"))
    b.on_event(final("las devoluciones se aceptan durante 30 días", 3,
                     "t1.s0"))
    assert out.of("interrupt") == [] and len(out.records) == 1
    assert b.skips == {"echo": 1}


def test_the_pages_voice_detector_stops_a_turn_in_flight() -> None:
    out = Output()
    executor = ManualExecutor()
    b = bot(output=out, executor=executor)
    assert b.barge_in() is False
    b.on_event(final("¿Puedo devolver un producto?"))
    assert b.barge_in("vad") is True
    for job in executor.jobs:
        job()
    assert out.of("interrupt") == [{"type": "interrupt", "turn": 1,
                                    "by": "vad"}]
    [record] = out.records
    assert record.status == "cancelled" and record.interrupted_by == "vad"
    assert out.of("speech") == []
