"""TurnTaking: the interviewer's patience, with simulated time (#213)."""

from typing import List, Optional, Tuple

import pytest

from orchestrator.domain.turn_taking import (
    BACKCHANNELS,
    Decision,
    TurnConfig,
    TurnTaking,
    looks_unfinished,
)


def run(policy: TurnTaking, start: float, end: float, step: float = 0.1,
        busy: bool = False, slot: Optional[str] = "s1"
        ) -> List[Tuple[float, Decision]]:
    """Tick from ``start`` to ``end``; return the decisions taken."""
    found: List[Tuple[float, Decision]] = []
    for i in range(int(round((end - start) / step)) + 1):
        t = round(start + i * step, 3)
        decision = policy.tick(t, busy, slot)
        if decision != "wait":
            found.append((t, decision))
    return found


def asked(config: TurnConfig = TurnConfig(), at: float = 0.0
          ) -> TurnTaking:
    policy = TurnTaking(config)
    policy.bot_spoke(at, "question")
    return policy


QUIET = TurnConfig(backchannel="off")


@pytest.mark.parametrize("text", [
    "Trabajé dos años en caja y", "Lo dejé porque", "Pues, o sea",
    "Eh", "Estuve en Madrid, en Sevilla,", "Me encargaba de",
    "Fue difícil pero", "Y entonces…", "Lo hice por ejemplo",
    "I worked there and", "The point is, you know",
])
def test_answers_cut_in_the_middle_look_unfinished(text: str) -> None:
    assert looks_unfinished(text)


@pytest.mark.parametrize("text", [
    "Trabajé dos años en caja.", "Sí.", "No sé qué.", "Nada más.",
    "¿Cómo?", "Lo resolví hablando con el cliente", "", "  ",
    "I fixed it myself.", "...!",
])
def test_whole_answers_do_not(text: str) -> None:
    assert not looks_unfinished(text)


def test_nothing_happens_before_the_bot_first_spoke() -> None:
    assert run(TurnTaking(TurnConfig()), 0.0, 30.0) == []


def test_a_finished_answer_ends_after_the_silence() -> None:
    policy = asked(QUIET)
    policy.heard(7.0, "u1", "Trabajé dos años en caja.", final=True)
    assert run(policy, 7.0, 20.0) == [(10.5, "end_turn")]
    assert policy.answer == "Trabajé dos años en caja."
    assert policy.last_utterance == "u1"


def test_finals_add_up_and_each_one_restarts_the_wait() -> None:
    policy = asked(QUIET)
    policy.heard(7.0, "u1", "Trabajé dos años en caja.", final=True)
    assert run(policy, 7.0, 10.0) == []
    policy.heard(10.0, "u2", "Luego pasé a reponer.", final=True)
    assert run(policy, 10.0, 13.4) == []
    policy.heard(13.4, "u2", "Luego pasé a reponer el almacén.",
                 final=True)
    assert run(policy, 13.4, 30.0) == [(16.9, "end_turn")]
    assert policy.answer == ("Trabajé dos años en caja. Luego pasé a "
                             "reponer el almacén.")


def test_partials_and_voice_activity_restart_the_wait_too() -> None:
    policy = asked(QUIET)
    policy.heard(7.0, "u1", "Trabajé dos años.", final=True)
    policy.heard(10.0, "u2", "en el", final=False)
    assert run(policy, 10.0, 13.0) == []
    policy.activity(13.0)
    assert run(policy, 13.0, 30.0)[0] == (16.5, "end_turn")


def test_think_time_holds_a_quick_answer() -> None:
    policy = asked(QUIET)
    policy.heard(0.5, "u1", "Sí.", final=True)
    assert run(policy, 0.5, 10.0) == [(6.0, "end_turn")]


def test_the_think_time_starts_when_the_question_ends() -> None:
    policy = asked(QUIET, at=4.0)
    policy.heard(4.2, "u1", "Sí.", final=True)
    assert run(policy, 4.2, 20.0) == [(10.0, "end_turn")]


def test_an_unfinished_answer_waits_up_to_the_longest_wait() -> None:
    policy = asked(QUIET)
    policy.heard(7.0, "u1", "Trabajé en caja y", final=True)
    assert run(policy, 7.0, 30.0, slot=None) == [(17.0, "end_turn")]


def test_silence_after_a_question_re_asks_it_once() -> None:
    policy = asked(QUIET)
    assert run(policy, 0.0, 30.0) == [(10.0, "reask")]
    policy.bot_spoke(14.0, "reask")
    assert run(policy, 14.0, 40.0) == [(24.0, "end_turn")]
    assert policy.answer == ""
    policy.bot_spoke(30.0, "question")
    assert run(policy, 30.0, 45.0) == [(40.0, "reask")]


def test_noise_restarts_the_re_ask_wait() -> None:
    policy = asked(QUIET)
    policy.activity(8.0)
    assert run(policy, 0.0, 30.0) == [(18.0, "reask")]


def test_a_stuck_candidate_gets_one_hint_per_slot() -> None:
    policy = asked(QUIET)
    policy.heard(7.0, "u1", "Pues en mi último trabajo, eh", final=True)
    assert run(policy, 7.0, 30.0) == [(14.0, "hint")]
    policy.bot_spoke(18.0, "hint")
    assert policy.answer == "Pues en mi último trabajo, eh"
    policy.heard(25.0, "u2", "organicé los turnos y", final=True)
    assert run(policy, 25.0, 40.0) == [(35.0, "end_turn")]
    assert policy.answer.endswith("organicé los turnos y")
    policy.bot_spoke(40.0, "question")
    policy.heard(47.0, "u3", "Lo que", final=True)
    assert run(policy, 47.0, 60.0, slot="s2") == [(54.0, "hint")]


def test_nothing_is_decided_while_the_bot_is_busy() -> None:
    policy = asked(QUIET)
    policy.heard(7.0, "u1", "Sí.", final=True)
    assert run(policy, 7.0, 30.0, busy=True) == []


def test_a_decided_turn_is_waited_for_until_speech_or_a_drop() -> None:
    policy = asked(QUIET)
    policy.heard(7.0, "u1", "Trabajé en caja.", final=True)
    assert run(policy, 7.0, 15.0) == [(10.5, "end_turn")]
    assert run(policy, 15.0, 30.0) == []
    assert policy.heard(30.0, "u2", "Y también", final=False) is True
    assert run(policy, 30.0, 45.0) == [(33.5, "end_turn")]
    policy.dropped()
    assert run(policy, 45.0, 46.0)[0] == (45.0, "end_turn")


def test_an_empty_text_is_not_speech() -> None:
    policy = asked(QUIET)
    assert policy.heard(1.0, "u1", " ... ", final=False) is False
    assert policy.heard(1.0, "u1", "", final=True) is False
    assert policy.answer == ""


def test_backchannels_come_at_pauses_of_a_long_answer() -> None:
    times = backchannel_times(asked(TurnConfig()))
    assert len(times) >= 3
    assert times[0] >= 6.0 + 8.0
    assert all(b - a >= 8.0 for a, b in zip(times, times[1:]))


def backchannel_times(policy: TurnTaking) -> List[float]:
    """Speak 3 s, pause 1.5 s, for 45 s; return the backchannels."""
    found: List[float] = []
    t = 6.0
    for i in range(10):
        for k in range(4):
            policy.heard(t + k, f"u{i}", "y entonces hice", final=False)
        policy.heard(t + 3.0, f"u{i}", "Lo hice así y", final=True)
        found += [at for at, d in run(policy, t + 3.0, t + 4.4)
                  if d == "backchannel"]
        t += 4.5
    return found


def test_one_backchannel_per_pause_and_never_after_the_answer() -> None:
    policy = asked(TurnConfig())
    for k in range(10):
        policy.heard(6.0 + k, "u1", "y luego estuve en", final=False)
    policy.heard(16.0, "u1", "Estuve en caja dos años.", final=True)
    found = run(policy, 16.0, 30.0)
    assert found == [(17.2, "backchannel"), (19.5, "end_turn")]


def test_backchannels_can_be_turned_off() -> None:
    policy = asked(QUIET)
    for k in range(10):
        policy.heard(6.0 + k, "u1", "y luego estuve en", final=False)
    policy.heard(16.0, "u1", "Estuve en caja dos años.", final=True)
    assert run(policy, 16.0, 30.0) == [(19.5, "end_turn")]


def test_a_short_answer_gets_no_backchannel() -> None:
    policy = asked(TurnConfig())
    policy.heard(6.5, "u1", "Sí, claro.", final=True)
    assert run(policy, 6.5, 20.0) == [(10.0, "end_turn")]


def test_the_tokens_are_short() -> None:
    for tokens in BACKCHANNELS.values():
        assert all(len(t.split()) <= 2 for t in tokens)
