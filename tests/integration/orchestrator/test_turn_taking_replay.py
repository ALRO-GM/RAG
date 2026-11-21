"""Replay of scripted answers against the interviewer (#213).

Runs ``experiments/voicebot/turn_taking_replay.py`` (the real
``VoiceBot`` and ``TurnTaking`` on a simulated clock, fake model and
voice) on a few answers with pauses of 2 to 5 seconds.
"""

import importlib.util
from pathlib import Path
from types import ModuleType

from orchestrator.domain.turn_taking import TurnConfig

SCRIPT = (Path(__file__).resolve().parents[3] / "experiments" / "voicebot"
          / "turn_taking_replay.py")


def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("turn_replay", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pauses_after_unfinished_fragments_never_end_the_answer() -> None:
    run = _script().replay("unfinished", 12, TurnConfig())
    assert run["pause_s"][0] >= 2.0 and run["pause_s"][1] <= 5.0
    assert run["pauses_over_4s"] > 0
    assert run["premature_decisions"] == 0
    assert run["premature_turns"] == 0
    assert run["end_delay_s"]["p50"] == 3.5


def test_a_longer_silence_covers_pauses_after_whole_sentences() -> None:
    run = _script().replay("finished", 12, TurnConfig(turn_silence_s=5.0))
    assert run["premature_decisions"] == 0 and run["premature_turns"] == 0


def test_the_default_silence_can_cut_a_long_pause_after_a_sentence() -> None:
    script = _script()
    with_vad = script.replay("finished", 12, TurnConfig())
    without = script.replay("finished", 12, TurnConfig(), vad_delay=None)
    assert with_vad["premature_turns"] <= with_vad["premature_decisions"]
    assert without["premature_turns"] > with_vad["premature_turns"]
