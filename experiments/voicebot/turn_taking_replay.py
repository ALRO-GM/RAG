"""Replay scripted answers against the interviewer's turn-taking (#213).

A simulation, not a recording: no audio, no model. Each scripted
answer is a few spoken fragments separated by pauses drawn uniformly
from 2 to 5 seconds (seeded). The events reach the real
:class:`~orchestrator.application.voicebot.VoiceBot` (with the
:class:`~orchestrator.domain.turn_taking.TurnTaking` policy, a fake
director, the fake TTS and the fake reasoner) as the live pipeline
would send them, on a simulated clock ticked every 0.1 s:

- the page's voice detector reports activity ``--vad-delay`` seconds
  after each fragment starts (its ``startMs`` is 150 ms; the network
  is assumed at 50 ms: an input, not a measurement); ``--no-vad``
  leaves it out;
- a partial every 0.8 s of speech (``partial_every_ms`` of the
  product), 0.3 s after its audio (assumed);
- the final ``--lag`` seconds after the fragment ends (default 0.674:
  the measured p50 of end of speech → final in
  ``docs/live/LIVE_ASR.md``).

- a turn decided at ``t`` speaks at ``t + --prep`` (default 0.912 s:
  the measured p50 of turn start → first audio published of the
  support bot in ``docs/live/VOICEBOT.md``; the interviewer's slot
  filler adds a call that was not measured, so the real wait is
  longer). Speech of the candidate in between drops the turn.

A turn the bot decides before the answer's last final arrived is a
**premature decision**; if it is not dropped and speaks before that
final, it is a **premature turn** (the bot cuts the candidate off).
Two kinds of pause are replayed: after a fragment that sounds unfinished («…y»,
«…porque», «…o sea», a comma) and after a complete sentence; every
answer ends with a complete sentence. For each the script reports the
premature turns and the delay from the last final to the decision::

    python experiments/voicebot/turn_taking_replay.py --out R.json
"""

import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from orchestrator.adapters.outbound.fakes import (
    FakeKnowledge,
    FakeReasoner,
    FakeTts,
)
from orchestrator.adapters.outbound.files import load_product
from orchestrator.application.ports import CancelToken, Document
from orchestrator.application.voicebot import VoiceBot
from orchestrator.domain.events import LiveEvent
from orchestrator.domain.records import TurnRecord
from orchestrator.domain.turn_taking import BotTurn, TurnConfig

ROOT = Path(__file__).resolve().parents[2]
TICK_S = 0.1
WORD_S = 0.4
"""Seconds per spoken word (about 150 words a minute)."""
UNFINISHED = [
    "Pues en mi último trabajo estaba en caja y",
    "llevaba también la reposición de la sala porque",
    "el encargado me lo pidió, o sea,",
    "al principio me costó un poco pero",
    "luego organizaba los turnos de los fines de semana y",
]
FINISHED = [
    "Estuve dos años en caja en un supermercado del barrio.",
    "También llevaba la reposición de la sala.",
    "El encargado me pidió que formara a los nuevos.",
    "Al principio me costó un poco.",
    "Luego organizaba los turnos de los fines de semana.",
]
CLOSING = "Y así fue como acabé llevando yo sola la tienda los domingos."


class Clock:
    """Simulated monotonic time."""

    def __init__(self) -> None:
        """Start at zero."""
        self.t = 0.0

    def now(self) -> float:
        """Return the simulated time."""
        return self.t

    def sleep(self, seconds: float) -> None:
        """Move the time on."""
        self.t += seconds


@dataclass
class _Plan:
    """A directed turn."""

    payload: Document
    query: str


@dataclass
class Director:
    """Counts the turns asked of it; asks a fixed question."""

    asked: List[Tuple[float, BotTurn]] = field(default_factory=list)
    clock: Optional[Clock] = None
    closed: bool = False
    target: Optional[str] = "s1"

    def heard(self, utterance: str, text: str, start: float,
              end: float) -> None:
        """Ignore the candidate's finals."""

    def plan_turn(self, answer: str, utterance: Optional[str],
                  cancel: CancelToken, kind: BotTurn = "question"
                  ) -> _Plan:
        """Record when a turn was decided."""
        assert self.clock is not None
        self.asked.append((self.clock.t, kind))
        return _Plan({"action": kind}, answer or "pregunta")

    def spoken(self, record: TurnRecord) -> None:
        """Ignore what the bot said."""

    def finish(self) -> Optional[Document]:
        """Nothing to hand over."""
        return None


class DelayedExecutor:
    """Runs each turn ``prep`` simulated seconds after it was started."""

    def __init__(self, clock: Clock, prep: float) -> None:
        """No turn yet."""
        self.clock = clock
        self.prep = prep
        self.jobs: List[Tuple[float, Callable[[], None]]] = []

    def submit(self, job: Callable[[], None]) -> None:
        """Keep ``job`` until it is due."""
        self.jobs.append((self.clock.t + self.prep, job))

    def run_due(self) -> None:
        """Run the jobs whose time has come."""
        while self.jobs and self.jobs[0][0] <= self.clock.t + 1e-9:
            _, job = self.jobs.pop(0)
            job()

    def wait_idle(self) -> None:
        """Run every job left."""
        while self.jobs:
            _, job = self.jobs.pop(0)
            job()


class _Question(FakeReasoner):
    """Always the same short question (no streaming)."""

    reason_stream = None  # type: ignore[assignment]

    def reason(self, request: Document, cancel: CancelToken) -> Document:
        """Say the next question."""
        return self._answer(request, "Cuéntame algo más de eso.")


def _event(seq: int, kind: str, uid: str, text: str, t: float
           ) -> LiveEvent:
    """One ``live.event.v1`` of the candidate."""
    return LiveEvent.model_validate({
        "schema": "live.event.v1", "session_id": "replay", "seq": seq,
        "type": kind, "at": "2026-11-21T10:00:00+01:00", "stream_time": t,
        "utterance": {"id": uid, "turn": 0, "index": 0, "speaker": "S1",
                      "start": t, "end": t, "text": text, "revision": 0},
    })


def _answer_events(
    fragments: List[str], pauses: List[float], start: float, lag: float,
    vad_delay: Optional[float], tag: str,
) -> Tuple[List[Tuple[float, str, str, str]], float]:
    """Return ``(time, kind, id, text)`` events and the last final's time."""
    events: List[Tuple[float, str, str, str]] = []
    t = start
    last_final = start
    for i, text in enumerate(fragments):
        words = text.split()
        length = len(words) * WORD_S
        uid = f"{tag}.{i}"
        if vad_delay is not None:
            events.append((t + vad_delay, "vad", uid, ""))
        k = 1
        while k * 0.8 < length:
            said = " ".join(words[:max(1, int(k * 0.8 / WORD_S))])
            events.append((t + k * 0.8 + 0.3, "partial", uid, said))
            k += 1
        last_final = t + length + lag
        events.append((last_final, "final", uid, text))
        t += length + (pauses[i] if i < len(pauses) else 0.0)
    return sorted(events), last_final


def replay(
    kind: str, answers: int, config: TurnConfig, seed: int = 213,
    lag: float = 0.674, vad_delay: Optional[float] = 0.2,
    first_words_s: float = 2.0, prep: float = 0.912,
) -> Dict[str, Any]:
    """Replay ``answers`` scripted answers; count premature turns.

    Args:
        kind: ``unfinished`` (pauses after cut fragments) or
            ``finished`` (pauses after complete sentences).
        answers: How many answers.
        config: The turn-taking thresholds.
        seed: The pauses' random seed.
        lag: End of speech → final, seconds.
        vad_delay: Start of speech → the page's activity report
            (``None``: no voice detector).
        first_words_s: Silence before the candidate starts answering.
        prep: Turn start → first audio, seconds.
    """
    rng = random.Random(seed)
    clock = Clock()
    director = Director(clock=clock)
    executor = DelayedExecutor(clock, prep)
    sink = _Sink(clock)
    product = load_product(ROOT / "config" / "products" /
                           "voicebot.interview.yaml")
    bot = VoiceBot(
        product, FakeKnowledge([], everything=True), _Question(),
        FakeTts(), sink, clock, executor, {"S1": "candidate"},
        director, config,
    )
    bot.open("replay")
    executor.wait_idle()

    def step() -> None:
        clock.t = round(clock.t + TICK_S, 6)
        executor.run_due()
        bot.tick()

    source = UNFINISHED if kind == "unfinished" else FINISHED
    premature = 0
    spoken_early = 0
    cut_answers = 0
    delays: List[float] = []
    pauses_used: List[float] = []
    seq = 0
    for n in range(answers):
        policy = bot.turn_taking
        assert policy is not None and policy.asked_at is not None
        count = rng.randint(2, 4)
        fragments = [source[(n + j) % len(source)] for j in range(count)]
        fragments.append(CLOSING)
        pauses = [round(rng.uniform(2.0, 5.0), 2) for _ in fragments[:-1]]
        pauses_used += pauses
        start = max(clock.t, policy.asked_at) + first_words_s
        events, last_final = _answer_events(
            fragments, pauses, start, lag, vad_delay, f"a{n}")
        before = len(sink.decided)
        said_before = len(sink.spoken)
        for at, what, uid, text in events:
            while clock.t + TICK_S <= at:
                step()
            clock.t = at
            executor.run_due()
            if what == "vad":
                bot.barge_in("vad")
            else:
                seq += 1
                bot.on_event(_event(seq, what, uid, text, at))
        early = [t for t in sink.decided[before:] if t < last_final]
        premature += len(early)
        cut = [t for t in sink.spoken[said_before:] if t < last_final]
        spoken_early += len(cut)
        cut_answers += 1 if cut else 0
        waited = 0.0
        while (len(sink.decided) - before == len(early)
               or executor.jobs) and waited < 60:
            step()
            waited += TICK_S
        delays.append(round(sink.decided[-1] - last_final, 2))
    delays.sort()
    return {
        "kind": kind, "answers": answers, "pauses": len(pauses_used),
        "pause_s": [min(pauses_used), max(pauses_used)],
        "pauses_over_4s": sum(1 for p in pauses_used if p > 4.0),
        "premature_decisions": premature, "premature_turns": spoken_early,
        "answers_cut": cut_answers,
        "end_delay_s": {"min": delays[0],
                        "p50": delays[len(delays) // 2],
                        "max": delays[-1]},
        "config": {"turn_silence_s": config.turn_silence_s,
                   "think_s": config.think_s,
                   "max_wait_s": config.max_wait_s,
                   "stuck_s": config.stuck_s,
                   "backchannel": config.backchannel},
        "lag_s": lag, "vad_delay_s": vad_delay, "prep_s": prep,
        "seed": seed,
    }


class _Sink:
    """Keeps when each turn spoke; drops the rest."""

    def __init__(self, clock: Clock) -> None:
        """Nothing spoken yet."""
        self.clock = clock
        self.spoken: List[float] = []
        self.decided: List[float] = []

    def publish(self, message: Document) -> None:
        """Note when a turn was decided (``bot``); drop the rest."""
        if message.get("type") == "bot":
            self.decided.append(self.clock.t)

    def record(self, record: TurnRecord) -> None:
        """Note the time of a turn that spoke."""
        if record.spoken:
            self.spoken.append(self.clock.t)


def main() -> None:
    """Run the replays and write the results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", type=int, default=40)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    runs: List[Dict[str, Any]] = []
    for silence in (3.5, 5.0, 6.0):
        config = TurnConfig(turn_silence_s=silence)
        for kind in ("unfinished", "finished"):
            runs.append(replay(kind, args.answers, config))
            runs.append(replay(kind, args.answers, config, vad_delay=None))
    for run in runs:
        print(f"{run['kind']:10} silence {run['config']['turn_silence_s']}"
              f" vad {run['vad_delay_s']}: premature decisions "
              f"{run['premature_decisions']}, turns "
              f"{run['premature_turns']} in {run['answers_cut']}/"
              f"{run['answers']} answers ({run['pauses']} pauses "
              f"{run['pause_s'][0]}–{run['pause_s'][1]} s), end delay "
              f"p50 {run['end_delay_s']['p50']} s")
    if args.out:
        args.out.write_text(json.dumps({"runs": runs}, indent=2) + "\n",
                            encoding="utf-8")


if __name__ == "__main__":
    main()
