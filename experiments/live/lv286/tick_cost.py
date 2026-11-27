"""Orchestrator-side cost of the live tick loop (#286), without a model.

Replays an invented round (the plan of
``experiments/interview/attribution_set.json``, invented lines) through
a :class:`TopicTracker` and a :class:`LiveTicker` whose reasoner answers
at once, and prints the time of ``TopicTracker.on_event`` per final and
of one tick call's handling around the model (request and reply). The
model's own latency is not part of it. Run it in two checkouts to
compare (``ROOT`` is the checkout whose code is on ``PYTHONPATH``)::

    CUDA_VISIBLE_DEVICES= PYTHONPATH=<checkout> \\
        python experiments/live/lv286/tick_cost.py <checkout>
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

from orchestrator.application.live_ticks import LiveTicker
from orchestrator.application.next_round import TopicTracker
from orchestrator.domain.events import LiveEvent
from orchestrator.domain.interview import InterviewPlan
from orchestrator.domain.next_round import topic_states

LINES = [
    ("S1", "Hola, gracias por venir. ¿Dónde trabajaste antes y cuánto "
           "tiempo?"),
    ("S2", "Estuve tres años en una zapatería del barrio, atendiendo "
           "clientes, reponiendo el género y preparando los pedidos."),
    ("S1", "Cuéntame qué tipos de pisada conoces."),
    ("S2", "Pues la verdad es que eso no lo tengo muy claro."),
    ("S1", "¿Y con la caja, cobras con tarjeta y efectivo?"),
    ("S2", "Sí, cobraba con tarjeta y en efectivo, y al cerrar cuadraba "
           "la caja."),
    ("S1", "¿Puedes trabajar fines de semana y con turnos rotativos?"),
    ("S2", "Los sábados sin problema y los turnos también me vienen bien."),
]
ROLES = {"S1": "interviewer", "S2": "candidate"}
REPLAYS = 200


def final(n: int, who: str, text: str) -> LiveEvent:
    """One final of the invented round."""
    return LiveEvent.model_validate({
        "schema": "live.event.v1", "session_id": "bench", "seq": n,
        "type": "final", "at": "2026-11-27T10:00:00Z",
        "stream_time": float(n * 5),
        "utterance": {"id": f"u{n}", "turn": n, "index": 0, "speaker": who,
                      "start": n * 5.0, "end": n * 5.0 + 4, "text": text,
                      "revision": 0}})


class Instant:
    """A reasoner that answers at once: two concepts treated, one asked."""

    def reason(self, request: Dict[str, Any], cancel: Any) -> Dict[str, Any]:
        """Tick the first two listed concepts on the last answer."""
        ids = [p["id"] for p in request["payload"].get("points", [])][:2]
        answer = request["payload"]["answers"][-1]
        asked = [x["id"] for x in request["context"]["window"]
                 if ROLES.get(x["speaker"]) == "interviewer"][-1:]
        return {"status": "ok", "data": {
            "points": [{"id": i, "utterance": answer} for i in ids],
            "asked": [{"id": i, "utterance": asked[0]} for i in ids[:1]]
            if asked else []}}


def quantile(values: List[float], q: float) -> float:
    """The ``q`` quantile in microseconds (nearest rank below)."""
    ordered = sorted(values)
    return round(ordered[int(q * (len(ordered) - 1))] * 1e6, 1)


def main() -> None:
    """Replay the round and print the two timings as JSON."""
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    plan = InterviewPlan.model_validate(json.loads(
        (root / "experiments/interview/attribution_set.json"
         ).read_text("utf-8"))["plan"])
    on_event: List[float] = []
    asks: List[float] = []
    for _ in range(REPLAYS):
        topics = TopicTracker(plan, topic_states(plan, []), dict(ROLES),
                              lambda message: None)
        jobs: List[Callable[[], None]] = []
        ticker = LiveTicker(Instant(), topics, dict(ROLES),
                            jobs.append, every_s=0.0,
                            sleep=lambda seconds: None)
        topics.start()
        for n, (who, text) in enumerate(LINES * 3):
            event = final(n, who, text)
            start = time.perf_counter()
            topics.on_event(event)
            on_event.append(time.perf_counter() - start)
            ticker.on_event(event)
            while jobs:
                job = jobs.pop(0)
                start = time.perf_counter()
                job()
                asks.append(time.perf_counter() - start)
    print(json.dumps({
        "finals": len(on_event), "tick_calls": len(asks),
        "tracker_on_event_us": {"p50": quantile(on_event, 0.5),
                                "p90": quantile(on_event, 0.9)},
        "tick_call_handling_us": {"p50": quantile(asks, 0.5),
                                  "p90": quantile(asks, 0.9)},
    }))


if __name__ == "__main__":
    main()
