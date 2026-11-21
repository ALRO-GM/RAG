"""Live events as the transcriber sends them: revisions, speakers, warnings."""

import json
from typing import Any, Dict, List

from orchestrator.adapters.outbound.fakes import FakeKnowledge, FakeReasoner
from orchestrator.adapters.outbound.jsonl_sink import MemorySink
from orchestrator.adapters.outbound.runtime import InlineExecutor
from orchestrator.application.live_assist import LiveAssist
from orchestrator.domain.events import LiveEvent
from tests.unit.orchestrator.fixtures import REPO, product


class Clock:
    def now(self) -> float:
        return 0.0

    def sleep(self, seconds: float) -> None:
        """Never waits."""


def event(seq: int, kind: str, **extra: Any) -> LiveEvent:
    data: Dict[str, Any] = {
        "schema": "live.event.v1", "session_id": "s", "seq": seq,
        "type": kind, "at": "2026-11-21T10:00:00Z", "stream_time": seq,
    }
    data.update(extra)
    return LiveEvent.model_validate(data)


def utterance(uid: str, speaker: str, text: str, revision: int = 0
              ) -> Dict[str, Any]:
    return {"id": uid, "turn": 0, "index": 0, "speaker": speaker,
            "start": 0.0, "end": 1.0, "text": text, "revision": revision,
            "confidence": 0.9, "speaker_certainty": None,
            "speaker_provisional": False}


def test_the_contract_examples_are_read() -> None:
    example = json.loads(
        (REPO / "contracts/examples/live.event.v1.json").read_text("utf-8"))
    assert LiveEvent.model_validate(example).utterance is not None
    warned = event(1, "warning",
                   warning={"code": "gpu_degraded", "message": "cpu"})
    assert warned.to_contract()["warning"]["code"] == "gpu_degraded"


def test_revisions_and_speaker_updates_change_the_window() -> None:
    reasoner = FakeReasoner()
    assist = LiveAssist(
        product(), FakeKnowledge(), reasoner, MemorySink(), Clock(),
        InlineExecutor(), {"S1": "agent", "S2": "customer"},
    )
    assist.on_event(event(1, "final", utterance=utterance(
        "t0.s0", "S1", "Me cobraron dos veces la factura")))
    assist.on_event(event(2, "speaker_update", utterance=utterance(
        "t0.s0", "S2", "Me cobraron dos veces la factura")))
    assist.on_event(event(5, "final", utterance=utterance(
        "t0.s0", "S2", "Me cobraron dos veces la factura de octubre", 1)))
    window: List[Dict[str, Any]] = reasoner.requests[-1]["context"]["window"]
    assert len(window) == 1
    assert window[0]["speaker"] == "S2"
    assert window[0]["text"].endswith("octubre")
    assert assist.calls == 1
