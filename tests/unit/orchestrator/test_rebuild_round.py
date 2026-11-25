"""The rebuild-round service (#264): an ended live round replayed from
what was kept, through the topics, the mandatory items and the lines to
avoid, then judged and stored as at the end of the round. Fakes only;
invented words."""

from typing import Any, Dict, List

from orchestrator.adapters.outbound.runtime import InlineExecutor
from orchestrator.application.ports import Document
from orchestrator.application.rebuild_round import (
    RoundRebuild,
    events_of_log,
    events_of_transcript,
)
from orchestrator.domain.events import Transcript
from tests.unit.orchestrator.fixtures import product
from tests.unit.orchestrator.test_avoid_demo import FakeAvoid
from tests.unit.orchestrator.test_employee_ux import FakeMandatory
from tests.unit.orchestrator.test_interview_kept import (
    SECRET,
    Judge,
    plan,
    transcript,
    wiring,
)
from tests.unit.orchestrator.test_interview_usecases import Store


def kept(extra: List[Any] = ()) -> Transcript:  # type: ignore[assignment]
    data = transcript("r9")
    for who, text in extra:
        n = len(data["utterances"])
        data["utterances"].append(
            {"id": f"t{n}.s0", "turn": n, "index": 0, "speaker": who,
             "start": float(n), "end": n + 0.9, "text": text})
    return Transcript.model_validate(data)


def test_a_round_is_rebuilt_judged_and_stored_from_its_transcript() -> None:
    store = Store()
    store.save(plan())
    events, roles = events_of_transcript(kept())
    published: List[Document] = []
    rebuilt = RoundRebuild(InlineExecutor(), wiring=wiring(store, Judge())
                           ).rebuild(
        "r9", events, roles, product=product("live_assist.interview"),
        plan=plan(), candidate="Ana Ruiz", candidate_id="ana-ruiz",
        publish=published.append)
    assert rebuilt.parts == ["topics", "judgement"]
    assert rebuilt.finals == 4
    assert (rebuilt.covered or []) + (rebuilt.revisit or []), \
        "the plan's slots were followed"
    assert rebuilt.accuracy is not None and rebuilt.accuracy["saved"]
    report = store.result("dev", "r9")
    assert report["candidate_id"] == "ana-ruiz" and report["kind"] == "live"
    summary = rebuilt.summary()
    assert summary["accuracy"] == "ready" and summary["saved"] is True
    assert SECRET not in str(summary) and "Ana" not in str(summary)


def test_the_checklist_and_the_lines_to_avoid_are_followed_again() -> None:
    mandatory, avoid = FakeMandatory(), FakeAvoid()
    events, roles = events_of_transcript(kept([
        ("S1", "¿Qué edad tienes?"),
        ("S1", "Tiene un plazo de 14 días para pensarlo."),
    ]))
    rebuilt = RoundRebuild(InlineExecutor(), mandatory=mandatory,
                           avoid=avoid).rebuild(
        "r9", events, roles, product=product("live_assist.interview"))
    assert rebuilt.parts == ["mandatory", "avoid"]
    assert rebuilt.covered is None and rebuilt.accuracy is None
    assert [f["category"] for f in rebuilt.avoid or []] == ["age"]
    assert "desistimiento" in (rebuilt.mandatory or {}).get("said", [])
    assert "¿Qué edad tienes?" in avoid.asked, "the interviewer's lines"
    assert not any(SECRET in t for t in avoid.asked), "not the candidate's"


def test_without_ports_or_plan_only_the_counts_come_back() -> None:
    events, roles = events_of_transcript(kept())
    rebuilt = RoundRebuild(InlineExecutor()).rebuild("r9", events, roles)
    assert rebuilt.parts == [] and rebuilt.events == len(events)
    assert rebuilt.summary()["accuracy"] is None


def test_a_round_is_rebuilt_from_a_live_log_too() -> None:
    lines: List[Dict[str, Any]] = [
        {"schema": "live.event.v1", "session_id": "r9", "seq": 0,
         "type": "session_started", "at": "2026-11-25T10:00:00Z",
         "stream_time": 0.0,
         "source": {"kind": "single_speaker", "sample_rate": 16000,
                    "channels": 1},
         "speakers": [{"id": "S1", "role": "interviewer"},
                      {"id": "S2", "role": "candidate"}]},
        {"seq": 1, "cut by a crash": True},
    ]
    for n, (who, text) in enumerate([
            ("S1", "¿Qué sabes de bases de datos tema1, por ejemplo?"),
            ("S2", "Índices y planes de consulta, con ejemplos.")], 2):
        lines.append({
            "schema": "live.event.v1", "session_id": "r9", "seq": n,
            "type": "final", "at": "2026-11-25T10:00:05Z",
            "stream_time": float(n),
            "utterance": {"id": f"t{n}.s0", "turn": n, "index": 0,
                          "speaker": who, "start": float(n),
                          "end": n + 0.9, "text": text, "revision": 0}})
    events, roles = events_of_log(lines)
    assert roles == {"S1": "interviewer", "S2": "candidate"}
    assert [e.type for e in events] == ["session_started", "final", "final"]
    store = Store()
    store.save(plan())
    rebuilt = RoundRebuild(InlineExecutor(), wiring=wiring(store, Judge())
                           ).rebuild("r9", events, roles, plan=plan(),
                                     candidate_id="ana-ruiz")
    assert rebuilt.finals == 2 and "topics" in rebuilt.parts


def test_the_rebuild_command_prints_numbers_and_keeps_the_hand_off(
    tmp_path: Any, capsys: Any,
) -> None:
    import json

    from orchestrator.adapters.inbound.cli import main
    from orchestrator.adapters.outbound.fs_candidates import (
        JsonCandidateStore,
    )
    from orchestrator.adapters.outbound.fs_interview_plans import (
        JsonPlanStore,
    )
    from orchestrator.domain.candidate import Round, new_profile
    from tests.unit.orchestrator.test_interview_kept import NOW

    JsonPlanStore(tmp_path / "interview").save(plan())
    ana = new_profile("dev", "Ana Ruiz", [], NOW).model_copy(update={
        "rounds": [Round(session_id="r9", at="2026-11-25T10:00:00",
                         kind="live")]})
    JsonCandidateStore(tmp_path / "interview").save(ana)
    path = tmp_path / "r9.transcript.v1.json"
    path.write_text(json.dumps(transcript()), "utf-8")
    common = ["--interview-dir", str(tmp_path / "interview"),
              "--handoff-dir", str(tmp_path / "handoff"),
              "--reasoner", "fake", "--knowledge", "fake",
              "--rag-token-file", str(tmp_path / "none")]
    code = main(["rebuild-round", "r9", "--transcript", str(path),
                 "--plan", "dev", "--candidate", ana.id, *common])
    out = capsys.readouterr().out
    assert code == 0
    printed = json.loads(out)
    assert printed["parts"] == ["topics", "judgement"]
    assert printed["saved"] is True and printed["finals"] == 4
    assert SECRET not in out and "Ana" not in out
    handoff = json.loads((tmp_path / "handoff"
                          / "r9.live_assist.interview.handoff.json")
                         .read_text())
    assert handoff["rebuilt"] is True and SECRET not in json.dumps(handoff)
    log = tmp_path / "events.jsonl"
    log.write_text("not json\n", "utf-8")
    assert main(["rebuild-round", "r9", "--events", str(log), *common]) == 2
    other = main(["rebuild-round", "zz", "--transcript", str(path), *common])
    assert other == 2
    nope = main(["rebuild-round", "r9", "--transcript", str(path),
                 "--plan", "nope", *common])
    assert nope == 2
