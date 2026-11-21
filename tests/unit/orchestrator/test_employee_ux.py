"""LV-16: employee, names, assistance modes and mandatory items."""

from typing import Any, Dict, List, Optional, Tuple

import pytest

from orchestrator.adapters.outbound.fakes import FakeKnowledge, FakeReasoner
from orchestrator.adapters.outbound.fs_mandatory import FileMandatory
from orchestrator.adapters.outbound.jsonl_sink import MemoryHandoff
from orchestrator.adapters.outbound.runtime import InlineExecutor
from orchestrator.application.mandatory import MandatoryTracker, TickRefused
from orchestrator.application.ports import Document, MandatoryUnavailable
from orchestrator.application.sessions import (
    LiveSessions,
    SessionError,
    check_employee,
    check_name,
    live_speakers,
    other_role,
    transcriber_body,
)
from orchestrator.domain.events import LiveEvent
from orchestrator.domain.live_choices import apply_live_choices
from orchestrator.domain.mandatory import Checklist, said_items
from tests.unit.orchestrator.fixtures import REPO, product, transcript
from tests.unit.orchestrator.test_sessions import (
    Clock,
    FakeEvents,
    FakeTranscriber,
    InlineRunner,
    Transcripts,
)

LISTED = {
    "domain": "support.demo", "version": 2, "threshold": 0.575,
    "remind_seconds": 30,
    "items": [
        {"id": "aviso-datos", "text": "Aviso de datos", "phase": "opening",
         "must_say": ["x"]},
        {"id": "desistimiento", "text": "14 días", "must_say": ["y"]},
        {"text": "sin id"},
    ],
}
NOTICE = ("Le recuerdo que tiene un plazo de 14 días para desistir del "
          "contrato sin coste.")


class Settings:
    """Saved «Ajustes» choices."""

    def __init__(self, choices: Optional[Dict[str, str]] = None) -> None:
        self.choices = choices

    def current(self) -> Optional[Dict[str, str]]:
        return self.choices


class FakeMandatory:
    """Says ``desistimiento`` for any sentence with "14 días"."""

    def __init__(self, fail: bool = False, listed: Any = None) -> None:
        self.fail = fail
        self.listed = LISTED if listed is None else listed
        self.checked: List[str] = []

    def items(self, domain: str) -> Document:
        if self.fail:
            raise MandatoryUnavailable("down")
        return dict(self.listed)

    def check(self, domain: str, text: str) -> Document:
        if self.fail:
            raise MandatoryUnavailable("down")
        self.checked.append(text)
        said = "14 días" in text
        return {"items": [
            {"id": "desistimiento", "score": 0.8 if said else 0.1,
             "said": said},
            {"id": "aviso-datos", "score": 0.2, "said": False},
        ]}


def final(uid: str, speaker: str, text: str,
          kind: str = "final") -> LiveEvent:
    """A final (or speaker update) event."""
    return LiveEvent.model_validate({
        "schema": "live.event.v1", "session_id": "s", "seq": 1,
        "type": kind, "at": "2026-11-21T10:00:00Z", "stream_time": 3.0,
        "utterance": {"id": uid, "turn": 0, "index": 0, "speaker": speaker,
                      "start": 0.0, "end": 1.0, "text": text,
                      "revision": 0},
    })


# ---- domain -----------------------------------------------------------------

def test_the_checklist_marks_ticks_and_sums_up() -> None:
    checklist = Checklist.from_list(LISTED)
    assert [i.id for i in checklist.items] == ["aviso-datos",
                                               "desistimiento"]
    assert checklist.version == 2 and checklist.remind_seconds == 30
    assert checklist.mark_said("desistimiento", "t1.s0", 0.8, 3.0)
    assert not checklist.mark_said("desistimiento", "t2.s0", 0.9, 4.0)
    assert not checklist.mark_said("nope", "t2.s0", 0.9, 4.0)
    assert checklist.pending() == ["aviso-datos"]
    assert checklist.tick("aviso-datos", 5.0)
    assert not checklist.tick("aviso-datos", 6.0)
    summary = checklist.summary()
    assert summary["said"] == ["desistimiento"]
    assert summary["ticked"] == ["aviso-datos"]
    assert summary["missing"] == []
    assert summary["ticks"] == [{"item": "aviso-datos", "at": 5.0}]
    empty = Checklist.from_list({})
    assert empty.items == [] and empty.remind_seconds == 45


def test_said_items_reads_only_explicit_hits() -> None:
    assert said_items({"items": [
        {"id": "a", "score": 0.7, "said": True},
        {"id": "b", "score": 0.9, "said": "yes"},
        {"score": 0.9, "said": True}, "junk",
    ]}) == [{"id": "a", "score": 0.7}]
    assert said_items({}) == []


# ---- use case ---------------------------------------------------------------

def tracker(port: FakeMandatory) -> Tuple[MandatoryTracker, List[Document]]:
    published: List[Document] = []
    roles = {"S1": "agent", "S2": "customer"}
    found = MandatoryTracker(port, "support.demo", roles, published.append,
                             InlineExecutor())
    return found, published


def test_only_the_employee_s_sentences_are_checked() -> None:
    port = FakeMandatory()
    track, published = tracker(port)
    track.start()
    assert track.active and published[0]["type"] == "mandatory"
    track.on_event(final("t0.s0", "S2", "Quiero 14 días de prueba"))
    assert port.checked == []
    track.on_event(final("t1.s0", "S1", "Buenos días"))
    track.on_event(final("t1.s0", "S1", "Buenos días"))
    track.on_event(final("t2.s0", "S1", NOTICE))
    assert port.checked == ["Buenos días", NOTICE]
    view = track.view()
    said = next(i for i in view["items"] if i["id"] == "desistimiento")
    assert said["state"] == "said" and said["utterance"] == "t2.s0"
    assert said["score"] == 0.8 and said["at"] == 3.0
    assert published[-1]["checklist"] == view
    summary = track.summary()
    assert summary is not None
    assert summary["missing"] == [{"id": "aviso-datos",
                                   "text": "Aviso de datos"}]


def test_a_sentence_given_to_the_employee_later_is_checked() -> None:
    port = FakeMandatory()
    track, _ = tracker(port)
    track.start()
    track.on_event(final("t0.s0", "S2", NOTICE))
    assert port.checked == []
    track.roles["ana"] = "agent"
    track.on_event(final("t0.s0", "ana", NOTICE, kind="speaker_update"))
    assert port.checked == [NOTICE]
    track.on_event(final("t9.s0", "ana", "", kind="speaker_update"))
    assert port.checked == [NOTICE]


def test_manual_ticks_need_confirmation_and_are_logged() -> None:
    track, published = tracker(FakeMandatory())
    track.start()
    with pytest.raises(TickRefused):
        track.tick("aviso-datos", False, 1.0)
    view = track.tick("aviso-datos", True, 2.0)
    assert view["ticks"] == [{"item": "aviso-datos", "at": 2.0}]
    assert published[-1]["type"] == "mandatory"
    with pytest.raises(TickRefused):
        track.tick("aviso-datos", True, 3.0)


def test_a_missing_service_warns_once_and_keeps_items_pending() -> None:
    down = FakeMandatory(fail=True)
    track, published = tracker(down)
    track.start()
    assert not track.active and track.summary() is None
    assert published == [{"type": "notice",
                          "code": "mandatory_unavailable",
                          "message": "down"}]
    up = FakeMandatory()
    track2, published2 = tracker(up)
    track2.start()
    up.fail = True
    track2.on_event(final("t1.s0", "S1", NOTICE))
    track2.on_event(final("t2.s0", "S1", "Otra frase"))
    assert [m["type"] for m in published2] == ["mandatory", "notice"]
    assert track2.checklist.pending() == ["aviso-datos", "desistimiento"]


def test_a_domain_without_items_publishes_nothing() -> None:
    track, published = tracker(FakeMandatory(listed={"items": []}))
    track.start()
    track.on_event(final("t1.s0", "S1", NOTICE))
    assert published == [] and not track.active


# ---- sessions ---------------------------------------------------------------

def sessions(
    transcriber: Optional[FakeTranscriber] = None,
    runner: Optional[InlineRunner] = None,
    settings: Optional[Settings] = None,
    handoffs: Optional[MemoryHandoff] = None,
    reasoner: Optional[FakeReasoner] = None,
) -> LiveSessions:
    lines = [
        ("a", 0.5, 2.0, "Buenos días, ¿en qué puedo ayudarle hoy?"),
        ("c", 2.5, 6.0, "Quiero contratar la tarifa de luz por teléfono."),
        ("a", 6.5, 9.0, NOTICE),
        ("c", 9.5, 13.0, "Vale, me parece bien todo lo que me cuenta."),
    ]
    sink = handoffs or MemoryHandoff()
    return LiveSessions(
        {"live_assist.support": product(),
         "live_assist.interview": product("live_assist.interview")},
        FakeKnowledge([{"id": "k1", "text": "contratar tarifa luz"}],
                      everything=True),
        reasoner or FakeReasoner(),
        Transcripts({"t": transcript(lines)}), Clock(),
        runner or InlineRunner(), InlineExecutor, transcriber,
        FakeEvents([]) if transcriber else None,
        settings=settings,
        mandatory=FileMandatory(REPO / "knowledge" / "domains"),
        handoff=lambda sid, name: sink,
    )


def feed(live: LiveSessions, session_id: str) -> List[Document]:
    found, _ = live.feed(session_id).read(0, 0.01)
    return [m for _, m in found]


def test_a_replay_with_an_employee_hands_off_the_mandatory_items() -> None:
    handoffs = MemoryHandoff()
    live = sessions(handoffs=handoffs)
    view = live.start_replay("demo-1", "live_assist.support", "t", None,
                             {"id": "ana", "name": "Ana  Pérez"})
    assert view["employee"] == {"id": "ana", "name": "Ana Pérez",
                                "role": "agent", "speaker": "a"}
    assert view["names"] == {"a": "Ana Pérez"}
    assert view["assist_mode"] == "pincelada"
    assert view["page"]["modes"] == ["pincelada", "guiado"]
    found = feed(live, "demo-1")
    kinds = [m["type"] for m in found]
    assert kinds[0] == "session" and kinds[1] == "mandatory"
    assert kinds[-2:] == ["handoff", "status"]
    record = handoffs.records[0]
    assert record["names"] == {"a": "Ana Pérez"}
    assert record["assist_mode"] == "pincelada"
    mandatory = record["mandatory"]
    assert mandatory["domain"] == "support.demo"
    assert mandatory["said"] == ["desistimiento"]
    missing = [m["id"] for m in mandatory["missing"]]
    assert missing == ["aviso-datos", "condiciones-contrato",
                       "aceptacion-expresa"]
    end = live.get("demo-1")
    assert end["handoff"] == record
    assert end["mandatory"]["items"][2]["state"] == "said"


def test_mode_names_and_ticks_during_a_session() -> None:
    runner = InlineRunner(hold=True)
    reasoner = FakeReasoner()
    live = sessions(runner=runner, reasoner=reasoner)
    live.start_replay("demo-2", "live_assist.support", "t", None)
    assert live.set_mode("demo-2", "guiado") == {"mode": "guiado"}
    with pytest.raises(SessionError) as bad:
        live.set_mode("demo-2", "loud")
    assert bad.value.code == "invalid_request"
    assert live.rename("demo-2", "c", " Carla ")["name"] == "Carla"
    assert live.rename("demo-2", "c", "Carla")["transcript"] is None
    for name in ("", "x" * 61, 3):
        with pytest.raises(SessionError):
            live.rename("demo-2", "c", name)
    with pytest.raises(SessionError):
        live.rename("demo-2", "../c", "Carla")
    with pytest.raises(SessionError) as unconfirmed:
        live.tick("demo-2", "aviso-datos", False)
    assert unconfirmed.value.code == "invalid_request"
    ticked = live.tick("demo-2", "aviso-datos", True)
    assert ticked["ticks"][0]["item"] == "aviso-datos"
    runner.jobs.pop()()
    requests = [r for r in reasoner.requests if r["role"] == "suggester"]
    assert requests and all(r["constraints"]["mode"] == "guiado"
                            for r in requests)
    found = feed(live, "demo-2")
    kinds = [m["type"] for m in found]
    assert "mode" in kinds and kinds.count("rename") == 2
    handoff = live.get("demo-2")["handoff"]
    assert handoff["assist_mode"] == "guiado"
    assert handoff["names"] == {"c": "Carla"}
    assert [r["name"] for r in handoff["renames"]] == ["Carla", "Carla"]
    assert handoff["mandatory"]["ticked"] == ["aviso-datos"]
    with pytest.raises(SessionError) as ended:
        live.rename("demo-2", "c", "Otra")
    assert ended.value.code == "wrong_state"


def test_an_interview_has_no_signals_but_modes_and_items() -> None:
    live = sessions()
    live.start_replay("int-1", "live_assist.interview", "t", None,
                      {"id": "eva"})
    view = live.get("int-1")
    assert view["employee"]["role"] == "interviewer"
    assert view["names"] == {"a": "eva"}
    assert view["page"]["signals"] is False
    assert view["handoff"]["mandatory"]["domain"] == "interview.generic"


def test_bad_employees_are_refused() -> None:
    assert check_employee(None) is None
    for bad in ({"id": "../x"}, {"name": "Ana"}, {"id": "a", "name": 3},
                "ana"):
        with pytest.raises(SessionError):
            check_employee(bad)  # type: ignore[arg-type]
    assert check_name("  Ana\tPérez ") == "Ana Pérez"


def test_a_stereo_employee_names_their_channel() -> None:
    transcriber = FakeTranscriber()
    live = sessions(transcriber=transcriber)
    view = live.start_live("live_assist.support", "stereo", "l-1",
                           {"id": "ana", "name": "Ana"})
    body = transcriber.bodies[0]
    assert body["speakers"][0] == {"channel": 0, "id": "S1",
                                   "role": "agent", "name": "Ana"}
    assert "diarize" not in body
    assert view["names"] == {"S1": "Ana"}
    assert view["employee"]["speaker"] == "S1"


def test_a_mic_employee_is_found_by_their_voiceprint() -> None:
    transcriber = FakeTranscriber()
    runner = InlineRunner(hold=True)
    live = sessions(transcriber=transcriber, runner=runner)
    view = live.start_live("live_assist.support", "mic", "l-2",
                           {"id": "ana", "name": "Ana"})
    body = transcriber.bodies[0]
    assert body["diarize"] is True and body["enrolled"] == ["ana"]
    assert {s["role"] for s in body["speakers"]} == {"customer"}
    assert view["page"]["roles"]["ana"] == "agent"
    assert view["names"] == {"ana": "Ana"}
    assert live.rename("l-2", "S2", "Carlos")["transcript"] is True
    assert transcriber.renamed == [("l-2", "S2", "Carlos")]
    transcriber.fail = True
    assert live.rename("l-2", "S2", "Carlos")["transcript"] is False


def test_diarization_off_keeps_a_mic_employee_as_one_voice() -> None:
    transcriber = FakeTranscriber()
    live = sessions(transcriber=transcriber,
                    settings=Settings({"live.diarization": "off",
                                       "live.assist_mode": "guiado"}))
    view = live.start_live("live_assist.support", "mic", "l-3",
                           {"id": "ana", "name": "Ana"})
    body = transcriber.bodies[0]
    assert "diarize" not in body and len(body["speakers"]) == 1
    assert view["employee"] is None


def test_speakers_and_body_helpers() -> None:
    support = product()
    assert other_role(support) == "customer"
    assert other_role(product("live_assist.interview")) == "candidate"
    mono = live_speakers(support, "mic", {"id": "ana", "name": "Ana"}, True)
    assert mono == [{"channel": 0, "id": "S1", "role": "customer"},
                    {"channel": 0, "id": "S2", "role": "customer"}]
    body = transcriber_body(support, "s", mono, diarize=True,
                            enrolled=["ana"])
    assert body["diarize"] is True and body["enrolled"] == ["ana"]
    plain = transcriber_body(support, "s", mono, enrolled=["ana"])
    assert "diarize" not in plain and "enrolled" not in plain


def test_the_saved_mode_reaches_the_reasoner_config() -> None:
    support = product()
    assert support.reasoner is not None and support.reasoner.mode is None
    guided = apply_live_choices(support, {"live.assist_mode": "guiado"})
    assert guided.reasoner is not None and guided.reasoner.mode == "guiado"
    same = apply_live_choices(support, {"live.assist_mode": "loud"})
    assert same.reasoner is not None and same.reasoner.mode is None


def test_an_ended_session_refuses_live_controls() -> None:
    live = sessions()
    live.start_replay("demo-3", "live_assist.support", "t", None)
    with pytest.raises(SessionError) as caught:
        live.set_mode("demo-3", "guiado")
    assert caught.value.code == "wrong_state"
    with pytest.raises(SessionError) as ticked:
        live.tick("demo-3", "aviso-datos", True)
    assert ticked.value.code == "wrong_state"
    with pytest.raises(SessionError):
        live.set_mode("nope", "guiado")
