"""A round whose lost audio was recovered is rebuilt (#265).

The transcriber merged the recovered sentences into the round's
transcript and asks for every indicator again: the candidate and the
plan come from the candidate's rounds, the judgement replaces the one
stored, a session still in memory sees it; deleting a candidate's file
asks the transcriber to delete any recovered audio. Fakes only;
invented words.
"""

import dataclasses
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from orchestrator.adapters.outbound.runtime import InlineExecutor
from orchestrator.application.ports import Document, TranscriberUnavailable
from orchestrator.application.recovered_round import (
    RecoveredRound,
    forget_recovery,
)
from tests.unit.orchestrator.fixtures import product
from tests.unit.orchestrator.test_interview_kept import (
    Judge,
    plan,
    transcript,
    wiring,
)
from tests.unit.orchestrator.test_interview_usecases import Store


INLINE = InlineExecutor()


class Book:
    """The candidates: one file whose rounds include ``r9``."""

    def __init__(self, found: bool = True) -> None:
        self.found = found
        self.store = self

    def by_session(self, session_id: str) -> Document:
        if not self.found:
            raise KeyError(session_id)
        return {"plan_id": "dev", "id": "ana-ruiz"}

    def load(self, plan_id: str, candidate_id: str) -> Any:
        return SimpleNamespace(id=candidate_id, name="Ana Ruiz", rounds=[])

    def owns(self, *args: Any) -> bool:
        return True

    def keep_report(self, plan_id: str, candidate_id: str,
                    save: Any) -> bool:
        save()
        return True


def sessions(store: Store, book: Optional[Book], live: Any = None) -> Any:
    wired = wiring(store, Judge())
    if book is not None:
        wired = dataclasses.replace(wired, candidates=book)  # type: ignore
    written: List[Any] = []
    return SimpleNamespace(
        interview=wired, products={
            "live_assist.interview": product("live_assist.interview")},
        mandatory=None, avoid=None,
        find=lambda sid: live, write_handoff=written.append,
        written=written)


def test_the_recovered_round_is_judged_again_for_its_candidate() -> None:
    store = Store()
    store.save(plan())
    world = sessions(store, Book())
    summary = RecoveredRound(world, INLINE).rebuild("r9", transcript("r9"))
    assert summary["accuracy"] == "ready" and summary["saved"] is True
    assert store.result("dev", "r9")["candidate_id"] == "ana-ruiz"
    with pytest.raises(ValueError):
        RecoveredRound(world, INLINE).rebuild("other", transcript("r9"))


def test_a_session_still_in_memory_sees_the_new_judgement() -> None:
    store = Store()
    store.save(plan())
    feed: List[Document] = []
    live = SimpleNamespace(
        product=product("live_assist.interview"),
        feed=SimpleNamespace(publish=feed.append), accuracy=None,
        handoff={"session_id": "r9"})
    world = sessions(store, Book(), live)
    RecoveredRound(world, INLINE).rebuild("r9", transcript("r9"))
    assert live.accuracy is not None and live.accuracy["state"] == "ready"
    assert live.handoff["recovered"] is True
    assert world.written == [live] and feed


def test_without_a_candidate_only_the_trackers_run() -> None:
    world = sessions(Store(), Book(found=False))
    summary = RecoveredRound(world, INLINE).rebuild("r9", transcript("r9"))
    assert summary["accuracy"] is None and summary["parts"] == []
    assert RecoveredRound(sessions(Store(), None), INLINE).rebuild(
        "r9", transcript("r9"))["parts"] == []


def test_deleting_a_file_asks_the_transcriber_to_delete_the_audio() -> None:
    asked: List[str] = []

    class Transcriber:
        def discard_recovery(self, session_id: str) -> None:
            if session_id == "down":
                raise TranscriberUnavailable("no answer")
            asked.append(session_id)

    assert forget_recovery(Transcriber(), ["r1", "down", "r2"]) == 2
    assert asked == ["r1", "r2"]
    assert forget_recovery(None, ["r1"]) == 0
    assert forget_recovery(object(), ["r1"]) == 0


def test_the_session_view_says_whether_the_page_keeps_audio() -> None:
    from orchestrator.application.sessions import page_info

    info: Dict[str, Any] = page_info(product("live_assist.interview"), {})
    assert set(info["buffer"]) == {"on", "max_hours"}
