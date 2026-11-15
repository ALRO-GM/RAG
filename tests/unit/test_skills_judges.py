"""Judge use cases: selection, fallback on quota, two judges compared."""

from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

import pytest

from src.application.skills.judges import (
    ActClassifier,
    FallbackJudge,
    JudgeSelection,
    act_accuracy,
    act_items,
    compare_act_judges,
    compare_confirmations,
    decision_label,
    release,
)
from src.application.skills.tagging import (
    SentenceTag,
    TagConfirmer,
    TagDecision,
    TaggedSentence,
)
from src.domain.errors import InvalidInputError, QuotaExhaustedError
from src.domain.skills.judging import JudgeState
from src.domain.skills.speech import Sentence
from tests.fakes import FakeChatModel, InMemoryCache
from tests.fakes import make_catalogue

NOW = datetime(2026, 11, 15, 12, 0, tzinfo=timezone.utc)


class MemoryStore:
    """A judge state store in memory."""

    def __init__(self, state: Optional[JudgeState] = None) -> None:
        """Start from ``state`` (empty by default)."""
        self.state = state or JudgeState()
        self.saves = 0

    def load(self) -> JudgeState:
        """Return the state."""
        return self.state

    def save(self, state: JudgeState) -> None:
        """Replace the state."""
        self.state = state
        self.saves += 1

    def update(self, change: Callable[[JudgeState], JudgeState]) -> JudgeState:
        """Apply ``change``."""
        self.save(change(self.state))
        return self.state


class OutOfQuota(FakeChatModel):
    """A Claude that refuses for quota after ``answers`` replies."""

    def __init__(self, answers: int = 0, reply: str = "{}") -> None:
        """Answer ``answers`` times, then refuse."""
        super().__init__([reply], "claude-code:opus")
        self.answers = answers

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        """Answer, or refuse once the quota is used."""
        if len(self.calls) >= self.answers:
            self.calls.append((system, user, max_tokens))
            raise QuotaExhaustedError(
                "limit reached", "3pm", NOW + timedelta(hours=2)
            )
        return super().complete(system, user, max_tokens)


class Releasable(FakeChatModel):
    """A local judge that counts its releases."""

    released = 0

    def release(self) -> None:
        """Count a release."""
        self.released += 1


def selection(state: Optional[JudgeState] = None) -> JudgeSelection:
    """Return a selection over a memory store with a fixed clock."""
    return JudgeSelection(MemoryStore(state), lambda: NOW)


def test_the_default_judge_is_claude_code_until_one_is_chosen() -> None:
    chosen = selection()
    status = chosen.status()
    assert (status.mode, status.chosen, status.claude_quota) == (
        "claude_code", False, "unknown",
    )
    assert status.modes == ["claude_code", "local", "both"]
    after = chosen.choose("both")
    assert (after.mode, after.chosen) == ("both", True)
    assert chosen.mode() == "both"
    with pytest.raises(InvalidInputError, match="Unknown judge"):
        chosen.choose("api")


def test_quota_events_are_recorded() -> None:
    chosen = selection()
    chosen.record_exhausted("3pm")
    status = chosen.status()
    assert status.claude_quota == "exhausted"
    assert (status.quota_exhausted_at, status.quota_resets) == (NOW, "3pm")
    assert not chosen.quota_available()
    chosen.record_answer()
    assert chosen.status().claude_quota == "available"


def test_fallback_switches_once_and_for_the_rest_of_the_job() -> None:
    claude, local = OutOfQuota(answers=1, reply="c"), FakeChatModel(["l"])
    chosen = selection()
    judge = FallbackJudge(claude, local, chosen)
    assert judge.model_name == "claude-code:opus"
    assert judge.complete("s", "u1", 9) == "c"
    assert chosen.status().quota_ok_at == NOW
    assert judge.complete("s", "u2", 9) == "l"
    assert judge.model_name == "fake-judge"
    assert judge.fallback_reason == "limit reached"
    assert judge.complete("s", "u3", 9) == "l"
    assert len(claude.calls) == 2 and len(local.calls) == 2
    status = chosen.status()
    assert status.quota_resets == "3pm"
    assert status.quota_resets_at == NOW + timedelta(hours=2)


def test_a_known_exhausted_quota_starts_on_the_local_judge() -> None:
    state = JudgeState(
        quota_exhausted_at=NOW - timedelta(hours=1), quota_resets="3pm"
    )
    claude, local = OutOfQuota(answers=5), FakeChatModel(["l"])
    judge = FallbackJudge(claude, local, selection(state))
    assert "usage limit was reached" in judge.fallback_reason
    assert "resets 3pm" in judge.fallback_reason
    assert judge.complete("s", "u", 9) == "l"
    assert claude.calls == []


def test_fallback_without_a_selection_and_release() -> None:
    local = Releasable(["l"])
    judge = FallbackJudge(OutOfQuota(), local)
    assert judge.complete("s", "u", 9) == "l"
    judge.release()
    assert local.released == 1
    release(FakeChatModel())  # no release method: nothing happens


def tagged_sample() -> List[TaggedSentence]:
    """Two sentences with candidates and one without."""
    rubric = make_catalogue()
    competency = rubric.competencies[0]
    indicator = competency.indicators[0]
    tag = SentenceTag(
        indicator_id=indicator.id, competency_id=competency.id,
        polarity=indicator.polarity, score=1.0,
    )
    sentences = [
        Sentence(id=f"t{n}.s0", turn=n, index=0, speaker="Ana",
                 text=f"frase número {n} con palabras", start=n, end=n + 1)
        for n in range(3)
    ]
    return [
        TaggedSentence(sentence=sentences[0], tags=[tag]),
        TaggedSentence(sentence=sentences[1], tags=[]),
        TaggedSentence(sentence=sentences[2], tags=[tag]),
    ]


def test_two_judges_confirm_the_same_candidates() -> None:
    rubric = make_catalogue()
    named = rubric.competencies[0].indicators[0].id
    first = TagConfirmer(FakeChatModel(
        [f'{{"indicator": "{named}", "reason": "r"}}'], "claude"
    ))
    local = Releasable(['{"indicator": null, "reason": "r"}'], "local")
    second = TagConfirmer(local)
    merged, comparison = compare_confirmations(
        tagged_sample(), rubric, first, second
    )
    assert merged[0].decision is not None
    assert merged[0].decision.model == "claude"
    assert merged[0].second_decision is not None
    assert merged[0].second_decision.status == "rejected"
    assert merged[1].decision is None and merged[1].second_decision is None
    assert comparison.items == 2 and comparison.agreement == 0.0
    assert comparison.confusion == {named: {"none": 2}}
    assert local.released == 1


def test_decision_labels() -> None:
    assert decision_label(None) == "?"
    assert decision_label(TagDecision(status="unparsed")) == "?"
    assert decision_label(TagDecision(status="rejected")) == "none"
    assert decision_label(
        TagDecision(status="confirmed", indicator_id="X.P1")
    ) == "X.P1"


def dialogues() -> List[Dict[str, Any]]:
    """Two tiny dialogues in the ``dialogues.json`` shape."""
    def utterance(n: int, speaker: str, text: str, act: Any) -> Dict[str, Any]:
        """One utterance."""
        return {"id": f"t{n}.s0", "speaker": speaker, "text": text,
                "act": act}

    return [
        {"session": {"id": "d1"}, "utterances": [
            utterance(0, "Ana", "Hola.", "relacion"),
            utterance(1, "Luis", "¿Empezamos?", "gestion"),
            utterance(2, "Ana", "Sí.", "respuesta"),
        ]},
        {"session": {"id": "d2"}, "utterances": [
            utterance(0, "Eva", "Mm.", None),
        ]},
    ]


def test_act_items_carry_context_and_reference() -> None:
    items = act_items(dialogues(), context=1)
    assert [i.id for i in items] == ["d1/t0.s0", "d1/t1.s0", "d1/t2.s0",
                                     "d2/t0.s0"]
    assert items[2].context == [("Luis", "¿Empezamos?")]
    assert items[2].reference == ["respuesta"]
    assert items[3].reference == []
    sample = act_items(dialogues(), sample=2)
    assert [i.id for i in sample] == ["d1/t0.s0", "d1/t2.s0"]


def test_act_classifier_caches_and_compares_two_judges() -> None:
    items = act_items(dialogues())
    cache = InMemoryCache()
    claude = FakeChatModel(['{"act": "relacion"}'], "claude")
    first = ActClassifier(claude, cache)
    verdicts = first.classify(items)
    assert [v.act for v in verdicts] == ["relacion"] * 4
    assert act_accuracy(items, verdicts) == round(1 / 3, 4)
    again = ActClassifier(claude, cache).classify(items)
    assert [v.act for v in again] == ["relacion"] * 4
    assert len(claude.calls) == 4
    local = Releasable(["nada"], "local")
    a, b, comparison = compare_act_judges(
        items, ActClassifier(claude, cache), ActClassifier(local)
    )
    assert [v.act for v in b] == [None] * 4
    assert comparison.confusion == {"relacion": {"?": 4}}
    assert (comparison.accuracy_a, comparison.accuracy_b) == (
        round(1 / 3, 4), 0.0,
    )
    assert local.released == 1
    assert act_accuracy([], []) is None


def test_a_known_reset_time_replaces_the_five_hours() -> None:
    state = JudgeState(
        quota_exhausted_at=NOW, quota_resets_at=NOW + timedelta(hours=1)
    )
    later = JudgeSelection(MemoryStore(state),
                           lambda: NOW + timedelta(hours=1, minutes=1))
    assert later.status().claude_quota == "available"
    longer = state.model_copy(
        update={"quota_resets_at": NOW + timedelta(hours=8)}
    )
    after_six = JudgeSelection(MemoryStore(longer),
                               lambda: NOW + timedelta(hours=6))
    assert after_six.status().claude_quota == "exhausted"


def test_both_mode_leaves_out_items_the_local_judge_answered_twice() -> None:
    items = act_items(dialogues())
    claude = OutOfQuota(answers=2, reply='{"act": "relacion"}')
    local = FakeChatModel(['{"act": "gestion"}'], "local")
    first = ActClassifier(FallbackJudge(claude, local))
    a, b, comparison = compare_act_judges(
        items, first, ActClassifier(local)
    )
    assert [v.model for v in a] == ["claude-code:opus"] * 2 + ["local"] * 2
    assert comparison.same_judge == 2 and comparison.items == 2
    assert comparison.agreement == 0.0, "only Claude vs local is compared"
    rubric = make_catalogue()
    named = rubric.competencies[0].indicators[0].id
    reply = f'{{"indicator": "{named}", "reason": "r"}}'
    claude_tags = OutOfQuota(answers=1, reply=reply)
    local_tags = FakeChatModel(['{"indicator": null, "reason": "r"}'],
                               "local")
    merged, compared = compare_confirmations(
        tagged_sample(), rubric,
        TagConfirmer(FallbackJudge(claude_tags, local_tags)),
        TagConfirmer(local_tags),
    )
    assert compared.same_judge == 1 and compared.items == 1
