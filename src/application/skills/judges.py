"""Use cases: choose the judge, fall back when Claude's quota runs out,
and run two judges side by side.

The judge is chosen from the web (``GET/PUT /analysis/judge``) and kept
by a :class:`~src.application.skills.ports.JudgeStateStore`, together
with what is known of Claude's quota. A job builds its judge from that
choice (composition root): Claude Code with the local model behind it
(:class:`FallbackJudge`), the local model alone, or both
(:func:`compare_confirmations`, :func:`compare_act_judges`).
"""

from datetime import datetime
from typing import (
    Any, Callable, Dict, List, Optional, Sequence, Tuple, get_args,
)

from pydantic import BaseModel

from src.application.caching import cache_key
from src.application.ports import Cache, NoProgress, Progress
from src.application.skills.ports import ChatModel, JudgeStateStore
from src.application.skills.tagging import (
    TagConfirmer,
    TagDecision,
    TaggedSentence,
)
from src.domain.errors import (
    InvalidInputError,
    LimitError,
    QuotaExhaustedError,
)
from src.domain.skills.act_prompting import (
    act_system_prompt,
    act_user_prompt,
    parse_act,
)
from src.domain.skills.judging import (
    DEFAULT_JUDGE_MODE,
    JUDGE_MODES,
    JudgeMode,
    LabelComparison,
    QuotaState,
    compare_labels,
    quota_status,
)
from src.domain.skills.rubric import RubricCatalogue
from src.domain.skills.usage import Usage, UsageLedger


class JudgeStatus(BaseModel):
    """What ``GET /analysis/judge`` answers.

    ``mode`` is the judge jobs use now; ``chosen`` tells whether it
    was picked from the web or is the default. ``claude_quota`` is
    ``available``, ``exhausted`` or ``unknown`` (Claude never called).
    """

    mode: JudgeMode
    chosen: bool
    modes: List[str]
    claude_quota: QuotaState
    quota_exhausted_at: Optional[datetime] = None
    quota_resets: str = ""
    quota_resets_at: Optional[datetime] = None
    quota_ok_at: Optional[datetime] = None


class JudgeSelection:
    """The judge chosen from the web and the quota seen by the jobs.

    Args:
        store: Where the choice and the quota record are kept.
        now: The clock (replaced in tests).
        default: The judge used while nothing was chosen.
    """

    def __init__(
        self,
        store: JudgeStateStore,
        now: Callable[[], datetime],
        default: JudgeMode = DEFAULT_JUDGE_MODE,
    ):
        """Wire the store and the clock."""
        self.store = store
        self.now = now
        self.default = default

    def mode(self) -> JudgeMode:
        """Return the judge jobs should use."""
        return self.store.load().mode or self.default

    def status(self) -> JudgeStatus:
        """Return the choice and Claude's quota as far as it is known."""
        state = self.store.load()
        return JudgeStatus(
            mode=state.mode or self.default,
            chosen=state.mode is not None,
            modes=list(JUDGE_MODES),
            claude_quota=quota_status(state, self.now()),
            quota_exhausted_at=state.quota_exhausted_at,
            quota_resets=state.quota_resets,
            quota_resets_at=state.quota_resets_at,
            quota_ok_at=state.quota_ok_at,
        )

    def choose(self, mode: str) -> JudgeStatus:
        """Store ``mode`` as the judge of the next jobs."""
        if mode not in get_args(JudgeMode):
            raise InvalidInputError(
                f"Unknown judge {mode!r}; choose one of "
                f"{', '.join(JUDGE_MODES)}."
            )
        self.store.update(lambda s: s.model_copy(update={"mode": mode}))
        return self.status()

    def quota_available(self) -> bool:
        """Tell whether Claude is worth trying (not known to be out)."""
        return self.status().claude_quota != "exhausted"

    def record_exhausted(
        self, resets: str, resets_at: Optional[datetime] = None
    ) -> None:
        """Remember that Claude refused for quota now."""
        now = self.now()
        self.store.update(lambda s: s.model_copy(update={
            "quota_exhausted_at": now, "quota_resets": resets,
            "quota_resets_at": resets_at,
        }))

    def record_answer(self) -> None:
        """Remember that Claude answered now."""
        now = self.now()
        self.store.update(
            lambda s: s.model_copy(update={"quota_ok_at": now})
        )


class FallbackJudge:
    """Claude first; the local judge once Claude's quota is used up.

    The switch is made once per job and kept for the rest of it: from
    the first quota refusal on, every item is answered by ``fallback``
    and :attr:`fallback_reason` says why. When the selection already
    knows the quota is out, the job starts on the local judge without
    calling Claude. :attr:`model_name` is always the judge that
    produced the last reply, so each item records its own judge.
    """

    def __init__(
        self,
        primary: ChatModel,
        fallback: ChatModel,
        selection: Optional[JudgeSelection] = None,
    ):
        """Wire both judges and the quota record."""
        self.primary = primary
        self.fallback = fallback
        self.selection = selection
        self.fallback_reason = ""
        self._answered = False
        if selection is not None and not selection.quota_available():
            status = selection.status()
            self.fallback_reason = (
                "Claude's usage limit was reached at "
                f"{status.quota_exhausted_at}"
                + (f" (resets {status.quota_resets})"
                   if status.quota_resets else "")
            )

    @property
    def active(self) -> ChatModel:
        """Return the judge answering now."""
        return self.fallback if self.fallback_reason else self.primary

    @property
    def model_name(self) -> str:
        """Return the identifier of the judge answering now."""
        return self.active.model_name

    @property
    def last_usage(self) -> Optional[Usage]:
        """Return the tokens of the last reply, if its judge reports them."""
        return last_usage(self.active)

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        """Answer with Claude, or with the local judge after a refusal."""
        if not self.fallback_reason:
            try:
                reply = self.primary.complete(system, user, max_tokens)
            except QuotaExhaustedError as exc:
                self.fallback_reason = str(exc)
                if self.selection is not None:
                    self.selection.record_exhausted(exc.resets, exc.resets_at)
            else:
                if not self._answered and self.selection is not None:
                    self.selection.record_answer()
                self._answered = True
                return reply
        return self.fallback.complete(system, user, max_tokens)

    def release(self) -> None:
        """Free the local judge's memory, if it was loaded."""
        release(self.fallback)


class CountingCache:
    """A :class:`~src.application.ports.Cache` that counts its hits.

    Every answer found in the wrapped cache is added to the ledger as
    a cache hit (it costs no tokens); everything else passes through.
    """

    def __init__(self, cache: Cache, ledger: UsageLedger):
        """Wrap ``cache``; count hits in ``ledger``."""
        self.cache = cache
        self.ledger = ledger

    def get(self, key: str) -> Optional[str]:
        """Return the value under ``key``, counting a hit."""
        value = self.cache.get(key)
        if value is not None:
            self.ledger.hit()
        return value

    def set(self, key: str, value: str) -> None:
        """Store ``value`` under ``key``."""
        self.cache.set(key, value)

    def clear(self) -> int:
        """Delete every entry; return how many were removed."""
        return self.cache.clear()


class CallBudget:
    """A judge that refuses calls past ``limit`` in one analysis.

    Wraps the real judge after the reply cache, so only calls that
    reach the model count (a resumed run answered from the cache costs
    nothing). Past the limit it raises
    :class:`~src.domain.errors.LimitError`; the step fails with that
    reason and the run can be resumed with a higher limit.
    """

    def __init__(
        self, judge: ChatModel, limit: int,
        ledger: Optional[UsageLedger] = None,
    ):
        """Wrap ``judge``; allow ``limit`` calls.

        ``ledger``, when given, gets every call that reaches the model
        with the tokens the judge reported for it.
        """
        self.judge = judge
        self.limit = max(limit, 0)
        self.ledger = ledger
        self._spent = [0]

    @property
    def calls(self) -> int:
        """Return how many calls reached the model (shared budgets too)."""
        return self._spent[0]

    def using(self, judge: ChatModel) -> "CallBudget":
        """Return a budget over ``judge`` that spends this one's calls.

        The synthesis step (#113) may ask another judge than the run's
        (Claude when the run's judge is the local model); its calls
        still count in the one limit of the analysis.
        """
        other = CallBudget(judge, self.limit, self.ledger)
        other._spent = self._spent
        return other

    @property
    def model_name(self) -> str:
        """Return the wrapped judge's identifier."""
        return self.judge.model_name

    @property
    def fallback_reason(self) -> str:
        """Return why the wrapped judge fell back to the local one, if so."""
        return str(getattr(self.judge, "fallback_reason", ""))

    @property
    def remaining(self) -> int:
        """Return how many calls are still allowed."""
        return max(self.limit - self.calls, 0)

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        """Ask the judge, unless the budget is spent."""
        if self.calls >= self.limit:
            raise LimitError(
                f"The judge-call limit of this analysis ({self.limit}) is "
                "reached; raise max_judge_calls to go on."
            )
        self._spent[0] += 1
        reply = self.judge.complete(system, user, max_tokens)
        if self.ledger is not None:
            self.ledger.record(self.judge.model_name, last_usage(self.judge))
        return reply

    def release(self) -> None:
        """Free the wrapped judge's memory, if it can."""
        release(self.judge)


def last_usage(judge: Any) -> Optional[Usage]:
    """Return the tokens of a judge's last reply (``None``: not reported)."""
    usage = getattr(judge, "last_usage", None)
    return usage if isinstance(usage, Usage) else None


def release(judge: Any) -> None:
    """Free a judge's memory when it can (a local model); else nothing."""
    method = getattr(judge, "release", None)
    if callable(method):
        method()


def decision_label(decision: Optional[TagDecision]) -> str:
    """Return a decision as one label: the indicator, ``none`` or ``?``."""
    if decision is None or decision.status == "unparsed":
        return "?"
    if decision.status == "rejected":
        return "none"
    return decision.indicator_id or "none"


def compare_confirmations(
    tagged: List[TaggedSentence],
    rubric: RubricCatalogue,
    first: TagConfirmer,
    second: TagConfirmer,
    language: Optional[str] = None,
) -> Tuple[List[TaggedSentence], LabelComparison]:
    """Confirm the same candidates with two judges and compare them.

    Each sentence keeps the first judge's ``decision`` and the second
    one's ``second_decision``. The comparison labels a decision by the
    indicator it confirms, ``none`` or ``?`` (not understood); there is
    no reference, so it measures agreement only. Items both decisions
    of which come from the same model (the first judge fell back to the
    local one) are left out of it and counted in ``same_judge``.
    """
    a = first.confirm(tagged, rubric, language)
    release(first.judge)
    b = second.confirm(tagged, rubric, language)
    release(second.judge)
    merged = [
        x.model_copy(update={"second_decision": y.decision})
        for x, y in zip(a, b)
    ]
    judged = [m for m in merged if m.tags]
    distinct = [
        m for m in judged
        if not same_model(m.decision, m.second_decision)
    ]
    comparison = compare_labels(
        [decision_label(m.decision) for m in distinct],
        [decision_label(m.second_decision) for m in distinct],
    )
    return merged, comparison.model_copy(
        update={"same_judge": len(judged) - len(distinct)}
    )


def same_model(
    first: Optional[TagDecision], second: Optional[TagDecision]
) -> bool:
    """Tell whether two decisions were made by the same model."""
    return (
        first is not None and second is not None
        and first.model == second.model
    )


class ActItem(BaseModel):
    """One sentence to classify, with its context and reference.

    ``context`` holds ``(speaker, text)`` pairs, oldest first;
    ``reference`` the categories counted as right (empty: none known).
    """

    id: str
    speaker: str
    text: str
    context: List[Tuple[str, str]]
    reference: List[str] = []


class ActVerdict(BaseModel):
    """A judge's category for one sentence (``None``: not understood)."""

    id: str
    act: Optional[str]
    model: str


def act_items(
    dialogues: Sequence[Dict[str, Any]],
    context: int = 3,
    sample: Optional[int] = None,
) -> List[ActItem]:
    """Build the items of the Spanish set (``dialogues.json`` documents).

    ``sample`` keeps that many sentences spread evenly over the whole
    set (every n-th), so a small run still covers every dialogue type.
    """
    items: List[ActItem] = []
    for dialogue in dialogues:
        utterances = dialogue.get("utterances", [])
        session = str(dialogue.get("session", {}).get("id", ""))
        for n, utterance in enumerate(utterances):
            before = utterances[max(n - context, 0):n]
            act = utterance.get("act")
            items.append(ActItem(
                id=f"{session}/{utterance['id']}",
                speaker=str(utterance["speaker"]),
                text=str(utterance["text"]),
                context=[(str(u["speaker"]), str(u["text"])) for u in before],
                reference=[str(act)] if act else [],
            ))
    if sample is not None and 0 < sample < len(items):
        step = len(items) / sample
        items = [items[int(i * step)] for i in range(sample)]
    return items


class ActClassifier:
    """Ask a judge for the dialogue act of each sentence (cached)."""

    def __init__(
        self,
        judge: ChatModel,
        cache: Optional[Cache] = None,
        progress: Optional[Progress] = None,
        max_tokens: int = 40,
    ):
        """Wire the judge and the optional reply cache."""
        self.judge = judge
        self.cache = cache
        self.progress = progress or NoProgress()
        self.max_tokens = max_tokens

    def classify(self, items: Sequence[ActItem]) -> List[ActVerdict]:
        """Return one verdict per item, in order."""
        system = act_system_prompt()
        verdicts: List[ActVerdict] = []
        for item in self.progress.track(
            list(items), len(items), "Classifying sentences"
        ):
            user = act_user_prompt(item.context, item.speaker, item.text)
            key = cache_key("act", self.judge.model_name, system, user)
            reply = self.cache.get(key) if self.cache else None
            if reply is None:
                reply = self.judge.complete(system, user, self.max_tokens)
                if self.cache:
                    self.cache.set(
                        cache_key("act", self.judge.model_name, system, user),
                        reply,
                    )
            verdicts.append(ActVerdict(
                id=item.id, act=parse_act(reply), model=self.judge.model_name,
            ))
        return verdicts


def compare_act_judges(
    items: Sequence[ActItem],
    first: ActClassifier,
    second: ActClassifier,
) -> Tuple[List[ActVerdict], List[ActVerdict], LabelComparison]:
    """Classify ``items`` with two judges; compare them and the reference.

    A reply that names no category counts as the label ``?``. Items
    both verdicts of which come from the same model (the first judge
    fell back to the local one) are left out of the comparison and
    counted in ``same_judge``.
    """
    a = first.classify(items)
    release(first.judge)
    b = second.classify(items)
    release(second.judge)
    kept = [n for n in range(len(items)) if a[n].model != b[n].model]
    comparison = compare_labels(
        [a[n].act or "?" for n in kept], [b[n].act or "?" for n in kept],
        [items[n].reference for n in kept],
    )
    return a, b, comparison.model_copy(
        update={"same_judge": len(items) - len(kept)}
    )


def act_accuracy(
    items: Sequence[ActItem], verdicts: Sequence[ActVerdict]
) -> Optional[float]:
    """Return the share of referenced items a judge got right."""
    scored = [
        (v.act, set(item.reference))
        for item, v in zip(items, verdicts) if item.reference
    ]
    if not scored:
        return None
    return round(sum(1 for act, ref in scored if act in ref) / len(scored), 4)
