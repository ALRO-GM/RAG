"""Use case: how accurate the answers of a live-page round were (#259).

A live-page interview with a role's plan and a candidate has no report
of its own, so its end result gets the same judgement as the AI
interviewer's report: when the round ends, the slots the interviewer
asked about, the candidate's answers to each and the concepts they
treated (:meth:`~orchestrator.application.next_round.TopicTracker.sheet_slots`)
go to the report template (``interview.report@2``) once; its reply is
read by :func:`~orchestrator.domain.interview_judgement.judge_slots`.
The result is published on the feed (``{"type": "accuracy"}``), kept in
the session's hand-off record without words, and stored **in full**
with the plan's results as an ``interview.sheet.v1`` of ``kind: "live"``
(#262), where the candidate's file and «Comparar candidatos» list it.
Only judgements, never a score: the live page does not score, and
nothing of it is shown to the candidate. One model call per
:data:`JUDGE_BATCH` answered slots; none when nothing was answered. A
failed call is logged with its status and error (never the candidate's
words) and stores nothing.
"""

import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from orchestrator.application.interview import (
    InterviewWiring,
    judge_chunks,
    report_request,
)
from orchestrator.application.next_round import TopicTracker
from orchestrator.application.replay import transcript_to_events
from orchestrator.application.ports import (
    CancelToken,
    Document,
    ReasonerUnavailable,
)
from orchestrator.domain.interview import (
    Answer,
    CheckItem,
    InterviewPlan,
    Sheet,
    SheetSlot,
)
from orchestrator.domain.events import Transcript
from orchestrator.domain.interview_judgement import MAX_CHECKS, judge_slots
from orchestrator.domain.next_round import topic_states
from orchestrator.domain.interview_points import quote

LOG = logging.getLogger(__name__)
LIVE_JUDGE_TEMPLATE = "interview.report@2"
JUDGE_BATCH = 4
"""Slots judged in one call: a long round goes in several calls, each
well inside the output cap and the 90 s deadline (12 short slots took
22–43 s in the #259 measurement; long rounds were not measured)."""
ANSWER_PARTS, PART_CHARS = 6, 700
"""Answer parts per slot and characters per part: what the report
template reads of each slot (``interview.report@2``)."""
WORDS = ("quote", "reason", "note")
"""Fields with the candidate's words or the model's prose about them."""


def without_words(result: Document) -> Document:
    """A round's judgement without quotes, reasons or notes.

    What the hand-off file keeps (#259 review): the labels, counts,
    bases, cited chunk ids and the utterance ids, never the candidate's
    words or what the model wrote about them.
    """
    def strip(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: strip(v) for k, v in value.items() if k not in WORDS}
        if isinstance(value, list):
            return [strip(v) for v in value]
        return value

    kept: Document = strip(result)
    return kept


class LiveJudge:
    """Judge one live-page round's answers when it ends, and keep it.

    Args:
        plan: The role's plan.
        wiring: The interviews' ports (reasoner, knowledge, plan store).
        session_id: The live session.
        publish: Sends one feed message.
        model: The model alias.
        template: The report template.
        candidate: The candidate's name, for the stored report.
        candidate_id: The candidate's profile, for the stored report.
        now: Wall-clock time.
    """

    def __init__(
        self,
        plan: InterviewPlan,
        wiring: InterviewWiring,
        session_id: str,
        publish: Callable[[Document], object],
        model: str = "haiku",
        template: str = LIVE_JUDGE_TEMPLATE,
        candidate: Optional[str] = None,
        candidate_id: Optional[str] = None,
        now: Callable[[], datetime] = datetime.now,
    ):
        """Nothing judged yet."""
        self.plan = plan
        self.wiring = wiring
        self.session_id = session_id
        self.publish = publish
        self.model = model
        self.template = template
        self.candidate = candidate
        self.candidate_id = candidate_id
        self.now = now

    def judge(self, topics: TopicTracker) -> Optional[Document]:
        """Judge the round (``None`` when nothing was answered)."""
        slots, texts = topics.sheet_slots()
        return self.judge_slots(slots, texts)

    def judge_slots(
        self, slots: List[SheetSlot], texts: Dict[str, str],
    ) -> Optional[Document]:
        """Judge the answered slots in batches, keep and publish it.

        Each slot's answers are joined into at most
        :data:`ANSWER_PARTS` parts of :data:`PART_CHARS` characters
        (:func:`merged_answers`) and at most :data:`JUDGE_BATCH` slots
        go in one call, so a long round stays within the template's
        limits and the deadline. Any failed batch fails the judgement:
        nothing is stored, the reason is logged (never the candidate's
        words) and ``{"state": "failed"}`` is published.
        """
        asked, texts = merged_slots([s for s in slots if s.answers], texts)
        if not asked:
            return None
        self.publish({"type": "accuracy", "state": "reporting"})
        judged: List[SheetSlot] = []
        checks: List[CheckItem] = []
        for n in range(0, len(asked), JUDGE_BATCH):
            batch = asked[n:n + JUDGE_BATCH]
            found = self._batch(batch, texts, n // JUDGE_BATCH)
            if found is None:
                result: Document = {"state": "failed",
                                    "judge": self.template, "saved": False}
                self.publish({"type": "accuracy", **result})
                return result
            judged += found[0]
            checks += [c for c in found[1] if c.quote not in
                       {k.quote for k in checks}]
        checks = checks[:MAX_CHECKS]
        saved = self._keep(judged, checks, len(texts))
        result = {
            "state": "ready", "judge": self.template,
            "plan_id": self.plan.id, "title": self.plan.title,
            "session_id": self.session_id, "saved": saved,
            "slots": [
                s.model_dump(mode="json", include={
                    "id", "label", "level", "status", "points",
                    "judgement", "accuracy"})
                for s in judged
            ],
            "to_check": [c.model_dump(mode="json") for c in checks],
        }
        self.publish({"type": "accuracy", **result})
        return result

    def _batch(
        self, batch: List[SheetSlot], texts: Dict[str, str], n: int,
    ) -> Optional[Tuple[List[SheetSlot], List[CheckItem]]]:
        """One report call on a few slots (``None``: it failed, logged)."""
        try:
            chunks = judge_chunks(self.wiring.planner.knowledge, self.plan,
                                  batch, texts, self.session_id)
        except Exception as exc:  # noqa: BLE001 - general knowledge
            LOG.warning("judge %s batch %d: knowledge failed (%s), "
                        "general knowledge used", self.session_id, n,
                        type(exc).__name__)
            chunks = {}
        request = report_request(
            self.plan, batch, texts, chunks, "", self.template, self.model,
            f"{self.session_id}-judge{n}"[-64:],
        )
        try:
            response = self.wiring.reasoner.reason(request, CancelToken())
        except ReasonerUnavailable as exc:
            LOG.warning("judge %s batch %d: reasoner unavailable: %s",
                        self.session_id, n, str(exc)[:200])
            return None
        data = response.get("data")
        if response.get("status") != "ok" or not isinstance(data, dict):
            error = response.get("error")
            error = error if isinstance(error, dict) else {}
            LOG.warning(
                "judge %s batch %d (%d slots, %d answer chars): status %s, "
                "error %s: %s, data %s", self.session_id, n, len(batch),
                sum(len(texts.get(a.utterance, "")) for s in batch
                    for a in s.answers),
                response.get("status"), error.get("kind"),
                str(error.get("message") or "")[:200],
                type(data).__name__)
            return None
        try:
            return judge_slots(
                data, batch, {sid: [c["id"] for c in found]
                              for sid, found in chunks.items()}, texts)
        except Exception as exc:  # noqa: BLE001 - logged, failed
            LOG.warning("judge %s batch %d: reply not read (%s)",
                        self.session_id, n, type(exc).__name__)
            return None

    def _keep(self, judged: List[SheetSlot], checks: List[CheckItem],
              utterances: int) -> bool:
        """Store the full report with the plan's results (#262)."""
        sheet = Sheet(
            session_id=self.session_id, plan_id=self.plan.id,
            plan_revision=self.plan.revision, title=self.plan.title,
            state="closed", kind="live", turns=utterances,
            slots=judged, filler="rules", candidate=self.candidate,
            candidate_id=self.candidate_id, to_check=checks,
            judge=self.template, finished_at=self.now(),
        )
        try:
            self.wiring.store.save_result(self.plan.id, sheet.to_contract())
        except (OSError, KeyError) as exc:
            LOG.warning("judge %s: report not kept (%s)", self.session_id,
                        type(exc).__name__)
            return False
        return True


def merged_answers(
    said: List[Tuple[str, str]], texts: Dict[str, str],
) -> List[Tuple[str, str]]:
    """Join a slot's consecutive utterances into a few answer parts.

    A live round splits an answer into many short utterances; the judge
    reads at most :data:`ANSWER_PARTS` parts of :data:`PART_CHARS`
    characters per slot. Each part keeps the id of its first utterance;
    parts beyond the limit are left out (and counted in the log).
    """
    parts: List[Tuple[str, str]] = []
    for utterance, first in said:
        text = " ".join((texts.get(utterance) or first).split())
        if not text:
            continue
        if parts and len(parts[-1][1]) + 1 + len(text) <= PART_CHARS:
            parts[-1] = (parts[-1][0], f"{parts[-1][1]} {text}")
        else:
            parts.append((utterance, text[:PART_CHARS]))
    return parts


def merged_slots(
    slots: List[SheetSlot], texts: Dict[str, str],
) -> Tuple[List[SheetSlot], Dict[str, str]]:
    """The slots with their answers joined (:func:`merged_answers`)."""
    joined: Dict[str, str] = dict(texts)
    found: List[SheetSlot] = []
    for slot in slots:
        parts = merged_answers([(a.utterance, a.quote) for a in slot.answers],
                               texts)
        if len(parts) > ANSWER_PARTS:
            LOG.info("judge: slot %s has %d answer parts, %d judged",
                     slot.id, len(parts), ANSWER_PARTS)
        parts = parts[:ANSWER_PARTS]
        joined.update(dict(parts))
        found.append(slot.model_copy(update={"answers": [
            Answer(utterance=u, quote=quote(t), status="partial")
            for u, t in parts]}))
    return found, joined


def rejudge(
    transcript: Transcript,
    plan: InterviewPlan,
    wiring: InterviewWiring,
    model: str = "haiku",
    candidate: Optional[str] = None,
    candidate_id: Optional[str] = None,
    publish: Callable[[Document], object] = lambda message: None,
    now: Callable[[], datetime] = datetime.now,
) -> Optional[Document]:
    """Judge a live round again from its stored ``transcript.v1`` (#262).

    For a round whose judgement failed or was lost: the conversation is
    replayed through a :class:`TopicTracker` (speakers' ``role`` says who
    interviews and who answers) and judged as at the end of the round;
    the full report is stored with the plan's results under the same
    session id. ``None`` when the candidate answered nothing the tracker
    could attribute to a topic of the plan.
    """
    roles = {s.id: s.role or "unknown" for s in transcript.speakers}
    topics = TopicTracker(plan, topic_states(plan, []), roles,
                          lambda message: None)
    for event in transcript_to_events(transcript, roles):
        topics.on_event(event)
    session_id = next(transcript_to_events(transcript, roles)).session_id
    judge = LiveJudge(plan, wiring, session_id, publish, model,
                      candidate=candidate, candidate_id=candidate_id,
                      now=now)
    return judge.judge(topics)
