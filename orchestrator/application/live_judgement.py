"""Use case: how accurate the answers of a live-page round were (#259).

A live-page interview with a role's plan and a candidate has no report
of its own, so its end result gets the same judgement as the AI
interviewer's report: when the round ends, the slots the interviewer
asked about, the candidate's answers to each and the concepts they
treated (:meth:`~orchestrator.application.next_round.TopicTracker.sheet_slots`)
go to the report template (``interview.report@2``) once; its reply is
read by :func:`~orchestrator.domain.interview_judgement.judge_slots`.
The result is published on the feed (``{"type": "accuracy"}``) and kept
in the session's hand-off record. Only judgements, never a score: the
live page does not score, and nothing of it is shown to the candidate.
One model call per round; none when nothing was answered.
"""

from typing import Any, Callable, Optional

from orchestrator.application.interview import (
    InterviewWiring,
    judge_chunks,
    report_request,
)
from orchestrator.application.next_round import TopicTracker
from orchestrator.application.ports import (
    CancelToken,
    Document,
    ReasonerUnavailable,
)
from orchestrator.domain.interview import InterviewPlan
from orchestrator.domain.interview_judgement import judge_slots

LIVE_JUDGE_TEMPLATE = "interview.report@2"
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
    """Judge one live-page round's answers when it ends.

    Args:
        plan: The role's plan.
        wiring: The interviews' ports (reasoner and knowledge).
        session_id: The live session.
        publish: Sends one feed message.
        model: The model alias.
        template: The report template.
    """

    def __init__(
        self,
        plan: InterviewPlan,
        wiring: InterviewWiring,
        session_id: str,
        publish: Callable[[Document], object],
        model: str = "haiku",
        template: str = LIVE_JUDGE_TEMPLATE,
    ):
        """Nothing judged yet."""
        self.plan = plan
        self.wiring = wiring
        self.session_id = session_id
        self.publish = publish
        self.model = model
        self.template = template

    def judge(self, topics: TopicTracker) -> Optional[Document]:
        """Judge the round (``None`` when nothing was answered).

        A failed call gives ``{"state": "failed"}``; the round's other
        results are not affected.
        """
        slots, texts = topics.sheet_slots()
        asked = [s for s in slots if s.answers]
        if not asked:
            return None
        self.publish({"type": "accuracy", "state": "reporting"})
        chunks = judge_chunks(self.wiring.planner.knowledge, self.plan,
                              asked, texts, self.session_id)
        request = report_request(
            self.plan, asked, texts, chunks, "", self.template, self.model,
            f"{self.session_id}-judge"[-64:],
        )
        try:
            response = self.wiring.reasoner.reason(request, CancelToken())
        except ReasonerUnavailable:
            response = {}
        data = response.get("data")
        if response.get("status") != "ok" or not isinstance(data, dict):
            result: Document = {"state": "failed", "judge": self.template}
            self.publish({"type": "accuracy", **result})
            return result
        judged, checks = judge_slots(
            data, asked, {sid: [c["id"] for c in found]
                          for sid, found in chunks.items()}, texts)
        result = {
            "state": "ready", "judge": self.template,
            "plan_id": self.plan.id, "title": self.plan.title,
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
