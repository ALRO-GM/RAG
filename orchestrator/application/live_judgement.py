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

**Who answered what** (#268). The word rules of the round attribute
few answers on real speech, so before judging, the model reads the
whole round (:meth:`TopicTracker.round_lines`: the interviewer's
questions and the candidate's answers in order, fenced) with the plan's
points and concepts and says which candidate utterances answer each
point and which concepts were treated (``interview.attribution@2``,
read by :mod:`orchestrator.domain.interview_attribution`): one call per
batch of at most 120 utterances / 16,000 characters. The judge then
gets those answers, and the stored report marks each concept the model
ticked ``by: "after_session"`` and the round ``attribution:
"after_session"``. When any attribution call fails (or the reply cannot
be read) the round falls back to the rules' attribution of the session,
as before, with ``attribution: "rules"`` and the reason (status and
error kind, never words) in ``attribution_error``.

**Coverage** (#286). The stored report also keeps, per topic of the
plan, its level and, per concept, whether its question was asked
(when, by whom, the interviewer's words) and whether the candidate
treated it (``coverage``, :func:`~orchestrator.domain.
interview_coverage.settled`): the session's marks (rules, the tick
call, by hand), the model's after-session reading for what the session
missed, and every mark, every undone mark and the level reached kept from an
earlier report of the same round, or from its hand-off (a rebuild or
``rejudge`` replays only the words). Every topic is
there, also one never answered, so «preguntado, sin respuesta» is told
apart from «no preguntado». The figure of the offer board does not
use it.
"""

import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

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
    ReasonerRejected,
    ReasonerUnavailable,
)
from orchestrator.domain.interview import (
    Answer,
    CheckItem,
    InterviewPlan,
    Judgement,
    Sheet,
    SheetSlot,
    TopicCoverage,
    fold,
)
from orchestrator.domain.events import Transcript
from orchestrator.domain.interview_attribution import (
    CANDIDATE,
    Attribution,
    RoundLine,
    asked_marks,
    attributed_slots,
    batches,
    with_session_ticks,
    merged,
    read_attribution,
    round_payload,
    used_ids,
)
from orchestrator.domain.interview_coverage import (
    earlier_coverage,
    settled,
)
from orchestrator.domain.interview_judgement import MAX_CHECKS, judge_slots
from orchestrator.domain.next_round import topic_states
from orchestrator.domain.interview_points import quote

LOG = logging.getLogger(__name__)
LIVE_JUDGE_TEMPLATE = "interview.report@2"
ATTRIBUTION_TEMPLATE = "interview.attribution@2"
ATTRIBUTION_FALLBACK = "interview.attribution@1"
"""Used when the reasoner refuses ``@2`` (a RAG not restarted with the
orchestrator, #286 review): no asked concepts after the session."""
"""Since #286 it also says which concepts the interviewer asked."""
ATTRIBUTION_DEADLINE_MS = 90000
"""Deadline of one attribution call (a slow template, #267)."""
JUDGE_BATCH = 4
"""Slots judged in one call: a long round goes in several calls, each
well inside the output cap and the 90 s deadline (12 short slots took
22–43 s in the #259 measurement; long rounds were not measured)."""
ANSWER_PARTS, PART_CHARS = 6, 700
"""Answer parts per slot and characters per part: what the report
template reads of each slot (``interview.report@2``)."""
PART_MARK = "#"
"""Joins an utterance id and a slot id in a joined part's key."""
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
        attribute: Whether the model attributes the round's answers
            before judging (#268); ``False`` keeps the rules' attribution.
        earlier: The round's coverage kept elsewhere (the hand-off's),
            used when no report of the round is stored (#286).
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
        attribute: bool = True,
        earlier: Optional[Mapping[str, TopicCoverage]] = None,
    ):
        """Nothing judged yet."""
        self.earlier = earlier
        self.plan = plan
        self.wiring = wiring
        self.session_id = session_id
        self.publish = publish
        self.model = model
        self.template = template
        self.candidate = candidate
        self.candidate_id = candidate_id
        self.now = now
        self.attribute = attribute

    def judge(self, topics: TopicTracker) -> Optional[Document]:
        """Attribute and judge the round (``None``: nothing answered).

        The model's attribution of the whole round when it works
        (:meth:`attribute_round`), else the rules' of the session.
        """
        lines = topics.round_lines()
        found, why = None, None
        earlier = self._earlier()
        if self.attribute and any(line.role == CANDIDATE for line in lines):
            self.publish({"type": "accuracy", "state": "reporting"})
            found, why = self.attribute_round(lines)
            if found is not None:
                texts = {line.id: line.text for line in lines
                         if line.role == CANDIDATE}
                session, _ = topics.sheet_slots()
                slots = with_session_ticks(
                    attributed_slots(self.plan, found, texts), session,
                    texts)
                coverage = settled(
                    self.plan.slots, topics.facts(), topics.reached(),
                    slots, texts, asked_marks(found, lines), earlier)
                return self.judge_slots(slots, texts, "after_session",
                                        announced=True, coverage=coverage)
        slots, texts = topics.sheet_slots()
        coverage = settled(self.plan.slots, topics.facts(),
                           topics.reached(), slots, texts, {}, earlier)
        return self.judge_slots(slots, texts, "rules", why,
                                announced=why is not None,
                                coverage=coverage)

    def _earlier(self) -> Dict[str, TopicCoverage]:
        """The coverage kept earlier for this round (#286 review).

        :func:`kept_coverage`: the stored report of the same session,
        else the coverage given (the hand-off's), else nothing.
        """
        return kept_coverage(self.wiring, self.plan.id, self.session_id,
                             self.earlier)

    def attribute_round(
        self, lines: List[RoundLine],
    ) -> Tuple[Optional[Attribution], Optional[str]]:
        """Ask the model who answered what, batch by batch (#268).

        Returns the merged attribution, or ``None`` and why: the status
        and error kind of a failed batch, ``unreadable`` (a reply of the
        wrong shape), ``ids_unknown`` (every id it gave was dropped) or
        ``empty`` (it attributed nothing); never words. Logs counts only.
        """
        parts = batches(lines)
        template = ATTRIBUTION_TEMPLATE
        found: List[Attribution] = []
        totals: Dict[str, int] = {}
        for n, batch in enumerate(parts):
            request = attribution_request(
                self.plan, batch, self.model,
                f"{self.session_id}-attr{n}"[-64:], template)
            try:
                try:
                    response = self.wiring.reasoner.reason(
                        request, CancelToken())
                except ReasonerRejected:
                    if template != ATTRIBUTION_TEMPLATE:
                        raise
                    template = ATTRIBUTION_FALLBACK
                    LOG.warning("attribution %s: the reasoner refused %s; "
                                "%s used (no asked concepts)",
                                self.session_id, ATTRIBUTION_TEMPLATE,
                                template)
                    request = {**request, "template": template}
                    response = self.wiring.reasoner.reason(
                        request, CancelToken())
            except ReasonerUnavailable as exc:
                LOG.warning("attribution %s batch %d: reasoner unavailable: "
                            "%s", self.session_id, n, str(exc)[:200])
                return None, "unavailable"
            data = response.get("data")
            if response.get("status") != "ok" or not isinstance(data, dict):
                error = response.get("error")
                kind = error.get("kind") if isinstance(error, dict) else None
                LOG.warning("attribution %s batch %d (%d utterances): "
                            "status %s, error %s", self.session_id, n,
                            len(batch), response.get("status"), kind)
                return None, f"{response.get('status')}: {kind or '-'}"
            stats: Dict[str, int] = {}
            try:
                found.append(read_attribution(data, self.plan, batch,
                                              stats))
            except Exception as exc:  # noqa: BLE001 - logged, rules used
                LOG.warning("attribution %s batch %d: reply not read (%s)",
                            self.session_id, n, type(exc).__name__)
                return None, f"unreadable: {type(exc).__name__}"
            LOG.info("attribution %s batch %d: %d entries given, %d kept, "
                     "%d dropped%s", self.session_id, n,
                     stats.get("given", 0), stats.get("kept", 0),
                     stats.get("given", 0) - stats.get("kept", 0),
                     "" if stats.get("shaped") else " (slots not a list)")
            for key, value in stats.items():
                totals[key] = totals.get(key, 0) + value
            totals["batches"] = totals.get("batches", 0) + 1
        result = merged(found, [line.id for line in lines])
        if not used_ids(result) and not any(s.asked
                                            for s in result.values()):
            why = ("unreadable" if totals.get("shaped", 0) < len(parts)
                   else "ids_unknown" if totals.get("given") else "empty")
            LOG.warning("attribution %s: nothing usable (%s), rules used",
                        self.session_id, why)
            return None, why
        LOG.info("attribution %s: %d batches, %d of %d candidate "
                 "utterances used, %d points answered, %d concepts",
                 self.session_id, len(parts), len(used_ids(result)),
                 sum(1 for line in lines if line.role == CANDIDATE),
                 sum(1 for s in result.values() if s.answers),
                 sum(len(s.points) for s in result.values()))
        return result, None

    def judge_slots(
        self, slots: List[SheetSlot], texts: Dict[str, str],
        attribution: str = "rules", why: Optional[str] = None,
        announced: bool = False,
        coverage: Optional[List[TopicCoverage]] = None,
    ) -> Optional[Document]:
        """Judge the answered slots in batches, keep and publish it.

        Each slot's answers are joined into at most
        :data:`ANSWER_PARTS` parts of :data:`PART_CHARS` characters
        (:func:`merged_answers`) and at most :data:`JUDGE_BATCH` slots
        go in one call, so a long round stays within the template's
        limits and the deadline. Any failed batch fails the judgement:
        nothing is stored, the reason is logged (never the candidate's
        words) and ``{"state": "failed"}`` is published. ``attribution``
        says who attributed the answers (``after_session`` or
        ``rules``; ``why``: why not the model); ``announced``: the
        ``reporting`` state was already published; ``coverage`` (#286)
        goes with the report.
        """
        originals = {s.id: [(a.utterance, texts.get(a.utterance) or a.quote)
                            for a in s.answers] for s in slots}
        asked, texts = merged_slots([s for s in slots if s.answers], texts)
        if not asked:
            if announced:
                self.publish({"type": "accuracy", "state": "empty",
                              **({"attribution_error": why} if why
                                 else {})})
            return None
        if not announced:
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
        judged, checks = to_originals(judged, checks, originals)
        utterances = len({k.split(PART_MARK)[0] for k in texts})
        saved, lost = self._keep(judged, checks, utterances, attribution,
                                 why, coverage)
        result = {
            "state": "ready", "judge": self.template,
            "plan_id": self.plan.id, "title": self.plan.title,
            "session_id": self.session_id, "saved": saved,
            **({"why": lost} if lost else {}),
            "attribution": attribution,
            **({"attribution_error": why} if why else {}),
            "slots": [
                s.model_dump(mode="json", include={
                    "id", "label", "level", "status", "points",
                    "judgement", "accuracy"})
                for s in judged
            ],
            "to_check": [c.model_dump(mode="json") for c in checks],
            **({"coverage": [c.model_dump(mode="json") for c in coverage]}
               if coverage is not None else {}),
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
              utterances: int, attribution: str = "rules",
              error: Optional[str] = None,
              coverage: Optional[List[TopicCoverage]] = None,
              ) -> Tuple[bool, Optional[str]]:
        """Store the full report with the plan's results (#262).

        Returns whether it was stored and, if not, why: ``no_file`` (the
        candidate's file was deleted: judging again would not help) or
        ``error`` (it may be retried).
        """
        sheet = Sheet(
            session_id=self.session_id, plan_id=self.plan.id,
            plan_revision=self.plan.revision, title=self.plan.title,
            state="closed", kind="live", turns=utterances,
            slots=judged, filler=attribution, candidate=self.candidate,
            candidate_id=self.candidate_id, to_check=checks,
            judge=self.template, finished_at=self.now(),
            attribution=attribution,  # type: ignore[arg-type]
            attribution_error=error,
            plan_slots=[s.id for s in self.plan.slots],
            coverage=coverage,
        )
        book = self.wiring.candidates
        try:
            if self.candidate_id and book is not None:
                # Never an orphan: the file may be deleted meanwhile.
                kept = book.keep_report(
                    self.plan.id, self.candidate_id,
                    lambda: self.wiring.store.save_result(
                        self.plan.id, sheet.to_contract()))
                return (True, None) if kept else (False, "no_file")
            self.wiring.store.save_result(self.plan.id, sheet.to_contract())
        except Exception as exc:  # noqa: BLE001 - logged, may be retried
            LOG.warning("judge %s: report not kept (%s)", self.session_id,
                        type(exc).__name__)
            return False, "error"
        return True, None


def attribution_request(
    plan: InterviewPlan, lines: List[RoundLine], model: str,
    request_id: str, template: str = ATTRIBUTION_TEMPLATE,
) -> Document:
    """The ``reasoner.request.v1`` of one attribution batch (#268).

    The plan's points (label, evidence, questions, concepts) and the
    batch's utterances in order (id, role, words); the template fences
    the words.
    """
    return {
        "schema": "reasoner.request.v1", "request_id": request_id,
        "role": "structured", "model": model,
        "template": template, "language": plan.language,
        "context": {"window": []},
        "payload": {
            "title": plan.title,
            "slots": [{"id": s.id, "label": s.label, "evidence": s.evidence,
                       "questions": s.questions[:3],
                       "points": [{"id": p.id, "text": p.text}
                                  for p in s.points]}
                      for s in plan.slots],
            "round": round_payload(lines),
        },
        "constraints": {"deadline_ms": ATTRIBUTION_DEADLINE_MS},
    }


def _origin(quote_: str, said: List[Tuple[str, str]],
            fallback: Optional[str]) -> Optional[str]:
    """The original utterance whose words contain ``quote_``.

    Without one, the fallback's own id (a part key made unique by
    :func:`merged_slots` loses its ``#slot`` suffix).
    """
    words = fold(quote_).strip(" .…\"'«»")
    found = next((u for u, text in said if words and words in fold(text)),
                 fallback)
    return found.split(PART_MARK)[0] if found else found


def to_originals(
    judged: List[SheetSlot], checks: List[CheckItem],
    originals: Dict[str, List[Tuple[str, str]]],
) -> Tuple[List[SheetSlot], List[CheckItem]]:
    """Point each quote at the utterance that said it, not the joined part.

    A joined answer part keeps the id of its first utterance; a quote
    from a later utterance of that part is moved to that utterance.
    """
    def moved(slot_id: str, j: Optional[Judgement]) -> Optional[Judgement]:
        if j is None:
            return j
        return j.model_copy(update={"utterance": _origin(
            j.quote, originals.get(slot_id, []), j.utterance)})

    slots = []
    for slot in judged:
        points = [
            p.model_copy(update={"judgement": moved(slot.id, p.judgement)})
            for p in slot.points
        ]
        answers = [a.model_copy(update={
            "utterance": a.utterance.split(PART_MARK)[0]})
            for a in slot.answers]
        slots.append(slot.model_copy(update={
            "judgement": moved(slot.id, slot.judgement), "points": points,
            "answers": answers}))
    every = [pair for found in originals.values() for pair in found]
    fixed = [c.model_copy(update={"utterance": _origin(
        c.quote, originals.get(c.slot or "", []) or every, c.utterance)})
        for c in checks]
    return slots, fixed


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
    """The slots with their answers joined (:func:`merged_answers`).

    One utterance can answer two slots (the model's attribution, #268)
    and start a different joined part in each: the second part's key
    is then ``<id>#<slot>``, so neither slot reads the other's words;
    :func:`to_originals` points the quotes back at the utterances.
    """
    joined: Dict[str, str] = dict(texts)
    used: Dict[str, str] = {}
    found: List[SheetSlot] = []
    for slot in slots:
        parts = merged_answers([(a.utterance, a.quote) for a in slot.answers],
                               texts)
        if len(parts) > ANSWER_PARTS:
            LOG.info("judge: slot %s has %d answer parts, %d judged",
                     slot.id, len(parts), ANSWER_PARTS)
        parts = [(u if used.get(u, t) == t else f"{u}{PART_MARK}{slot.id}",
                  t) for u, t in parts[:ANSWER_PARTS]]
        used.update({u: t for u, t in parts})
        joined.update(dict(parts))
        found.append(slot.model_copy(update={"answers": [
            Answer(utterance=u, quote=quote(t), status="partial")
            for u, t in parts]}))
    return found, joined


def kept_coverage(
    wiring: InterviewWiring, plan_id: str, session_id: str,
    fallback: Optional[Mapping[str, TopicCoverage]] = None,
) -> Dict[str, TopicCoverage]:
    """The coverage kept for a round (#286): its stored report's, else
    ``fallback`` (the hand-off's), else nothing."""
    try:
        found = earlier_coverage(wiring.store.result(plan_id, session_id))
    except Exception:  # noqa: BLE001 - none stored, or unreadable
        found = {}
    return found or dict(fallback or {})


def replayed_tracker(
    transcript: Transcript, plan: InterviewPlan,
    earlier: Optional[Mapping[str, TopicCoverage]] = None,
) -> TopicTracker:
    """The topic tracker of a round rebuilt from its ``transcript.v1``.

    The speakers' ``role`` says who interviews and who answers. For the
    same finals it attributes answers as the live round did; it can
    differ when the live round revised a final after the tracker read
    it, or when roles were set otherwise during the session.
    """
    roles = {s.id: s.role or "unknown" for s in transcript.speakers}
    topics = TopicTracker(plan, topic_states(plan, []), roles,
                          lambda message: None)
    if earlier:
        topics.seed_undone(earlier)  # an undone mark is never redone
    for event in transcript_to_events(transcript, roles):
        topics.on_event(event)
    return topics


def rejudge(
    transcript: Transcript,
    plan: InterviewPlan,
    wiring: InterviewWiring,
    model: str = "haiku",
    candidate: Optional[str] = None,
    candidate_id: Optional[str] = None,
    publish: Callable[[Document], object] = lambda message: None,
    now: Callable[[], datetime] = datetime.now,
    attribute: bool = True,
) -> Optional[Document]:
    """Judge a live round again from its stored ``transcript.v1`` (#262).

    For a round whose judgement failed or was lost, or to redo an older
    round with the model's attribution (#268): the conversation is
    replayed through a :class:`TopicTracker` (speakers' ``role`` says who
    interviews and who answers) and attributed and judged as at the end
    of the round (``attribute=False``: the rules' attribution only); the
    full report is stored with the plan's results under the same
    session id. ``None`` when nothing the candidate said was attributed
    to a point of the plan.
    """
    roles = {s.id: s.role or "unknown" for s in transcript.speakers}
    session_id = next(transcript_to_events(transcript, roles)).session_id
    topics = replayed_tracker(transcript, plan,
                              kept_coverage(wiring, plan.id, session_id))
    judge = LiveJudge(plan, wiring, session_id, publish, model,
                      candidate=candidate, candidate_id=candidate_id,
                      now=now, attribute=attribute)
    return judge.judge(topics)
