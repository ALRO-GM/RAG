"""Use cases: the candidate profile across interview rounds (#235).

:class:`CandidateBook` keeps the candidates of each role (an interview
plan): it creates them, shows them with the core questions of KN-4,
takes the interviewer's confirmations, corrections and discards,
deletes a candidate on request and applies the retention of the
interview products (``privacy.store_text_days``). :class:`CoreTracker`
follows one session (the live page or the AI interviewer): when a core
question still open is asked, it reads the candidate's answer right
after it with the fixed rules of :mod:`orchestrator.domain.candidate`
and keeps what they are sure of as a **proposal** for the interviewer
to confirm («Confirmar datos»). The candidate's file also lists their
stored interview reports (#262: the AI interviewer's sheets and the
live-page reports, in full); a profile with reports never expires by
days, and deleting the file deletes those reports too: nothing else
deletes them. Each change is published as
``{"type": "core", "profile": ...}``; only confirmed or corrected
answers are screened out of the live suggestions.

No model is called here: capturing is deterministic.
"""

import logging
import threading
from datetime import datetime
from typing import Callable, Dict, List, Optional, Set

from orchestrator.application.interview import PlanError, PlanStore
from orchestrator.application.ports import (
    CandidateStore,
    CoreQuestionSource,
    Document,
)
from orchestrator.domain.candidate import (
    CandidateProfile,
    Captured,
    CoreQuestion,
    Round,
    age_unknown,
    asked_core,
    capture,
    clean_name,
    expired,
    new_profile,
    open_questions,
    stamp,
    view,
    with_capture,
    with_confirmed,
    with_correction,
    with_round,
    without_answer,
    without_old_proposals,
    without_proposal,
)
from orchestrator.domain.events import LiveEvent
from orchestrator.domain.next_round import round_from_sheet

LOG = logging.getLogger(__name__)
INTERVIEWER, CANDIDATE = "interviewer", "candidate"
ANSWER_FINALS = 3
"""Candidate sentences read for an answer before the question is let go."""
ANSWER_WINDOW_S = 30.0
"""Seconds after the question within which the answer must start."""
SUBSTANTIVE_WORDS = 3
"""An interviewer sentence of this many words lets a pending question go."""
SUGGESTION_KEYS = ("text", "example", "say")
"""Fields of a suggestion that may word a question."""


class CandidateBook:
    """Keep, show, correct and delete the candidates of each role.

    Args:
        store: Where the profiles are kept.
        core: The core questions.
        plan_exists: Tells whether a plan (role) exists.
        retention_days: Days a profile is kept after its last change
            (``None``: no limit), from the interview products' privacy;
            a profile with stored interview reports is kept whatever
            its age (#262).
        now: Wall-clock time.
        plans: The plans' stored interview reports (``None``: the file
            lists none and deleting it deletes no report).
    """

    def __init__(
        self,
        store: CandidateStore,
        core: CoreQuestionSource,
        plan_exists: Callable[[str], bool],
        retention_days: Optional[int] = None,
        now: Callable[[], datetime] = datetime.now,
        plans: Optional[PlanStore] = None,
    ):
        """Wire the store; nothing is read yet."""
        self.store = store
        self.core = core
        self.plan_exists = plan_exists
        self.retention_days = retention_days
        self.now = now
        self.plans = plans
        self._lock = threading.Lock()

    # ---- queries -------------------------------------------------------

    def questions(self) -> List[CoreQuestion]:
        """The core questions, in order."""
        return self.core.questions()

    def list(self, plan_id: str) -> Document:
        """The candidates of a role, newest change first."""
        self._check_plan(plan_id)
        self.purge()
        rows = sorted(
            (self._view(p) for p in self.store.list(plan_id)),
            key=lambda r: str(r["updated_at"]), reverse=True,
        )
        questions = self.questions()
        return {
            "plan_id": plan_id, "candidates": rows,
            "questions": [q.model_dump() for q in questions],
            "retention_days": self.retention_days,
        }

    def get(self, plan_id: str, candidate_id: str) -> Document:
        """One candidate with the core questions, answers and reports."""
        return self._view(self._load(plan_id, candidate_id))

    def reports(self, profile: CandidateProfile) -> List[Document]:
        """The candidate's stored interview reports, newest first.

        A report is theirs when it names the profile or its session is
        one of the profile's rounds (older sheets name no profile).
        """
        if self.plans is None:
            return []
        sessions = {r.session_id for r in profile.rounds}
        try:
            found = self.plans.results(profile.plan_id)
        except KeyError:
            return []
        return [
            {"session_id": r.get("session_id"),
             "kind": "live" if r.get("kind") == "live" else "ai",
             "finished_at": r.get("finished_at"), "title": r.get("title"),
             "plan_id": profile.plan_id}
            for r in found
            if r.get("candidate_id") == profile.id
            or r.get("session_id") in sessions
        ]

    def profile(self, plan_id: str, candidate_id: str) -> CandidateProfile:
        """One candidate's profile (``PlanError`` when there is none)."""
        return self._load(plan_id, candidate_id)

    def by_session(self, session_id: str) -> Document:
        """The candidate interviewed in a session (its rounds say it)."""
        for plan_id in self.store.plans():
            for found in self.store.list(plan_id):
                if self._expired(found):
                    continue
                if any(r.session_id == session_id for r in found.rounds):
                    return view(found, self.questions())
        raise PlanError("not_found", f"no candidate for {session_id!r}")

    # ---- commands ------------------------------------------------------

    def create(self, plan_id: str, name: object) -> Document:
        """Add a candidate to a role."""
        self._check_plan(plan_id)
        try:
            clean = clean_name(name)
        except ValueError as exc:
            raise PlanError("invalid_request", str(exc)) from exc
        self.purge()
        with self._lock:
            taken = [p.id for p in self.store.list(plan_id)]
            profile = new_profile(plan_id, clean, taken, self.now())
            self.store.save(profile)
        return view(profile, self.questions())

    def confirm(
        self, plan_id: str, candidate_id: str,
        question_ids: Optional[List[str]] = None,
    ) -> Document:
        """Confirm proposals (``None``: all of them) as answers."""
        with self._lock:
            profile = self._load(plan_id, candidate_id)
            wanted = (list(profile.proposals) if question_ids is None
                      else list(question_ids))
            missing = [q for q in wanted if q not in profile.proposals]
            if missing:
                raise PlanError("not_found",
                                f"no proposal for {missing[0]!r}")
            changed = with_confirmed(profile, wanted, self.now())
            self.store.save(changed)
        return view(changed, self.questions())

    def discard(
        self, plan_id: str, candidate_id: str, question_id: str
    ) -> Document:
        """Discard one proposal (the question stays open)."""
        with self._lock:
            profile = self._load(plan_id, candidate_id)
            if question_id not in profile.proposals:
                raise PlanError("not_found",
                                f"no proposal for {question_id!r}")
            changed = without_proposal(profile, question_id, self.now())
            self.store.save(changed)
        return view(changed, self.questions())

    def forget_new(self, plan_id: str, candidate_id: str) -> None:
        """Delete a profile made for a session that did not start."""
        with self._lock:
            try:
                profile = self.store.load(plan_id, candidate_id)
            except KeyError:
                return
            if not (profile.answers or profile.proposals or profile.rounds):
                self.store.delete(plan_id, candidate_id)

    def correct(
        self, plan_id: str, candidate_id: str, question_id: str,
        text: object,
    ) -> Document:
        """Keep the interviewer's answer to one core question."""
        question = self._question(question_id)
        with self._lock:
            profile = self._load(plan_id, candidate_id)
            try:
                changed = with_correction(profile, question, str(text or ""),
                                          self.now())
            except ValueError as exc:
                raise PlanError("invalid_request", str(exc)) from exc
            self.store.save(changed)
        return view(changed, self.questions())

    def clear(
        self, plan_id: str, candidate_id: str, question_id: str
    ) -> Document:
        """Forget one answer (the question is open again)."""
        self._question(question_id)
        with self._lock:
            profile = self._load(plan_id, candidate_id)
            changed = without_answer(profile, question_id, self.now())
            self.store.save(changed)
        return view(changed, self.questions())

    def keep_report(
        self, plan_id: str, candidate_id: str, save: Callable[[], None],
    ) -> bool:
        """Store a report of a candidate only while their file exists.

        ``save`` runs under the same lock as :meth:`delete`, so a file
        deleted while a judgement ran never gets an orphan report.
        ``False`` (nothing saved) when the file is gone.
        """
        with self._lock:
            if candidate_id not in self.store.ids(plan_id):
                LOG.info("candidate %s/%s gone: report not kept", plan_id,
                         candidate_id)
                return False
            save()  # a file that cannot be read still exists: it is kept
        return True

    def owns(self, plan_id: str) -> Callable[[Document], bool]:
        """Tell, for the reports of a role, which are in a candidate's file.

        A report is in a file when it names a stored profile (readable
        or not) or its session is a round of a readable profile. While a
        profile of the role cannot be read, every report counts as in a
        file: its rounds are unknown, so none is offered for deletion.
        """
        try:
            ids = set(self.store.ids(plan_id))
            readable = self.store.list(plan_id)
        except KeyError:
            return lambda report: False
        unknown = len(ids) > len(readable)
        sessions = {r.session_id for p in readable for r in p.rounds}

        def owned(report: Document) -> bool:
            return unknown or report.get("candidate_id") in ids or \
                report.get("session_id") in sessions

        return owned

    def person_of(self, plan_id: str) -> Callable[[Document], Optional[str]]:
        """Tell, for the reports of a role, whose file each is in (#271).

        The profile id the report names, else the profile one of whose
        rounds is the report's session (older sheets name no profile);
        ``None`` when it is in no file. The offer page joins a person's
        interviews into one row with it.
        """
        try:
            readable = self.store.list(plan_id)
        except KeyError:
            return lambda report: None
        by_session = {r.session_id: p.id for p in readable for r in p.rounds}

        def person(report: Document) -> Optional[str]:
            named = report.get("candidate_id")
            if isinstance(named, str) and named:
                return named
            return by_session.get(str(report.get("session_id") or ""))

        return person

    def delete_unowned_report(self, plan_id: str, session_id: str) -> Document:
        """Delete a stored report that belongs to no candidate's file.

        Only for a report in no file by :meth:`owns` (those go with
        their file): ``conflict`` otherwise.
        """
        if self.plans is None:
            raise PlanError("unavailable", "no stored reports here")
        with self._lock:
            try:
                report = self.plans.result(plan_id, session_id)
            except (KeyError, ValueError) as exc:
                raise PlanError("not_found",
                                f"no report {session_id!r}") from exc
            if self.owns(plan_id)(report):
                raise PlanError(
                    "conflict", "this report is in a candidate's file (or a "
                    "file that cannot be read); it goes with the file")
            try:
                gone = self.plans.delete_result(plan_id, session_id)
            except OSError as exc:
                raise PlanError("delete_failed",
                                "the report could not be deleted") from exc
        return {"deleted": gone, "plan_id": plan_id,
                "session_id": session_id}

    def delete(self, plan_id: str, candidate_id: str) -> Document:
        """Delete a candidate's file: profile, answers, rounds and reports.

        The only way their stored interview reports are deleted (#262).
        The reports go first: if one cannot be deleted the profile is
        kept, so deleting again finishes it.
        """
        with self._lock:
            try:
                profile = self.store.load(plan_id, candidate_id)
            except KeyError as exc:
                raise PlanError("not_found", "no such candidate") from exc
            reports = 0
            for row in self.reports(profile):
                try:
                    reports += int(self.plans.delete_result(  # type: ignore
                        plan_id, str(row["session_id"])))
                except KeyError:
                    continue
                except OSError as exc:
                    raise PlanError(
                        "delete_failed", "a report could not be deleted; "
                        "delete the file again to finish") from exc
            try:
                gone = self.store.delete(plan_id, candidate_id)
            except KeyError as exc:
                raise PlanError("not_found", "no such candidate") from exc
        if not gone:
            raise PlanError("not_found", f"no candidate {candidate_id!r}")
        LOG.info("candidate %s/%s deleted with %d reports", plan_id,
                 candidate_id, reports)
        return {"deleted": True, "plan_id": plan_id, "id": candidate_id,
                "reports_deleted": reports}

    def keep_capture(
        self, plan_id: str, candidate_id: str, question_id: str,
        found: Captured, session_id: str,
    ) -> Optional[CandidateProfile]:
        """Keep a captured answer as a proposal unless it is answered.

        The profile is read again first, so an answer corrected or
        confirmed on another page meanwhile is never overwritten.
        """
        with self._lock:
            try:
                profile = self.store.load(plan_id, candidate_id)
            except KeyError:
                return None
            changed = with_capture(profile, question_id, found, session_id,
                                   self.now())
            if changed is None:
                return None
            self.store.save(changed)
            return changed

    def keep_round(
        self, plan_id: str, candidate_id: str, kept: Round
    ) -> Optional[CandidateProfile]:
        """Add one round to a candidate (``None``: the profile is gone)."""
        with self._lock:
            try:
                profile = self.store.load(plan_id, candidate_id)
            except KeyError:
                return None
            changed = with_round(profile, kept, self.now())
            self.store.save(changed)
            return changed

    def purge(self) -> int:
        """Apply the retention to every profile; return how many went.

        A profile unchanged for longer than the retention is deleted, and
        older unconfirmed proposals are dropped from the others. A
        profile whose last change cannot be read is kept and logged
        (never deleted on a guess). Runs when the orchestrator starts,
        on every new candidate and whenever profiles are listed.
        """
        if self.retention_days is None:
            return 0
        removed = 0
        now = self.now()
        with self._lock:
            for plan_id in self.store.plans():
                for found in self.store.list(plan_id):
                    if age_unknown(found):
                        LOG.warning("candidate %s/%s: unreadable updated_at "
                                    "%r, kept", plan_id, found.id,
                                    found.updated_at)
                        continue
                    if expired(found, now, self.retention_days) and \
                            not self.reports(found):
                        removed += int(self.store.delete(plan_id, found.id))
                        continue
                    trimmed = without_old_proposals(found, now,
                                                    self.retention_days)
                    if trimmed is not None:
                        self.store.save(trimmed)
        return removed

    # ---- internals -----------------------------------------------------

    def _expired(self, profile: CandidateProfile) -> bool:
        """Past the retention, unless it has stored reports (#262)."""
        return expired(profile, self.now(), self.retention_days) and \
            not self.reports(profile)

    def _view(self, profile: CandidateProfile) -> Document:
        """The profile's view with its stored reports."""
        found = view(profile, self.questions())
        found["reports"] = self.reports(profile)
        return found

    def _check_plan(self, plan_id: str) -> None:
        if not self.plan_exists(plan_id):
            raise PlanError("not_found", f"no plan {plan_id!r}")

    def _load(self, plan_id: str, candidate_id: str) -> CandidateProfile:
        try:
            profile = self.store.load(plan_id, candidate_id)
        except KeyError as exc:
            raise PlanError("not_found",
                            f"no candidate {candidate_id!r}") from exc
        if self._expired(profile):
            self.store.delete(plan_id, candidate_id)
            raise PlanError("not_found", f"no candidate {candidate_id!r}")
        return profile

    def _question(self, question_id: str) -> CoreQuestion:
        for question in self.questions():
            if question.id == question_id:
                return question
        raise PlanError("not_found", f"no core question {question_id!r}")


class CoreTracker:
    """Capture a candidate's answers to the core questions in one session.

    Args:
        book: The candidates.
        profile: The candidate's profile when the session starts.
        session_id: The session.
        roles: Speaker id → role (the same mapping the session updates).
        publish: Sends one feed message.
        kind: ``live`` (the live page) or ``bot`` (the AI interviewer).
    """

    def __init__(
        self,
        book: CandidateBook,
        profile: CandidateProfile,
        session_id: str,
        roles: Dict[str, str],
        publish: Callable[[Document], object],
        kind: str = "live",
    ):
        """Nothing asked yet."""
        self.book = book
        self.profile = profile
        self.session_id = session_id
        self.roles = roles
        self.publish = publish
        self.kind = kind
        self.questions = book.questions()
        self.pending: Optional[str] = None
        self.pending_at: Optional[float] = None
        self.heard: List[str] = []
        self.captured: List[str] = []
        self.failed = False
        self.covered: List[str] = []
        self.weak: List[str] = []
        self.revisit: List[str] = []
        self.seconds: Dict[str, float] = {}
        self._lock = threading.Lock()

    @property
    def plan_id(self) -> str:
        """The role."""
        return self.profile.plan_id

    def answered(self) -> Set[str]:
        """The core questions with a confirmed or corrected answer."""
        return set(self.profile.answers)

    def open(self) -> List[CoreQuestion]:
        """The core questions still to ask, in order."""
        return open_questions(self.profile, self.questions)

    def start(self) -> None:
        """Say who the candidate is and what is already answered."""
        self._publish()

    def on_event(self, event: LiveEvent) -> None:
        """Follow the finals of the interviewer and the candidate."""
        u = event.utterance
        if event.type != "final" or u is None:
            return
        role = self.roles.get(u.speaker, "unknown")
        if role in (INTERVIEWER, CANDIDATE):
            self.add_talk(role, u.end - u.start)
        if role == INTERVIEWER:
            self.asked(u.text, u.end)
        elif role == CANDIDATE:
            self.answer(u.text, u.start)

    def asked(self, text: str, at: Optional[float] = None) -> None:
        """An interviewer's sentence on the live page.

        A core question still open waits for the candidate's answer;
        any other substantive sentence (an instruction, another
        question) lets a waiting one go.
        """
        with self._lock:
            found = asked_core(text, self.open())
            if found is not None:
                self._wait(found, at)
            elif len(text.split()) >= SUBSTANTIVE_WORDS:
                self._release()

    def expect(self, question_id: Optional[str],
               at: Optional[float] = None) -> None:
        """The AI interviewer asked ``question_id`` (``None``: another thing).

        The bot's own words are not matched: it says which core question
        its turn asks, so a reworded question still counts.
        """
        with self._lock:
            if question_id is not None and question_id not in self.answered():
                self._wait(question_id, at)
            else:
                self._release()

    def answer(self, text: str, at: Optional[float] = None) -> None:
        """A candidate's sentence: read it when a core question waits.

        Only the sentences right after the question count: at most
        :data:`ANSWER_FINALS`, starting within :data:`ANSWER_WINDOW_S`.
        """
        with self._lock:
            pending = self.pending
            if pending is None:
                return
            if (at is not None and self.pending_at is not None
                    and at - self.pending_at > ANSWER_WINDOW_S):
                self._release()
                return
            self.heard.append(text)
            found = capture(pending, " ".join(self.heard))
            if found is None:
                if len(self.heard) >= ANSWER_FINALS:
                    self._release()
                return
            self._release()
        self._keep(pending, found)

    def _wait(self, question_id: str, at: Optional[float]) -> None:
        self.pending, self.pending_at, self.heard = question_id, at, []

    def _release(self) -> None:
        self.pending, self.pending_at, self.heard = None, None, []

    def screen(self, suggestions: List[Document]) -> List[Document]:
        """Drop the suggestions that ask a core question already answered.

        Only confirmed or corrected answers count; each field of a
        suggestion is matched on its own.
        """
        answered = [q for q in self.questions if q.id in self.answered()]
        if not answered:
            return suggestions
        return [item for item in suggestions if not any(
            asked_core(str(item.get(k) or ""), answered)
            for k in SUGGESTION_KEYS)]

    def add_talk(self, role: str, seconds: float) -> None:
        """Count ``seconds`` of speech of ``role`` (the talk balance)."""
        if seconds > 0:
            with self._lock:
                self.seconds[role] = self.seconds.get(role, 0.0) + seconds

    def note_sheet(self, sheet: Document) -> None:
        """Take the covered and weak slots of the AI interviewer's sheet."""
        self.covered, self.weak = round_from_sheet(sheet)

    def talk(self) -> Dict[str, float]:
        """Each role's share of the speaking time (0–1, three decimals)."""
        with self._lock:
            total = sum(self.seconds.values())
            return {role: round(s / total, 3)
                    for role, s in sorted(self.seconds.items())} \
                if total > 0 else {}

    def finish(self) -> None:
        """Note this session as one of the candidate's rounds.

        The round keeps the slots covered and left weak (for the next
        round's plan) and the talk balance; never a rating.
        """
        done = self.book.keep_round(self.plan_id, self.profile.id, Round(
            session_id=self.session_id, at=stamp(self.book.now()),
            kind="bot" if self.kind == "bot" else "live",
            covered=list(self.covered), weak=list(self.weak),
            revisit=list(self.revisit), talk=self.talk(),
        ))
        if done is not None:
            self.profile = done

    def _keep(self, question_id: str, found: Captured) -> None:
        try:
            changed = self.book.keep_capture(
                self.plan_id, self.profile.id, question_id, found,
                self.session_id,
            )
        except OSError as exc:
            if not self.failed:
                self.failed = True
                self.publish({"type": "notice", "code": "core_not_kept",
                              "message": f"answer not kept: {exc}"[:200]})
            return
        if changed is None:
            return
        self.profile = changed
        self.captured.append(question_id)
        self._publish(question_id)

    def _publish(self, captured: Optional[str] = None) -> None:
        message: Document = {
            "type": "core", "profile": view(self.profile, self.questions),
        }
        if captured:
            message["captured"] = captured
        self.publish(message)
