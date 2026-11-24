"""Use cases: the candidate profile across interview rounds (#235).

:class:`CandidateBook` keeps the candidates of each role (an interview
plan): it creates them, shows them with the core questions of KN-4,
takes the interviewer's corrections, deletes a candidate on request and
drops the profiles older than the retention of the interview products
(``privacy.store_text_days``). :class:`CoreTracker` follows one session
(the live page or the AI interviewer): when the interviewer asks a core
question that is still open, it reads the candidate's next answer with
the fixed rules of :mod:`orchestrator.domain.candidate` and keeps what
they are sure of. Each change is published as ``{"type": "core",
"profile": ...}``; answered questions are screened out of the live
suggestions.

No model is called here: capturing is deterministic.
"""

import threading
from datetime import datetime
from typing import Callable, Dict, List, Optional, Set

from orchestrator.application.interview import PlanError
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
    asked_core,
    capture,
    clean_name,
    expired,
    new_profile,
    open_questions,
    stamp,
    view,
    with_capture,
    with_correction,
    with_round,
    without_answer,
)
from orchestrator.domain.events import LiveEvent
from orchestrator.domain.next_round import round_from_sheet

INTERVIEWER, CANDIDATE = "interviewer", "candidate"
ANSWER_FINALS = 3
"""Candidate sentences read for an answer before the question is let go."""
SUGGESTION_KEYS = ("text", "example", "say")
"""Fields of a suggestion that may word a question."""


class CandidateBook:
    """Keep, show, correct and delete the candidates of each role.

    Args:
        store: Where the profiles are kept.
        core: The core questions.
        plan_exists: Tells whether a plan (role) exists.
        retention_days: Days a profile is kept after its last change
            (``None``: no limit), from the interview products' privacy.
        now: Wall-clock time.
    """

    def __init__(
        self,
        store: CandidateStore,
        core: CoreQuestionSource,
        plan_exists: Callable[[str], bool],
        retention_days: Optional[int] = None,
        now: Callable[[], datetime] = datetime.now,
    ):
        """Wire the store; nothing is read yet."""
        self.store = store
        self.core = core
        self.plan_exists = plan_exists
        self.retention_days = retention_days
        self.now = now
        self._lock = threading.Lock()

    # ---- queries -------------------------------------------------------

    def questions(self) -> List[CoreQuestion]:
        """The core questions, in order."""
        return self.core.questions()

    def list(self, plan_id: str) -> Document:
        """The candidates of a role, newest change first."""
        self._check_plan(plan_id)
        self.purge()
        questions = self.questions()
        rows = sorted(
            (view(p, questions) for p in self.store.list(plan_id)),
            key=lambda r: str(r["updated_at"]), reverse=True,
        )
        return {
            "plan_id": plan_id, "candidates": rows,
            "questions": [q.model_dump() for q in questions],
            "retention_days": self.retention_days,
        }

    def get(self, plan_id: str, candidate_id: str) -> Document:
        """One candidate with the core questions and their answers."""
        return view(self._load(plan_id, candidate_id), self.questions())

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
        with self._lock:
            taken = [p.id for p in self.store.list(plan_id)]
            profile = new_profile(plan_id, clean, taken, self.now())
            self.store.save(profile)
        return view(profile, self.questions())

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

    def delete(self, plan_id: str, candidate_id: str) -> Document:
        """Delete a candidate's profile (answers and rounds)."""
        with self._lock:
            try:
                gone = self.store.delete(plan_id, candidate_id)
            except KeyError as exc:
                raise PlanError("not_found", "no such candidate") from exc
        if not gone:
            raise PlanError("not_found", f"no candidate {candidate_id!r}")
        return {"deleted": True, "plan_id": plan_id, "id": candidate_id}

    def keep_capture(
        self, plan_id: str, candidate_id: str, question_id: str,
        found: Captured, session_id: str,
    ) -> Optional[CandidateProfile]:
        """Keep a captured answer unless the question has one already.

        The profile is read again first, so an answer corrected on
        another page meanwhile is never overwritten.
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
        """Delete the profiles older than the retention; return how many."""
        if self.retention_days is None:
            return 0
        removed = 0
        now = self.now()
        with self._lock:
            for plan_id in self.store.plans():
                for found in self.store.list(plan_id):
                    if expired(found, now, self.retention_days):
                        removed += int(self.store.delete(plan_id, found.id))
        return removed

    # ---- internals -----------------------------------------------------

    def _expired(self, profile: CandidateProfile) -> bool:
        return expired(profile, self.now(), self.retention_days)

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
        self.heard: List[str] = []
        self.captured: List[str] = []
        self.failed = False
        self.covered: List[str] = []
        self.weak: List[str] = []
        self.seconds: Dict[str, float] = {}
        self._lock = threading.Lock()

    @property
    def plan_id(self) -> str:
        """The role."""
        return self.profile.plan_id

    def answered(self) -> Set[str]:
        """The core questions with a kept answer."""
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
            self.asked(u.text)
        elif role == CANDIDATE:
            self.answer(u.text)

    def asked(self, text: str) -> None:
        """An interviewer's sentence: maybe a core question still open."""
        with self._lock:
            found = asked_core(text, self.open())
            if found is not None:
                self.pending, self.heard = found, []
            elif self.pending is not None and "?" in text:
                self.pending, self.heard = None, []

    def answer(self, text: str) -> None:
        """A candidate's sentence: read it when a core question waits."""
        with self._lock:
            pending = self.pending
            if pending is None:
                return
            self.heard.append(text)
            found = capture(pending, " ".join(self.heard))
            if found is None:
                if len(self.heard) >= ANSWER_FINALS:
                    self.pending, self.heard = None, []
                return
            self.pending, self.heard = None, []
        self._keep(pending, found)

    def screen(self, suggestions: List[Document]) -> List[Document]:
        """Drop the suggestions that ask a core question already answered."""
        answered = [q for q in self.questions if q.id in self.answered()]
        if not answered:
            return suggestions
        kept = []
        for item in suggestions:
            words = " ".join(str(item.get(k) or "") for k in SUGGESTION_KEYS)
            if asked_core(words, answered) is None:
                kept.append(item)
        return kept

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
            talk=self.talk(),
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
