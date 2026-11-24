"""Use case: the role's topics in a live-page round (#235).

A live-page interview with a role and a candidate shows the plan's slots
as its topics (``{"type": "topics", "topics": [...]}``), in the next
round's order: what earlier rounds left weak first, then what is still
open, then what is covered (each chip says which with ``earlier``).
During the round a slot is marked done when the interviewer asked about
it (two content words of the slot, or its only one) and the candidate's
answer covers it by the cheap filler of the AI interviewer
(:func:`~orchestrator.domain.interview_agenda.heuristic_fill`,
``complete``); a slot asked but not covered that way is weak for the
next round. No model
is called and nothing here rates the person.
"""

import re
import threading
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from orchestrator.application.ports import Document
from orchestrator.domain.events import LiveEvent
from orchestrator.domain.interview import InterviewPlan, Slot, fold
from orchestrator.domain.interview_agenda import heuristic_fill
from orchestrator.domain.next_round import TopicState

INTERVIEWER, CANDIDATE = "interviewer", "candidate"
_STOP = frozenset(
    "sobre entre desde hasta tienes puedes cuando donde porque tambien "
    "alguna alguno algun cuentame cuenta puesto persona personas "
    "hiciste ejemplo concreto concreta respuesta situacion".split()
)


def keywords(text: str) -> Set[str]:
    """The content words of ``text`` (folded, five letters or more)."""
    return {w for w in re.findall(r"[a-z0-9+#]{5,}", fold(text))
            if w not in _STOP}


def slot_asked(text: str, slots: Sequence[Slot]) -> Optional[Slot]:
    """The slot an interviewer's sentence asks about, if one clearly.

    Two shared content words with the slot's label and questions (one
    when the label has a single content word); the slot sharing most
    wins, ties go to the plan's order.
    """
    heard = keywords(text)
    best: Optional[Slot] = None
    best_n = 0
    for slot in slots:
        label = keywords(slot.label)
        topic = label | keywords(" ".join(slot.questions))
        shared = len(heard & topic)
        needed = 1 if len(label) == 1 and heard & label else 2
        if shared >= needed and shared > best_n:
            best, best_n = slot, shared
    return best


class TopicTracker:
    """Follow the role's topics through one live-page round.

    Args:
        plan: The role's plan.
        states: The slots with what earlier rounds left of them.
        roles: Speaker id → role (the session's mapping).
        publish: Sends one feed message.
    """

    def __init__(
        self,
        plan: InterviewPlan,
        states: Sequence[TopicState],
        roles: Dict[str, str],
        publish: Callable[[Document], object],
    ):
        """Nothing asked yet."""
        self.plan = plan
        self.states = list(states)
        self.roles = roles
        self.publish = publish
        self.target: Optional[Slot] = None
        self.covered: Set[str] = set()
        self.touched: Set[str] = set()
        self._lock = threading.Lock()

    def start(self) -> None:
        """Show the topics in the next round's order."""
        self._publish()

    def on_event(self, event: LiveEvent) -> None:
        """Follow the interviewer's questions and the candidate's answers."""
        u = event.utterance
        if event.type != "final" or u is None:
            return
        role = self.roles.get(u.speaker, "unknown")
        changed = False
        with self._lock:
            if role == INTERVIEWER:
                asked = slot_asked(u.text, self.plan.slots)
                if asked is not None:
                    self.target = asked
                    self.touched.add(asked.id)
            elif role == CANDIDATE and self.target is not None:
                for fill in heuristic_fill(u.text, self.target,
                                           self.plan.slots):
                    if fill.slot != self.target.id:
                        continue
                    self.touched.add(fill.slot)
                    if fill.status == "complete" and \
                            fill.slot not in self.covered:
                        self.covered.add(fill.slot)
                        changed = True
        if changed:
            self._publish()

    def result(self) -> Tuple[List[str], List[str]]:
        """The covered and the weak slot ids of this round."""
        with self._lock:
            order = [s.id for s in self.plan.slots]
            covered = [s for s in order if s in self.covered]
            weak = [s for s in order
                    if s in self.touched and s not in self.covered]
        return covered, weak

    def _publish(self) -> None:
        self.publish({"type": "topics", "topics": [
            t.to_topic(t.slot.id in self.covered) for t in self.states
        ]})
