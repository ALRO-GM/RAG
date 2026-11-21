"""The interviewer's agenda: which slot to ask next, and when to close.

:class:`Agenda` follows an
:class:`~orchestrator.domain.interview.InterviewPlan`
through the interview. The model only *words* each turn; what the turn
does is decided here, so it can be tested without a model:

* ``open``: greet and ask the first slot (``minimo`` slots first, in
  plan order).
* After an answer to the slot asked: ``complete`` fills it and the
  agenda moves on; the candidate saying they do not know the topic
  marks it ``unknown`` («no lo conoce») and moves on; otherwise at most
  ``max_probes`` follow-ups (``probe``, one by default). When the
  follow-ups did not cover it, ``clarify`` offers the choice: refine
  the answer or go to the next question. A skip marks it ``skipped``;
  after the clarification the agenda moves on whatever the answer.
* A question of the candidate is answered (``answer``) and the slot is
  asked again, without spending an attempt (once per slot).
* ``close`` when every ``minimo`` slot is settled (filled, partial after
  its attempts, skipped or unknown) and there is no time for the
  ``deseable`` (30 % of the budget left) or ``opcional`` (50 %) ones,
  or when the time is up, or after :data:`MAX_TURNS` turns.

:func:`heuristic_fill` is the cheap slot filler (no model): it says
whether an answer covers the slot asked, from its length and markers of
a concrete first-person example (past-tense verbs, figures, a result).
Plain data and functions: no I/O.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Set

from orchestrator.domain.interview import (
    QUOTE_CHARS,
    Answer,
    InterviewPlan,
    SheetSlot,
    Slot,
    fold,
)

Action = Literal["open", "ask", "probe", "clarify", "answer", "close"]
FillStatus = Literal["complete", "partial"]
MAX_TURNS = 40
"""Turns after which the interview closes whatever is left."""
DESEABLE_TIME, OPCIONAL_TIME = 0.30, 0.50
"""Share of the time budget that must be left to ask a deseable or an
opcional slot once the minimos are settled."""
QUOTE_WORDS = 30

_UNKNOWN = re.compile(
    r"\b(no (lo|la|los|las) (conozco|domino|he usado|he utilizado|he "
    r"tocado)|no se (nada|que es|lo que es|de eso|sobre eso)|desconozco|"
    r"no tengo (ni idea|experiencia (con|en|de))|nunca (he|lo he|la he) "
    r"(trabajado|usado|utilizado|tocado)|no sabria (decirte|responder|"
    r"contestar)|no he trabajado (nunca )?(con|en) (eso|esto|ello))"
)
_SKIP = re.compile(
    r"\b(pasamos|pasemos|pasar a la siguiente|siguiente pregunta|"
    r"(la )?siguiente,|saltar|saltemos|sigamos|prefiero (pasar|seguir|no "
    r"contestar)|mejor (pasamos|seguimos|sigamos)|lo dejamos)"
)
_PAST = re.compile(
    r"\b(\w{2,}(é|í)|hice|estuve|tuve|fui|puse|propuse|dije|traje|"
    r"pude|supe|quise|conduje|hicimos|estuvimos|tuvimos|fuimos)\b"
)
_CONCRETE = re.compile(
    r"(\d|\b(anos?|meses|semanas|por ejemplo|una vez|en mi (anterior|"
    r"ultimo|antiguo)|cuando trabajaba|mi equipo y yo|me encargaba|"
    r"me encargue|era responsable|certificado|titulo|carne|nivel)\b)"
)
_RESULT = re.compile(
    r"\b(al final|consegui|conseguimos|logre|logramos|resultado|se "
    r"resolvio|funciono|redujimos|reduje|mejoro|mejoramos|gracias a eso|"
    r"desde entonces|aprendi|acabamos|terminamos|cerramos|quedo "
    r"(resuelto|contento|satisfecho))"
)
_STOP = frozenset(
    "sobre entre desde hasta tienes puedes cuando donde porque tambien "
    "alguna alguno algun experiencia cuentame cuenta trabajo puesto "
    "persona personas hiciste ejemplo concreto concreta respuesta "
    "situacion".split()
)


def says_unknown(text: str) -> bool:
    """Tell whether the candidate says they do not know the topic."""
    folded = fold(text)
    if _UNKNOWN.search(folded):
        return True
    words = folded.replace(",", " ").replace(".", " ").split()
    return len(words) <= 6 and "no" in words and "se" in words


def wants_skip(text: str) -> bool:
    """Tell whether the candidate asks to go to the next question."""
    return bool(_SKIP.search(fold(text)))


def is_question(text: str) -> bool:
    """Tell whether the candidate asked something (ends with ``?``)."""
    return text.strip().endswith("?")


def quote(text: str) -> str:
    """Return the first words of ``text`` as a quote."""
    words = text.split()
    cut = " ".join(words[:QUOTE_WORDS])
    return (cut + (" …" if len(words) > QUOTE_WORDS else ""))[:QUOTE_CHARS]


def _keywords(text: str) -> Set[str]:
    """Return the content words of ``text`` (folded, 5 letters or more)."""
    return {
        w for w in re.findall(r"[a-z0-9+#]{5,}", fold(text))
        if w not in _STOP
    }


@dataclass(frozen=True)
class Fill:
    """One slot an answer gives evidence for."""

    slot: str
    status: FillStatus
    quote: str


def heuristic_fill(
    text: str, target: Optional[Slot], slots: List[Slot]
) -> List[Fill]:
    """Say which slots ``text`` covers, without a model.

    The slot asked (``target``) is covered when the answer is on it at
    all: ``complete`` for a competency when it has 25 words or more, a
    first-person past-tense action and a result; for a requirement
    when it has 10 words or more and a figure, a concrete marker or a
    word of the slot. Shorter or vaguer answers are ``partial``. Any
    other slot sharing two content words with a 15-word answer is
    ``partial``.
    """
    folded = fold(text)
    words = len(text.split())
    if words < 3 or says_unknown(text):
        return []
    found: List[Fill] = []
    said = quote(text)
    if target is not None:
        past = bool(_PAST.search(text.lower()))
        if target.kind == "competency":
            done = (words >= 25 and past
                    and bool(_RESULT.search(folded)))
            if done or words >= 8:
                found.append(Fill(target.id,
                                  "complete" if done else "partial", said))
        else:
            topic = _keywords(
                " ".join([target.label, target.evidence, *target.questions])
            )
            done = words >= 10 and (
                bool(_CONCRETE.search(folded)) or past
                or len(topic & _keywords(text)) >= 1
            )
            if done or words >= 4:
                found.append(Fill(target.id,
                                  "complete" if done else "partial", said))
    if words >= 15:
        heard = _keywords(text)
        for slot in slots:
            if target is not None and slot.id == target.id:
                continue
            if len(heard & _keywords(slot.label + " " + slot.evidence)) >= 2:
                found.append(Fill(slot.id, "partial", said))
    return found


@dataclass(frozen=True)
class Step:
    """What the interviewer does next."""

    action: Action
    slot: Optional[Slot] = None
    missing: str = ""
    minutes_left: float = 0.0


_MISSING_COMPETENCY = (
    "un ejemplo real: qué hizo él o ella exactamente y cómo acabó"
)
_MISSING_REQUIREMENT = "un dato concreto: dónde, cuánto tiempo, qué hacía"
_DONE = ("filled", "skipped", "unknown")


class Agenda:
    """Follow one interview's plan turn by turn.

    Args:
        plan: The plan being covered.
        started: Monotonic seconds when the interview started.
        max_probes: Follow-ups per slot before the clarification offer.
    """

    def __init__(
        self, plan: InterviewPlan, started: float, max_probes: int = 1
    ):
        """Every slot starts empty; nothing has been asked."""
        self.plan = plan
        self.started = started
        self.max_probes = max(0, max_probes)
        self.slots: Dict[str, Slot] = {s.id: s for s in plan.slots}
        self.state: Dict[str, SheetSlot] = {
            s.id: SheetSlot(id=s.id, label=s.label, level=s.level,
                            weight=s.weight, competency=s.competency)
            for s in plan.slots
        }
        self.order = [s.id for s in plan.ordered()]
        self.target: Optional[str] = None
        self.stage: Literal["none", "asked", "probed", "clarify"] = "none"
        self.answered_questions: Set[str] = set()
        self.pending_question = False
        self.opened = False
        self.closed_by: Optional[str] = None
        self.turns = 0
        self.turns_to_minimums: Optional[int] = None

    # ---- time -------------------------------------------------------

    def minutes_left(self, now: float) -> float:
        """Minutes of the budget still left (never below 0)."""
        used = (now - self.started) / 60.0
        return max(0.0, round(self.plan.minutes - used, 1))

    # ---- answers ----------------------------------------------------

    def answer(
        self, utterance: str, text: str, fills: List[Fill]
    ) -> None:
        """Take the candidate's answer to the slot asked."""
        target = self.target
        self.pending_question = False
        for fill in fills:
            self._record(fill, utterance)
        if target is None or self.closed_by is not None:
            return
        state = self.state[target]
        if state.status in _DONE:
            return
        if says_unknown(text):
            state.status = "unknown"
            return
        if self.stage == "clarify":
            if wants_skip(text) and state.status != "filled":
                state.status = "skipped"
            elif state.status == "empty":
                state.status = "skipped"
            self.stage = "none"
            return
        covered = any(f.slot == target for f in fills)
        if (not covered and is_question(text)
                and target not in self.answered_questions):
            self.pending_question = True

    def _record(self, fill: Fill, utterance: str) -> None:
        """Add one fill to its slot's answers and status."""
        state = self.state.get(fill.slot)
        if state is None or state.status in ("skipped", "unknown"):
            return
        state.answers.append(Answer(
            utterance=utterance, quote=fill.quote[:QUOTE_CHARS],
            status=fill.status,
        ))
        if fill.status == "complete":
            state.status = "filled"
        elif state.status == "empty":
            state.status = "partial"

    # ---- decisions --------------------------------------------------

    def minimums_settled(self) -> bool:
        """Tell whether every minimo slot needs nothing more."""
        return all(
            self._settled(sid) for sid in self.order
            if self.slots[sid].level == "minimo"
        )

    def _settled(self, sid: str) -> bool:
        """A slot is settled when done, or asked and left (not current)."""
        state = self.state[sid]
        if state.status in _DONE:
            return True
        return state.asked > 0 and sid != self.target

    def step(self, now: float) -> Step:
        """Decide the next turn and count it."""
        left = self.minutes_left(now)
        self.turns += 1
        if self.closed_by is not None:
            return Step("close", minutes_left=left)
        if not self.opened:
            self.opened = True
            first = self._next_slot(left)
            return self._ask(first, "open", left)
        if left <= 0 or self.turns > MAX_TURNS:
            return self._close("time", left)
        target = self.target
        if target is not None:
            state = self.state[target]
            slot = self.slots[target]
            if self.pending_question:
                self.answered_questions.add(target)
                return Step("answer", slot, minutes_left=left)
            if state.status not in _DONE and self.stage != "none":
                if self.stage == "asked" and state.probes < self.max_probes:
                    state.probes += 1
                    return Step("probe", slot, self._missing(slot), left)
                if self.stage in ("asked", "probed"):
                    self.stage = "clarify"
                    return Step("clarify", slot, self._missing(slot), left)
            self.target = None
            self.stage = "none"
        self._note_minimums()
        nxt = self._next_slot(left)
        if nxt is None:
            return self._close("covered", left)
        return self._ask(nxt, "ask", left)

    def _note_minimums(self) -> None:
        """Remember the turn at which the minimos were first settled."""
        if self.turns_to_minimums is None and self.minimums_settled():
            self.turns_to_minimums = self.turns - 1

    def _ask(self, slot: Optional[Slot], action: Action, left: float) -> Step:
        """Ask ``slot`` (open with it), or close when there is none."""
        if slot is None:
            return self._close("covered", left)
        state = self.state[slot.id]
        state.asked += 1
        self.target = slot.id
        self.stage = "asked"
        if state.probes >= self.max_probes:
            self.stage = "probed"
        return Step(action, slot, minutes_left=left)

    def _close(self, why: str, left: float) -> Step:
        """Close the interview."""
        self.closed_by = why
        self.target = None
        self._note_minimums()
        return Step("close", minutes_left=left)

    def _next_slot(self, left: float) -> Optional[Slot]:
        """The next slot to ask, by level and plan order, time allowing."""
        share = left / self.plan.minutes if self.plan.minutes else 0.0
        for sid in self.order:
            slot = self.slots[sid]
            state = self.state[sid]
            if state.status in _DONE or state.asked > 0:
                continue
            if slot.level == "deseable" and share < DESEABLE_TIME:
                continue
            if slot.level == "opcional" and share < OPCIONAL_TIME:
                continue
            if slot.level != "minimo" and not self.minimums_settled():
                continue
            return slot
        return None

    @staticmethod
    def _missing(slot: Slot) -> str:
        """What a follow-up asks for."""
        if slot.kind == "competency":
            return _MISSING_COMPETENCY
        return _MISSING_REQUIREMENT

    # ---- results ----------------------------------------------------

    def sheet_slots(self) -> List[SheetSlot]:
        """The slots' states in plan order (copies)."""
        return [self.state[s.id].model_copy(deep=True)
                for s in self.plan.slots]
