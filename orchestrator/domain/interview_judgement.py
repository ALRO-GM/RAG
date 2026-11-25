"""How accurate a candidate's answers were, per point and concept (#259).

The report template (``interview.report@2``) judges each answered slot
of a sheet, and each treated concept of it, with one label of
:data:`~orchestrator.domain.interview.JUDGE_LABELS`, the candidate's
quote and a one-sentence reason, and lists the statements worth
checking. This module reads that reply defensively and applies the
rules that do not depend on the model:

* an unknown label becomes ``unclear``; an utterance id that is not one
  of the candidate's answers to that slot is dropped, and the quote is
  always the candidate's own words;
* a claim about the candidate's own experience («trabajé tres años en
  X») cannot be checked from an interview: it is ``to_verify``, never
  ``wrong``, whatever the model says;
* a reason or note that reads as a verdict on the person («miente», «se
  lo inventa», «apto / no apto», «contratar») is dropped: the report
  judges answers, never the person;
* the **score rule** (:func:`capped_score`): the model's 0–100 score of
  a slot is capped by its accuracy, so a wrong answer never scores as
  covered: ``wrong`` 30, ``off_topic`` 20, ``unclear`` 50,
  ``correct_basic`` 75, and 50 when any treated concept is ``wrong``;
  ``solid`` and ``to_verify`` keep the model's score. The weighted
  suitability is computed from the capped scores, as before.

Never said or shown to the candidate. Plain data and functions: no I/O.
"""

import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from orchestrator.domain.interview import (
    JUDGE_LABELS,
    QUOTE_CHARS,
    REASON_CHARS,
    CheckItem,
    JudgeLabel,
    Judgement,
    SheetSlot,
    fold,
)

CAPS: Dict[str, int] = {
    "wrong": 30, "off_topic": 20, "unclear": 50, "correct_basic": 75,
}
"""Highest score a slot keeps for its judgement (``solid``,
``to_verify``: no cap)."""
WRONG_POINT_CAP = 50
"""Highest score of a slot with a treated concept judged ``wrong``."""
MAX_CHECKS = 10
MIN_QUOTE_WORDS = 3
"""Words a quote needs to find its utterance without an id."""
COUNT_KEYS: Tuple[str, ...] = (*JUDGE_LABELS, "not_treated", "does_not_know")

_SYNONYMS = {
    "basic": "correct_basic", "correct": "correct_basic",
    "superficial": "correct_basic", "incorrect": "wrong",
    "offtopic": "off_topic", "off-topic": "off_topic",
    "unverifiable": "to_verify", "verify": "to_verify",
}
_EXPERIENCE = re.compile(
    r"\b(trabaje|trabajaba|he trabajado|estuve|estaba (en|de|como)|"
    r"llevo \w+ (anos?|meses)|fui (jefe|jefa|encargad[oa]|responsable)|"
    r"me encargaba|me encargue|he sido|era (jefe|jefa|responsable)|"
    r"tengo (\w+ )?(anos?|meses) de experiencia|mi (anterior|ultimo) "
    r"(puesto|trabajo|empleo)|en mi (anterior|ultimo) (puesto|trabajo|"
    r"empleo)|dirigi|lidere|gestione"
    r"|i worked|i've worked|i have worked|i was (a|an|the|in charge|"
    r"responsible)|i used to (work|be|run|manage|lead)|i led|i managed|"
    r"i ran|i've been (a|an|working)|i have been (a|an|working)|"
    r"i spent \w+ (years?|months?) (at|in|as|with)|"
    r"worked (at|in|for) (\w+ ){1,4}for \w+ (years?|months?)|"
    r"he estado \w+ (anos?|meses) (en|de|como)|"
    r"llevo (mas de |casi )?\w+ (anos?|meses) (de|como|en|trabajando)|"
    r"(\w+ )?(years?|months?) of experience|in my (last|previous|former) "
    r"(job|role|position|company))\b"
)
"""First-person claims about one's own experience, in Spanish and
English (accents folded)."""
_VERDICT = re.compile(
    r"\b(miente|mienta|mientan|mintio|mentia|mintiendo|ha mentido|"
    r"nos miente|(es )?(un|una) mentiros[oa]|mentiros[oa]|"
    r"se lo (ha )?inventad[oa]|se lo inventa|se lo invento|"
    r"(se )?(ha|habria) inventado (su|la|lo|el|esto|eso)|"
    r"(esta |estaria )?inventando|"
    r"inventa(ndo|do|ba)? (los|las|el|la|datos|cifras|todo|eso|esto)|"
    r"fabric\w+ (la|el|los|las|that|it|this|the answer)|"
    r"(no )?(l[aoe]s? )?(contrataria|descartaria|rechazaria)|"
    r"dishonest|invented (that|it|this)|"
    r"no es apt[oa] para el puesto|(es )?apt[oa] para el puesto|"
    r"(no )?(es|seria) (un|una) (buen|buena|mal|mala) candidat[oa]|"
    r"(no )?(contratar|contratarl[aoe]s?|descartarl[aoe]s?|"
    r"rechazarl[aoe]s?)|(no )?(recomiendo|aconsejo) (contratar|"
    r"descartar|rechazar)|"
    r"is lying|was lying|lied about|is a liar|made (it|this|that) up|"
    r"fabricated (it|this|that|the|his|her|their)|"
    r"(not )?suitable for the (role|job|position)|(do not |don't )?"
    r"hire (him|her|them|this candidate)|should (not )?be hired|"
    r"reject (him|her|them|this candidate))\b"
)
"""Phrases of a verdict on the person (accents folded), not on an
answer: «miente», «se lo ha inventado», «no es apto para el puesto»,
«contratarla», "is lying", "made it up", "not suitable for the role".
Words that only look alike («inventario», «apto para alimentos»,
«rechazo de pedidos», "suitable for outdoor use") are kept."""


def _str(value: object) -> str:
    """A reply field as text: strings and numbers, anything else empty."""
    if isinstance(value, bool):
        return ""
    if isinstance(value, (str, int, float)):
        return str(value)
    return ""


def clean_text(value: object, limit: int = REASON_CHARS) -> str:
    """One trimmed line, or empty when it reads as a verdict on the person."""
    text = " ".join(_str(value).split())[:limit]
    return "" if _VERDICT.search(fold(text)) else text


def label_of(value: object) -> Optional[JudgeLabel]:
    """The judgement label of a model's value (unknown: ``unclear``).

    ``None`` when there is no value at all.
    """
    if value is None or value == "":
        return None
    word = _str(value).strip().lower().replace(" ", "_")
    word = _SYNONYMS.get(word, word)
    known: Dict[str, JudgeLabel] = {label: label for label in JUDGE_LABELS}
    return known.get(word, "unclear")


def is_experience(text: str) -> bool:
    """Whether ``text`` is a claim about the candidate's own experience."""
    return bool(_EXPERIENCE.search(fold(text)))


def judgement_from(
    item: object, said: Mapping[str, str], chunks: Set[str]
) -> Optional[Judgement]:
    """Read one judgement of the model against what the candidate said.

    Args:
        item: The model's ``{"label", "reason", "utterance", "quote",
            "kind", "cites"}``.
        said: The candidate's utterances for this slot (id → text).
        chunks: Ids of the chunks the judge was given for this slot.
    """
    if not isinstance(item, dict):
        return None
    label = label_of(item.get("label"))
    if label is None:
        return None
    utterance = _str(item.get("utterance"))
    quote = " ".join(_str(item.get("quote")).split())
    if utterance not in said:
        # Without a known id the quote names its utterance only when it
        # is long enough to be telling (a single letter matches anything).
        telling = len(quote.split()) >= MIN_QUOTE_WORDS
        utterance = next((u for u, text in said.items() if telling and fold(
            quote).strip(" .…\"'«»") in fold(text)), "")
    if utterance:
        text = said[utterance]
        if not quote or fold(quote).strip(" .…\"'«»") not in fold(text):
            quote = text
    else:
        quote = ""
    experience = _str(item.get("kind")) == "experience" or (
        bool(quote) and is_experience(quote))
    if label == "wrong" and experience:
        label = "to_verify"
    raw_cites = item.get("cites")
    cites = [c for c in (raw_cites if isinstance(raw_cites, list) else [])
             if isinstance(c, str) and c in chunks][:4]
    return Judgement(
        label=label, reason=clean_text(item.get("reason")),
        quote=quote[:QUOTE_CHARS], utterance=utterance or None,
        basis="documents" if cites else "general", cites=cites,
    )


def capped_score(score: Optional[int], slot: SheetSlot) -> Optional[int]:
    """The slot's score after the accuracy caps (``None`` stays)."""
    if score is None:
        return None
    cap = 100
    if slot.judgement is not None:
        cap = CAPS.get(slot.judgement.label, 100)
    if any(p.judgement is not None and p.judgement.label == "wrong"
           for p in slot.points):
        cap = min(cap, WRONG_POINT_CAP)
    return min(score, cap)


def accuracy_counts(slot: SheetSlot) -> Dict[str, int]:
    """How many of the slot's concepts got each label or status.

    A slot without concepts counts its own judgement. Treated concepts
    without a judgement are not counted under any label.
    """
    found = {k: 0 for k in COUNT_KEYS}
    if not slot.points:
        if slot.judgement is not None:
            found[slot.judgement.label] += 1
        return {k: v for k, v in found.items() if v}
    for point in slot.points:
        if point.status == "pending":
            found["not_treated"] += 1
        elif point.status == "unknown":
            found["does_not_know"] += 1
        elif point.judgement is not None:
            found[point.judgement.label] += 1
    return {k: v for k, v in found.items() if v}


def judge_slots(
    data: Mapping[str, Any],
    slots: Sequence[SheetSlot],
    chunks: Mapping[str, Sequence[str]],
    texts: Optional[Mapping[str, str]] = None,
) -> Tuple[List[SheetSlot], List[CheckItem]]:
    """Merge the report's judgements into the slots; read the checks.

    Only answered slots and treated concepts are judged; the scores are
    capped by :func:`capped_score` and every ``wrong`` judgement is also
    a statement worth checking. ``texts`` gives the whole words of each
    candidate utterance (the sheet keeps only their first words).
    """
    full = texts or {}
    given: Dict[str, Dict[str, Any]] = {}
    raw = data.get("slots")
    for item in raw if isinstance(raw, list) else []:
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            given[item["id"]] = item
    merged: List[SheetSlot] = []
    said_all: Dict[str, str] = {}
    owner: Dict[str, str] = {}
    for slot in slots:
        said = {a.utterance: full.get(a.utterance, a.quote)
                for a in slot.answers}
        said.update({p.utterance: full.get(p.utterance, p.quote)
                     for p in slot.points
                     if p.utterance and p.utterance not in said})
        said_all.update(said)
        owner.update({u: slot.id for u in said})
        item = given.get(slot.id)
        if item is None or not said:
            merged.append(slot.model_copy(update={
                "accuracy": accuracy_counts(slot)}))
            continue
        known = set(chunks.get(slot.id, ()))
        judged = judgement_from(item.get("judgement"), said, known)
        by_point: Dict[str, Any] = {}
        raw_points = item.get("points")
        for p in raw_points if isinstance(raw_points, list) else []:
            if isinstance(p, dict) and isinstance(p.get("id"), str):
                by_point[p["id"]] = p
        points = [
            p.model_copy(update={"judgement": judgement_from(
                by_point.get(p.id), said, known)})
            if p.status == "treated" and p.id in by_point else p
            for p in slot.points
        ]
        updated = slot.model_copy(update={"judgement": judged,
                                          "points": points})
        updated = updated.model_copy(update={
            "score": capped_score(updated.score, updated),
            "above": updated.above and (
                judged is None or judged.label == "solid"),
            "accuracy": accuracy_counts(updated),
        })
        merged.append(updated)
    return merged, _checks(data.get("to_check"), merged, said_all, owner)


def _checks(
    raw: object, slots: Sequence[SheetSlot], said: Mapping[str, str],
    owner: Mapping[str, str],
) -> List[CheckItem]:
    """The statements worth checking: the model's, then every ``wrong``."""
    found: List[CheckItem] = []
    seen: Set[str] = set()

    def add(utterance: str, quote: str, why: str, note: str,
            slot: Optional[str]) -> None:
        if not quote or quote in seen or len(found) >= MAX_CHECKS:
            return
        seen.add(quote)
        found.append(CheckItem(slot=slot, quote=quote[:QUOTE_CHARS],
                               utterance=utterance or None,
                               why=why,  # type: ignore[arg-type]
                               note=clean_text(note)))

    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        why = _str(item.get("why"))
        if why not in ("wrong", "contradiction", "vague"):
            continue
        words = fold(" ".join(_str(item.get("quote")).split())).strip(
            " .…\"'«»")
        judged = judgement_from({"label": "unclear", **item}, said, set())
        if judged is None or not judged.utterance or not words or (
                words not in fold(said[judged.utterance])):
            continue  # not the candidate's own words: never listed
        if why == "wrong" and is_experience(said[judged.utterance]):
            continue  # an experience claim is to verify, not wrong
        add(judged.utterance, judged.quote, why, _str(item.get("note")),
            owner.get(judged.utterance))
    for slot in slots:
        judged_all = [slot.judgement, *(p.judgement for p in slot.points)]
        for j in judged_all:
            if j is not None and j.label == "wrong" and j.utterance:
                add(j.utterance, j.quote, "wrong", j.reason, slot.id)
    return found
