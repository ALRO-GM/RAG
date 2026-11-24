"""Always an opinion: verdicts per skill and overall, read by situation.

The owner asked (#231) that the analysis of a person never ends in
«insufficient evidence»: from whatever was found it gives a verdict and
a short conclusion for every skill, an overall score, one verdict line
and the main strengths and improvements. How firm that is stays in the
data (``confidence``, ``caveats``, ``basis``) for the interface to show
as secondary detail, never as the headline.

The conversation type selects the question the verdict answers (a
coaching session: has the person acquired the knowledge?; an interview:
fit for the role; a support call: was it resolved well?), see
:data:`FRAMES`. Scores are the judge's reading on the rubric scale, or
the analysis's own level when it has one; the cut-offs and weights here
are choices for display, not measured values, and nothing about their
validity is established. Plain data and rules: no I/O, no model.
"""

from typing import Dict, List, Literal, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

OPINION_FORMAT_VERSION = 2
"""Version of the synthesis JSON. 2 added the opinions (#231): files
without them are version 1 and still read."""

Verdict = Literal["strong", "adequate", "weak"]
"""A skill's or a person's verdict: a reading of a score, not a decision."""
OpinionBasis = Literal["firm", "provisional", "observation", "participation"]
"""What a skill's opinion rests on, firmest first: a firm level, a
provisional estimate, quoted observations, or only how much the person
took part."""
Source = Literal["judge", "rules"]
"""Who wrote the opinion: the judge, or the rules below when there was
no judge reply (no quote, the model failed, a competency it skipped)."""
Certainty = Literal["low", "medium", "high"]

STRONG_FROM = 3.75
"""A score at or above this reads as ``strong`` (a display choice)."""
ADEQUATE_FROM = 2.5
"""A score at or above this (and below :data:`STRONG_FROM`) reads as
``adequate``; below it, ``weak`` (a display choice)."""

MAX_CITES = 3
"""Most quotes kept with one skill opinion."""
MAX_TOP_STRENGTHS = 3
"""Most skills listed as strengths."""
MAX_TOP_IMPROVEMENTS = 2
"""Most skills listed as improvements."""

WEIGHTS: Dict[str, float] = {
    "firm": 1.0, "provisional": 0.6, "observation": 0.4,
    "participation": 0.2,
}
"""How much a skill's score counts in the overall one, by its basis:
firm levels weigh most (a display choice, not a measured value)."""

FRAMES: Dict[str, Dict[str, str]] = {
    "coaching": {
        "en": "has the person acquired and applied the knowledge or "
              "skills being coached?",
        "es": "¿ha adquirido y aplicado la persona los conocimientos o "
              "habilidades que se trabajan?",
    },
    "entrevista": {
        "en": "which demonstrated strengths match what the role (or kind of "
              "tasks) asks for, and where are the gaps? Describe the "
              "match skill by skill; never say whether to hire or move "
              "the person forward.",
        "es": "¿qué fortalezas demostradas coinciden con lo que pide el "
              "puesto (o el tipo de tareas) y dónde están las carencias? "
              "Describe la coincidencia competencia por competencia; "
              "nunca digas si contratar o avanzar a la persona.",
    },
    "venta": {
        "en": "did the person handle the customer well: needs found, "
              "objections answered, a clear next step or resolution?",
        "es": "¿atendió bien la persona al cliente: necesidades "
              "detectadas, objeciones atendidas, siguiente paso o "
              "resolución claros?",
    },
    "feedback": {
        "en": "did the person give or take feedback in a way that can "
              "change behaviour?",
        "es": "¿dio o recibió la persona el feedback de forma que "
              "pueda cambiar conductas?",
    },
    "negociacion": {
        "en": "did the person defend their interests while moving the "
              "negotiation towards an agreement?",
        "es": "¿defendió la persona sus intereses mientras acercaba la "
              "negociación a un acuerdo?",
    },
    "conflicto": {
        "en": "did the person help the conflict move towards a "
              "resolution or make it worse?",
        "es": "¿ayudó la persona a encaminar el conflicto hacia una "
              "resolución o lo empeoró?",
    },
    "reunion_decision": {
        "en": "did the person help the meeting reach clear decisions "
              "and next steps?",
        "es": "¿ayudó la persona a que la reunión llegase a decisiones "
              "y próximos pasos claros?",
    },
    "lluvia_ideas": {
        "en": "did the person contribute and build on ideas so the "
              "group moved forward?",
        "es": "¿aportó la persona ideas y construyó sobre las de los "
              "demás para que el grupo avanzase?",
    },
    "resolucion_problemas": {
        "en": "did the person help the group understand and solve the "
              "problem?",
        "es": "¿ayudó la persona al grupo a entender y resolver el "
              "problema?",
    },
}
"""The question the verdict answers, by conversation type and language.
Types not listed (``conversacion_libre``, ``otro``, none) use
:data:`DEFAULT_FRAME`. Reading a call as «was it resolved well?» is a
framing of the prompt, not a validated criterion."""

DEFAULT_FRAME: Dict[str, str] = {
    "en": "how well did the person communicate and work with others in "
          "this conversation?",
    "es": "¿cómo se comunicó y colaboró la persona en esta "
          "conversación?",
}
"""The question for a conversation of no known type."""

LABELS: Dict[str, Dict[str, str]] = {
    "strong": {"en": "strong", "es": "sólido"},
    "adequate": {"en": "adequate", "es": "correcto"},
    "weak": {"en": "weak", "es": "flojo"},
}
"""Words for the verdicts in the fallback sentences."""


def scaled(score: float, minimum: int, maximum: int) -> float:
    """Return ``score`` on the 1 to 5 scale the cut-offs are set on."""
    if maximum <= minimum or (minimum, maximum) == (1, 5):
        return score
    return 1 + (score - minimum) * 4 / (maximum - minimum)


def frame(conversation_type: Optional[str], language: str) -> str:
    """Return the question the verdict answers for this conversation."""
    chosen = FRAMES.get(conversation_type or "", DEFAULT_FRAME)
    return chosen.get(language, chosen["en"])


def verdict_of(score: float) -> Verdict:
    """Return the verdict a score reads as."""
    if score >= STRONG_FROM:
        return "strong"
    if score >= ADEQUATE_FROM:
        return "adequate"
    return "weak"


class OpinionQuote(BaseModel):
    """A quote behind a skill opinion: what was said and when."""

    text: str
    turn: Optional[int] = None
    start: Optional[float] = None


class SkillOpinion(BaseModel):
    """The opinion about one person on one skill.

    ``score`` is on the rubric scale (1 to the maximum); ``verdict`` is
    its reading; ``conclusion`` is one or two sentences. ``basis`` and
    ``caveats`` (codes: ``provisional``, ``observations_only``,
    ``no_quotes``, ``short_participation``) tell the interface how
    firm it is, to show apart from the headline. ``cites`` are the
    quotes behind it (at most :data:`MAX_CITES`), when there are any.
    """

    competency_id: str
    name: str
    score: int
    verdict: Verdict
    conclusion: str
    basis: OpinionBasis
    confidence: Certainty = "low"
    caveats: List[str] = Field(default_factory=list)
    source: Source = "rules"
    cites: List[OpinionQuote] = Field(default_factory=list)


class OverallOpinion(BaseModel):
    """The opinion about the whole person in this conversation.

    ``score`` is the weighted mean of the skill scores (one decimal,
    see :data:`WEIGHTS`) on a scale up to ``maximum``; ``line`` is the
    one-line verdict; ``frame`` the question it answers.
    """

    score: float
    maximum: int
    minimum: int = 1
    percent: float = 0.0
    verdict: Verdict
    line: str
    frame: str
    confidence: Certainty = "low"
    caveats: List[str] = Field(default_factory=list)
    source: Source = "rules"


def overall_score(skills: Sequence[SkillOpinion]) -> float:
    """Return the weighted mean of the skill scores, to one decimal."""
    total = sum(WEIGHTS[s.basis] for s in skills)
    if not total:
        return 0.0
    mean = sum(s.score * WEIGHTS[s.basis] for s in skills) / total
    return round(mean, 1)


def top_picks(
    skills: Sequence[SkillOpinion], minimum: int = 1, maximum: int = 5,
) -> Tuple[List[str], List[str]]:
    """Return the ids of the main strengths (≤3) and improvements (≤2).

    Only skills with something behind them count: a firm or provisional
    level, or observations that lean one way (a score away from the
    middle of the scale). Skills read from participation alone are
    never listed, so a person with nothing quoted has empty lists.
    Strengths are the highest scores at ``adequate`` or above,
    improvements the lowest scores below ``strong`` (cut-offs on the
    scaled score); ties go to the firmer basis, then to the rubric's
    order. A skill is never both.
    """
    rank = list(WEIGHTS)
    order = {s.competency_id: i for i, s in enumerate(skills)}
    middle = (minimum + maximum + 1) // 2

    def value(s: SkillOpinion) -> float:
        return scaled(s.score, minimum, maximum)

    def firmer(s: SkillOpinion) -> int:
        return rank.index(s.basis)

    backed = [
        s for s in skills
        if s.basis in ("firm", "provisional")
        or (s.basis == "observation" and s.score != middle)
    ]
    strong = sorted(
        (s for s in backed if value(s) >= ADEQUATE_FROM),
        key=lambda s: (-s.score, firmer(s), order[s.competency_id]),
    )[:MAX_TOP_STRENGTHS]
    taken = {s.competency_id for s in strong}
    weak = sorted(
        (s for s in backed
         if s.competency_id not in taken and value(s) < STRONG_FROM),
        key=lambda s: (s.score, firmer(s), order[s.competency_id]),
    )[:MAX_TOP_IMPROVEMENTS]
    return (
        [s.competency_id for s in strong],
        [s.competency_id for s in weak],
    )


def overall_caveats(skills: Sequence[SkillOpinion]) -> List[str]:
    """Return the codes of what limits the whole opinion."""
    found: List[str] = []
    if not any(s.basis == "firm" for s in skills):
        found.append("no_firm_levels")
    if any(s.basis == "observation" for s in skills) and all(
        s.basis in ("participation", "observation") for s in skills
    ):
        found.append("observations_only")
    if any("short_participation" in s.caveats for s in skills):
        found.append("short_participation")
    if skills and not any(s.cites for s in skills):
        found.append("no_quotes")
    return found
