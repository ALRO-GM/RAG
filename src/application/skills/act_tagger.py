"""Use cases: tag dialogue acts automatically and write them out (#82).

Three approaches behind one port
(:class:`~src.application.skills.ports.ActTagger`):

- :class:`RuleTagger` (A): the rule key of
  :mod:`src.domain.skills.act_rules` through a calibrated outcome table;
- :class:`LlmTagger` (B): the single label the judge names
  (:class:`~src.application.skills.judges.ActClassifier`) through an
  outcome table of that judge;
- the fine-tuned encoder (D), an adapter with temperature scaling
  (``src/adapters/outbound/encoder_act_tagger.py``);

and :class:`CombinedTagger`, which trusts each approach for the
categories it was measured best at (:class:`~src.domain.skills.
act_tagging.Route`). :func:`tag_acts` turns any of them into the
weighted format of #73: the top 3 calibrated labels in whole %, the
explicit rest, and "dudoso" on a label below its category's base rate
plus the margin. The settings of the default tagger are a
:class:`TaggerCard` (JSON) written by the measurement that chose them.
"""

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field

from src.application.skills.judges import ActClassifier, ActItem
from src.application.skills.ports import ActTagger
from src.domain.skills.act_agreement import doubt_thresholds, tagger_labels
from src.domain.skills.act_calibration import OutcomeTable
from src.domain.skills.act_rules import RULES_VERSION, rule_key
from src.domain.skills.act_tagging import (
    ActInput,
    Probabilities,
    Route,
    check_probabilities,
    route,
)
from src.domain.skills.dialogue_acts import WeightedLabel

ACTS_FORMAT = "acts.v1"
"""Format name written in every ``<session>.acts.json``."""


class RuleTagger:
    """Approach A: surface rules, calibrated per rule key."""

    def __init__(self, table: OutcomeTable):
        """Keep the outcome table learned on held-out sentences."""
        self.table = table

    @property
    def name(self) -> str:
        """Return ``rules``."""
        return "rules"

    @property
    def version(self) -> str:
        """Return the version of the cue lists."""
        return RULES_VERSION

    def keys(self, items: Sequence[ActInput]) -> List[str]:
        """Return the rule key of every item."""
        return [rule_key(item) for item in items]

    def predict(self, items: Sequence[ActInput]) -> List[Probabilities]:
        """Return the calibrated distribution of each item's rule key."""
        return [check_probabilities(self.table.probabilities(k))
                for k in self.keys(items)]


def act_item(item: ActInput) -> ActItem:
    """Return the judge's item for a tagger input."""
    return ActItem(id=item.id, speaker=item.speaker, text=item.text,
                   context=[(c.speaker, c.text) for c in item.context])


def llm_outcome(act: Optional[str]) -> str:
    """Return the outcome key of a judge's answer (``?``: not understood)."""
    return f"llm:{act or '?'}"


class LlmTagger:
    """Approach B: a judge's single label, calibrated per label.

    The judge (local Qwen3 or Claude Code) names one category; the
    outcome table, learned on held-out sentences of the same judge,
    says how often each category is right when it names that one.
    """

    def __init__(self, classifier: ActClassifier, table: OutcomeTable):
        """Wire the classifier (and its cache) and the table."""
        self.classifier = classifier
        self.table = table

    @property
    def name(self) -> str:
        """Return ``llm``."""
        return "llm"

    @property
    def version(self) -> str:
        """Return the judge's model name."""
        return self.classifier.judge.model_name

    def outcomes(self, items: Sequence[ActInput]) -> List[str]:
        """Ask the judge and return its outcome keys."""
        verdicts = self.classifier.classify([act_item(i) for i in items])
        return [llm_outcome(v.act) for v in verdicts]

    def predict(self, items: Sequence[ActInput]) -> List[Probabilities]:
        """Return the calibrated distribution of each answer."""
        return [check_probabilities(self.table.probabilities(o))
                for o in self.outcomes(items)]


class CombinedTagger:
    """Each approach for the acts it was measured best at.

    ``taggers`` by name; ``routes`` are tried in order and the
    ``fallback`` tagger answers the rest. Rule keys are passed to the
    routes when a :class:`RuleTagger` is among the taggers.
    """

    def __init__(self, taggers: Mapping[str, ActTagger],
                 routes: Sequence[Route], fallback: str):
        """Check that every route names a known tagger."""
        names = {r.tagger for r in routes} | {fallback}
        missing = names - set(taggers)
        if missing:
            raise ValueError(f"no tagger named {', '.join(sorted(missing))}")
        self.taggers = dict(taggers)
        self.routes = list(routes)
        self.fallback = fallback

    @property
    def name(self) -> str:
        """Return ``combined``."""
        return "combined"

    @property
    def version(self) -> str:
        """Return the versions of the parts, ``name=version`` joined."""
        return " ".join(f"{n}={t.version}"
                        for n, t in sorted(self.taggers.items()))

    def predict_with_sources(
        self, items: Sequence[ActInput]
    ) -> Tuple[List[Probabilities], List[str]]:
        """Return each item's distribution and the tagger that gave it."""
        outputs = {n: t.predict(items) for n, t in self.taggers.items()}
        keys: Dict[str, List[str]] = {
            n: t.keys(items) for n, t in self.taggers.items()
            if isinstance(t, RuleTagger)
        }
        result: List[Probabilities] = []
        sources: List[str] = []
        for n in range(len(items)):
            source, out = route(
                self.routes, self.fallback,
                {name: o[n] for name, o in outputs.items()},
                {name: k[n] for name, k in keys.items()})
            result.append(out)
            sources.append(source)
        return result, sources

    def predict(self, items: Sequence[ActInput]) -> List[Probabilities]:
        """Return each item's routed distribution."""
        return self.predict_with_sources(items)[0]


class TaggerCard(BaseModel):
    """The settings of a tagger, as chosen by a measurement.

    ``rules`` is the outcome table of approach A; ``encoder`` the
    directory of the fine-tuned weights (relative to the repository,
    git-ignored) and ``temperature`` their calibration; ``routes`` and
    ``fallback`` the combination; ``base_rates`` and ``margin`` the
    "dudoso" thresholds. ``source`` says where the figures come from.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    rules: OutcomeTable
    encoder: str = ""
    temperature: float = Field(default=1.0, gt=0)
    routes: List[Route] = []
    fallback: str = "rules"
    base_rates: Dict[str, float]
    margin: float = Field(ge=0)
    source: str = ""

    def thresholds(self) -> Dict[str, float]:
        """Return the least non-doubtful probability per category."""
        return doubt_thresholds(self.base_rates, self.margin)


class ActSentence(BaseModel):
    """One sentence's labels as the tagger writes them.

    ``labels`` are the top 3 in whole % (``doubtful`` below the
    threshold), ``rest`` what is left to 100, ``by`` the approach that
    answered (a combined tagger's part).
    """

    id: str
    speaker: str
    text: str
    labels: List[WeightedLabel]
    rest: int = Field(ge=0, le=100)
    by: str


class ActsDocument(BaseModel):
    """What ``skills acts`` writes next to a transcript."""

    format: str = ACTS_FORMAT
    session: str
    tagger: str
    version: str
    margin: float
    thresholds: Dict[str, float]
    note: str = ""
    sentences: List[ActSentence]


def tag_acts(
    session: str,
    items: Sequence[ActInput],
    tagger: ActTagger,
    thresholds: Mapping[str, float],
    margin: float,
    note: str = "",
) -> ActsDocument:
    """Tag every sentence and return the labels in the #73 format."""
    if isinstance(tagger, CombinedTagger):
        predictions, sources = tagger.predict_with_sources(items)
    else:
        predictions = tagger.predict(items)
        sources = [tagger.name] * len(items)
    sentences = []
    for item, p, source in zip(items, predictions, sources):
        shown = tagger_labels(p, thresholds=thresholds)
        sentences.append(ActSentence(
            id=item.id, speaker=item.speaker, text=item.text,
            labels=shown.labels, rest=shown.rest, by=source))
    return ActsDocument(
        session=session, tagger=tagger.name, version=tagger.version,
        margin=margin, thresholds=dict(thresholds), note=note,
        sentences=sentences)
