"""Steps d and e of ``skills analyze`` over the existing modules.

:class:`RuleEventExtractor` (step e) derives decisions, action items,
open questions and the fate of proposals from the dialogue acts of
step d with the fixed rules of ``docs/skills/CONVERSATION_EVENTS.md``
(#83). :class:`ProvisionalActTagger` (step d, opt-in) stands in for the
dialogue-act tagger (#82) until it exists: it asks the selected judge
for one act per sentence (the route of ``skills judge_acts``), one call
per sentence, and says in its output that it is provisional. Both
write the ``acts.v1`` / events JSON the analysis stores.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence

from src.application.skills.conversation_events import analyse
from src.application.skills.judges import ActClassifier, ActItem, release
from src.application.skills.ports import StepOutput
from src.domain.errors import InvalidInputError
from src.domain.skills.act_sequence import ACTS_SCHEMA, session_from_json
from src.domain.skills.conversation_events import EventRules
from src.domain.skills.speech import Sentence
from src.domain.skills.transcript import Transcript

RULES_NAME = "rules (CONVERSATION_EVENTS.md)"
"""Who produced the events: fixed rules, no model."""


class RuleEventExtractor:
    """Derive the conversation events from step d's acts (#83)."""

    def __init__(self, rules: EventRules) -> None:
        """Keep the ``event_*`` parameters."""
        self.rules = rules

    def extract(
        self,
        transcript: Transcript,
        sentences: Sequence[Sentence],
        acts: Optional[Any],
    ) -> StepOutput:
        """Return the links, events and per-person counts."""
        if acts is None:
            raise InvalidInputError(
                "The events need the dialogue acts of step d."
            )
        session = session_from_json(acts, "acts.json")
        result = analyse(session, self.rules)
        counts = {k: v for k, v in result.summary.counts.items() if v}
        return StepOutput(result.model_dump(mode="json"), counts, RULES_NAME)


class ProvisionalActTagger:
    """One act per sentence from the selected judge (until #82 exists).

    Args:
        classifier: Builds the act classifier over the judge, when the
            step runs (so nothing is loaded before).
        context: Sentences shown before each one.
    """

    def __init__(
        self, classifier: Callable[[], ActClassifier], context: int = 3
    ) -> None:
        """Keep the factory; nothing is built yet."""
        self.classifier = classifier
        self.context = max(context, 0)

    def tag(
        self, transcript: Transcript, sentences: Sequence[Sentence]
    ) -> StepOutput:
        """Return the sentences as ``acts.v1`` with the judge's act."""
        items = [
            ActItem(
                id=s.id, speaker=s.speaker, text=s.text,
                context=[
                    (b.speaker, b.text)
                    for b in sentences[max(n - self.context, 0):n]
                ],
            )
            for n, s in enumerate(sentences)
        ]
        classifier = self.classifier()
        try:
            verdicts = classifier.classify(items)
        finally:
            release(classifier.judge)
        rows: List[Dict[str, Any]] = []
        unparsed = 0
        for sentence, verdict in zip(sentences, verdicts):
            label: Dict[str, Any] = {"label": verdict.act or "otro",
                                     "weight": 100}
            if verdict.act is None:
                unparsed += 1
                label["doubtful"] = True
            rows.append({
                "id": sentence.id, "turn": sentence.turn,
                "speaker": sentence.speaker, "text": sentence.text,
                "start": sentence.start, "end": sentence.end,
                "labels": [label],
            })
        models = sorted({v.model for v in verdicts})
        name = ", ".join(models) + " (provisional tagger)"
        return StepOutput(
            {
                "schema": ACTS_SCHEMA,
                "session": {"id": transcript.session_id},
                "tagger": name, "provisional": True,
                "utterances": rows,
            },
            {"sentences": len(rows), "unparsed": unparsed},
            name,
        )
