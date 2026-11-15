"""Steps d and e of ``skills analyze`` over the existing modules.

:class:`TaggerActStep` (step d) labels every sentence with the
dialogue-act tagger of #82 (``skills acts``: by default the fine-tuned
encoder, the rules when its weights are missing) and returns the
``acts.v1`` document, recording which approach answered, the "dudoso"
margin and how many first labels are doubtful.
:class:`RuleEventExtractor` (step e) derives decisions, action items,
open questions and the fate of proposals from those acts with the
fixed rules of ``docs/skills/CONVERSATION_EVENTS.md`` (#83).
"""

import json
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from src.application.skills.act_tagger import ActPlace, TaggerCard, tag_acts
from src.application.skills.conversation_events import analyse
from src.application.skills.judges import release
from src.application.skills.ports import ActTagger, StepOutput
from src.domain.errors import InvalidInputError
from src.domain.skills.act_sequence import session_from_json
from src.domain.skills.act_tagging import sentence_inputs
from src.domain.skills.conversation_events import EventRules
from src.domain.skills.speech import Sentence
from src.domain.skills.transcript import Transcript

RULES_NAME = "rules (CONVERSATION_EVENTS.md)"
"""Who produced the events: fixed rules, no model."""


class TaggerActStep:
    """Step d: the dialogue-act tagger of #82 on every sentence.

    Args:
        tagger: Builds the tagger and the note on how it was chosen
            (e.g. "no fine-tuned weights: the rules answered"), when the
            step runs, so nothing is loaded before.
        card: The tagger's measured settings ("dudoso" thresholds).
        context: Sentences before each one the tagger reads.
    """

    def __init__(
        self,
        tagger: Callable[[], Tuple[ActTagger, str]],
        card: Callable[[], TaggerCard],
        context: int = 3,
    ) -> None:
        """Keep the factories; nothing is built yet."""
        self.tagger = tagger
        self.card = card
        self.context = max(context, 0)

    def tag(
        self, transcript: Transcript, sentences: Sequence[Sentence]
    ) -> StepOutput:
        """Return the sentences as ``acts.v1`` with their weighted labels."""
        card = self.card()
        model, note = self.tagger()
        items = sentence_inputs(
            [(s.id, s.speaker, s.text) for s in sentences], self.context
        )
        places = [
            ActPlace(turn=s.turn, start=s.start, end=s.end) for s in sentences
        ]
        try:
            document = tag_acts(
                transcript.session_id, items, model, card.thresholds(),
                card.margin, note, places,
            )
        finally:
            release(model)
        counts: Dict[str, int] = {
            "sentences": len(document.utterances),
            "doubtful": sum(
                1 for u in document.utterances if u.labels[0].doubtful
            ),
        }
        for u in document.utterances:
            key = f"by_{u.by}"
            counts[key] = counts.get(key, 0) + 1
        info = document.tagger
        text = f"Doubtful margin {info.margin}."
        if note:
            text += f" {note}"
        return StepOutput(
            json.loads(document.to_json()), counts,
            f"{info.name} ({info.version})", text,
        )


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
