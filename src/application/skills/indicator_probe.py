"""Ask a judge about the reference sentences, in either listing order.

Uses the prompts of ``skills tag --confirm`` as they are, so what is
measured is what that command does. Reversing the order lists the
competencies and, inside each, the indicators the other way round: a
judge that follows the content gives the same answer, one that leans on
position does not.
"""

from typing import List, Sequence

from src.application.skills.ports import ChatModel
from src.domain.skills.judge_reference import Order, Outcome, ReferenceItem
from src.domain.skills.rubric import Competency, RubricCatalogue
from src.domain.skills.speech import Sentence
from src.domain.skills.tag_prompting import (
    confirm_system_prompt,
    confirm_user_prompt,
    parse_confirmation,
)


def _sentence(position: int, speaker: str, text: str) -> Sentence:
    """Wrap a line of the reference as a sentence of a made-up turn."""
    return Sentence(
        id=f"t{position}.s0", turn=position, index=0, speaker=speaker,
        text=text,
    )


def _context(lines: Sequence[str]) -> List[Sentence]:
    """Turn ``"Speaker: text"`` lines into sentences."""
    out: List[Sentence] = []
    for position, line in enumerate(lines):
        speaker, _, text = line.partition(": ")
        out.append(_sentence(position, speaker, text))
    return out


class IndicatorProbe:
    """Run a judge over reference items and record what it named."""

    def __init__(self, judge: ChatModel, rubric: RubricCatalogue,
                 max_tokens: int = 120):
        """Wire the judge and the rubric whose indicators are offered."""
        self.judge = judge
        self.rubric = rubric
        self.max_tokens = max_tokens

    def offered(self, item: ReferenceItem, order: Order) -> List[Competency]:
        """Return the competencies of ``item`` as listed in ``order``."""
        found = [self.rubric.competency(c) for c in item.offered]
        if order == "forward":
            return found
        return [
            c.model_copy(update={"indicators": list(reversed(c.indicators))})
            for c in reversed(found)
        ]

    def ask(self, item: ReferenceItem, order: Order) -> Outcome:
        """Return the judge's answer to one item in one order."""
        offered = self.offered(item, order)
        context = _context(item.context)
        target = _sentence(len(context), item.speaker, item.text)
        reply = self.judge.complete(
            confirm_system_prompt(self.rubric.language),
            confirm_user_prompt(context, target, offered),
            self.max_tokens,
        )
        first = offered[0].indicators[0].id
        named, _, ok = parse_confirmation(reply)
        if not ok:
            return Outcome(item_id=item.id, order=order, parsed=False,
                           first_listed=first)
        if named is not None and not any(
            c.indicator(named) for c in offered
        ):
            return Outcome(item_id=item.id, order=order, outside=True,
                           first_listed=first)
        return Outcome(item_id=item.id, order=order, named=named,
                       first_listed=first)

    def run(self, items: Sequence[ReferenceItem],
            order: Order) -> List[Outcome]:
        """Return one outcome per item in the given order."""
        return [self.ask(item, order) for item in items]
