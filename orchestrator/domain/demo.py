"""A scripted live-page demo: one fictitious interview, fixed and replayable.

LV-17 (#232): the live page can play a ready-made interview (``live.demo.v1``,
``examples/demo/*.demo.json``) through the replay path. The file holds
the conversation (a ``transcript.v1``) and everything the page would
otherwise get from the services: the suggestions, prepared in advance
and attached to the sentence after which they appear; the line better
avoided with its hint; and the topics of the job with the sentence that
covers each. Nothing is asked to a model, so every run is the same.
``demo`` is always true: the file is demo data, never a real person,
and the page says so. ``result_session`` names the stored analysis the
page links to at the end.

:func:`with_partials` adds growing ``partial`` events before each final,
so a replayed sentence types in as a live one does. Plain data and
rules: no I/O.
"""

import math
from datetime import datetime, timedelta
from typing import Dict, Iterable, Iterator, List, Literal, Optional, Set

from pydantic import BaseModel, ConfigDict, Field, model_validator

from orchestrator.domain.events import LiveEvent, Transcript, Utterance

DEMO_ID = r"^[a-z0-9][a-z0-9._-]{0,47}$"
PARTIAL_EVERY_S = 0.25
"""Stream seconds between two partials of a replayed demo sentence."""


class DemoTopic(BaseModel):
    """A topic of the job, ticked when ``covered_by`` has been said."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    covered_by: str


class DemoSuggestion(BaseModel):
    """A prepared suggestion shown once ``after`` has been said."""

    model_config = ConfigDict(extra="forbid")

    after: str
    kind: str = Field(min_length=1)
    text: str = Field(min_length=1)
    example: Optional[str] = None
    script_phase: Optional[str] = None


class DemoAvoid(BaseModel):
    """The prepared hint of a line better avoided."""

    model_config = ConfigDict(extra="forbid")

    utterance: str
    category: str = Field(min_length=1)
    title: str = Field(min_length=1)
    why: str = Field(min_length=1)
    alternative: str = Field(min_length=1)


class DemoScript(BaseModel):
    """A ``live.demo.v1`` file."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_: Literal["live.demo.v1"] = Field(alias="schema")
    demo: Literal[True]
    id: str = Field(pattern=DEMO_ID)
    note: str = Field(min_length=1)
    product: str
    title: str
    candidate: str
    result_session: Optional[str] = Field(
        default=None, pattern=r"^demo-[A-Za-z0-9._-]{1,59}$"
    )
    topics: List[DemoTopic] = Field(default_factory=list)
    suggestions: List[DemoSuggestion] = Field(default_factory=list)
    avoid: List[DemoAvoid] = Field(default_factory=list)
    transcript: Transcript

    @model_validator(mode="after")
    def _known_lines(self) -> "DemoScript":
        """Every reference names a sentence of the transcript."""
        known: Set[str] = {u.id for u in self.transcript.utterances}
        named = ([t.covered_by for t in self.topics]
                 + [s.after for s in self.suggestions]
                 + [a.utterance for a in self.avoid])
        missing = sorted(set(named) - known)
        if missing:
            raise ValueError(f"unknown sentences {missing}")
        if not self.transcript.utterances:
            raise ValueError("the demo has no sentences")
        return self

    def summary(self) -> Dict[str, object]:
        """What a page lists of the demo."""
        last = self.transcript.utterances[-1].end
        return {
            "id": self.id, "demo": True, "title": self.title,
            "candidate": self.candidate, "product": self.product,
            "seconds": round(last, 1), "result_session": self.result_session,
            "note": self.note,
        }


def _words_at(text: str, share: float) -> str:
    """Return the first words of ``text`` making up ``share`` of it."""
    words = text.split()
    count = max(1, math.ceil(len(words) * share))
    return " ".join(words[:count])


def with_partials(
    events: Iterable[LiveEvent], every: float = PARTIAL_EVERY_S
) -> Iterator[LiveEvent]:
    """Yield ``events`` with growing partials before each final.

    The partials of a sentence go from its start to its end every
    ``every`` seconds; sequence numbers are given again in order.
    """
    seq = 0
    for event in events:
        u = event.utterance
        if event.type == "final" and u is not None and u.end > u.start:
            steps = int((u.end - u.start) / every)
            shown = ""
            for i in range(1, steps + 1):
                at = u.start + i * every
                text = _words_at(u.text, (at - u.start) / (u.end - u.start))
                if text == shown or text == u.text:
                    continue
                shown = text
                yield event.model_copy(update={
                    "seq": seq, "type": "partial", "stream_time": at,
                    "at": _shift(event.at, at - event.stream_time),
                    "timings": None,
                    "utterance": Utterance(**{
                        **u.model_dump(), "text": text, "end": at,
                    }),
                })
                seq += 1
        yield event.model_copy(update={"seq": seq})
        seq += 1


def _shift(stamp: str, seconds: float) -> str:
    """Return the ISO time ``stamp`` moved by ``seconds``."""
    try:
        moved = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return stamp
    return (moved + timedelta(seconds=seconds)).isoformat(
        timespec="milliseconds"
    ).replace("+00:00", "Z")
