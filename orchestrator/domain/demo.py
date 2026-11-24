"""A scripted live-page demo: one fictitious interview, fixed and replayable.

LV-17 (#232): the live page can play a ready-made interview (``live.demo.v1``,
``examples/demo/*.demo.json``) through the replay path. The file holds
the conversation (a ``transcript.v1``) and everything the page would
otherwise get from the services: the suggestions, prepared in advance
and attached to the sentence after which they appear; the line better
avoided with its hint; and the topics of the job with the sentence that
covers each. Nothing is asked to a model, so every run is the same.
A support demo (#250) also carries the knowledge chunks its suggestions
cite, the mandatory items of the domain with the sentence that says each
one (or none), a prepared call report (:mod:`.call_report`) and a voice
assistant example (:mod:`.voice_demo`).
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

from orchestrator.domain.call_report import CallReport
from orchestrator.domain.events import LiveEvent, Transcript, Utterance
from orchestrator.domain.voice_demo import VoiceDemo

DEMO_ID = r"^[a-z0-9][a-z0-9._-]{0,47}$"
PARTIAL_EVERY_S = 0.25
"""Stream seconds between two partials of a replayed demo sentence."""


class DemoTopic(BaseModel):
    """A topic of the job, ticked when ``covered_by`` has been said."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    covered_by: str
    en: Optional[Dict[str, str]] = None
    """The same texts in English (``name``), for a page in English."""


class DemoSuggestion(BaseModel):
    """A prepared suggestion shown once ``after`` has been said."""

    model_config = ConfigDict(extra="forbid")

    after: str
    kind: str = Field(min_length=1)
    text: str = Field(min_length=1)
    example: Optional[str] = None
    script_phase: Optional[str] = None
    expressed: List[str] = Field(default_factory=list)
    """Signals the customer expressed in the ``after`` sentence (support)."""
    cites: List[str] = Field(default_factory=list)
    """Ids of the demo's ``sources`` the suggestion comes from (support)."""
    en: Optional[Dict[str, str]] = None
    """``text`` and ``example`` in English (the conversation stays Spanish)."""


class DemoAvoid(BaseModel):
    """The prepared hint of a line better avoided."""

    model_config = ConfigDict(extra="forbid")

    utterance: str
    category: str = Field(min_length=1)
    title: str = Field(min_length=1)
    why: str = Field(min_length=1)
    alternative: str = Field(min_length=1)
    en: Optional[Dict[str, str]] = None
    """``title``, ``why`` and ``alternative`` in English."""


class DemoSource(BaseModel):
    """A knowledge chunk a prepared suggestion cites, copied as it is."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    metadata: Dict[str, object] = Field(default_factory=dict)


class DemoMandatoryItem(BaseModel):
    """A mandatory item of the call and the sentence that says it, if any."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    phase: Optional[str] = None
    said_by: Optional[str] = None
    """The agent sentence that says it; none: the agent never says it."""


class DemoMandatory(BaseModel):
    """The mandatory items of a demo (as the product's domain lists them)."""

    model_config = ConfigDict(extra="forbid")

    domain: str = Field(min_length=1)
    version: int = 1
    remind_seconds: int = Field(default=45, ge=5)
    items: List[DemoMandatoryItem] = Field(min_length=1)


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
    en: Optional[Dict[str, str]] = None
    """``title`` and ``note`` in English."""
    result_session: Optional[str] = Field(
        default=None, pattern=r"^demo-[A-Za-z0-9._-]{1,59}$"
    )
    topics: List[DemoTopic] = Field(default_factory=list)
    suggestions: List[DemoSuggestion] = Field(default_factory=list)
    avoid: List[DemoAvoid] = Field(default_factory=list)
    sources: List[DemoSource] = Field(default_factory=list)
    mandatory: Optional[DemoMandatory] = None
    report: Optional[CallReport] = None
    voicebot: Optional[VoiceDemo] = None
    transcript: Transcript

    @model_validator(mode="after")
    def _known_lines(self) -> "DemoScript":
        """Every reference names a sentence of the transcript."""
        known: Set[str] = {u.id for u in self.transcript.utterances}
        named = ([t.covered_by for t in self.topics]
                 + [s.after for s in self.suggestions]
                 + [a.utterance for a in self.avoid])
        if self.mandatory:
            named += [i.said_by for i in self.mandatory.items if i.said_by]
        if self.report:
            named += sorted(self.report.sentences())
        missing = sorted(set(named) - known)
        if missing:
            raise ValueError(f"unknown sentences {missing}")
        self._known_sources()
        if not self.transcript.utterances:
            raise ValueError("the demo has no sentences")
        return self

    def _known_sources(self) -> None:
        """Every cited source and mandatory item of the report exists."""
        sources = {s.id for s in self.sources}
        cited = [c for s in self.suggestions for c in s.cites]
        if self.voicebot:
            cited += [c for e in self.voicebot.exchanges for c in e.cites]
        missing = sorted(set(cited) - sources)
        if missing:
            raise ValueError(f"unknown sources {missing}")
        if self.report:
            items = {i.id for i in self.mandatory.items} \
                if self.mandatory else set()
            said = {i.id: i.said_by for i in self.mandatory.items} \
                if self.mandatory else {}
            for result in self.report.mandatory:
                if result.item not in items:
                    raise ValueError(f"unknown item {result.item!r}")
                if (said[result.item] or None) != result.utterance:
                    raise ValueError(
                        f"report and call disagree on {result.item!r}"
                    )

    def summary(self) -> Dict[str, object]:
        """What a page lists of the demo."""
        last = self.transcript.utterances[-1].end
        return {
            "id": self.id, "demo": True, "title": self.title,
            "candidate": self.candidate, "product": self.product,
            "seconds": round(last, 1), "result_session": self.result_session,
            "note": self.note, "en": dict(self.en or {}),
            "report": self.report is not None,
            "voicebot": self.voicebot is not None,
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
