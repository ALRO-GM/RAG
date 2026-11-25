"""A scripted live-page demo: one fictitious interview, fixed and replayable.

LV-17 (#232): the live page can play a ready-made interview (``live.demo.v1``,
``examples/demo/*.demo.json``) through the replay path. The file holds
the conversation (a ``transcript.v1``) and everything the page would
otherwise get from the services: the suggestions, prepared in advance
and attached to the sentence after which they appear; the line better
avoided with its hint; and the topics of the job with the sentence that
covers each, the sentence that asks about it and its concepts with the
sentence that treats each, if any (#258). Nothing is asked to a model,
so every run is the same.
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

import copy
import json
import math
import re
from datetime import datetime, timedelta
from typing import (
    Any, Callable, Dict, Iterable, Iterator, List, Literal, Optional, Set,
)

from pydantic import BaseModel, ConfigDict, Field, model_validator

from orchestrator.domain.call_report import CallReport
from orchestrator.domain.events import LiveEvent, Transcript, Utterance
from orchestrator.domain.voice_demo import VoiceDemo

DEMO_ID = r"^[a-z0-9][a-z0-9._-]{0,47}$"
PARTIAL_EVERY_S = 0.25
"""Stream seconds between two partials of a replayed demo sentence."""


class DemoPoint(BaseModel):
    """A concept of a demo topic, ticked when ``covered_by`` is said."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    covered_by: Optional[str] = None
    """The sentence that treats it; none: it is never treated."""
    en: Optional[Dict[str, str]] = None
    """``text`` in English."""


class DemoTopic(BaseModel):
    """A topic of the job, ticked when ``covered_by`` has been said."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    covered_by: str
    asked_by: Optional[str] = None
    """The sentence that asks about it (it becomes the current topic)."""
    points: List[DemoPoint] = Field(default_factory=list)
    """Its concepts (#258), each ticked by its own sentence."""
    en: Optional[Dict[str, str]] = None
    """The same texts in English (``name``), for a page in English."""

    def sentences(self) -> List[str]:
        """Every sentence the topic names."""
        named = [self.covered_by]
        if self.asked_by:
            named.append(self.asked_by)
        return named + [p.covered_by for p in self.points if p.covered_by]


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
    title: str = Field(min_length=1)
    """A short readable name of the chunk, shown instead of its id."""
    text: str = Field(min_length=1)
    """The chunk as the knowledge base has it (Spanish)."""
    metadata: Dict[str, object] = Field(default_factory=dict)
    en: Optional[Dict[str, str]] = None
    """``title`` in English; the text stays as the knowledge base has it."""


class DemoMandatoryItem(BaseModel):
    """A mandatory item of the call and the sentence that says it, if any."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    phase: Optional[str] = None
    en: Optional[Dict[str, str]] = None
    """``text`` in English, for the report."""
    said_by: Optional[str] = None
    """The agent sentence that says it; none: the agent never says it."""


class DemoMandatory(BaseModel):
    """The mandatory items of a demo (as the product's domain lists them)."""

    model_config = ConfigDict(extra="forbid")

    domain: str = Field(min_length=1)
    version: int = 1
    remind_seconds: int = Field(default=45, ge=5)
    items: List[DemoMandatoryItem] = Field(min_length=1)


Gender = Literal["female", "male"]


class DemoCharacter(BaseModel):
    """A scripted person: the speaker id of their lines and their name.

    ``gender`` is the one their name (and the voice that reads them)
    has; ``None``: any (the voice assistant). ``gendered_text`` says the
    script's wording about them is gendered («señor», «harto», «he»), so
    renaming them to the other gender needs rewording, not a rename.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    gender: Optional[Gender] = None
    gendered_text: bool = False


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
    cast: List[DemoCharacter] = Field(default_factory=list)
    """Who speaks (#254); empty: the transcript's speakers, no gender."""
    transcript: Transcript

    @model_validator(mode="after")
    def _known_lines(self) -> "DemoScript":
        """Every reference names a sentence of the transcript."""
        known: Set[str] = {u.id for u in self.transcript.utterances}
        named = ([n for t in self.topics for n in t.sentences()]
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
        self._known_cast()
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

    def _known_cast(self) -> None:
        """The cast names every speaker once, with the transcript's name."""
        if not self.cast:
            return
        cast = {c.id: c for c in self.cast}
        if len(cast) != len(self.cast):
            raise ValueError("a person is twice in the cast")
        speaking = {u.speaker for u in self.transcript.utterances}
        if self.voicebot:
            speaking |= {self.voicebot.user, self.voicebot.bot}
        missing = sorted(speaking - set(cast))
        if missing:
            raise ValueError(f"not in the cast: {missing}")
        for speaker in self.transcript.speakers:
            if speaker.id in cast and speaker.name != cast[speaker.id].name:
                raise ValueError(
                    f"{speaker.id!r} is {speaker.name!r} in the transcript"
                    f" but {cast[speaker.id].name!r} in the cast"
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


def recast(
    data: Dict[str, Any], person: str, name: str,
    gender: Optional[str] = None,
) -> Dict[str, Any]:
    """Return a demo file's data with ``person`` renamed everywhere.

    ``data`` is a ``live.demo.v1`` file as plain JSON data; every text
    that says the old name (whole, first word, last word) says the new
    one, and the cast entry gets the new name and, when given, the new
    ``gender``. The new name has as many words as the old one and none
    of its words is already in the file, so a rename can be undone by
    the opposite rename.

    Raises:
        ValueError: ``person`` is not in the cast, the name does not
            fit, the demo has a stored analysis that names people too,
            or the gender changes for a person worded by gender.
    """
    cast = data.get("cast") or []
    entry = next((c for c in cast if isinstance(c, dict)
                  and c.get("id") == person), None)
    if entry is None:
        raise ValueError(f"{person!r} is not in the cast")
    if data.get("result_session"):
        raise ValueError("this demo's stored analysis names the people too")
    if gender not in (None, "female", "male"):
        raise ValueError("gender is female or male")
    if (gender is not None and gender != entry.get("gender")
            and entry.get("gendered_text")):
        raise ValueError(
            f"the script words {person!r} as {entry.get('gender')}:"
            " rewording, not a rename, changes that"
        )
    old, new = str(entry["name"]).split(), name.split()
    if len(new) != len(old) or not new:
        raise ValueError(f"the new name has {len(old)} word(s), like the old")
    pairs = {" ".join(old): " ".join(new), old[0]: new[0], old[-1]: new[-1]}
    pairs = {k: v for k, v in pairs.items() if k != v}
    text = json.dumps(data, ensure_ascii=False)
    taken = [w for w in new if w not in old
             and re.search(rf"(?<!\w){re.escape(w)}(?!\w)", text)]
    if taken:
        raise ValueError(f"already in the demo: {taken}")
    if not pairs:
        renamed = copy.deepcopy(data)
    else:
        pattern = re.compile("(?<!\\w)(?:" + "|".join(
            re.escape(k) for k in sorted(pairs, key=len, reverse=True)
        ) + ")(?!\\w)")
        renamed = _map_strings(
            data, lambda s: pattern.sub(lambda m: pairs[m.group(0)], s))
    for item in renamed.get("cast") or []:
        if isinstance(item, dict) and item.get("id") == person:
            item["name"] = " ".join(new)
            if gender is not None:
                item["gender"] = gender
    return dict(renamed)


def _map_strings(value: Any, change: Callable[[str], str]) -> Any:
    """Return ``value`` (JSON data) with ``change`` applied to every text."""
    if isinstance(value, str):
        return change(value)
    if isinstance(value, list):
        return [_map_strings(v, change) for v in value]
    if isinstance(value, dict):
        return {k: _map_strings(v, change) for k, v in value.items()}
    return value
