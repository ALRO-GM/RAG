"""The ``transcript.v1`` of a live session, from its final events (LV-11).

A live session's finals are already sentences with speakers and times;
a sentence may have been revised (``revision`` grows). At the end of
the session the last revision of each sentence is kept, empty ones are
dropped, and the rest, in time order, become the utterances of an
ordinary ``transcript.v1`` document: consecutive sentences of one
speaker form a turn and the ids are renumbered without gaps (live ids
may skip numbers a partial reserved). No audio, words or roles leave.
"""

from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence

from transcriber.domain.live import LiveSpeaker
from transcriber.domain.models import (
    Pipeline,
    Session,
    SessionState,
    Speaker,
    Transcript,
    Utterance,
)
from transcriber.domain.sentences import regroup_utterances

UNDETERMINED = "und"
"""Language of a transcript whose language is not known (ISO 639-2)."""


def last_finals(events: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Return the last revision of each final sentence, in time order."""
    latest: Dict[str, Dict[str, Any]] = {}
    for event in events:
        if event.get("type") != "final":
            continue
        utterance = dict(event["utterance"])
        known = latest.get(utterance["id"])
        if known is None or utterance["revision"] >= known["revision"]:
            latest[utterance["id"]] = utterance
    return sorted(
        latest.values(),
        key=lambda u: (u["start"], u["turn"], u["index"]),
    )


def _language(
    finals: Sequence[Mapping[str, Any]], asked: Optional[str]
) -> str:
    """Return the language asked for, else the most heard, else ``und``."""
    if asked:
        return asked
    heard = Counter(u["language"] for u in finals if u.get("language"))
    return heard.most_common(1)[0][0] if heard else UNDETERMINED


def live_transcript(
    session_id: str,
    events: Sequence[Mapping[str, Any]],
    speakers: Sequence[LiveSpeaker],
    pipeline: Pipeline,
    language: Optional[str] = None,
    started_at: Optional[datetime] = None,
) -> Transcript:
    """Return the transcript of a live session's events.

    ``duration`` is the stream time of ``session_ended`` (or of the
    last event). Only speakers who said something are listed; an id a
    final names but ``speakers`` does not is listed anonymously.
    """
    finals = [u for u in last_finals(events) if u["text"].strip()]
    pending = [
        Utterance(
            id=f"t{n}.s0", turn=n, index=0, speaker=u["speaker"],
            start=u["start"], end=max(u["end"], u["start"]),
            text=u["text"].strip(), previous_speaker=None,
            previous_utterance=None, confidence=u.get("confidence"),
            language=u.get("language"),
        )
        for n, u in enumerate(finals)
    ]
    utterances = regroup_utterances(pending)
    heard = {u.speaker for u in utterances}
    named = {s.id: s for s in speakers}
    listed = [
        Speaker(id=s.id, name=s.name, enrolled=False)
        for s in speakers if s.id in heard
    ] + [
        Speaker(id=who, enrolled=False)
        for who in sorted(heard - set(named))
    ]
    duration = max(
        [float(e.get("stream_time", 0.0)) for e in events] + [0.0]
    )
    session = Session(
        id=session_id, language=_language(finals, language),
        started_at=started_at, duration=duration, state=SessionState.DONE,
    )
    return Transcript(
        session=session, speakers=listed, utterances=utterances,
        pipeline=pipeline,
    )
