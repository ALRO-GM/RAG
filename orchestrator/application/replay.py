"""Use case: replay a ``transcript.v1`` as a ``live.event.v1`` stream.

The transcript's sentences become ``final`` events stamped with the
time each sentence ended (``turn_end`` when the speaker changes), so a
recorded or simulated conversation can drive the live-assist use case
with real timing (``speed`` 1), faster (``speed`` > 1) or as fast as
the services answer (``speed`` ``None``). Speakers get the roles of
the product: given explicitly, or by order over the product's channel
roles (the first speaker takes the first role; further speakers take
the last one).
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, Iterator, List, Optional

from orchestrator.application.ports import CancelToken, Clock, EventHandler
from orchestrator.domain.config import ProductConfig
from orchestrator.domain.events import (
    LiveEvent,
    Source,
    Speaker,
    Timings,
    Transcript,
    Utterance,
)

SAMPLE_RATE = 16000
STOP_CHECK = 0.25
"""Longest wait (seconds) before a replay looks at its stop flag again."""


def default_roles(
    config: ProductConfig, transcript: Transcript
) -> Dict[str, str]:
    """Give each speaker a role by order over the product's channels."""
    declared = [
        config.source.channels[c].role for c in sorted(config.source.channels)
    ] or ["unknown"]
    return {
        s.id: declared[min(i, len(declared) - 1)]
        for i, s in enumerate(transcript.speakers)
    }


def transcript_to_events(
    transcript: Transcript, roles: Dict[str, str]
) -> Iterator[LiveEvent]:
    """Yield the events of the conversation, in stream order."""
    session = transcript.session
    start = _start(session.started_at)
    seq = 0

    def stamp(offset: float) -> str:
        return (start + timedelta(seconds=offset)).isoformat(
            timespec="milliseconds"
        ).replace("+00:00", "Z")

    channels = {s.id: i for i, s in enumerate(transcript.speakers)}
    stereo = len(transcript.speakers) <= 2
    yield LiveEvent(
        session_id=_session_id(session.id), seq=seq, type="session_started",
        at=stamp(0.0), stream_time=0.0,
        source=Source(
            kind="stereo_channels" if stereo else "mono_diarized",
            sample_rate=SAMPLE_RATE, channels=2 if stereo else 1,
        ),
        speakers=[
            Speaker(id=s.id, name=s.name, enrolled=s.enrolled,
                    channel=channels[s.id] if stereo else None,
                    role=roles.get(s.id, "unknown"))
            for s in transcript.speakers
        ],
    )
    previous: Optional[str] = None
    for item in transcript.utterances:
        seq += 1
        if previous is not None and item.speaker != previous:
            yield LiveEvent(
                session_id=_session_id(session.id), seq=seq,
                type="turn_end", at=stamp(item.start),
                stream_time=item.start,
            )
            seq += 1
        previous = item.speaker
        yield LiveEvent(
            session_id=_session_id(session.id), seq=seq, type="final",
            at=stamp(item.end), stream_time=item.end,
            utterance=Utterance(
                id=item.id, turn=item.turn, index=item.index,
                speaker=item.speaker,
                channel=channels[item.speaker] if stereo else None,
                start=item.start, end=item.end, text=item.text,
                revision=0, language=item.language or session.language,
            ),
            timings=Timings(lag_ms=0.0),
        )
    seq += 1
    last = transcript.utterances[-1].end if transcript.utterances else 0.0
    if previous is not None:
        yield LiveEvent(
            session_id=_session_id(session.id), seq=seq, type="turn_end",
            at=stamp(last), stream_time=last,
        )
        seq += 1
    yield LiveEvent(
        session_id=_session_id(session.id), seq=seq, type="session_ended",
        at=stamp(last), stream_time=last,
    )


def _session_id(raw: str) -> str:
    """Return ``raw`` as a valid, short ``live.event.v1`` session id."""
    cleaned = "".join(c if c.isalnum() or c in "._-" else "-" for c in raw)
    return (cleaned.lstrip("._-") or "replay")[:48]


def _start(started_at: Optional[str]) -> datetime:
    """Return the session's start (UTC), or the epoch when unknown."""
    if started_at:
        try:
            parsed = datetime.fromisoformat(started_at)
        except ValueError:
            parsed = datetime(1970, 1, 1, tzinfo=timezone.utc)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    return datetime(1970, 1, 1, tzinfo=timezone.utc)


class Replay:
    """Feed a conversation to a handler at a chosen speed.

    Args:
        assist: What receives the events (the live-assist use case, or
            a session that also shows them).
        clock: Monotonic time and waiting.
        speed: Stream seconds per wall second; ``None`` for no waiting.
        stop: Set to end the replay early (checked between events and
            while waiting).
    """

    def __init__(
        self,
        assist: EventHandler,
        clock: Clock,
        speed: Optional[float],
        stop: Optional[CancelToken] = None,
    ):
        """Keep the collaborators."""
        if speed is not None and speed <= 0:
            raise ValueError("speed must be positive or None (max)")
        self.assist = assist
        self.clock = clock
        self.speed = speed
        self.stop = stop or CancelToken()

    def run(self, events: Iterator[LiveEvent]) -> List[LiveEvent]:
        """Deliver ``events`` on time; return them (for an events file)."""
        sent: List[LiveEvent] = []
        origin = self.clock.now()
        for event in events:
            if self.speed is not None:
                due = event.stream_time / self.speed
                while not self.stop.cancelled:
                    wait = due - (self.clock.now() - origin)
                    if wait <= 0:
                        break
                    self.clock.sleep(min(wait, STOP_CHECK))
            if self.stop.cancelled:
                break
            sent.append(event)
            self.assist.on_event(event)
        self.assist.finish()
        return sent
