"""Fakes for the recovery of a live session's lost audio (#265).

An in-memory store, a scripted batch transcriber, a settable clock, and
a helper that writes invented ``transcript.v1`` documents. No audio is
decoded and no words of a real person appear anywhere.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from transcriber.domain.live_recovery import RecoveryJob
from transcriber.domain.models import (
    Pipeline,
    Session,
    SessionState,
    Speaker,
    Transcript,
    Utterance,
)
from transcriber.domain.sentences import regroup_utterances

T0 = datetime(2026, 11, 26, 10, 0, tzinfo=timezone.utc)

Said = Tuple[str, float, float, str]
"""(speaker, start, end, text) of one invented sentence."""


def transcript(
    said: Sequence[Said], speakers: Sequence[Speaker] = (),
    session_id: str = "iv-1", duration: Optional[float] = None,
) -> Transcript:
    """An invented transcript of ``said``."""
    pending = [
        Utterance(id=f"t{n}.s0", turn=n, index=0, speaker=who, start=a,
                  end=b, text=text, previous_speaker=None,
                  previous_utterance=None)
        for n, (who, a, b, text) in enumerate(said)
    ]
    listed = list(speakers) + [
        Speaker(id=who, enrolled=False)
        for who in sorted({s[0] for s in said} - {s.id for s in speakers})
    ]
    return Transcript(
        session=Session(
            id=session_id, language="es", state=SessionState.DONE,
            duration=duration if duration is not None
            else max([b for _, _, b, _ in said] + [0.0])),
        speakers=listed, utterances=regroup_utterances(pending),
        pipeline=Pipeline(recogniser="fake", voice_model="none",
                          transcriber="test"),
    )


class Clock:
    """A wall clock that only moves when told."""

    def __init__(self) -> None:
        self.at = T0

    def now(self) -> datetime:
        return self.at

    def advance(self, seconds: float) -> None:
        self.at += timedelta(seconds=seconds)


class MemoryRecoveryStore:
    """The recovery store in memory."""

    def __init__(self) -> None:
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.chunks: Dict[str, Dict[int, bytes]] = {}
        self.texts: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.wavs: List[Tuple[str, Any, Optional[int]]] = []

    def ids(self) -> List[str]:
        return sorted(set(self.jobs) | set(self.chunks))

    def load(self, session_id: str) -> Optional[RecoveryJob]:
        found = self.jobs.get(session_id)
        return RecoveryJob.from_dict(found) if found is not None else None

    def save(self, job: RecoveryJob) -> None:
        self.jobs[job.session_id] = dict(job.to_dict(),
                                         chunks=dict(job.chunks),
                                         meta=dict(job.meta))

    def put_chunk(self, session_id: str, number: int, data: bytes) -> None:
        self.chunks.setdefault(session_id, {})[number] = data

    def wav(self, session_id: str, chunks: Mapping[str, Sequence[int]],
            spans: Sequence[Tuple[int, int]], channels: int,
            channel: Optional[int]) -> bytes:
        self.wavs.append((session_id, list(spans), channel))
        return b"RIFF-fake"

    def audio_bytes(self) -> int:
        return sum(len(d) for c in self.chunks.values() for d in c.values())

    def delete_audio(self, session_id: str) -> int:
        return len(self.chunks.pop(session_id, {}))

    def save_text(self, session_id: str, name: str,
                  document: Mapping[str, Any]) -> None:
        self.texts.setdefault(session_id, {})[name] = dict(document)

    def read_text(self, session_id: str,
                  name: str) -> Optional[Dict[str, Any]]:
        found = self.texts.get(session_id, {}).get(name)
        return dict(found) if found is not None else None

    def delete_texts(self, session_id: str) -> None:
        self.texts.pop(session_id, None)

    def delete(self, session_id: str) -> None:
        self.jobs.pop(session_id, None)
        self.chunks.pop(session_id, None)
        self.texts.pop(session_id, None)


class ScriptedTranscriber:
    """The batch pipeline, scripted: each part answers what it is told.

    ``results`` maps a channel (``None`` for mono) to the transcript a
    part of it gets; ``plan`` the states the next statuses return
    (default: done at once); ``fail_next`` makes that many parts fail.
    """

    def __init__(self, results: Mapping[Optional[int], Transcript]) -> None:
        self.results = dict(results)
        self.parts: Dict[str, Optional[int]] = {}
        self.submitted: List[Tuple[str, str, Optional[int], Any]] = []
        self.forgotten: List[str] = []
        self.plan: List[Tuple[str, float]] = []
        self.fail_next = 0

    def submit(self, part_id: str, wav: bytes, language: str,
               num_speakers: Optional[int],
               expected: Optional[List[str]]) -> None:
        channel = None if part_id.endswith("-m") else int(part_id[-1])
        self.parts[part_id] = channel
        self.submitted.append((part_id, language, num_speakers, expected))

    def status(self, part_id: str) -> Tuple[str, float]:
        if part_id not in self.parts:
            return "lost", 0.0
        if self.fail_next:
            self.fail_next -= 1
            return "failed", 0.0
        if self.plan:
            return self.plan.pop(0)
        return "done", 1.0

    def transcript(self, part_id: str) -> Optional[Dict[str, Any]]:
        if part_id not in self.parts:
            return None
        return self.results[self.parts[part_id]].to_contract()

    def forget(self, part_id: str) -> None:
        if self.parts.pop(part_id, "absent") != "absent":
            self.forgotten.append(part_id)


class RecordingRebuilder:
    """Keeps the rebuilds asked; may refuse."""

    def __init__(self, error: Optional[Exception] = None) -> None:
        self.asked: List[Tuple[str, Dict[str, Any]]] = []
        self.error = error

    def rebuild(self, session_id: str,
                transcript: Mapping[str, Any]) -> None:
        if self.error is not None:
            raise self.error
        self.asked.append((session_id, dict(transcript)))
