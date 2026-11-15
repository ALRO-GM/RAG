"""Fakes and builders for the search for speakers' names (#102)."""

import json
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from transcriber.domain.errors import ModelLoadError
from transcriber.domain.models import (
    Pipeline,
    Session,
    SessionState,
    Speaker,
    Transcript,
    Utterance,
)
from transcriber.domain.sentences import regroup_utterances
from transcriber.domain.speaker_names import SpeakerNames

NOW = datetime(2026, 11, 15, 12, 0, tzinfo=timezone.utc)


def conversation(
    lines: Sequence[Tuple[str, str]],
    names: Optional[Dict[str, str]] = None,
    enrolled: Sequence[str] = (),
    session_id: str = "s1",
) -> Transcript:
    """A finished transcript: one sentence per ``(speaker, text)``.

    Every sentence lasts 3 s. ``names`` gives speakers a display name;
    ids in ``enrolled`` are enrolled people.
    """
    rough = [
        Utterance(
            id="t0.s0", turn=0, index=0, speaker=speaker, start=3.0 * n,
            end=3.0 * n + 2.5, text=text, previous_speaker=None,
            previous_utterance=None,
        )
        for n, (speaker, text) in enumerate(lines)
    ]
    utterances = regroup_utterances(rough)
    ids = list(dict.fromkeys(speaker for speaker, _ in lines))
    speakers = [
        Speaker(id=i, name=(names or {}).get(i), enrolled=i in enrolled)
        for i in ids
    ]
    session = Session(
        id=session_id, language="es", duration=3.0 * len(lines) + 1,
        state=SessionState.DONE,
    )
    return Transcript(
        session=session, speakers=speakers, utterances=utterances,
        pipeline=Pipeline(recogniser="r", voice_model="v", transcriber="t"),
    )


def reply(
    name: Optional[str], kind: str = "self", confidence: float = 0.95,
    quote: Optional[str] = None,
) -> str:
    """A model answer in the JSON the prompt asks for."""
    return json.dumps({
        "is_person_name": name is not None,
        "name": name, "kind": kind if name else "none",
        "quote": quote if quote is not None else (name or ""),
        "confidence": confidence,
    }, ensure_ascii=False)


class FakeChatModel:
    """A chat model whose answers a function gives from the prompt.

    ``answer`` receives the user message; ``fail`` makes every call
    raise :class:`ModelLoadError` (a GPU with no room).
    """

    def __init__(
        self,
        answer: Callable[[str], str] = lambda user: reply(None),
        fail: Optional[Exception] = None,
        name: str = "fake-qwen",
    ) -> None:
        """Answer with ``answer``; nothing is loaded."""
        self.answer = answer
        self.fail = fail
        self.name = name
        self.prompts: List[str] = []
        self.released = 0
        self.loaded = False

    @property
    def model_name(self) -> str:
        """Return the fake model's name."""
        return self.name

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        """Record the prompt and answer it (or fail)."""
        if self.fail is not None:
            raise self.fail
        self.loaded = True
        self.prompts.append(user)
        return self.answer(user)

    def release(self) -> None:
        """Count the release."""
        self.released += 1
        self.loaded = False


def marked(user: str) -> str:
    """Return the text of the marked line of a prompt."""
    line = next(x for x in user.splitlines() if x.startswith(">>"))
    return line.split(":", 1)[1].strip()


def judged(user: str) -> str:
    """Return the word the prompt asks about."""
    return user.splitlines()[-1].split(":", 1)[1].strip()


def no_room() -> ModelLoadError:
    """The error of a GPU without memory for the model."""
    return ModelLoadError("Waited 600 s for 6.7 GB of GPU memory.")


class FakeNameStore:
    """Records of the search in a dictionary."""

    def __init__(self) -> None:
        """Start empty."""
        self.records: Dict[str, SpeakerNames] = {}

    def save_names(self, names: SpeakerNames) -> None:
        """Keep the record."""
        self.records[names.session] = names

    def names(self, session_id: str) -> Optional[SpeakerNames]:
        """Return the record, if any."""
        return self.records.get(session_id)
