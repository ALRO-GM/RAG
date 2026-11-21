"""Ports of the orchestrator: what it needs from the outside.

Knowledge and reasoner are reached with the JSON documents of their
contracts (``knowledge.query.v1`` / ``knowledge.result.v1``,
``reasoner.request.v1`` / ``reasoner.response.v1``), so no adapter has
to import the services' own code.
"""

from typing import Any, Callable, Dict, Iterator, List, Protocol

from orchestrator.domain.events import LiveEvent, Transcript
from orchestrator.domain.records import SuggestionRecord, TurnRecord
from orchestrator.domain.speech import SpeechChunk

Document = Dict[str, Any]
"""A contract document as plain JSON."""


class CancelToken:
    """A flag a caller sets to say its answer is no longer wanted."""

    def __init__(self) -> None:
        """Start not cancelled."""
        self.cancelled = False

    def cancel(self) -> None:
        """Mark the call as stale."""
        self.cancelled = True


class KnowledgePort(Protocol):
    """Per-domain retrieval."""

    def query(self, query: Document) -> Document:
        """Answer a ``knowledge.query.v1`` with a ``knowledge.result.v1``.

        Raises:
            KnowledgeUnavailable: The service could not answer.
        """
        ...


class ReasonerPort(Protocol):
    """The reasoner service."""

    def reason(self, request: Document, cancel: CancelToken) -> Document:
        """Answer a ``reasoner.request.v1`` with a ``reasoner.response.v1``.

        A model failure is a ``status`` of the response, not an
        exception; an adapter checks ``cancel`` before it sends.

        Raises:
            ReasonerUnavailable: The service could not be reached.
        """
        ...


class TextToSpeech(Protocol):
    """Text to speech (Piper in a process of its own)."""

    def synthesize(
        self, request: Document, cancel: CancelToken
    ) -> Iterator[SpeechChunk]:
        """Speak a ``tts.request.v1``: one chunk per sentence, in order.

        With ``stream`` true each sentence is yielded as soon as it is
        synthesised; no more are made once ``cancel`` is set, or once a
        later request names this one in its ``interrupt_id`` (barge-in).

        Raises:
            TtsUnavailable: The synthesiser could not be run.
        """
        ...


class StreamingReasoner(Protocol):
    """A reasoner that hands its output over while it is written."""

    def reason_stream(
        self, request: Document, cancel: CancelToken,
        on_text: Callable[[str], None],
    ) -> Document:
        """Like :meth:`ReasonerPort.reason`, passing text pieces on.

        ``on_text`` receives the model's raw output piece by piece;
        the return value is the whole ``reasoner.response.v1``. Once
        ``cancel`` is set no more pieces are passed.

        Raises:
            ReasonerUnavailable: The service could not be reached.
        """
        ...


class Clock(Protocol):
    """Monotonic time and waiting (replaced by a fake in tests)."""

    def now(self) -> float:
        """Return monotonic seconds."""
        ...

    def sleep(self, seconds: float) -> None:
        """Wait ``seconds``."""
        ...


class Executor(Protocol):
    """Runs a call apart from the event loop."""

    def submit(self, job: Callable[[], None]) -> None:
        """Run ``job`` (inline or on a thread)."""
        ...

    def wait_idle(self) -> None:
        """Return when every submitted job has finished."""
        ...


class SuggestionSink(Protocol):
    """Where the suggestion records go (a JSONL file, a UI)."""

    def write(self, record: SuggestionRecord) -> None:
        """Store one record."""
        ...


class BotOutput(Protocol):
    """Where the voice bot's messages go (the session's feed and log)."""

    def publish(self, message: Document) -> None:
        """Send one message to the page (speech, interrupt, turn state)."""
        ...

    def record(self, record: TurnRecord) -> None:
        """Keep the record of a finished turn."""
        ...


class TurnSink(Protocol):
    """Where the voice bot's turn records are logged (a JSONL file)."""

    def write(self, record: TurnRecord) -> None:
        """Store one record."""
        ...


class EventHandler(Protocol):
    """What receives a conversation's events one by one."""

    def on_event(self, event: LiveEvent) -> None:
        """Handle one ``live.event.v1``."""
        ...

    def finish(self) -> None:
        """Called once after the last event."""
        ...


class LiveTranscriber(Protocol):
    """The transcriber's live sessions (``/api/live/sessions``)."""

    def start(self, body: Document) -> Document:
        """Start a live session; return the transcriber's session.

        Raises:
            TranscriberUnavailable: The transcriber did not start it.
        """
        ...

    def stop(self, session_id: str) -> None:
        """Stop a live session (nothing if it already ended).

        Raises:
            TranscriberUnavailable: The transcriber could not be asked.
        """
        ...


class EventSource(Protocol):
    """The ``live.event.v1`` stream of a running session."""

    def follow(
        self, session_id: str, stop: CancelToken
    ) -> Iterator[Document]:
        """Yield the session's events until it ends or ``stop`` is set.

        Raises:
            TranscriberUnavailable: The stream could not be opened.
        """
        ...


class TranscriptStore(Protocol):
    """Recorded conversations a replay can use."""

    def list(self) -> List[Document]:
        """Return ``[{"id", "title", "utterances", "speakers"}]``."""
        ...

    def load(self, transcript_id: str) -> Transcript:
        """Return one ``transcript.v1``.

        Raises:
            KeyError: There is no transcript with that id.
        """
        ...


class Runner(Protocol):
    """Runs a long job (a session) apart from the caller."""

    def start(self, job: Callable[[], None]) -> None:
        """Run ``job`` in the background."""
        ...


class KnowledgeUnavailable(Exception):
    """The knowledge service did not answer."""


class ReasonerUnavailable(Exception):
    """The reasoner service could not be reached."""


class TtsUnavailable(Exception):
    """The text-to-speech process could not speak."""


class TranscriberUnavailable(Exception):
    """The transcriber refused or could not be reached."""
