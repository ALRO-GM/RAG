"""Ports of the orchestrator: what it needs from the outside.

Knowledge and reasoner are reached with the JSON documents of their
contracts (``knowledge.query.v1`` / ``knowledge.result.v1``,
``reasoner.request.v1`` / ``reasoner.response.v1``), so no adapter has
to import the services' own code.
"""

from typing import (
    AbstractSet, Any, Callable, Dict, Iterator, List, Optional, Protocol,
    Tuple,
)

from orchestrator.domain.events import LiveEvent, Transcript
from orchestrator.domain.records import SuggestionRecord, TurnRecord
from orchestrator.domain.speech import SpeechChunk
from orchestrator.domain.voices import EngineVoice, VoiceProfile

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

    def rename(self, session_id: str, speaker: str, name: str) -> None:
        """Give a speaker of a live session a name (for its hand-off).

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


class SettingsSource(Protocol):
    """The pipeline settings saved on the web's «Ajustes» (``settings.v1``)."""

    def current(self) -> Optional[Dict[str, str]]:
        """Return the saved choices, or ``None`` (keep the product's)."""
        ...


class MandatoryUnavailable(Exception):
    """The mandatory items could not be read or checked."""


class MandatoryPort(Protocol):
    """A knowledge domain's mandatory items (LV-16, served by the RAG)."""

    def items(self, domain: str) -> Document:
        """Return the domain's list: ``domain``, ``version``,
        ``threshold``, ``remind_seconds`` and ``items`` (``id``,
        ``text``, ``phase``, ``must_say``); no items when it has none.

        Raises:
            MandatoryUnavailable: The list could not be read.
        """
        ...

    def check(self, domain: str, text: str) -> Document:
        """Return ``items: [{"id", "score", "said"}]`` for one sentence.

        Raises:
            MandatoryUnavailable: The sentence could not be checked.
        """
        ...


class VoiceProfiles(Protocol):
    """The enrolled people's voice profiles (kept by the transcriber)."""

    def get(self, speaker_id: str) -> Optional[VoiceProfile]:
        """Return the profile of ``speaker_id``, or ``None``.

        Never raises: an unreadable profile reads as no profile.
        """
        ...


class HandoffSink(Protocol):
    """Where a session's hand-off record goes when it ends (LV-16)."""

    def write(self, record: Document) -> None:
        """Keep the record: names, mode and the mandatory items."""
        ...


class VoiceCloner(Protocol):
    """Speech in an enrolled person's own voice (a cloning model)."""

    def speak(
        self, text: str, reference: str, language: str, cancel: CancelToken
    ) -> Tuple[bytes, int]:
        """Say one sentence like ``reference``; return PCM s16le and rate.

        Raises:
            TtsUnavailable: The cloner could not speak.
        """
        ...

    def warm(self, reference: str, language: str) -> None:
        """Load the model and the reference voice, off the hot path."""
        ...


class SpeechStream(Protocol):
    """A natural-voice engine that streams a sentence's audio (#214)."""

    @property
    def loaded(self) -> bool:
        """Whether the model is loaded (on the GPU: memory is taken)."""
        ...

    def stream(
        self, text: str, voice: EngineVoice, language: str,
        cancel: CancelToken,
    ) -> Iterator[Tuple[bytes, int]]:
        """Yield PCM s16le pieces of one sentence and their rate.

        Stops early once ``cancel`` is set.

        Raises:
            TtsUnavailable: The engine could not speak.
        """
        ...

    def warm(self, voice: EngineVoice) -> None:
        """Load the model (and the voice), off the hot path.

        Raises:
            TtsUnavailable: The engine could not be loaded.
        """
        ...


class SpeechEngines(Protocol):
    """Where the natural-voice engines come from."""

    def installed(self) -> AbstractSet[str]:
        """The engines whose worker is installed (``pocket``, ``qwen3``)."""
        ...

    def get(self, voice: EngineVoice) -> SpeechStream:
        """Return the engine that speaks ``voice``.

        An engine on the CPU is shared and stays loaded; one on the GPU
        is the caller's until :meth:`release`.
        """
        ...

    def release(self, engine: SpeechStream) -> None:
        """Give an engine back (a GPU one is stopped: its memory freed)."""
        ...


class GpuGate(Protocol):
    """May a natural voice use the GPU now? (``gpu-lease.v1``)"""

    def check(self, need_mib: int) -> Optional[str]:
        """Return why not (``None``: it may), ``need_mib`` free on it.

        Never takes the lease: during a voice-bot session the live
        session's lease is what keeps other GPU work off the card.
        """
        ...

    def take(self, need_mib: int) -> Optional[str]:
        """Hold the lease if nobody does (a sample); return why not."""
        ...

    def drop(self) -> None:
        """Release the lease :meth:`take` wrote, if any."""
        ...
