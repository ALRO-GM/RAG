"""Ports of the orchestrator: what it needs from the outside.

Knowledge and reasoner are reached with the JSON documents of their
contracts (``knowledge.query.v1`` / ``knowledge.result.v1``,
``reasoner.request.v1`` / ``reasoner.response.v1``), so no adapter has
to import the services' own code.
"""

from typing import (
    AbstractSet, Any, Callable, Dict, Iterator, List, Mapping, Optional,
    Protocol, Tuple,
)

from orchestrator.domain.candidate import CandidateProfile, CoreQuestion
from orchestrator.domain.demo import DemoScript
from orchestrator.domain.demo_voices import DemoVoice, DemoVoiceManifest
from orchestrator.domain.events import LiveEvent, Transcript
from orchestrator.domain.mandatory_lists import OwnerList
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
        self, session_id: str, stop: CancelToken, from_seq: int = 0,
    ) -> Iterator[Document]:
        """Yield the session's events until it ends or ``stop`` is set.

        ``from_seq``: the first event wanted (the stream is resumed
        there after a drop, #264).

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


class ReasonerRejected(ReasonerUnavailable):
    """The reasoner refused the request as invalid (HTTP 422), e.g. a
    template revision it does not know yet (#286: a RAG not restarted
    with the orchestrator)."""


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

    def check_list(self, listed: Document, text: str) -> Document:
        """Like :meth:`check`, against a whole list sent with the text.

        ``listed`` has the shape of :meth:`items` (#294: the list a
        session uses, the general one with an owner's list).

        Raises:
            MandatoryUnavailable: The sentence could not be checked.
        """
        ...

    def review(self, listed: Document) -> Document:
        """Return ``close: [{"item", "example", "other",
        "other_example", "score"}]``: the examples of ``listed`` that
        come close to an example of another of its items (#294).

        Raises:
            MandatoryUnavailable: The list could not be checked.
        """
        ...


class MandatoryListStore(Protocol):
    """The owner's lists of «Lo que hay que decir» (#294)."""

    def list(self) -> List[OwnerList]:
        """Return every readable list (current versions)."""
        ...

    def load(self, list_id: str, version: Optional[int] = None) -> OwnerList:
        """Return a list, its current or an earlier ``version``.

        Raises:
            KeyError: There is no such list (or version).
        """
        ...

    def save(self, owned: OwnerList) -> None:
        """Keep ``owned`` as the current version (earlier ones stay)."""
        ...

    def delete(self, list_id: str) -> bool:
        """Delete a list (its versions stay readable by
        :meth:`load` with a version); tell whether it existed."""
        ...

    def taken(self) -> List[str]:
        """Return every id ever used, deleted lists' too (never reused)."""
        ...

    def offer_list(self, plan_id: str) -> Optional[str]:
        """Return the id of the list a job offer uses (``None``: general)."""
        ...

    def assign(self, plan_id: str, list_id: Optional[str]) -> None:
        """Make a job offer use ``list_id`` (``None``: the general list)."""
        ...

    def offers(self) -> Dict[str, str]:
        """Return every job offer with a list: plan id → list id."""
        ...


class DemoScripts(Protocol):
    """The scripted demos of the live page (``live.demo.v1``, LV-17)."""

    def list(self) -> List[Document]:
        """Return every demo as a page lists it."""
        ...

    def load(self, demo_id: str) -> DemoScript:
        """Return one demo.

        Raises:
            KeyError: There is no such demo.
        """
        ...


class AvoidUnavailable(Exception):
    """The check of questions better avoided could not be made.

    ``reason`` is ``no_rules`` (the RAG has no interview rules) or
    ``unreachable`` (the RAG did not answer or answered wrongly).
    """

    def __init__(self, message: str, reason: str = "unreachable"):
        """Keep the message and why."""
        super().__init__(message)
        self.reason = reason


class AvoidPort(Protocol):
    """The RAG's check of interview questions better avoided (LV-17)."""

    def check(self, text: str, language: str) -> Document:
        """Return ``hints: [{"category", "title", "why", "alternative"}]``.

        Raises:
            AvoidUnavailable: The sentence could not be checked (no
                rules, no service).
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

    @property
    def gpu(self) -> bool:
        """Whether the model runs on the GPU (it then needs the lease)."""
        ...

    @property
    def loaded(self) -> bool:
        """Whether the model is loaded (on the GPU: memory is taken)."""
        ...

    def speak(
        self, text: str, reference: str, language: str, cancel: CancelToken,
        controls: Optional[Mapping[str, float]] = None,
    ) -> Tuple[bytes, int]:
        """Say one sentence like ``reference``; return PCM s16le and rate.

        ``controls`` are the voice's Piper controls (``length_scale``,
        ``noise_scale``, ``noise_w``, ``sentence_silence``); a cloner
        that speaks over Piper applies them, one that does not ignores
        them.

        Raises:
            TtsUnavailable: The cloner could not speak.
        """
        ...

    def warm(
        self, reference: str, language: str,
        controls: Optional[Mapping[str, float]] = None,
    ) -> None:
        """Load the model and the reference voice, off the hot path."""
        ...

    @property
    def generation(self) -> int:
        """Which model start is current (it grows with every start).

        A caller that decides to stop the model reads this *before* it
        looks at why, and passes it to :meth:`unload`.
        """
        ...

    def unload(self, generation: Optional[int] = None) -> None:
        """Stop the model now, whoever holds it (a load in progress is
        killed): its (GPU) memory is free when this returns.

        With ``generation`` the stop is for that start only: if another
        start has begun since, it is a no-op, so a stale decision cannot
        kill a model a newer session just started.
        """
        ...

    def release(self) -> None:
        """This holder is done with the model: a GPU one is stopped once
        no holder remains."""
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


class CandidateStore(Protocol):
    """Where the candidate profiles of each role are kept (#235)."""

    def load(self, plan_id: str, candidate_id: str) -> CandidateProfile:
        """Return one profile (``KeyError`` when there is none)."""
        ...

    def save(self, profile: CandidateProfile) -> None:
        """Keep ``profile`` (replacing its older copy)."""
        ...

    def list(self, plan_id: str) -> List[CandidateProfile]:
        """Every readable profile of a role."""
        ...

    def plans(self) -> List[str]:
        """The roles (plan ids) that have profiles."""
        ...

    def ids(self, plan_id: str) -> List[str]:
        """The ids of every stored profile of a role, readable or not."""
        ...

    def delete(self, plan_id: str, candidate_id: str) -> bool:
        """Delete one profile; ``False`` when there was none."""
        ...


class CoreQuestionSource(Protocol):
    """The core questions of KN-4 (``interview.core.v1``)."""

    def questions(self) -> List[CoreQuestion]:
        """Return them in order (empty when the file cannot be read)."""
        ...


class DemoVoiceStore(Protocol):
    """Where the scripted examples' clips are kept (#254).

    Outside the repository, files only the owner may read. A file name
    is a clip's, never a path: the store checks it.
    """

    def manifest(self, demo_id: str) -> Optional[DemoVoiceManifest]:
        """The demo's manifest; ``None`` when there is none (or unreadable)."""
        ...

    def save_manifest(self, manifest: DemoVoiceManifest) -> None:
        """Write the manifest (atomically)."""
        ...

    def write_clip(self, demo_id: str, name: str, wav: bytes) -> None:
        """Write one clip (atomically)."""
        ...

    def read_clip(self, demo_id: str, name: str) -> bytes:
        """Return one clip; ``KeyError`` when there is none."""
        ...

    def has_clip(self, demo_id: str, name: str) -> bool:
        """Whether the clip's file is there."""
        ...

    def delete_clip(self, demo_id: str, name: str) -> None:
        """Delete one clip (nothing when it is gone already)."""
        ...

    def clear(self, demo_id: str) -> int:
        """Delete every clip and the manifest of a demo; how many clips."""
        ...

    def sweep(self, demo_id: str, keep: AbstractSet[str]) -> int:
        """Delete the demo's clip files (and half-written ones) not in
        ``keep``; return how many."""
        ...


class ExclusiveGpu(Protocol):
    """The GPU lease for work that must run alone (a demo render, #254)."""

    def take_alone(self, need_mib: int) -> Optional[str]:
        """Hold the lease only if nobody holds it (a live session
        included), wait for the memory; return why not. Paired with one
        :meth:`drop` whenever it returns."""
        ...

    def drop(self) -> None:
        """Give the lease back."""
        ...


class DemoSpeaker(Protocol):
    """Says a demo line with a chosen voice, ahead of time (#254)."""

    def engine(self, voice: DemoVoice) -> str:
        """What will speak ``voice`` (``pocket-cpu``, ``qwen3-cuda``)."""
        ...

    def need_mib(self, voice: DemoVoice) -> int:
        """GPU memory ``voice`` needs (0: none, no lease)."""
        ...

    def speak(
        self, text: str, voice: DemoVoice, reference: Optional[str],
        language: str,
    ) -> Tuple[bytes, int]:
        """Return PCM s16le mono of the whole line and its rate.

        ``reference`` is the clone's reference clip (``None`` for a
        stock voice).

        Raises:
            TtsUnavailable: The engine could not speak.
        """
        ...

    def close(self) -> None:
        """Stop every engine this speaker started."""
        ...
