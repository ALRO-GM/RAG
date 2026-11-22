"""Use case: the voice bot speaks with a natural voice (#214).

:class:`NaturalVoice` is a :class:`~orchestrator.application.ports.
TextToSpeech` over a chain of streaming engines
(:func:`~orchestrator.domain.voices.voice_chain`: Qwen3-TTS or Pocket
TTS on the GPU, then Pocket TTS on the CPU) with Piper last. For each
sentence the first engine that can speak it does:

- an engine on the GPU is only used while the GPU gate allows it (the
  live session's ``gpu-lease.v1`` lease is held and, before it loads,
  the memory it needs is free); when the lease goes, a loaded GPU
  engine is stopped at once, so its memory is free again;
- the first audio of a sentence must come within ``budget_ms``;
  otherwise that sentence is cancelled on that engine and the next one
  says it; while an engine is still busy (loading, or finishing a
  cancelled sentence) the next sentences skip it;
- an engine that fails is not asked again in this session;
- a sentence is streamed as the engine speaks it, in growing pieces
  (:class:`~orchestrator.domain.speech.AudioJoiner`); a piece that
  does not come within ``chunk_timeout_ms`` ends the sentence there.

Whenever the engine that speaks changes (from the first of the chain,
at the start), ``report`` is told which one and why the ones before it
did not (the session's feed says so).

A session's first sentences wait for the voice to get ready: once
:meth:`NaturalVoice.warming` (or :meth:`NaturalVoice.warm`) has started
the warm-up, :meth:`NaturalVoice.synthesize` waits until every engine
of the chain has loaded, at most ``ready_wait_s`` from that start, and
only then applies the budget (the opening line of an interview used to
be said while Qwen3-TTS was still loading, so Pocket or Piper said it;
``docs/live/NATURAL_TTS.md`` §8). After :meth:`NaturalVoice.close`
nothing more is said, quietly.

Only the text reaches the natural engines: Piper's controls (speed,
``noise_scale``, ``noise_w``, ``sentence_silence``, and the speed that
``match_rate`` sets) reach Piper alone, when it says a sentence.
"""

import queue
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import (
    Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union,
)

from orchestrator.application.ports import (
    CancelToken,
    Document,
    GpuGate,
    SpeechEngines,
    SpeechStream,
    TextToSpeech,
    TtsUnavailable,
)
from orchestrator.application.voice import DEFAULT_BUDGET_MS
from orchestrator.domain.speech import (
    AudioJoiner,
    SpeechChunk,
    split_sentences,
    wav_bytes,
)
from orchestrator.domain.voices import EngineVoice

CHUNK_TIMEOUT_MS = 4000.0
"""Longest wait for the next piece of a sentence already started."""
WARM_WAIT_S = 30.0
"""How long the warm-up waits for the GPU (the live session takes the
lease and waits up to 20 s for the image worker to free the card)."""
READY_WAIT_S = 12.0
"""How long a session's first sentences wait for the voice to load
(from the start of the warm-up); after that, the fallbacks speak."""
GATE_POLL_S = 0.25
"""How often the warm-up asks the GPU gate again while it refuses."""
PIPER = "piper"

_Item = Union[Tuple[bytes, int], BaseException, None]


@dataclass
class _Job:
    """One sentence being spoken by one engine (internal)."""

    cancel: CancelToken = field(default_factory=CancelToken)
    pieces: "queue.Queue[_Item]" = field(default_factory=queue.Queue)


class _Tier:
    """One engine of the chain and its state in this session (internal)."""

    def __init__(self, voice: EngineVoice):
        """Nothing loaded, nothing running."""
        self.voice = voice
        self.engine: Optional[SpeechStream] = None
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.busy: Optional["Future[None]"] = None
        self.failed = ""

    @property
    def name(self) -> str:
        """The engine's label."""
        return self.voice.name

    def running(self) -> bool:
        """Whether a sentence (or the loading) is still in progress."""
        return self.busy is not None and not self.busy.done()


class NaturalVoice:
    """Speak with the first engine of a chain that can, Piper last.

    Args:
        chain: The engines, in order (see the module).
        engines: Where they come from.
        fallback: Piper.
        gate: The GPU gate (``None``: GPU engines are never used).
        budget_ms: Longest wait for a sentence's first audio.
        chunk_timeout_ms: Longest wait for each later piece.
        language: The sentences' language.
        report: Told ``(engine, reason)`` when the speaker changes.
        warm_wait_s: How long :meth:`warm` waits for the GPU.
        sleep: Waits (tests replace it).
        ready_wait_s: How long the first sentences wait for the warm-up.
    """

    def __init__(
        self,
        chain: Sequence[EngineVoice],
        engines: SpeechEngines,
        fallback: TextToSpeech,
        gate: Optional[GpuGate] = None,
        budget_ms: float = DEFAULT_BUDGET_MS,
        chunk_timeout_ms: float = CHUNK_TIMEOUT_MS,
        language: str = "es",
        report: Optional[Callable[[str, str], None]] = None,
        warm_wait_s: float = WARM_WAIT_S,
        sleep: Callable[[float], None] = time.sleep,
        ready_wait_s: float = READY_WAIT_S,
    ):
        """Keep the chain; nothing is loaded until :meth:`warm` or use."""
        self.tiers = [_Tier(voice) for voice in chain]
        self.engines = engines
        self.fallback = fallback
        self.gate = gate
        self.budget_ms = budget_ms
        self.chunk_timeout_ms = chunk_timeout_ms
        self.language = language
        self.report = report
        self.warm_wait_s = warm_wait_s
        self.sleep = sleep
        self.spoken: Dict[str, int] = {}
        self.first_speaker = ""
        self._speaker = self.tiers[0].name if self.tiers else PIPER
        self.ready_wait_s = ready_wait_s
        self._lock = threading.Lock()
        self._closed = False
        self._ready = threading.Event()
        self._ready_by: Optional[float] = None

    @property
    def ready(self) -> bool:
        """Whether the warm-up has finished (every engine tried)."""
        return self._ready.is_set()

    def warming(self) -> None:
        """Say a warm-up starts now: the first sentences will wait for it.

        Called just before :meth:`warm` is run in the background, so a
        sentence asked for before that thread runs waits too.
        """
        if self._ready_by is None:
            self._ready_by = time.monotonic() + self.ready_wait_s

    def wait_ready(self) -> bool:
        """Wait for the warm-up (at most until its bound); whether it ended.

        Returns at once when no warm-up was started or the voice was
        closed.
        """
        while not self._ready.is_set() and not self._closed:
            left = self._left()
            if left <= 0:
                break
            self._ready.wait(min(left, 0.05))
        return self._ready.is_set()

    def warm(self) -> None:
        """Load every engine of the chain (in the background).

        The engines load side by side (Pocket on the CPU never waits for
        a GPU model or its lease). A GPU engine waits up to
        ``warm_wait_s`` for the gate; one that cannot load is marked
        failed for this session. An engine that is already loaded
        (Pocket on the CPU, kept by the orchestrator for every session)
        is not asked again. :attr:`ready` once every engine was tried.
        """
        self.warming()
        try:
            threads = [threading.Thread(target=self._warm, args=(tier,),
                                        daemon=True)
                       for tier in self.tiers[1:]]
            for thread in threads:
                thread.start()
            if self.tiers:
                self._warm(self.tiers[0])
            for thread in threads:
                thread.join()
        finally:
            self._ready.set()

    def _warm(self, tier: _Tier) -> None:
        """Load one tier's engine (see :meth:`warm`)."""
        need = tier.voice.need_mib
        waited = 0.0
        while need and self._gpu_refusal(tier) is not None:
            if waited >= self.warm_wait_s or self._closed:
                break
            self.sleep(GATE_POLL_S)
            waited += GATE_POLL_S
        if need and self._gpu_refusal(tier) is not None:
            return
        if self._closed:
            return
        engine = self._engine(tier)
        if not engine.loaded:
            try:
                engine.warm(tier.voice)
            except TtsUnavailable as exc:
                tier.failed = str(exc)[:120] or "could not load"
        if self._closed:  # the session ended while it loaded
            self._unload(tier)

    def _left(self) -> float:
        """Seconds the first sentences may still wait for the warm-up."""
        if self._ready_by is None:
            return 0.0
        return self._ready_by - time.monotonic()

    def _await_ready(self, cancel: CancelToken) -> None:
        """Wait for the warm-up, unless cancelled, closed or past bound."""
        while (not self._ready.is_set() and not cancel.cancelled
               and not self._closed):
            left = self._left()
            if left <= 0:
                return
            self._ready.wait(min(left, 0.05))

    def synthesize(
        self, request: Document, cancel: CancelToken
    ) -> Iterator[SpeechChunk]:
        """Speak ``request`` sentence by sentence (see the module)."""
        text = str(request.get("text") or "")
        sentences = (split_sentences(text) if request.get("stream", True)
                     else [" ".join(text.split())])
        audio_format = str(request.get("format") or "pcm_s16le")
        request_id = str(request["request_id"])
        self._await_ready(cancel)
        for seq, sentence in enumerate(s for s in sentences if s):
            if cancel.cancelled or self._closed:
                return
            started = time.perf_counter()
            reasons: List[str] = []
            for tier in self.tiers:
                pieces = self._start(tier, sentence, cancel, reasons)
                if pieces is None:
                    continue
                self._spoke(tier.name, reasons)
                text = sentence  # later pieces of it carry no text
                for pcm, rate in pieces:
                    audio = wav_bytes(pcm, rate) if audio_format == "wav" \
                        else pcm
                    yield SpeechChunk(
                        request_id=request_id, seq=seq, text=text,
                        audio=audio, format=audio_format, sample_rate=rate,
                        synth_ms=round(
                            (time.perf_counter() - started) * 1000, 1),
                    )
                    text = ""
                break
            else:
                if cancel.cancelled or self._closed:
                    return
                self._spoke(PIPER, reasons)
                yield from self._piper(request, sentence, seq, cancel)

    def close(self) -> None:
        """Stop waiting, and give every engine back (GPU memory freed).

        An engine still loading is given back once it has loaded (by
        :meth:`warm`, or here in the background), so ending a session
        never waits for a model.
        """
        self._closed = True
        for tier in self.tiers:
            tier.pool.shutdown(wait=False, cancel_futures=True)
        done = threading.Thread(target=self._unload_all, daemon=True)
        done.start()
        done.join(timeout=2.0)

    def _unload_all(self) -> None:
        """Give every tier's engine back."""
        for tier in self.tiers:
            self._unload(tier)

    # ---------------------------------------------------------- internal

    def _engine(self, tier: _Tier) -> SpeechStream:
        """Return the tier's engine, getting it the first time."""
        with self._lock:
            if tier.engine is None:
                tier.engine = self.engines.get(tier.voice)
            return tier.engine

    def _unload(self, tier: _Tier) -> None:
        """Give the tier's engine back."""
        with self._lock:
            engine, tier.engine = tier.engine, None
        if engine is not None:
            self.engines.release(engine)

    def _gpu_refusal(self, tier: _Tier) -> Optional[str]:
        """Why a GPU tier may not run now (``None``: it may)."""
        if not tier.voice.need_mib:
            return None
        if self.gate is None:
            return "no GPU gate"
        loaded = tier.engine is not None and tier.engine.loaded
        return self.gate.check(0 if loaded else tier.voice.need_mib)

    def _start(
        self, tier: _Tier, sentence: str, cancel: CancelToken,
        reasons: List[str],
    ) -> Optional[Iterator[Tuple[bytes, int]]]:
        """Start ``sentence`` on ``tier``; its pieces, or ``None``.

        ``None`` too once the voice is closed (nothing more is said).
        """
        if self._closed:
            return None
        if tier.failed:
            reasons.append(f"{tier.name} failed ({tier.failed})")
            return None
        if tier.running():
            reasons.append(f"{tier.name} busy")
            return None
        refusal = self._gpu_refusal(tier)
        if refusal is not None:
            reasons.append(f"{tier.name}: {refusal}")
            if tier.engine is not None:
                self._unload(tier)  # the lease is gone: free the card
            return None
        job = _Job()
        engine = self._engine(tier)
        try:
            tier.busy = tier.pool.submit(self._run, engine, tier.voice,
                                         sentence, job)
        except RuntimeError:  # closed meanwhile: the pool is shut down
            return None
        first = self._next(job, self.budget_ms, cancel)
        if first is None:
            job.cancel.cancel()
            if not cancel.cancelled:
                reasons.append(f"{tier.name} missed the "
                               f"{self.budget_ms:g} ms budget")
            return None
        if isinstance(first, BaseException):
            tier.failed = str(first)[:120] or type(first).__name__
            reasons.append(f"{tier.name} failed ({tier.failed})")
            return None
        return self._pieces(first, job, cancel)

    def _run(self, engine: SpeechStream, voice: EngineVoice, sentence: str,
             job: _Job) -> None:
        """Speak one sentence into the job's queue (the tier's thread)."""
        try:
            for piece in engine.stream(sentence, voice, self.language,
                                       job.cancel):
                job.pieces.put(piece)
            job.pieces.put(None)
        except TtsUnavailable as exc:
            job.pieces.put(exc)

    def _next(
        self, job: _Job, timeout_ms: float, cancel: CancelToken
    ) -> Optional[_Item]:
        """The job's next item; ``None`` on timeout, cancel, close or end."""
        deadline = time.monotonic() + timeout_ms / 1000.0
        while not cancel.cancelled and not self._closed:
            left = deadline - time.monotonic()
            if left <= 0:
                return None
            try:
                return job.pieces.get(timeout=min(left, 0.05))
            except queue.Empty:
                continue
        return None

    def _pieces(
        self, first: Tuple[bytes, int], job: _Job, cancel: CancelToken
    ) -> Iterator[Tuple[bytes, int]]:
        """Yield the sentence's audio in growing pieces until it ends."""
        rate = first[1]
        joiner = AudioJoiner(rate)
        item: Optional[_Item] = first
        while isinstance(item, tuple):
            if item[1] != rate:  # never mixed within one sentence
                break
            ready = joiner.add(item[0])
            if ready is not None:
                yield ready, rate
            if cancel.cancelled or self._closed:
                job.cancel.cancel()
                return
            item = self._next(job, self.chunk_timeout_ms, cancel)
        if cancel.cancelled or self._closed:
            job.cancel.cancel()
            return
        if item is None or isinstance(item, BaseException):
            job.cancel.cancel()
        rest = joiner.flush()
        if rest is not None:
            yield rest, rate

    def _spoke(self, name: str, reasons: List[str]) -> None:
        """Count who spoke; report when the speaker changes."""
        self.spoken[name] = self.spoken.get(name, 0) + 1
        if not self.first_speaker:
            self.first_speaker = name
        if name != self._speaker:
            self._speaker = name
            if self.report is not None:
                self.report(name, "; ".join(reasons))

    def _piper(
        self, request: Document, sentence: str, seq: int,
        cancel: CancelToken,
    ) -> Iterator[SpeechChunk]:
        """Say one sentence with Piper, numbered as the others."""
        single = dict(request, text=sentence, stream=False,
                      interrupt_id=None,
                      request_id=f"{request['request_id']}-p{seq}"[-64:])
        for chunk in self.fallback.synthesize(single, cancel):
            yield SpeechChunk(
                request_id=str(request["request_id"]), seq=seq,
                text=chunk.text, audio=chunk.audio, format=chunk.format,
                sample_rate=chunk.sample_rate, synth_ms=chunk.synth_ms,
            )


@dataclass
class NaturalWiring:
    """What the natural voices need (``None`` in the sessions: off).

    ``reference`` is Qwen3's default voice (clip, transcript; ``None``:
    the ``qwen3`` option falls back); ``budget_ms`` the longest wait
    for a sentence's first audio in a session; ``ready_wait_s`` how
    long a session's first sentences wait for its voice to load.
    """

    engines: SpeechEngines
    gate: Optional[GpuGate] = None
    reference: Optional[Tuple[str, str]] = None
    budget_ms: float = DEFAULT_BUDGET_MS
    ready_wait_s: float = READY_WAIT_S


SAMPLE_BUDGET_MS = 120000.0
"""«Escuchar ejemplo» is not live: it waits for a GPU model to load."""


def gpu_need(voice: NaturalVoice) -> int:
    """The most GPU memory one engine of ``voice``'s chain needs."""
    return max((tier.voice.need_mib for tier in voice.tiers), default=0)
