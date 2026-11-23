"""Use case: the voice bot's voice (Piper, or a cloned voice) and samples.

:class:`ClonedVoice` is a :class:`~orchestrator.application.ports.
TextToSpeech` that speaks each sentence in an enrolled person's voice
(``voicebot.voice = clone:<person>``, ``docs/live/VOICE_CLONING.md``)
and falls back to Piper:

- it is only built for a profile that is clonable (consent to cloning
  and a reference clip; :func:`~orchestrator.domain.voices.plan_voice`
  decides, and the constructor refuses anything else);
- a sentence the cloner does not finish within ``budget_ms`` is said
  by Piper (the cloner's late audio is dropped), and while the cloner
  is still busy with it the next sentences are said by Piper too;
- a cloner that fails (:class:`TtsUnavailable`) is not asked again in
  this session: Piper says the rest;
- a cloner on the GPU runs **under the live session's GPU lease**
  (``gpu-lease.v1``, :class:`~orchestrator.application.ports.GpuGate`),
  like the natural voices: it loads only while a ``live`` lease is held
  and the memory it needs (:data:`CLONE_NEED_MIB`) is free, it is
  stopped (memory freed) as soon as the lease goes and when the
  session ends, and meanwhile Piper speaks. Without a gate a GPU cloner
  is never used (``docs/live/VOICE_CLONING.md`` §4);
- Piper's controls of the request (speed, ``noise_scale``, ``noise_w``,
  ``sentence_silence``) reach the cloner, which speaks over Piper and
  so keeps the prosody the converter then recolours.

:func:`speak_sample` renders the fixed sentence of the «Ajustes» page's
«Escuchar ejemplo» button with the controls being edited, as one WAV.
No notice is spoken when the bot speaks with a cloned voice (owner's
decision, #207).
"""

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Callable, Dict, Iterator, List, Optional, Tuple

from orchestrator.application.ports import (
    CancelToken,
    Document,
    GpuGate,
    TextToSpeech,
    TtsUnavailable,
    VoiceCloner,
)
from orchestrator.domain.config import Tts
from orchestrator.domain.speech import (
    SpeechChunk,
    pcm_of,
    split_sentences,
    tts_request,
    wav_bytes,
)
from orchestrator.domain.voices import (
    SAMPLE_TEXT,
    VoicePlan,
    VoiceProfile,
)

DEFAULT_BUDGET_MS = 800.0
"""Longest wait for one cloned sentence before Piper says it.

The voice bot's target is first audio within 2.5 s of the end of the
user's sentence; without the synthesis it took 1,638 ms at p90 (1,731
ms end to end minus Piper's 93 ms, ``docs/live/VOICEBOT.md``), which
leaves 862 ms, rounded down."""
CLONE_NEED_MIB = 800
"""GPU memory a cloner needs (OpenVoice v2 peaked at 782 MiB,
``docs/live/VOICE_CLONING.md`` §3, rounded up)."""
WARM_WAIT_S = 30.0
"""How long the warm-up waits for the GPU gate."""
GATE_POLL_S = 0.25
"""How often the warm-up asks the gate again while it refuses."""
CONTROLS = ("length_scale", "noise_scale", "noise_w", "sentence_silence")
"""The Piper controls a request carries."""


def request_controls(request: Document) -> Dict[str, float]:
    """The Piper controls ``request`` sets (the ones that are numbers)."""
    return {name: float(request[name]) for name in CONTROLS
            if isinstance(request.get(name), (int, float))}


def tts_controls(tts: Tts, length_scale: float) -> Dict[str, float]:
    """The Piper controls of ``tts`` with ``length_scale`` as the speed.

    The ones left unset (``None``) are left out: Piper's own apply.
    """
    controls = {"length_scale": length_scale,
                "noise_scale": tts.noise_scale, "noise_w": tts.noise_w,
                "sentence_silence": tts.sentence_silence}
    return {k: v for k, v in controls.items() if v is not None}


class ClonedVoice:
    """Speak in an enrolled person's voice, Piper when it cannot.

    Args:
        cloner: The cloning model.
        fallback: Piper.
        profile: The person (must be clonable).
        budget_ms: Longest wait for one cloned sentence.
        language: The language of the sentences.
        gate: The GPU gate; a cloner on the GPU is only used under it.
        controls: The Piper controls the warm-up speaks with (the
            session's own), so the first sentence meets a warm model.
        sleep: Waits (tests).
    """

    def __init__(
        self,
        cloner: VoiceCloner,
        fallback: TextToSpeech,
        profile: VoiceProfile,
        budget_ms: float = DEFAULT_BUDGET_MS,
        language: str = "es",
        gate: Optional[GpuGate] = None,
        controls: Optional[Dict[str, float]] = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        """Refuse a profile without consent to cloning or a clip."""
        if not profile.clonable or profile.reference is None:
            raise ValueError(
                "a voice is only cloned with the person's consent"
            )
        self.cloner = cloner
        self.fallback = fallback
        self.reference = profile.reference
        self.budget_ms = budget_ms
        self.language = language
        self.gate = gate
        self.controls = controls
        self.sleep = sleep
        self.need_mib = CLONE_NEED_MIB if cloner.gpu else 0
        self.refusal = ""
        self.failed = False
        self._closed = False
        self.cloned = 0
        self.fell_back = 0
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._busy: Optional["Future[Tuple[bytes, int]]"] = None
        self._lock = threading.Lock()

    def warm(self) -> None:
        """Load the cloner and the reference voice (in the background).

        A cloner on the GPU waits up to :data:`WARM_WAIT_S` for the gate
        (the live session takes the lease and the image worker frees
        the card); if it never allows it, Piper speaks the session.
        """
        waited = 0.0
        while self._gpu_refusal() is not None:
            if waited >= WARM_WAIT_S or self._closed:
                return
            self.sleep(GATE_POLL_S)
            waited += GATE_POLL_S
        try:
            self.cloner.warm(self.reference, self.language, self.controls)
        except TtsUnavailable:
            self.failed = True
        if self._closed:  # the session ended while it loaded
            self._unload()

    def synthesize(
        self, request: Document, cancel: CancelToken
    ) -> Iterator[SpeechChunk]:
        """Speak ``request`` sentence by sentence (see the module)."""
        text = str(request.get("text") or "")
        sentences = (split_sentences(text) if request.get("stream", True)
                     else [" ".join(text.split())])
        audio_format = str(request.get("format") or "pcm_s16le")
        request_id = str(request["request_id"])
        for seq, sentence in enumerate(s for s in sentences if s):
            if cancel.cancelled or self._closed:
                return
            started = time.perf_counter()
            cloned = self._clone(sentence, cancel, request_controls(request))
            if cancel.cancelled:
                return
            if cloned is None:
                self.fell_back += 1
                yield from self._piper(request, sentence, seq, cancel)
                continue
            self.cloned += 1
            pcm, rate = cloned
            audio = wav_bytes(pcm, rate) if audio_format == "wav" else pcm
            yield SpeechChunk(
                request_id=request_id, seq=seq, text=sentence, audio=audio,
                format=audio_format, sample_rate=rate,
                synth_ms=round((time.perf_counter() - started) * 1000, 1),
            )

    def close(self) -> None:
        """Stop waiting for the cloner; a GPU one is stopped (memory free).

        The stop runs in the background so ending a session never waits
        for a sentence still being cloned.
        """
        self._closed = True
        self._pool.shutdown(wait=False, cancel_futures=True)
        if self.need_mib:
            threading.Thread(target=self._unload, daemon=True).start()

    def _unload(self) -> None:
        """Stop the cloner's model (its GPU memory is freed)."""
        try:
            self.cloner.unload()
        except TtsUnavailable:  # pragma: no cover - best effort
            pass

    def _gpu_refusal(self) -> Optional[str]:
        """Why a GPU cloner may not run now (``None``: it may)."""
        if not self.need_mib:
            return None
        if self.gate is None:
            self.refusal = "no GPU gate"
        else:
            self.refusal = self.gate.check(
                0 if self.cloner.loaded else self.need_mib) or ""
        return self.refusal or None

    def _clone(
        self, sentence: str, cancel: CancelToken,
        controls: Optional[Dict[str, float]] = None,
    ) -> Optional[Tuple[bytes, int]]:
        """Return the cloned sentence, or ``None`` (Piper says it)."""
        if self._closed:
            return None
        if self._gpu_refusal() is not None:
            if self.cloner.loaded:  # the lease is gone: free the card
                threading.Thread(target=self._unload, daemon=True).start()
            return None
        with self._lock:
            if self.failed or (self._busy is not None
                               and not self._busy.done()):
                return None
            job = self._pool.submit(self.cloner.speak, sentence,
                                    self.reference, self.language, cancel,
                                    controls)
            self._busy = job
        try:
            return job.result(timeout=self.budget_ms / 1000.0)
        except FutureTimeout:
            return None
        except TtsUnavailable:
            self.failed = True
            return None

    def _piper(
        self, request: Document, sentence: str, seq: int,
        cancel: CancelToken,
    ) -> Iterator[SpeechChunk]:
        """Say one sentence with Piper, numbered as the cloned ones."""
        single = dict(request, text=sentence, stream=False,
                      interrupt_id=None,
                      request_id=f"{request['request_id']}-p{seq}"[-64:])
        for chunk in self.fallback.synthesize(single, cancel):
            yield SpeechChunk(
                request_id=str(request["request_id"]), seq=seq,
                text=chunk.text, audio=chunk.audio, format=chunk.format,
                sample_rate=chunk.sample_rate, synth_ms=chunk.synth_ms,
            )


def voice_request(
    request_id: str, text: str, tts: Tts, language: Optional[str],
    interrupt_id: Optional[str] = None, stream: bool = True,
    audio_format: str = "wav",
) -> Document:
    """Build the ``tts.request.v1`` of ``text`` with ``tts``'s controls."""
    return tts_request(
        request_id, text, tts.voice, language, interrupt_id=interrupt_id,
        audio_format=audio_format, length_scale=tts.length_scale,
        stream=stream, noise_scale=tts.noise_scale, noise_w=tts.noise_w,
        sentence_silence=tts.sentence_silence,
    )


def render_sample(
    tts: Tts, length_scale: float, speaker: TextToSpeech, language: str,
) -> bytes:
    """Say :data:`SAMPLE_TEXT` with ``speaker``; return one WAV.

    Pieces at another sample rate than the first (Piper among the
    other voice's sentences) are left out: one WAV has one rate.

    Raises:
        TtsUnavailable: Nothing could be said.
    """
    used = tts.model_copy(update={"length_scale": length_scale})
    request = voice_request("sample", SAMPLE_TEXT, used, language,
                            audio_format="pcm_s16le")
    pieces: List[bytes] = []
    rate = 0
    for chunk in speaker.synthesize(request, CancelToken()):
        if rate and chunk.sample_rate != rate:
            continue
        rate = chunk.sample_rate
        pieces.append(pcm_of(chunk))
    if not pieces:
        raise TtsUnavailable("nothing was said")
    return wav_bytes(b"".join(pieces), rate)


def speak_sample(
    tts: Tts,
    plan: VoicePlan,
    piper: TextToSpeech,
    clone: Optional[Callable[[VoiceProfile], ClonedVoice]] = None,
    language: str = "es",
) -> Tuple[bytes, Document]:
    """Render :data:`SAMPLE_TEXT` with Piper or ``plan``'s clone; one WAV.

    Return the WAV and what was used (``voice``, ``length_scale``,
    ``clone``, ``reason``, ``engine``).

    Raises:
        TtsUnavailable: Nothing could be said.
    """
    speaker: TextToSpeech = piper
    voice: Optional[ClonedVoice] = None
    if plan.clone is not None and clone is not None:
        voice = clone(plan.clone)
        speaker = voice
    try:
        wav = render_sample(tts, plan.length_scale, speaker, language)
    finally:
        if voice is not None:
            voice.close()
    reason = plan.reason
    if plan.clone is not None and clone is None:
        reason = "; ".join(r for r in (reason, "no cloner installed") if r)
    cloned = voice is not None and voice.cloned > 0
    return wav, {
        "voice": tts.voice, "length_scale": plan.length_scale,
        "clone": cloned, "reason": reason,
        "engine": "openvoice" if cloned else "piper",
    }
