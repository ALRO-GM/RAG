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
  this session: Piper says the rest.

:func:`speak_sample` renders the fixed sentence of the «Ajustes» page's
«Escuchar ejemplo» button with the controls being edited, as one WAV.
No notice is spoken when the bot speaks with a cloned voice (owner's
decision, #207).
"""

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Callable, Iterator, List, Optional, Tuple

from orchestrator.application.ports import (
    CancelToken,
    Document,
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


class ClonedVoice:
    """Speak in an enrolled person's voice, Piper when it cannot.

    Args:
        cloner: The cloning model.
        fallback: Piper.
        profile: The person (must be clonable).
        budget_ms: Longest wait for one cloned sentence.
        language: The language of the sentences.
    """

    def __init__(
        self,
        cloner: VoiceCloner,
        fallback: TextToSpeech,
        profile: VoiceProfile,
        budget_ms: float = DEFAULT_BUDGET_MS,
        language: str = "es",
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
        self.failed = False
        self.cloned = 0
        self.fell_back = 0
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._busy: Optional["Future[Tuple[bytes, int]]"] = None
        self._lock = threading.Lock()

    def warm(self) -> None:
        """Load the cloner and the reference voice (in the background)."""
        try:
            self.cloner.warm(self.reference, self.language)
        except TtsUnavailable:
            self.failed = True

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
            if cancel.cancelled:
                return
            started = time.perf_counter()
            cloned = self._clone(sentence, cancel)
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
        """Stop waiting for the cloner."""
        self._pool.shutdown(wait=False, cancel_futures=True)

    def _clone(
        self, sentence: str, cancel: CancelToken
    ) -> Optional[Tuple[bytes, int]]:
        """Return the cloned sentence, or ``None`` (Piper says it)."""
        with self._lock:
            if self.failed or (self._busy is not None
                               and not self._busy.done()):
                return None
            job = self._pool.submit(self.cloner.speak, sentence,
                                    self.reference, self.language, cancel)
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


def speak_sample(
    tts: Tts,
    plan: VoicePlan,
    piper: TextToSpeech,
    clone: Optional[Callable[[VoiceProfile], ClonedVoice]] = None,
    language: str = "es",
) -> Tuple[bytes, Document]:
    """Render :data:`SAMPLE_TEXT` with ``tts`` and ``plan``; one WAV.

    Return the WAV and what was used (``voice``, ``length_scale``,
    ``clone``, ``reason``).

    Raises:
        TtsUnavailable: Nothing could be said.
    """
    speaker: TextToSpeech = piper
    voice: Optional[ClonedVoice] = None
    if plan.clone is not None and clone is not None:
        voice = clone(plan.clone)
        speaker = voice
    used = tts.model_copy(update={"length_scale": plan.length_scale})
    request = voice_request("sample", SAMPLE_TEXT, used, language,
                            audio_format="pcm_s16le")
    pieces: List[bytes] = []
    rate = 0
    try:
        for chunk in speaker.synthesize(request, CancelToken()):
            if rate and chunk.sample_rate != rate:
                continue  # Piper among cloned sentences: keep one rate
            rate = chunk.sample_rate
            pieces.append(pcm_of(chunk))
    finally:
        if voice is not None:
            voice.close()
    if not pieces:
        raise TtsUnavailable("nothing was said")
    reason = plan.reason
    if plan.clone is not None and clone is None:
        reason = "; ".join(r for r in (reason, "no cloner installed") if r)
    return wav_bytes(b"".join(pieces), rate), {
        "voice": tts.voice, "length_scale": plan.length_scale,
        "clone": voice is not None and voice.cloned > 0, "reason": reason,
    }
