"""Say a demo line with a chosen voice, through the voice bot's engines.

The :class:`~orchestrator.application.ports.DemoSpeaker` of the render
command (#254). It adds no engine of its own: Piper
(:class:`~.piper_tts.PiperTts`), Pocket TTS and Qwen3-TTS
(:class:`~.subprocess_speech.WorkerEngines`) and the OpenVoice cloner
(:class:`~.subprocess_cloner.SubprocessCloner`) speak as they do for the
voice bot. A line is cut into sentences, each said in turn and joined
with a short pause, so every engine gets what it was measured with.

Only one GPU engine is loaded at a time: a different GPU voice stops the
one before (the render orders its lines by voice, so each loads once).
:class:`ToneDemoSpeaker` answers with tones, for tests and a dry run.
"""

from typing import List, Optional, Tuple

from orchestrator.adapters.outbound.fakes import tone_wav
from orchestrator.application.ports import (
    CancelToken,
    SpeechEngines,
    SpeechStream,
    TextToSpeech,
    TtsUnavailable,
    VoiceCloner,
)
from orchestrator.application.voice import CLONE_NEED_MIB
from orchestrator.domain.demo_voices import DemoVoice
from orchestrator.domain.speech import pcm_of, split_sentences, tts_request
from orchestrator.domain.voices import NEED_MIB, EngineVoice

PAUSE_S = 0.18
"""Silence between two sentences of one line."""


def _silence(rate: int, seconds: float = PAUSE_S) -> bytes:
    """``seconds`` of 16-bit silence at ``rate``."""
    return b"\x00\x00" * int(rate * seconds)


class EngineDemoSpeaker:
    """Speak with Piper, Pocket TTS, Qwen3-TTS or OpenVoice.

    Args:
        tts: Piper (``None``: no Piper voice may be used).
        engines: Pocket TTS and Qwen3-TTS (``None``: none).
        cloner: The OpenVoice cloner (``None``: none).
        pocket_device: Where Pocket TTS runs (``cpu`` needs no lease).
    """

    def __init__(
        self,
        tts: Optional[TextToSpeech] = None,
        engines: Optional[SpeechEngines] = None,
        cloner: Optional[VoiceCloner] = None,
        pocket_device: str = "cpu",
    ):
        """Keep the engines; nothing is started yet."""
        self.tts = tts
        self.engines = engines
        self.cloner = cloner
        self.pocket_device = pocket_device
        self._gpu: Optional[Tuple[EngineVoice, SpeechStream]] = None
        self._count = 0

    def _engine_voice(
        self, voice: DemoVoice, reference: Optional[str]
    ) -> EngineVoice:
        """The worker voice of a Pocket or Qwen3 voice."""
        if voice.engine == "pocket":
            return EngineVoice("pocket", self.pocket_device,
                               voice=voice.voice or "lola")
        return EngineVoice("qwen3", "cuda", reference=reference)

    def engine(self, voice: DemoVoice) -> str:
        """What speaks ``voice``: ``piper``, ``pocket-cpu``, ``qwen3-cuda``,
        ``openvoice-cuda``."""
        if voice.engine == "piper":
            return "piper"
        if voice.engine == "openvoice":
            gpu = self.cloner is None or self.cloner.gpu
            return f"openvoice-{'cuda' if gpu else 'cpu'}"
        return self._engine_voice(voice, None).name

    def need_mib(self, voice: DemoVoice) -> int:
        """GPU memory ``voice`` needs (the voice bot's measured figures)."""
        if voice.engine == "piper":
            return 0
        if voice.engine == "openvoice":
            gpu = self.cloner is None or self.cloner.gpu
            return CLONE_NEED_MIB if gpu else 0
        if voice.engine == "pocket" and self.pocket_device == "cpu":
            return 0
        return NEED_MIB.get(voice.engine, 0)

    def speak(
        self, text: str, voice: DemoVoice, reference: Optional[str],
        language: str,
    ) -> Tuple[bytes, int]:
        """The whole line as PCM s16le mono and its rate."""
        if voice.engine == "piper":
            return self._piper(text, voice, language)
        sentences = split_sentences(text) or [text]
        pieces: List[bytes] = []
        rate = 0
        for sentence in sentences:
            if voice.engine == "openvoice":
                if self.cloner is None or not reference:
                    raise TtsUnavailable("no OpenVoice cloner")
                pcm, rate = self.cloner.speak(sentence, reference, language,
                                              CancelToken())
            else:
                pcm, rate = self._stream(sentence, voice, reference,
                                         language)
            if pieces:
                pieces.append(_silence(rate))
            pieces.append(pcm)
        return b"".join(pieces), rate

    def _piper(
        self, text: str, voice: DemoVoice, language: str
    ) -> Tuple[bytes, int]:
        """Piper says the line, sentence by sentence (its own pause)."""
        if self.tts is None:
            raise TtsUnavailable("no Piper")
        self._count += 1
        request = tts_request(f"demo-{self._count}", text, voice.voice,
                              language=language, audio_format="wav",
                              stream=False)
        request["speaker"] = voice.speaker
        pieces: List[bytes] = []
        rate = 0
        for chunk in self.tts.synthesize(request, CancelToken()):
            if pieces:
                pieces.append(_silence(chunk.sample_rate))
            pieces.append(pcm_of(chunk))
            rate = chunk.sample_rate
        return b"".join(pieces), rate

    def _stream(
        self, sentence: str, voice: DemoVoice, reference: Optional[str],
        language: str,
    ) -> Tuple[bytes, int]:
        """Pocket TTS or Qwen3-TTS says one sentence."""
        engines = self.engines
        if engines is None or voice.engine not in engines.installed():
            raise TtsUnavailable(f"{voice.engine} is not installed")
        wanted = self._engine_voice(voice, reference)
        if wanted.device == "cuda":
            if self._gpu is not None and self._gpu[0] != wanted:
                self._drop_gpu()
            if self._gpu is None:
                self._gpu = (wanted, engines.get(wanted))
            stream = self._gpu[1]
        else:
            stream = engines.get(wanted)
        pieces: List[bytes] = []
        rate = 0
        for pcm, rate in stream.stream(sentence, wanted, language,
                                       CancelToken()):
            pieces.append(pcm)
        return b"".join(pieces), rate

    def _drop_gpu(self) -> None:
        """Stop the GPU engine in use, if any."""
        if self._gpu is not None and self.engines is not None:
            self.engines.release(self._gpu[1])
        self._gpu = None

    def close(self) -> None:
        """Stop the GPU engine and the cloner this render started."""
        self._drop_gpu()
        if self.cloner is not None:
            self.cloner.release()


class ToneDemoSpeaker:
    """A :class:`~orchestrator.application.ports.DemoSpeaker` of tones.

    Each voice gets its own pitch; a line lasts ``seconds_per_word`` per
    word. ``need`` is the GPU memory it claims for clones (to exercise
    the lease without a GPU).
    """

    def __init__(self, seconds_per_word: float = 0.3, rate: int = 16000,
                 need: int = 0):
        """Keep the options."""
        self.seconds_per_word = seconds_per_word
        self.rate = rate
        self.need = need
        self.said: List[Tuple[str, str]] = []

    def engine(self, voice: DemoVoice) -> str:
        """Always ``tone``."""
        return "tone"

    def need_mib(self, voice: DemoVoice) -> int:
        """``need`` for a clone, 0 for a stock voice."""
        return self.need if voice.person else 0

    def speak(
        self, text: str, voice: DemoVoice, reference: Optional[str],
        language: str,
    ) -> Tuple[bytes, int]:
        """A tone as long as the line, pitched by the voice."""
        self.said.append((voice.spec, text))
        pitch = 160.0 + (sum(map(ord, voice.spec)) % 12) * 25.0
        wav = tone_wav(self.seconds_per_word * len(text.split()), self.rate,
                       pitch)
        return wav[44:], self.rate

    def close(self) -> None:
        """Nothing to stop."""
