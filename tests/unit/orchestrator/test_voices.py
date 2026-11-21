"""The voice bot's voice: Piper's controls from «Ajustes», rate matching,
cloning only with consent, the fallback to Piper and the sample (#207)."""

import threading
import wave
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import pytest

from orchestrator.adapters.outbound.fakes import FakeTts
from orchestrator.application.ports import CancelToken, TtsUnavailable
from orchestrator.application.voice import (
    ClonedVoice,
    speak_sample,
    voice_request,
)
from orchestrator.domain.config import Tts
from orchestrator.domain.live_choices import apply_live_choices, voice_choices
from orchestrator.domain.speech import pcm_of, tts_request, wav_bytes
from orchestrator.domain.voices import (
    PIPER_WORDS_PER_SECOND,
    VoicePlan,
    VoiceProfile,
    matched_length_scale,
    plan_voice,
)
from tests.unit.orchestrator.fixtures import product
from tests.unit.orchestrator.test_voicebot_sessions import (
    bot_sessions,
    user_final,
)
from tests.unit.orchestrator.test_sessions import messages

DAVEFX = "es_ES-davefx-medium"
AGREED = VoiceProfile("ana", 3.0, clone_consent=True, reference="/r.wav")
REFUSED = VoiceProfile("bea", 5.0, clone_consent=False, reference=None)


class FakeCloner:
    """A cloner that answers a 24 kHz tone, or waits, or fails."""

    def __init__(self, wait: Optional[threading.Event] = None,
                 fail: bool = False) -> None:
        self.wait = wait
        self.fail = fail
        self.said: List[Tuple[str, str]] = []
        self.warmed = 0

    def speak(self, text: str, reference: str, language: str,
              cancel: CancelToken) -> Tuple[bytes, int]:
        if self.fail:
            raise TtsUnavailable("cloner crashed")
        if self.wait is not None:
            self.wait.wait(5)
        self.said.append((text, reference))
        return b"\x02\x00" * 240, 24000

    def warm(self, reference: str, language: str) -> None:
        self.warmed += 1
        if self.fail:
            raise TtsUnavailable("no model")


def profiles(*found: VoiceProfile) -> Any:
    table = {p.speaker_id: p for p in found}
    return table.get


def test_saved_piper_controls_and_clone_reach_the_bot_s_tts() -> None:
    bot = product("voicebot")
    chosen = apply_live_choices(bot, {
        "voicebot.voice": "clone:ana", "voicebot.length_scale": "1.2",
        "voicebot.noise_scale": "0.3", "voicebot.noise_w": "0.6",
        "voicebot.sentence_silence": "0.2", "voicebot.match_rate": "ana",
    })
    assert chosen.tts is not None
    assert chosen.tts.voice == DAVEFX and chosen.tts.clone_of == "ana"
    assert (chosen.tts.length_scale, chosen.tts.noise_scale,
            chosen.tts.noise_w, chosen.tts.sentence_silence) == (
        1.2, 0.3, 0.6, 0.2)
    assert chosen.tts.match_rate_of == "ana"
    bad = voice_choices({
        "voicebot.voice": "clone:../etc", "voicebot.length_scale": "3",
        "voicebot.noise_w": "abc", "voicebot.match_rate": "off",
    })
    assert bad == {}, "invalid values keep the product's"
    assert voice_choices({"voicebot.voice": "sharvard"}) == {
        "voice": "es_ES-sharvard-medium", "engine": "piper"}


def test_the_operator_rate_sets_the_length_scale_within_bounds() -> None:
    piper = PIPER_WORDS_PER_SECOND[DAVEFX]
    assert matched_length_scale(piper, DAVEFX) == 1.0
    assert matched_length_scale(piper / 1.2, DAVEFX) == 1.2
    assert matched_length_scale(1.0, DAVEFX) == 1.5, "clamped (slow)"
    assert matched_length_scale(20.0, DAVEFX) == 0.7, "clamped (fast)"
    assert matched_length_scale(None, DAVEFX) is None
    assert matched_length_scale(0.1, DAVEFX) is None, "implausible"
    assert matched_length_scale(3.0, "es_ES-unknown") is None


def test_without_consent_cloning_is_impossible() -> None:
    plan = plan_voice(DAVEFX, 1.0, "bea", None, profiles(REFUSED))
    assert plan.clone is None and plan.reason == "no consent to cloning"
    clipless = VoiceProfile("cid", clone_consent=True, reference=None)
    plan = plan_voice(DAVEFX, 1.0, "cid", None, profiles(clipless))
    assert plan.clone is None and plan.reason == "no reference clip"
    plan = plan_voice(DAVEFX, 1.0, "nobody", None, profiles())
    assert plan.clone is None and plan.reason == "no voice profile"
    with pytest.raises(ValueError, match="consent"):
        ClonedVoice(FakeCloner(), FakeTts(), REFUSED)
    with pytest.raises(ValueError, match="consent"):
        ClonedVoice(FakeCloner(), FakeTts(), clipless)
    plan = plan_voice(DAVEFX, 1.1, "ana", "ana", profiles(AGREED))
    assert plan.clone == AGREED and plan.reason == ""
    assert plan.length_scale == round(PIPER_WORDS_PER_SECOND[DAVEFX] / 3, 3)
    plan = plan_voice(DAVEFX, 1.1, None, "bea-x", profiles())
    assert plan.length_scale == 1.1
    assert plan.reason == "no measured speaking rate"


def request(text: str, fmt: str = "wav") -> Dict[str, Any]:
    return tts_request("r1", text, DAVEFX, "es", audio_format=fmt)


def test_a_cloned_voice_speaks_each_sentence_like_the_reference() -> None:
    cloner = FakeCloner()
    voice = ClonedVoice(cloner, FakeTts(), AGREED)
    voice.warm()
    chunks = list(voice.synthesize(
        request("Primera frase completa. Segunda frase completa."),
        CancelToken()))
    assert [c.seq for c in chunks] == [0, 1]
    assert all(c.sample_rate == 24000 and c.format == "wav" for c in chunks)
    assert len(pcm_of(chunks[0])) == 480
    assert cloner.said[0] == ("Primera frase completa.", "/r.wav")
    assert cloner.warmed == 1 and voice.cloned == 2
    voice.close()


def test_a_slow_cloner_falls_back_to_piper_and_a_failed_one_stays_out(
) -> None:
    gate = threading.Event()
    piper = FakeTts()
    slow = ClonedVoice(FakeCloner(wait=gate), piper, AGREED, budget_ms=20)
    chunks = list(slow.synthesize(
        request("Primera frase completa. Segunda frase completa.",
                "pcm_s16le"), CancelToken()))
    gate.set()
    assert [c.sample_rate for c in chunks] == [16000, 16000], "Piper"
    assert [c.seq for c in chunks] == [0, 1]
    assert {c.request_id for c in chunks} == {"r1"}
    assert slow.fell_back == 2
    assert piper.requests[0]["stream"] is False
    assert piper.requests[0]["interrupt_id"] is None
    slow.close()
    broken = FakeCloner(fail=True)
    voice = ClonedVoice(broken, FakeTts(), AGREED)
    voice.warm()
    assert voice.failed
    out = list(voice.synthesize(request("Una frase bastante larga."),
                                CancelToken()))
    assert [c.sample_rate for c in out] == [16000]
    cancel = CancelToken()
    cancel.cancel()
    assert list(ClonedVoice(FakeCloner(), FakeTts(), AGREED).synthesize(
        request("Nada que decir aquí."), cancel)) == []


def test_the_sample_says_the_fixed_sentence_with_the_draft() -> None:
    piper = FakeTts()
    tts = Tts(kind="piper", voice=DAVEFX, length_scale=1.3, noise_w=0.5)
    wav, used = speak_sample(tts, VoicePlan(DAVEFX, 1.3), piper)
    with wave.open(BytesIO(wav), "rb") as reader:
        assert reader.getframerate() == 16000 and reader.getnframes() > 0
    assert used == {"voice": DAVEFX, "length_scale": 1.3, "clone": False,
                    "reason": "", "engine": "piper"}
    assert piper.requests[0]["noise_w"] == 0.5
    assert piper.requests[0]["length_scale"] == 1.3
    cloner = FakeCloner()
    wav, used = speak_sample(
        tts, VoicePlan(DAVEFX, 1.0, AGREED), piper,
        lambda p: ClonedVoice(cloner, piper, p))
    assert used["clone"] is True and cloner.said
    _, used = speak_sample(tts, VoicePlan(DAVEFX, 1.0, AGREED), piper)
    assert used["reason"] == "no cloner installed"

    class Mute(FakeTts):
        def synthesize(self, request: Any, cancel: Any) -> Any:
            return iter(())

    with pytest.raises(TtsUnavailable):
        speak_sample(tts, VoicePlan(DAVEFX, 1.0), Mute())


def test_requests_carry_the_controls_and_refuse_out_of_range() -> None:
    tts = Tts(kind="piper", voice=DAVEFX, noise_scale=0.4,
              sentence_silence=0.1)
    doc = voice_request("r", "Hola a todos.", tts, "es")
    assert doc["noise_scale"] == 0.4 and doc["sentence_silence"] == 0.1
    assert "noise_w" not in doc
    with pytest.raises(ValueError, match="noise_w"):
        tts_request("r", "Hola.", DAVEFX, noise_w=2.0)
    assert pcm_of(next(iter(FakeTts().synthesize(
        request("Hola a todos.", "pcm_s16le"), CancelToken())))) != b""
    assert wav_bytes(b"", 8000)[:4] == b"RIFF"


class Settings:
    def __init__(self, choices: Dict[str, str]) -> None:
        self.choices = choices

    def current(self) -> Optional[Dict[str, str]]:
        return self.choices


def test_a_session_speaks_with_the_clone_only_when_consented() -> None:
    events = [user_final("t0.s0", "¿Puedo devolver un producto?", 1)]
    for found, cloner, engine, cloned, reason in (
        (AGREED, FakeCloner(), "openvoice", True, ""),
        (AGREED, FakeCloner(), "qwen3", True, "Qwen3-TTS not installed"),
        (REFUSED, FakeCloner(), "openvoice", False, "no consent to cloning"),
        (AGREED, None, "openvoice", False, "no cloner installed"),
    ):
        live = bot_sessions(events, tts=FakeTts())
        live.settings = Settings({"voicebot.voice":
                                  f"clone:{found.speaker_id}",
                                  "voicebot.clone_engine": engine})
        live.profiles = type("P", (), {"get": staticmethod(
            profiles(found))})()
        live.cloner = cloner
        live.start_live("voicebot", "mic")
        voice = [m for m in messages(live, "vb-1") if m["type"] == "voice"]
        assert voice == [{"type": "voice", "clone": cloned,
                          "engine": "openvoice" if cloned else "piper",
                          "length_scale": 1.0, "reason": reason}]
        if cloner is not None:
            assert bool(cloner.said) is cloned


def test_the_sessions_sample_uses_the_unsaved_choices() -> None:
    live = bot_sessions([], tts=FakeTts())
    wav, used = live.sample({"voicebot.length_scale": "1.4"})
    assert wav[:4] == b"RIFF" and used["length_scale"] == 1.4
    from orchestrator.application.sessions import SessionError

    with pytest.raises(SessionError):
        bot_sessions([]).sample({})
