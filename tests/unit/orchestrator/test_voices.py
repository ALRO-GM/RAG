"""The voice bot's voice: Piper's controls from «Ajustes», rate matching,
cloning only with consent, the fallback to Piper and the sample (#207)."""

import threading
import time
import wave
from io import BytesIO
from typing import Any, Dict, List, Mapping, Optional, Tuple

import pytest

from orchestrator.adapters.outbound.fakes import FakeTts
from orchestrator.application.ports import CancelToken, TtsUnavailable
from orchestrator.application.voice import (
    CLONE_NEED_MIB,
    ClonedVoice,
    SharedCloner,
    request_controls,
    speak_sample,
    tts_controls,
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
                 fail: bool = False, gpu: bool = False) -> None:
        self.wait = wait
        self.fail = fail
        self.gpu = gpu
        self.loaded = False
        self.said: List[Tuple[str, str]] = []
        self.controls: List[Optional[Mapping[str, float]]] = []
        self.warmed = 0
        self.unloaded = threading.Event()
        self.generation = 1

    def speak(self, text: str, reference: str, language: str,
              cancel: CancelToken,
              controls: Optional[Mapping[str, float]] = None,
              ) -> Tuple[bytes, int]:
        if self.fail:
            raise TtsUnavailable("cloner crashed")
        if self.wait is not None:
            self.wait.wait(5)
        self.loaded = True
        self.said.append((text, reference))
        self.controls.append(controls)
        return b"\x02\x00" * 240, 24000

    def warm(self, reference: str, language: str,
             controls: Optional[Mapping[str, float]] = None) -> None:
        self.warmed += 1
        self.controls.append(controls)
        if self.fail:
            raise TtsUnavailable("no model")
        self.loaded = True

    def unload(self, generation: Optional[int] = None) -> None:
        if generation is not None and generation != self.generation:
            return
        self.loaded = False
        self.unloaded.set()

    def release(self) -> None:
        if self.gpu:
            self.unload()


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


class FakeGate:
    """A GPU gate that allows or refuses, and counts the leases taken."""

    def __init__(self, refusal: Optional[str] = None) -> None:
        self.refusal = refusal
        self.checked: List[int] = []
        self.taken: List[int] = []
        self.dropped = 0

    def check(self, need_mib: int) -> Optional[str]:
        self.checked.append(need_mib)
        return self.refusal

    def take(self, need_mib: int) -> Optional[str]:
        self.taken.append(need_mib)
        return self.refusal

    def drop(self) -> None:
        self.dropped += 1


def test_a_gpu_cloner_only_runs_under_the_live_lease() -> None:
    cloner = FakeCloner(gpu=True)
    gate = FakeGate()
    voice = ClonedVoice(cloner, FakeTts(), AGREED, gate=gate)
    voice.warm()
    assert cloner.warmed == 1 and gate.checked == [CLONE_NEED_MIB]
    out = list(voice.synthesize(request("Una frase bastante larga."),
                                CancelToken()))
    assert voice.cloned == 1 and out[0].sample_rate == 24000
    assert gate.checked[-1] == 0, "loaded: the memory is already ours"
    gate.refusal = "no live GPU lease"
    out = list(voice.synthesize(request("Otra frase bastante larga."),
                                CancelToken()))
    assert [c.sample_rate for c in out] == [16000], "Piper speaks"
    assert cloner.unloaded.wait(2), "the lease went: the card is freed"
    assert voice.refusal == "no live GPU lease"
    voice.close()


def test_a_gpu_cloner_without_a_gate_or_lease_is_never_loaded() -> None:
    sleeps: List[float] = []
    for gate in (None, FakeGate("no live GPU lease")):
        cloner = FakeCloner(gpu=True)
        voice = ClonedVoice(cloner, FakeTts(), AGREED, gate=gate,
                            sleep=sleeps.append)
        voice.warm()
        out = list(voice.synthesize(request("Una frase bastante larga."),
                                    CancelToken()))
        assert cloner.warmed == 0 and not cloner.said
        assert [c.sample_rate for c in out] == [16000]
        voice.close()
    assert sleeps, "the warm-up waited for the lease before giving up"
    assert sum(sleeps) >= 30.0 - 0.3


def test_a_cpu_cloner_needs_no_lease_and_close_frees_a_gpu_one() -> None:
    cpu = FakeCloner(gpu=False)
    gate = FakeGate("held")
    voice = ClonedVoice(cpu, FakeTts(), AGREED, gate=gate)
    voice.warm()
    assert cpu.warmed == 1 and gate.checked == []
    voice.close()
    assert not cpu.unloaded.wait(0.1), "a CPU model stays loaded"
    gpu = FakeCloner(gpu=True)
    voice = ClonedVoice(gpu, FakeTts(), AGREED, gate=FakeGate())
    voice.warm()
    voice.close()
    assert gpu.unloaded.wait(2), "the session ended: memory freed"
    assert list(voice.synthesize(request("Nada que decir aquí."),
                                 CancelToken())) == []


def test_the_pipers_controls_reach_the_cloner() -> None:
    cloner = FakeCloner()
    tts = Tts(kind="piper", voice=DAVEFX, noise_scale=0.4, noise_w=0.6,
              sentence_silence=0.2)
    controls = tts_controls(tts, 1.25)
    assert controls == {"length_scale": 1.25, "noise_scale": 0.4,
                        "noise_w": 0.6, "sentence_silence": 0.2}
    assert tts_controls(Tts(kind="piper", voice=DAVEFX), 1.0) == {
        "length_scale": 1.0}
    voice = ClonedVoice(cloner, FakeTts(), AGREED, controls=controls)
    voice.warm()
    list(voice.synthesize(
        voice_request("r1", "Una frase bastante larga.", tts, "es"),
        CancelToken()))
    assert cloner.controls[0] == controls
    assert cloner.controls[1] == {"length_scale": 1.0, "noise_scale": 0.4,
                                  "noise_w": 0.6, "sentence_silence": 0.2}
    assert request_controls({"length_scale": True, "noise_w": "x",
                             "noise_scale": 0.3}) == {
        "noise_scale": 0.3}, "numbers only, a boolean is not one"
    voice.close()


def test_the_sample_takes_the_lease_for_a_gpu_cloner() -> None:
    gate = FakeGate()
    cloner = FakeCloner(gpu=True)
    live = bot_sessions([], tts=FakeTts())
    live.profiles = type("P", (), {"get": staticmethod(profiles(AGREED))})()
    live.cloner = cloner
    live.clone_gate = gate
    pick = {"voicebot.voice": "clone:ana",
            "voicebot.clone_engine": "openvoice"}
    wav, used = live.sample({**pick, "voicebot.length_scale": "1.2"})
    assert used["clone"] is True and used["engine"] == "openvoice"
    assert gate.taken == [CLONE_NEED_MIB] and gate.dropped == 1
    assert cloner.controls[0] == {"length_scale": 1.2}
    assert cloner.unloaded.wait(2)
    gate.refusal = "the GPU lease is held by ragputin-img"
    _, used = live.sample(pick)
    assert used["clone"] is False and used["engine"] == "piper"
    assert "ragputin-img" in used["reason"] and gate.dropped == 2


def test_a_session_gives_the_cloner_its_gate_and_controls() -> None:
    events = [user_final("t0.s0", "¿Puedo devolver un producto?", 1)]
    cloner = FakeCloner(gpu=True)
    live = bot_sessions(events, tts=FakeTts())
    live.settings = Settings({"voicebot.voice": "clone:ana",
                              "voicebot.clone_engine": "openvoice",
                              "voicebot.length_scale": "1.3"})
    live.profiles = type("P", (), {"get": staticmethod(profiles(AGREED))})()
    live.cloner = cloner
    live.clone_gate = FakeGate()
    live.start_live("voicebot", "mic")
    first = cloner.controls[0]
    assert cloner.said and first is not None
    assert first["length_scale"] == 1.3


def settle(check: Any, seconds: float = 2.0) -> bool:
    """Wait until ``check()`` is true (threads of the voices)."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if check():
            return True
        time.sleep(0.01)
    return bool(check())


def test_a_shared_cloner_is_stopped_only_by_its_last_holder() -> None:
    cloner = FakeCloner(gpu=True)
    shared = SharedCloner(cloner)
    first = ClonedVoice(shared.lend(), FakeTts(), AGREED, gate=FakeGate())
    second = ClonedVoice(shared.lend(), FakeTts(), AGREED, gate=FakeGate())
    first.warm()
    assert cloner.loaded and shared.holders == 2
    first.close()
    assert settle(lambda: shared.holders == 1)
    assert not cloner.unloaded.wait(0.2), "a session still holds it"
    assert cloner.loaded
    second.close()
    assert cloner.unloaded.wait(2) and shared.holders == 0
    # a holder is given back once, however many times it closes
    first.close()
    assert shared.holders == 0


def test_a_stop_scheduled_by_one_session_spares_the_next_ones_worker(
) -> None:
    cloner = FakeCloner(gpu=True)
    shared = SharedCloner(cloner)
    first = ClonedVoice(shared.lend(), FakeTts(), AGREED, gate=FakeGate())
    first.warm()
    first.cloner.release()  # the last holder goes: a stop is scheduled...
    assert settle(lambda: cloner.unloaded.is_set())
    cloner.unloaded.clear()
    second = ClonedVoice(shared.lend(), FakeTts(), AGREED, gate=FakeGate())
    second.warm()  # ...the next session warms its worker...
    shared._stop_if_idle()  # ...and the late stop runs: it re-checks
    assert shared.holders == 1 and cloner.loaded
    assert not cloner.unloaded.is_set(), "the new session's worker lives"
    second.close()
    assert cloner.unloaded.wait(2)


def test_a_sample_during_a_session_does_not_stop_its_worker() -> None:
    gate = FakeGate()
    cloner = FakeCloner(gpu=True)
    live = bot_sessions([], tts=FakeTts())
    live.profiles = type("P", (), {"get": staticmethod(profiles(AGREED))})()
    live.cloner = cloner
    live.clone_gate = gate
    pick = {"voicebot.voice": "clone:ana",
            "voicebot.clone_engine": "openvoice"}
    session_voice = live._cloned(AGREED)
    session_voice.warm()
    assert cloner.loaded
    gate.taken.clear()
    _, used = live.sample(pick)
    assert used["clone"] is True
    assert gate.taken == [0], "loaded already: no 800 MiB to wait for"
    assert not cloner.unloaded.wait(0.3) and cloner.loaded
    session_voice.close()
    assert cloner.unloaded.wait(2)


def test_the_lease_lost_while_the_bot_is_silent_frees_the_card() -> None:
    cloner = FakeCloner(gpu=True)
    gate = FakeGate()
    voice = ClonedVoice(cloner, FakeTts(), AGREED, gate=gate, watch_s=0.01)
    voice.warm()
    assert cloner.loaded and not cloner.unloaded.is_set()
    gate.refusal = "no live GPU lease"
    assert cloner.unloaded.wait(2), "unloaded with nobody speaking"
    voice.close()


def test_a_load_in_progress_is_stopped_when_the_lease_goes() -> None:
    class Loading(FakeCloner):
        def warm(self, reference: str, language: str,
                 controls: Optional[Mapping[str, float]] = None) -> None:
            self.loaded = True  # the process exists, the model loads
            self.unloaded.wait(5)
            raise TtsUnavailable("the cloner stopped")

    cloner = Loading(gpu=True)
    gate = FakeGate()
    voice = ClonedVoice(cloner, FakeTts(), AGREED, gate=gate, watch_s=0.01)
    worker = threading.Thread(target=voice.warm)
    worker.start()
    assert settle(lambda: cloner.loaded)
    gate.refusal = "no live GPU lease"
    worker.join(3)
    assert not worker.is_alive() and cloner.unloaded.is_set()
    assert voice.failed is False, "the lease going is not a failure"
    voice.close()


def test_a_refused_warm_up_keeps_the_clone_out_of_the_session() -> None:
    cloner = FakeCloner(gpu=True)
    gate = FakeGate("no live GPU lease")
    heard: List[Tuple[str, str]] = []
    voice = ClonedVoice(cloner, FakeTts(), AGREED, gate=gate,
                        sleep=lambda _: None,
                        report=lambda e, w: heard.append((e, w)))
    voice.warm()
    assert voice.failed and cloner.warmed == 0
    gate.refusal = None  # the lease turns up mid-session: still no load
    out = list(voice.synthesize(request("Una frase bastante larga."),
                                CancelToken()))
    assert [c.sample_rate for c in out] == [16000]
    assert cloner.warmed == 0 and not cloner.said, "no cold load mid-talk"
    assert heard == [("piper", "openvoice: no live GPU lease")]
    voice.close()


def test_the_feed_hears_when_the_clone_gives_way_to_piper() -> None:
    cloner = FakeCloner(gpu=True)
    gate = FakeGate()
    heard: List[Tuple[str, str]] = []
    voice = ClonedVoice(cloner, FakeTts(), AGREED, gate=gate,
                        report=lambda e, w: heard.append((e, w)))
    voice.warm()
    list(voice.synthesize(request("Una frase bastante larga."),
                          CancelToken()))
    assert heard == [], "the first message already said openvoice"
    gate.refusal = "the GPU lease is held by ragputin-img"
    list(voice.synthesize(request("Otra frase bastante larga."),
                          CancelToken()))
    gate.refusal = None
    list(voice.synthesize(request("Y otra frase bastante larga."),
                          CancelToken()))
    assert heard == [
        ("piper", "openvoice: the GPU lease is held by ragputin-img"),
        ("openvoice", "")]
    voice.close()


def test_a_session_ends_with_the_clone_counts_in_its_feed() -> None:
    events = [user_final("t0.s0", "¿Puedo devolver un producto?", 1)]
    live = bot_sessions(events, tts=FakeTts())
    live.settings = Settings({"voicebot.voice": "clone:ana",
                              "voicebot.clone_engine": "openvoice"})
    live.profiles = type("P", (), {"get": staticmethod(profiles(AGREED))})()
    live.cloner = FakeCloner()
    live.start_live("voicebot", "mic")
    summary = [m for m in messages(live, "vb-1")
               if m["type"] == "voice_summary"]
    assert len(summary) == 1 and summary[0]["engine"] == "openvoice"
    assert summary[0]["cloned"] >= 1 and summary[0]["fell_back"] == 0


def test_a_warm_up_that_ends_after_close_spares_another_holders_model(
) -> None:
    class SlowWarm(FakeCloner):
        def __init__(self) -> None:
            super().__init__(gpu=True)
            self.go = threading.Event()
            self.first = True

        def warm(self, reference: str, language: str,
                 controls: Optional[Mapping[str, float]] = None) -> None:
            if self.first:
                self.first = False
                self.go.wait(5)
            super().warm(reference, language, controls)

    inner = SlowWarm()
    shared = SharedCloner(inner)
    gate = FakeGate()
    a = ClonedVoice(shared.lend(), FakeTts(), AGREED, gate=gate, watch_s=60)
    loading = threading.Thread(target=a.warm)
    loading.start()
    time.sleep(0.1)
    a.close()  # session A ends while its worker loads
    b = ClonedVoice(shared.lend(), FakeTts(), AGREED, gate=gate, watch_s=60)
    assert settle(lambda: shared.holders == 1)
    b.warm()
    assert inner.loaded
    inner.unloaded.clear()
    inner.go.set()
    loading.join(3)  # A's warm returns and sees it was closed
    time.sleep(0.2)
    assert inner.loaded and not inner.unloaded.is_set(), "B's model lives"
    b.close()
    assert inner.unloaded.wait(2)


def test_a_sample_whose_lease_cannot_be_taken_leaves_nothing_taken() -> None:
    class Failing(FakeGate):
        def take(self, need_mib: int) -> Optional[str]:
            raise OSError("disk full")

    gate = Failing()
    live = bot_sessions([], tts=FakeTts())
    live.profiles = type("P", (), {"get": staticmethod(profiles(AGREED))})()
    live.cloner = FakeCloner(gpu=True)
    live.clone_gate = gate
    from orchestrator.application.sessions import SessionError

    with pytest.raises((OSError, SessionError)):
        live.sample({"voicebot.voice": "clone:ana",
                     "voicebot.clone_engine": "openvoice"})
    assert gate.dropped == 0, "no drop for a take that never happened"


def test_a_stale_lease_lost_decision_cannot_kill_a_newer_worker() -> None:
    """The refusal is seen for start 1; start 2 begins before it acts."""
    cloner = FakeCloner(gpu=True)
    cloner.loaded = True

    class Racing(FakeGate):
        def check(self, need_mib: int) -> Optional[str]:
            cloner.generation += 1  # another session starts a worker now
            return "no live GPU lease"

    voice = ClonedVoice(cloner, FakeTts(), AGREED, gate=Racing(),
                        watch_s=60)
    assert voice._clone("Hola.", CancelToken()) is None
    assert not cloner.unloaded.wait(0.3), "the new worker was spared"
    assert cloner.loaded
    voice._unload(cloner.generation)  # a decision about the current one
    assert cloner.unloaded.is_set()
    voice.close()


def test_the_watcher_reads_the_generation_before_the_gate() -> None:
    """Reading it after the gate would unload a newer worker."""
    cloner = FakeCloner(gpu=True)
    cloner.loaded = True
    holder: List[ClonedVoice] = []

    class Racing(FakeGate):
        def check(self, need_mib: int) -> Optional[str]:
            cloner.generation += 1  # a new worker starts meanwhile
            holder[0]._stop.set()  # one watcher round only
            return "no live GPU lease"

    voice = ClonedVoice(cloner, FakeTts(), AGREED, gate=Racing(),
                        watch_s=0.01)
    holder.append(voice)
    voice._watch()
    assert cloner.loaded and not cloner.unloaded.is_set()
    voice.close()
