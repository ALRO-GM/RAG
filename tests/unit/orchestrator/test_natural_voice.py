"""The natural voices (#214): who speaks each sentence, the fallbacks to
Pocket on the CPU and Piper, the GPU gate, streaming in growing pieces,
the settings that choose them and the sample."""

import threading
import wave
from io import BytesIO
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import pytest

from orchestrator.adapters.outbound.fakes import FakeTts
from orchestrator.application.natural_voice import (
    NaturalVoice,
    NaturalWiring,
    gpu_need,
)
from orchestrator.application.ports import (
    CancelToken,
    SpeechStream,
    TtsUnavailable,
)
from orchestrator.domain.live_choices import voice_choices
from orchestrator.domain.speech import (
    AudioJoiner,
    SpeechChunk,
    tts_request,
    wav_bytes,
    whole_chunk,
)
from orchestrator.domain.voices import (
    EngineVoice,
    VoiceProfile,
    voice_chain,
)
from tests.unit.orchestrator.test_sessions import messages
from tests.unit.orchestrator.test_voicebot_sessions import (
    bot_sessions,
    user_final,
)

DAVEFX = "es_ES-davefx-medium"
RATE = 24000
PIECE = b"\x01\x00" * 1920  # 80 ms at 24 kHz
POCKET_CPU = EngineVoice("pocket", "cpu", voice="lola")
POCKET_GPU = EngineVoice("pocket", "cuda", voice="lola")
QWEN3 = EngineVoice("qwen3", "cuda", reference="/ref.wav",
                    reference_text="Hola.")
AGREED = VoiceProfile("ana", 3.0, clone_consent=True, reference="/ana.wav")


class Engine:
    """A streaming engine: ``pieces`` per sentence, or waits, or fails."""

    def __init__(self, pieces: int = 3, fail: bool = False,
                 gate: Optional[threading.Event] = None,
                 fail_after: Optional[int] = None) -> None:
        self.pieces = pieces
        self.fail = fail
        self.gate = gate
        self.fail_after = fail_after
        self.said: List[Tuple[str, EngineVoice]] = []
        self.warmed = 0
        self.cancelled = 0
        self.is_loaded = False

    @property
    def loaded(self) -> bool:
        return self.is_loaded

    def stream(self, text: str, voice: EngineVoice, language: str,
               cancel: CancelToken) -> Iterator[Tuple[bytes, int]]:
        if self.fail:
            raise TtsUnavailable("engine crashed")
        if self.gate is not None:
            self.gate.wait(5)
            if cancel.cancelled:
                self.cancelled += 1
                return
        self.is_loaded = True
        self.said.append((text, voice))
        for n in range(self.pieces):
            if self.fail_after is not None and n >= self.fail_after:
                raise TtsUnavailable("stopped mid-sentence")
            if cancel.cancelled:
                self.cancelled += 1
                return
            yield PIECE, RATE

    def warm(self, voice: EngineVoice) -> None:
        self.warmed += 1
        if self.fail:
            raise TtsUnavailable("no model")
        self.is_loaded = True


class Engines:
    """Hands out one fake engine per name; remembers releases."""

    def __init__(self, table: Dict[str, Engine],
                 installed: Optional[Set[str]] = None) -> None:
        self.table = table
        self.found = installed if installed is not None else {
            "pocket", "qwen3"}
        self.released: List[str] = []

    def installed(self) -> Set[str]:
        return self.found

    def get(self, voice: EngineVoice) -> SpeechStream:
        return self.table[voice.name]

    def release(self, engine: SpeechStream) -> None:
        for name, known in self.table.items():
            if known is engine:
                self.released.append(name)


class Gate:
    """A GPU gate that refuses with ``why`` (``None``: allows)."""

    def __init__(self, why: Optional[str] = None,
                 take: Optional[str] = None) -> None:
        self.why = why
        self.take_why = take
        self.asked: List[int] = []
        self.taken: List[int] = []
        self.dropped = 0

    def check(self, need_mib: int) -> Optional[str]:
        self.asked.append(need_mib)
        return self.why

    def take(self, need_mib: int) -> Optional[str]:
        self.taken.append(need_mib)
        return self.take_why

    def drop(self) -> None:
        self.dropped += 1


def request(text: str, audio_format: str = "pcm_s16le") -> Dict[str, Any]:
    return tts_request("r1", text, DAVEFX, "es", audio_format=audio_format)


def voice(chain: List[EngineVoice], table: Dict[str, Engine],
          gate: Optional[Gate] = None, budget_ms: float = 500.0,
          **more: Any) -> Tuple[NaturalVoice, List[Tuple[str, str]], FakeTts]:
    reports: List[Tuple[str, str]] = []
    piper = FakeTts()
    spoken = NaturalVoice(
        chain, Engines(table), piper, gate, budget_ms=budget_ms,
        report=lambda name, why: reports.append((name, why)),
        sleep=lambda _: None, **more)
    return spoken, reports, piper


# ---------------------------------------------------------------- domain


def test_the_chain_follows_the_choice_and_what_is_installed() -> None:
    both = {"pocket", "qwen3"}
    reference = ("/ref.wav", "Hola.")
    assert voice_chain("piper", "lola", "cpu", None, both, reference) == (
        [], "")
    assert voice_chain("pocket", "lola", "cpu", None, both, reference) == (
        [POCKET_CPU], "")
    assert voice_chain("pocket", "lola", "cuda", None, both, reference) == (
        [POCKET_GPU, POCKET_CPU], "")
    assert voice_chain("qwen3", "lola", "cpu", None, both, reference) == (
        [QWEN3, POCKET_CPU], "")
    cloned, _ = voice_chain("piper", "lola", "cpu", AGREED, both, None)
    assert cloned == [EngineVoice("qwen3", "cuda", reference="/ana.wav"),
                      POCKET_CPU], "a person's clip: no transcript"
    assert voice_chain("qwen3", "lola", "cpu", None, both, None) == (
        [POCKET_CPU], "no reference voice for Qwen3-TTS")
    assert voice_chain("qwen3", "lola", "cpu", None, {"pocket"},
                       reference) == ([POCKET_CPU],
                                      "Qwen3-TTS not installed")
    assert voice_chain("pocket", "vera", "cuda", None, set(), None) == (
        [], "Pocket TTS not installed")
    assert voice_chain("pocket", "vera", "cpu", None, both, None)[0] == [
        POCKET_CPU], "an unknown catalogue voice is lola"


def test_engine_names_and_their_gpu_memory() -> None:
    assert (QWEN3.name, QWEN3.need_mib) == ("qwen3-cuda", 4400)
    assert (POCKET_GPU.name, POCKET_GPU.need_mib) == ("pocket-cuda", 800)
    assert (POCKET_CPU.name, POCKET_CPU.need_mib) == ("pocket-cpu", 0)


def test_the_joiner_sends_the_first_piece_at_once_then_growing_ones() -> None:
    joiner = AudioJoiner(RATE, max_ms=200.0)
    sizes = []
    for _ in range(12):
        out = joiner.add(PIECE)
        if out is not None:
            sizes.append(len(out) // 2 * 1000 // RATE)
    rest = joiner.flush()
    assert sizes == [80, 80, 160, 240, 240]
    assert rest is not None and len(rest) // 2 * 1000 // RATE == 160
    assert joiner.flush() is None


def test_a_short_text_s_pieces_become_one_chunk() -> None:
    def chunk(text: str, rate: int,
              audio_format: str = "pcm_s16le") -> SpeechChunk:
        return SpeechChunk("r", 0, text, PIECE, audio_format, rate, 1.0)

    assert whole_chunk([]) is None
    one = chunk("Ajá.", RATE)
    assert whole_chunk([one]) is one
    joined = whole_chunk([one, chunk("", RATE), chunk("", 22050)])
    assert joined is not None and joined.text == "Ajá."
    assert len(joined.audio) == 2 * len(PIECE)
    as_wav = SpeechChunk("r", 0, "Ajá.", wav_bytes(PIECE, RATE), "wav",
                         RATE, 1.0)
    both = whole_chunk([as_wav, as_wav])
    assert both is not None and both.audio[:4] == b"RIFF"
    assert both.duration_ms == 160.0


def test_the_settings_choose_natural_voices_and_the_cloner() -> None:
    assert voice_choices({"voicebot.voice": "pocket:lola"}) == {
        "voice": DAVEFX, "engine": "pocket", "pocket_voice": "lola"}
    assert voice_choices({"voicebot.voice": "qwen3",
                          "voicebot.clone_engine": "openvoice",
                          "voicebot.pocket_device": "cuda"}) == {
        "voice": DAVEFX, "engine": "qwen3", "clone_engine": "openvoice",
        "pocket_device": "cuda"}
    assert voice_choices({"voicebot.voice": "pocket:vera",
                          "voicebot.clone_engine": "chatterbox",
                          "voicebot.pocket_device": "tpu"}) == {}


# ------------------------------------------------------------- use case


def test_a_streaming_engine_says_each_sentence_in_growing_pieces() -> None:
    pocket = Engine(pieces=4)
    spoken, reports, piper = voice([POCKET_CPU], {"pocket-cpu": pocket})
    chunks = list(spoken.synthesize(
        request("Hola, ¿qué tal estás? Cuéntame algo de ti.", "wav"),
        CancelToken()))
    assert [c.seq for c in chunks] == [0, 0, 0, 1, 1, 1]
    assert [c.text for c in chunks] == [
        "Hola, ¿qué tal estás?", "", "", "Cuéntame algo de ti.", "", ""]
    assert [c.duration_ms for c in chunks] == [80.0, 80.0, 160.0] * 2
    assert all(c.audio[:4] == b"RIFF" and c.sample_rate == RATE
               for c in chunks)
    assert spoken.spoken == {"pocket-cpu": 2}
    assert spoken.first_speaker == "pocket-cpu"
    assert reports == [] and piper.requests == []


def test_a_gpu_engine_runs_only_through_the_gate() -> None:
    qwen3, pocket = Engine(), Engine()
    gate = Gate(why="no live GPU lease")
    spoken, reports, _ = voice([QWEN3, POCKET_CPU],
                               {"qwen3-cuda": qwen3, "pocket-cpu": pocket},
                               gate)
    list(spoken.synthesize(request("Buenos días a todos."), CancelToken()))
    assert qwen3.said == [] and len(pocket.said) == 1
    assert gate.asked == [4400], "the memory is asked before it loads"
    assert reports == [("pocket-cpu", "qwen3-cuda: no live GPU lease")]
    gate.why = None
    list(spoken.synthesize(request("Buenos días a todos."), CancelToken()))
    assert len(qwen3.said) == 1
    assert reports[-1] == ("qwen3-cuda", "")
    list(spoken.synthesize(request("Otra frase más larga."), CancelToken()))
    assert gate.asked[-1] == 0, "loaded: only the lease is checked"
    gate.why = "no live GPU lease"
    engines = spoken.engines
    list(spoken.synthesize(request("Y una más para acabar."),
                           CancelToken()))
    assert isinstance(engines, Engines)
    assert engines.released == ["qwen3-cuda"], "the lease went: unloaded"
    spoken_none, _, _ = voice([QWEN3], {"qwen3-cuda": Engine()})
    assert [c.text for c in spoken_none.synthesize(
        request("Sin puerta a la GPU."), CancelToken())] == [
        "Sin puerta a la GPU."], "no gate: Piper"
    assert spoken_none.spoken == {"piper": 1}


def test_a_slow_first_audio_goes_to_the_next_voice_and_is_cancelled() -> None:
    hold = threading.Event()
    slow, pocket = Engine(gate=hold), Engine()
    spoken, reports, _ = voice(
        [POCKET_GPU, POCKET_CPU], {"pocket-cuda": slow, "pocket-cpu": pocket},
        Gate(), budget_ms=50.0)
    first = list(spoken.synthesize(request("Una frase que tarda."),
                                   CancelToken()))
    assert first[0].text == "Una frase que tarda."
    assert reports == [("pocket-cpu",
                        "pocket-cuda missed the 50 ms budget")]
    second = list(spoken.synthesize(request("Mientras sigue ocupado."),
                                    CancelToken()))
    assert second and len(pocket.said) == 2
    hold.set()
    spoken.tiers[0].busy.result(timeout=5)  # type: ignore[union-attr]
    assert slow.cancelled == 1 and slow.said == []
    spoken.close()


def test_a_failed_engine_is_not_asked_again() -> None:
    broken, pocket = Engine(fail=True), Engine()
    spoken, reports, _ = voice([QWEN3, POCKET_CPU],
                               {"qwen3-cuda": broken, "pocket-cpu": pocket},
                               Gate())
    for _ in range(2):
        list(spoken.synthesize(request("Hola a todos, buenas."),
                               CancelToken()))
    assert reports == [("pocket-cpu", "qwen3-cuda failed (engine crashed)")]
    assert spoken.tiers[0].failed == "engine crashed"
    assert spoken.spoken == {"pocket-cpu": 2}


def test_piper_speaks_when_no_engine_can() -> None:
    spoken, reports, piper = voice(
        [POCKET_CPU], {"pocket-cpu": Engine(fail=True)})
    chunks = list(spoken.synthesize(
        request("Primera frase aquí. Segunda frase aquí."), CancelToken()))
    assert [c.seq for c in chunks] == [0, 1]
    assert all(c.request_id == "r1" for c in chunks)
    assert len(piper.requests) == 2
    assert reports == [("piper", "pocket-cpu failed (engine crashed)")]


def test_a_sentence_cut_mid_way_ends_there() -> None:
    cut = Engine(pieces=5, fail_after=2)
    spoken, _, _ = voice([POCKET_CPU], {"pocket-cpu": cut})
    chunks = list(spoken.synthesize(request("Se corta a la mitad."),
                                    CancelToken()))
    assert sum(c.duration_ms for c in chunks) == 160.0
    hold = threading.Event()
    stuck = Engine(pieces=3)
    spoken, _, _ = voice([POCKET_CPU], {"pocket-cpu": stuck},
                         chunk_timeout_ms=20.0)
    original = stuck.stream

    def slow_after_first(text: str, v: EngineVoice, language: str,
                         cancel: CancelToken) -> Iterator[Tuple[bytes, int]]:
        pieces = original(text, v, language, cancel)
        yield next(pieces)
        hold.wait(0.3)
        yield from pieces

    stuck.stream = slow_after_first  # type: ignore[method-assign,assignment]
    chunks = list(spoken.synthesize(request("Se queda parada."),
                                    CancelToken()))
    assert len(chunks) == 1, "the next piece was late: the sentence ends"


def test_a_barge_in_stops_the_sentence_and_the_rest() -> None:
    pocket = Engine(pieces=50)
    spoken, _, piper = voice([POCKET_CPU], {"pocket-cpu": pocket})
    cancel = CancelToken()
    seen = []
    for chunk in spoken.synthesize(
            request("Una frase muy larga. Y otra más detrás."), cancel):
        seen.append(chunk)
        cancel.cancel()
    assert len(seen) == 1 and piper.requests == []
    done = CancelToken()
    done.cancel()
    assert list(spoken.synthesize(request("Nada que decir."), done)) == []
    hold = threading.Event()
    waiting = Engine(gate=hold)
    spoken, reports, piper = voice([POCKET_CPU], {"pocket-cpu": waiting})
    cancel = CancelToken()
    threading.Timer(0.05, cancel.cancel).start()
    assert list(spoken.synthesize(request("Cancelada antes de sonar."),
                                  cancel)) == []
    assert reports == [] and piper.requests == []
    hold.set()


def test_the_whole_text_as_one_sentence_when_not_streamed() -> None:
    spoken, _, _ = voice([POCKET_CPU], {"pocket-cpu": Engine(pieces=1)})
    doc = dict(request("Uno. Dos tres cuatro."), stream=False)
    assert [c.text for c in spoken.synthesize(doc, CancelToken())] == [
        "Uno. Dos tres cuatro."]


def test_warming_waits_for_the_gate_and_marks_failures() -> None:
    qwen3, pocket = Engine(), Engine(fail=True)
    gate = Gate(why="no live GPU lease")
    spoken, _, _ = voice([QWEN3, POCKET_CPU],
                         {"qwen3-cuda": qwen3, "pocket-cpu": pocket}, gate,
                         warm_wait_s=3.0)
    spoken.warm()
    assert qwen3.warmed == 0 and len(gate.asked) >= 4
    assert spoken.tiers[1].failed == "no model"
    gate.why = None
    spoken.warm()
    assert qwen3.warmed == 1
    spoken.close()
    assert spoken.engines.released == ["qwen3-cuda",  # type: ignore
                                       "pocket-cpu"]
    assert gpu_need(spoken) == 4400


# ------------------------------------------------------------- sessions


class Settings:
    def __init__(self, choices: Dict[str, str]) -> None:
        self.choices = choices

    def current(self) -> Optional[Dict[str, str]]:
        return self.choices


def natural_sessions(choices: Dict[str, str], table: Dict[str, Engine],
                     gate: Optional[Gate] = None,
                     installed: Optional[Set[str]] = None,
                     events: Optional[List[Any]] = None) -> Any:
    live = bot_sessions(events or [], tts=FakeTts())
    live.settings = Settings(choices)
    live.natural = NaturalWiring(Engines(table, installed), gate,
                                 ("/ref.wav", "Hola."), 500.0)
    live.profiles = type("P", (), {"get": staticmethod(
        {"ana": AGREED}.get)})()
    return live


def test_a_session_speaks_with_pocket_by_default_and_says_so() -> None:
    events = [user_final("t0.s0", "¿Puedo devolver un producto?", 1)]
    pocket = Engine()
    live = natural_sessions({}, {"pocket-cpu": pocket}, events=events)
    live.start_live("voicebot", "mic")
    feed = messages(live, "vb-1")
    assert [m for m in feed if m["type"] == "voice"] == [{
        "type": "voice", "engine": "pocket-cpu", "clone": False,
        "length_scale": 1.0, "reason": ""}]
    speech = [m for m in feed if m["type"] == "speech"]
    assert speech and speech[0]["sample_rate"] == RATE
    assert pocket.warmed == 1 and pocket.said
    records = [m["record"] for m in feed if m["type"] == "turn"]
    assert records[0]["spoken"] and "" not in records[0]["spoken"]


def test_a_session_reports_when_the_speaker_changes() -> None:
    events = [user_final("t0.s0", "¿Puedo devolver un producto?", 1)]
    gate = Gate(why="no live GPU lease")
    live = natural_sessions({"voicebot.voice": "qwen3"},
                            {"qwen3-cuda": Engine(),
                             "pocket-cpu": Engine()}, gate, events=events)
    live.start_live("voicebot", "mic")
    voices = [m for m in messages(live, "vb-1") if m["type"] == "voice"]
    assert voices[0]["engine"] == "qwen3-cuda"
    assert voices[1] == {"type": "voice", "engine": "pocket-cpu",
                         "clone": False, "length_scale": 1.0,
                         "reason": "qwen3-cuda: no live GPU lease"}


def test_cloning_with_qwen3_keeps_the_consent_rules() -> None:
    qwen3 = Engine()
    live = natural_sessions({"voicebot.voice": "clone:ana"},
                            {"qwen3-cuda": qwen3, "pocket-cpu": Engine()},
                            Gate())
    live.start_live("voicebot", "mic")
    voices = [m for m in messages(live, "vb-1") if m["type"] == "voice"]
    assert voices == [{"type": "voice", "engine": "qwen3-cuda",
                       "clone": True, "length_scale": 1.0, "reason": ""}]
    assert qwen3.warmed == 1
    refused = natural_sessions({"voicebot.voice": "clone:bea"},
                               {"qwen3-cuda": Engine(),
                                "pocket-cpu": Engine()}, Gate())
    refused.start_live("voicebot", "mic")
    voices = [m for m in messages(refused, "vb-1") if m["type"] == "voice"]
    assert voices == [{"type": "voice", "engine": "piper", "clone": False,
                       "length_scale": 1.0, "reason": "no voice profile"}]
    off = bot_sessions([], tts=FakeTts())
    off.settings = Settings({"voicebot.voice": "qwen3"})
    off.start_live("voicebot", "mic")
    voices = [m for m in messages(off, "vb-1") if m["type"] == "voice"]
    assert voices[0]["engine"] == "piper"
    assert voices[0]["reason"] == "natural voices off"
    missing = natural_sessions({"voicebot.voice": "pocket:lola"}, {},
                               installed=set())
    missing.start_live("voicebot", "mic")
    voices = [m for m in messages(missing, "vb-1") if m["type"] == "voice"]
    assert voices[0]["reason"] == "Pocket TTS not installed"


def test_the_sample_takes_the_lease_for_a_gpu_voice() -> None:
    gate = Gate()
    qwen3 = Engine()
    live = natural_sessions({}, {"qwen3-cuda": qwen3,
                                 "pocket-cpu": Engine()}, gate)
    wav, used = live.sample({"voicebot.voice": "qwen3"})
    with wave.open(BytesIO(wav), "rb") as reader:
        assert reader.getframerate() == RATE
    assert used["engine"] == "qwen3-cuda" and used["clone"] is False
    assert used["reason"] == ""
    assert gate.taken == [4400] and gate.dropped == 1
    assert live.natural.engines.released == ["qwen3-cuda"]
    busy = Gate(why="the GPU lease is held by batch", take="no GPU found")
    live = natural_sessions({}, {"qwen3-cuda": Engine(),
                                 "pocket-cpu": Engine()}, busy)
    _, used = live.sample({"voicebot.voice": "clone:ana"})
    assert used["engine"] == "pocket-cpu" and used["clone"] is False
    assert used["reason"] == ("no GPU found; qwen3-cuda: the GPU lease is "
                              "held by batch")
    _, used = live.sample({"voicebot.voice": "pocket:lola"})
    assert used == {"voice": DAVEFX, "length_scale": 1.0, "clone": False,
                    "reason": "", "engine": "pocket-cpu"}
    assert busy.taken == [4400], "a CPU voice takes no lease"


def test_the_sample_falls_back_and_reports_failures() -> None:
    live = natural_sessions({}, {"pocket-cpu": Engine(fail=True)})
    _, used = live.sample({"voicebot.voice": "pocket:lola"})
    assert used["engine"] == "piper"
    assert used["reason"] == "pocket-cpu failed (engine crashed)"
    from orchestrator.application.sessions import SessionError

    class Mute(FakeTts):
        def synthesize(self, request: Any, cancel: Any) -> Any:
            return iter(())

    live.tts = Mute()
    with pytest.raises(SessionError):
        live.sample({"voicebot.voice": "pocket:lola"})


def test_a_session_that_ends_while_a_model_loads_frees_it() -> None:
    qwen3 = Engine()
    spoken, _, _ = voice([QWEN3], {"qwen3-cuda": qwen3}, Gate())
    original = qwen3.warm

    def ends_meanwhile(v: EngineVoice) -> None:
        original(v)
        spoken._closed = True

    qwen3.warm = ends_meanwhile  # type: ignore[method-assign]
    spoken.warm()
    assert spoken.engines.released == ["qwen3-cuda"]  # type: ignore
    later, _, _ = voice([QWEN3], {"qwen3-cuda": Engine()}, Gate())
    later.close()
    later.warm()
    assert later.tiers[0].engine is None, "closed: nothing is loaded"
