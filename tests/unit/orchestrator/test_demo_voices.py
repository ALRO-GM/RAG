"""The scripted examples' voices (DM-3, #254): rules and use cases."""

import json
import re
from array import array
from pathlib import Path
from typing import AbstractSet, Any, Dict, List, Optional, Tuple

import pytest

from orchestrator.adapters.outbound.demo_speaker import ToneDemoSpeaker
from orchestrator.adapters.outbound.fs_demos import DirDemos
from orchestrator.application.demo_voices import (
    DemoVoiceError,
    DemoVoiceLibrary,
    RenderDemoVoices,
)
from orchestrator.application.ports import TtsUnavailable
from orchestrator.domain.demo import DemoScript, recast
from orchestrator.domain.demo_voices import (
    PEAK,
    STOCK_VOICES,
    TARGET_RMS,
    DemoClip,
    DemoVoice,
    DemoVoiceManifest,
    check_voices,
    demo_lines,
    even_loudness,
    page_view,
    parse_voice,
    playable,
    retime,
    text_hash,
)
from orchestrator.domain.voices import VoiceProfile

REPO = Path(__file__).resolve().parents[3]
DEMOS = REPO / "examples" / "demo"
CALL = "llamada-lumbre"
STOCK = {"agente": "pocket:lola", "cliente": "piper:davefx",
         "asistente": "piper:sharvard-f"}


def raw(demo: str = CALL) -> Dict[str, Any]:
    """A demo file as plain data."""
    data: Dict[str, Any] = json.loads(
        (DEMOS / f"{demo}.demo.json").read_text("utf-8"))
    return data


def script(demo: str = CALL) -> DemoScript:
    return DirDemos(DEMOS).load(demo)


class Scripts:
    """A DemoScripts port over scripts in memory."""

    def __init__(self, *scripts: DemoScript):
        self.scripts = {s.id: s for s in scripts}

    def list(self) -> List[Dict[str, object]]:
        return [s.summary() for s in self.scripts.values()]

    def load(self, demo_id: str) -> DemoScript:
        return self.scripts[demo_id]


class Store:
    """A DemoVoiceStore in memory."""

    def __init__(self) -> None:
        self.manifests: Dict[str, DemoVoiceManifest] = {}
        self.files: Dict[Tuple[str, str], bytes] = {}
        self.saves = 0

    def manifest(self, demo_id: str) -> Optional[DemoVoiceManifest]:
        return self.manifests.get(demo_id)

    def save_manifest(self, manifest: DemoVoiceManifest) -> None:
        self.saves += 1
        self.manifests[manifest.demo] = manifest

    def write_clip(self, demo_id: str, name: str, wav: bytes) -> None:
        self.files[(demo_id, name)] = wav

    def read_clip(self, demo_id: str, name: str) -> bytes:
        return self.files[(demo_id, name)]

    def has_clip(self, demo_id: str, name: str) -> bool:
        return (demo_id, name) in self.files

    def delete_clip(self, demo_id: str, name: str) -> None:
        self.files.pop((demo_id, name), None)

    def sweep(self, demo_id: str, keep: AbstractSet[str]) -> int:
        gone = [k for k in self.files if k[0] == demo_id and k[1] not in keep]
        for k in gone:
            del self.files[k]
        return len(gone)

    def clear(self, demo_id: str) -> int:
        gone = [k for k in self.files if k[0] == demo_id]
        for k in gone:
            del self.files[k]
        self.manifests.pop(demo_id, None)
        return len(gone)


class Profiles:
    """VoiceProfiles from a dict."""

    def __init__(self, **people: VoiceProfile):
        self.people = dict(people)

    def get(self, speaker_id: str) -> Optional[VoiceProfile]:
        return self.people.get(speaker_id)


def agreed(person: str) -> VoiceProfile:
    return VoiceProfile(person, clone_consent=True,
                        reference=f"/x/{person}.clone.wav")


class Gate:
    """An ExclusiveGpu whose answers the test sets."""

    def __init__(self, live: bool = False, refuse: Optional[str] = None):
        self.live = live
        self.refuse = refuse
        self.taken: List[int] = []
        self.dropped = 0
        self.log: List[str] = []

    def take_alone(self, need_mib: int) -> Optional[str]:
        self.taken.append(need_mib)
        self.log.append("take")
        if self.live:
            return "the GPU lease is held by transcriber-live"
        return self.refuse

    def drop(self) -> None:
        self.dropped += 1
        self.log.append("drop")


class Failing(ToneDemoSpeaker):
    """Fails on the ``after``-th line."""

    def __init__(self, after: int):
        super().__init__()
        self.after = after

    def speak(self, text: str, voice: DemoVoice, reference: Optional[str],
              language: str) -> Tuple[bytes, int]:
        if len(self.said) >= self.after:
            raise TtsUnavailable("worker died")
        return super().speak(text, voice, reference, language)


def renderer(store: Store, speaker: Optional[ToneDemoSpeaker] = None,
             gate: Optional[Gate] = None, profiles: Optional[Profiles] = None,
             demo: Optional[DemoScript] = None) -> RenderDemoVoices:
    return RenderDemoVoices(
        Scripts(demo or script()), store, speaker or ToneDemoSpeaker(),
        profiles or Profiles(), gate, now=lambda: "2026-11-24T10:00:00Z",
    )


# ---- the cast and the names ----

def test_every_speaker_is_in_the_cast_with_the_same_name() -> None:
    demo = script()
    cast = {c.id: c for c in demo.cast}
    assert set(cast) == {"agente", "cliente", "asistente"}
    for speaker in demo.transcript.speakers:
        assert cast[speaker.id].name == speaker.name
    assert demo.candidate == cast["cliente"].name
    assert demo.voicebot is not None
    assert (demo.voicebot.user, demo.voicebot.bot) == ("cliente", "asistente")
    interview = script("entrevista-lucia")
    assert {c.id for c in interview.cast} == {"entrevistador", "candidata"}


def test_a_cast_that_disagrees_with_the_transcript_is_refused() -> None:
    data = raw()
    data["cast"][0]["name"] = "Otra"
    with pytest.raises(ValueError, match="in the cast"):
        DemoScript.model_validate(data)
    data = raw()
    data["cast"] = data["cast"][:1]
    with pytest.raises(ValueError, match="not in the cast"):
        DemoScript.model_validate(data)
    data = raw()
    data["cast"].append(dict(data["cast"][0]))
    with pytest.raises(ValueError, match="twice"):
        DemoScript.model_validate(data)


def _names(data: Dict[str, Any]) -> str:
    """Every text of the demo but its cast, as one string."""
    return json.dumps({k: v for k, v in data.items() if k != "cast"},
                      ensure_ascii=False)


def test_the_report_names_only_the_cast() -> None:
    data = raw()
    report = json.dumps(data["report"], ensure_ascii=False)
    assert "Andrés Ferrer" in report and "Marta" in report
    capitalised = set(re.findall(r"\b[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+\b", report))
    # The only people the report names are the cast's.
    for name in ("Lucía", "Pablo", "Luis", "Irene", "Ana", "Mario"):
        assert name not in capitalised


def test_a_rename_reaches_every_text_and_can_be_undone() -> None:
    data = raw()
    renamed = recast(data, "agente", "Mario", "male")
    text = _names(renamed)
    assert "Marta" not in text and text.count("Mario") >= 4
    demo = DemoScript.model_validate(renamed)
    assert demo.transcript.speakers[0].name == "Mario"
    assert demo.cast[0].gender == "male"
    assert "Mario" in demo.transcript.utterances[0].text
    assert demo.report is not None
    report = json.dumps(demo.report.model_dump(), ensure_ascii=False)
    assert "Mario" in report and "Marta" not in report
    assert recast(renamed, "agente", "Marta", "female") == data
    customer = recast(data, "cliente", "Pablo Gil")
    assert "Ferrer" not in _names(customer)
    assert "señor Gil" in _names(customer)
    assert recast(customer, "cliente", "Andrés Ferrer") == data


def test_a_rename_that_cannot_be_undone_is_refused() -> None:
    data = raw()
    with pytest.raises(ValueError, match="rewording"):
        recast(data, "cliente", "Ana Ferrer", "female")
    with pytest.raises(ValueError, match="word"):
        recast(data, "cliente", "Pablo")
    with pytest.raises(ValueError, match="already in the demo"):
        recast(data, "agente", "Lumbre")
    with pytest.raises(ValueError, match="not in the cast"):
        recast(data, "nadie", "X")
    with pytest.raises(ValueError, match="analysis"):
        recast(raw("entrevista-lucia"), "candidata", "Ana Gil")
    with pytest.raises(ValueError, match="gender"):
        recast(data, "agente", "Mario", "other")
    assert recast(data, "agente", "Marta") == data


# ---- voices ----

def test_voices_are_read_as_the_operator_writes_them() -> None:
    assert parse_voice("pocket:lola").gender == "female"
    assert parse_voice("piper:sharvard-m").speaker == 0
    assert parse_voice("piper:sharvard-f").speaker == 1
    clone = parse_voice("clone:ana:female", "openvoice")
    assert (clone.person, clone.gender, clone.engine, clone.kind) == (
        "ana", "female", "openvoice", "clone")
    assert parse_voice("clone:ana").gender is None
    assert parse_voice("clone:ana").spec == "clone:ana"
    for bad in ("davefx", "clone:", "clone:../x", "clone:a:other",
                "pocket:anna"):
        with pytest.raises(ValueError):
            parse_voice(bad)
    with pytest.raises(ValueError):
        parse_voice("clone:ana", "chatterbox")
    assert all(v.kind == "stock" for v in STOCK_VOICES.values())


def _voices(**specs: str) -> Dict[str, DemoVoice]:
    return {p: parse_voice(s) for p, s in {**STOCK, **specs}.items()}


def test_stock_voices_that_fit_the_cast_pass() -> None:
    assert check_voices(script(), _voices(), Profiles().get) == []


def test_a_voice_of_the_other_gender_is_refused() -> None:
    problems = check_voices(script(), _voices(agente="piper:sharvard-m"),
                            Profiles().get)
    assert any("male voice" in p and "Marta" in p for p in problems)
    # A clone of unknown gender cannot be checked; a declared one is.
    assert check_voices(script(), _voices(agente="clone:ana"),
                        Profiles(ana=agreed("ana")).get) == []
    assert check_voices(script(), _voices(agente="clone:ana:male"),
                        Profiles(ana=agreed("ana")).get)


def test_two_people_of_one_example_never_share_a_voice() -> None:
    problems = check_voices(script(), _voices(cliente="pocket:lola"),
                            Profiles().get)
    assert any("share pocket:lola in the call" in p for p in problems)
    shared = _voices(cliente="clone:ana", asistente="clone:ana")
    problems = check_voices(script(), shared,
                            Profiles(ana=agreed("ana")).get)
    assert problems == [
        "'asistente' and 'cliente' would share clone:ana in the voicebot"]
    # The agent and the assistant never talk to each other.
    assert check_voices(script(), _voices(asistente="pocket:lola"),
                        Profiles().get) == []


def test_a_clone_needs_consent_and_a_reference_clip() -> None:
    voices = _voices(agente="clone:ana")
    assert "clone:ana: no enrolled person" in check_voices(
        script(), voices, Profiles().get)
    assert "clone:ana: no consent to cloning" in check_voices(
        script(), voices, Profiles(ana=VoiceProfile("ana")).get)
    assert "clone:ana: no reference clip" in check_voices(
        script(), voices,
        Profiles(ana=VoiceProfile("ana", clone_consent=True)).get)


def test_every_speaking_person_needs_a_voice_and_nobody_else() -> None:
    voices = _voices()
    del voices["asistente"]
    voices["nadie"] = parse_voice("pocket:lola")
    problems = check_voices(script(), voices, Profiles().get)
    assert "no voice for 'asistente'" in problems
    assert any("'nadie' says nothing" in p for p in problems)


def test_lines_cover_both_examples_in_order() -> None:
    lines = demo_lines(script())
    assert [x.key for x in lines[:2]] == ["t0.s0", "t1.s0"]
    assert [x.key for x in lines[-2:]] == ["vb3.user", "vb3.bot"]
    assert {x.example for x in lines} == {"call", "voicebot"}
    assert len(lines) == 19 + 8
    assert text_hash(" Hola  ") == text_hash("Hola")


# ---- render ----

def test_render_writes_one_clip_per_line_and_a_manifest() -> None:
    store = Store()
    speaker = ToneDemoSpeaker()
    result = renderer(store, speaker).render(CALL, STOCK)
    manifest = store.manifests[CALL]
    assert len(result.rendered) == 27 and result.lease is None
    assert manifest.voices == STOCK
    assert set(manifest.clips) == {x.key for x in demo_lines(script())}
    clip = manifest.clips["t0.s0"]
    assert (clip.voice, clip.kind, clip.cloned_from, clip.engine) == (
        "pocket:lola", "stock", None, "tone")
    assert clip.seconds > 0 and store.has_clip(CALL, clip.file)
    assert store.files[(CALL, clip.file)][:4] == b"RIFF"
    assert len(store.files) == 27


def test_render_again_renders_only_what_changed() -> None:
    store = Store()
    renderer(store).render(CALL, STOCK)
    saves = store.saves
    speaker = ToneDemoSpeaker()
    again = renderer(store, speaker).render(CALL, {})
    assert again.rendered == [] and len(again.kept) == 27
    assert speaker.said == [] and store.saves == saves
    one = renderer(store, speaker).render(CALL, {"agente": "piper:sharvard-f",
                                                 "asistente": "pocket:lola"})
    agent = [x.key for x in demo_lines(script())
             if x.person in ("agente", "asistente")]
    assert sorted(one.rendered) == sorted(agent)
    assert len(store.files) == 27  # the replaced clips are gone
    data = raw()
    data["transcript"]["utterances"][3]["text"] += " Gracias."
    changed = DemoScript.model_validate(data)
    text = renderer(store, demo=changed).render(CALL, {})
    assert text.rendered == ["t3.s0"] and len(text.removed) == 1


def test_a_missing_voice_is_never_picked_silently() -> None:
    with pytest.raises(DemoVoiceError, match="no voice for 'asistente'"):
        renderer(Store()).render(CALL, {"agente": "pocket:lola",
                                        "cliente": "piper:davefx"})
    with pytest.raises(DemoVoiceError, match="unknown voice"):
        renderer(Store()).render(CALL, {**STOCK, "agente": "lola"})
    with pytest.raises(DemoVoiceError, match="no demo"):
        renderer(Store()).render("nada", STOCK)


def test_a_clone_takes_the_lease_only_when_it_needs_the_gpu() -> None:
    profiles = Profiles(ana=agreed("ana"))
    gate = Gate()
    store = Store()
    speaker = ToneDemoSpeaker(need=4400)
    result = renderer(store, speaker, gate, profiles).render(
        CALL, {**STOCK, "agente": "clone:ana"})
    assert gate.taken == [4400] and gate.dropped == 1
    assert result.lease == 4400
    clip = store.manifests[CALL].clips["t0.s0"]
    assert (clip.kind, clip.cloned_from) == ("clone", "ana")
    stock = Gate()
    renderer(Store(), ToneDemoSpeaker(need=4400), stock).render(CALL, STOCK)
    assert stock.taken == []


def test_a_refused_lease_keeps_the_previous_clips() -> None:
    store = Store()
    renderer(store).render(CALL, STOCK)
    before = (dict(store.files), store.manifests[CALL])
    profiles = Profiles(ana=agreed("ana"))
    for gate, why in ((Gate(refuse="the GPU lease is held by img"),
                       "refused"),
                      (Gate(live=True), "transcriber-live"),
                      (None, "no lease")):
        with pytest.raises(DemoVoiceError, match=why):
            renderer(store, ToneDemoSpeaker(need=800), gate,
                     profiles).render(CALL, {"agente": "clone:ana"})
        assert (store.files, store.manifests[CALL]) == before
        if gate is not None and gate.taken:
            assert gate.dropped == len(gate.taken)


def test_an_engine_failure_keeps_the_previous_clips() -> None:
    store = Store()
    renderer(store).render(CALL, STOCK)
    before = (dict(store.files), store.manifests[CALL])
    with pytest.raises(DemoVoiceError, match="worker died"):
        renderer(store, Failing(after=3)).render(
            CALL, {"agente": "piper:sharvard-f", "asistente": "pocket:lola"})
    assert (store.files, store.manifests[CALL]) == before


def test_a_consent_withdrawn_mid_render_stops_it() -> None:
    profiles = Profiles(ana=agreed("ana"))

    class Withdraws(ToneDemoSpeaker):
        def speak(self, text: str, voice: DemoVoice,
                  reference: Optional[str],
                  language: str) -> Tuple[bytes, int]:
            profiles.people["ana"] = VoiceProfile("ana")
            return super().speak(text, voice, reference, language)

    store = Store()
    with pytest.raises(DemoVoiceError, match="consent or clip gone"):
        renderer(store, Withdraws(), Gate(), profiles).render(
            CALL, {**STOCK, "cliente": "clone:ana"})
    assert store.files == {} and CALL not in store.manifests


# ---- what the page may play ----

def library(store: Store, profiles: Optional[Profiles] = None,
            demo: Optional[DemoScript] = None) -> DemoVoiceLibrary:
    return DemoVoiceLibrary(Scripts(demo or script()), store,
                            profiles or Profiles())


def test_the_page_sees_lengths_and_kinds_never_people_or_files() -> None:
    store = Store()
    profiles = Profiles(ana=agreed("ana"))
    renderer(store, gate=Gate(), profiles=profiles).render(
        CALL, {**STOCK, "cliente": "clone:ana"})
    view = library(store, profiles).view(CALL)
    examples: Dict[str, Any] = view["examples"]
    assert examples["call"]["available"] is True
    assert len(examples["call"]["clips"]) == 19
    assert len(examples["voicebot"]["clips"]) == 8
    assert view["voices"] == {"agente": "stock", "asistente": "stock",
                              "cliente": "clone"}
    assert view["kinds"] == ["clone", "stock"]
    text = json.dumps(view)
    assert "ana" not in text and ".wav" not in text
    assert library(store, profiles).clip(CALL, "t1.s0")[:4] == b"RIFF"
    with pytest.raises(KeyError):
        library(store, profiles).clip(CALL, "t99.s0")


def test_a_withdrawn_consent_deletes_the_clips_and_falls_back() -> None:
    store = Store()
    profiles = Profiles(ana=agreed("ana"))
    renderer(store, gate=Gate(), profiles=profiles).render(
        CALL, {**STOCK, "cliente": "clone:ana"})
    profiles.people["ana"] = VoiceProfile("ana")  # withdrawn
    view = library(store, profiles).view(CALL)
    examples: Dict[str, Any] = view["examples"]
    assert examples["call"]["available"] is False
    assert examples["voicebot"]["available"] is False
    clips = store.manifests[CALL].clips
    assert all(c.cloned_from != "ana" for c in clips.values())
    # The customer says 9 lines of the call and 4 of the voice example.
    assert len(store.files) == len(clips) == 27 - 9 - 4
    with pytest.raises(KeyError):
        library(store, profiles).clip(CALL, "t1.s0")
    del profiles.people["ana"]  # forgotten: nothing more to delete
    assert library(store, profiles).view(CALL)["voices"] == {}


def test_a_changed_text_is_not_played() -> None:
    store = Store()
    renderer(store).render(CALL, STOCK)
    data = raw()
    data["voicebot"]["exchanges"][0]["bot"] += " Gracias."
    changed = DemoScript.model_validate(data)
    view = library(store, demo=changed).view(CALL)
    examples: Dict[str, Any] = view["examples"]
    assert examples["voicebot"]["available"] is False
    assert examples["call"]["available"] is True
    with pytest.raises(KeyError):
        library(store, demo=changed).clip(CALL, "vb0.user")


def test_an_example_is_heard_whole_or_not_at_all() -> None:
    store = Store()
    renderer(store).render(CALL, STOCK)
    manifest = store.manifests[CALL]
    lines = demo_lines(script())
    clip = manifest.clips["t5.s0"]
    store.delete_clip(CALL, clip.file)
    found = playable(lines, manifest, lambda n: store.has_clip(CALL, n))
    assert "call" not in found and len(found["voicebot"]) == 8
    other = manifest.model_copy(update={"voices": {**STOCK,
                                                   "agente": "x:y"}})
    assert "voicebot" in playable(lines, other, lambda n: True)
    assert "call" not in playable(lines, other, lambda n: True)
    assert playable(lines, None, lambda n: True) == {}


def test_nothing_rendered_reads_as_not_available() -> None:
    view = library(Store()).view(CALL)
    assert view["examples"] == {
        "call": {"available": False, "clips": {}},
        "voicebot": {"available": False, "clips": {}}}
    assert page_view(script(), {})["kinds"] == []


# ---- timing ----

def test_the_call_takes_the_clips_lengths_and_keeps_its_pauses() -> None:
    demo = script()
    first, second = demo.transcript.utterances[:2]
    timed = retime(demo.transcript, {"t0.s0": 2.0, "t1.s0": 12.5})
    a, b, c = timed.utterances[:3]
    assert (a.start, a.end) == (first.start, first.start + 2.0)
    assert b.start == pytest.approx(a.end + (second.start - first.end))
    assert b.end == pytest.approx(b.start + 12.5)
    original = demo.transcript.utterances[2]
    assert c.end - c.start == pytest.approx(original.end - original.start)
    assert retime(demo.transcript, {}).utterances == \
        demo.transcript.utterances


def test_a_heard_call_is_replayed_at_its_clips_pace() -> None:
    store = Store()
    renderer(store, ToneDemoSpeaker(seconds_per_word=0.5)).render(CALL, STOCK)
    loaded = library(store).load(CALL)
    clips = store.manifests[CALL].clips
    for u in loaded.transcript.utterances:
        assert u.end - u.start == pytest.approx(clips[u.id].seconds,
                                                abs=0.002)
    assert library(store).list()[0]["seconds"] == round(
        loaded.transcript.utterances[-1].end, 1)
    assert library(Store()).load(CALL) == script()


def test_a_listed_demo_that_cannot_be_read_is_left_out() -> None:
    class Broken(Scripts):
        def list(self) -> List[Dict[str, object]]:
            return [{"id": "gone"}, *super().list()]

    found = DemoVoiceLibrary(Broken(script()), Store(), Profiles()).list()
    assert [d["id"] for d in found] == [CALL]


def test_a_clip_that_does_not_fit_is_rendered_again() -> None:
    store = Store()
    renderer(store).render(CALL, STOCK)
    manifest = store.manifests[CALL]
    clip = manifest.clips["t2.s0"]
    store.manifests[CALL] = manifest.model_copy(update={"clips": {
        **manifest.clips,
        "t2.s0": DemoClip(**{**clip.model_dump(), "engine": "piper"})}})
    again = renderer(store).render(CALL, {})
    assert again.rendered == ["t2.s0"]


def test_every_clip_is_brought_to_one_level_without_clipping() -> None:
    quiet = array("h", [100, -100] * 500).tobytes()
    loud = array("h", [20000, -20000] * 500).tobytes()
    for pcm in (quiet, loud):
        out = array("h")
        out.frombytes(even_loudness(pcm))
        rms = (sum(s * s for s in out) / len(out)) ** 0.5
        assert rms == pytest.approx(TARGET_RMS, rel=0.01)
    spiky = array("h", [0] * 999 + [1000]).tobytes()
    out = array("h")
    out.frombytes(even_loudness(spiky))
    assert max(out) == PEAK, "the gain stops at the peak"
    assert even_loudness(b"\x00\x00" * 10) == b"\x00\x00" * 10
    assert even_loudness(b"") == b"" and even_loudness(b"\x01") == b""


def test_the_lease_goes_back_only_after_the_engine_stopped() -> None:
    gate = Gate()

    class Logged(ToneDemoSpeaker):
        def close(self) -> None:
            gate.log.append("close")

    renderer(Store(), Logged(need=4400), gate,
             Profiles(ana=agreed("ana"))).render(
        CALL, {**STOCK, "agente": "clone:ana"})
    assert gate.log == ["take", "close", "drop"]
    refused = Gate(refuse="busy")
    with pytest.raises(DemoVoiceError):
        renderer(Store(), Logged(need=4400), refused,
                 Profiles(ana=agreed("ana"))).render(
            CALL, {**STOCK, "agente": "clone:ana"})
    assert refused.log == ["take", "drop"], "nothing started, nothing closed"


def test_a_consent_withdrawn_after_her_lines_publishes_none_of_them() -> None:
    profiles = Profiles(ana=agreed("ana"))
    store = Store()
    renderer(store).render(CALL, STOCK)
    before = (dict(store.files), store.manifests[CALL])

    class Late(ToneDemoSpeaker):
        def speak(self, text: str, voice: DemoVoice,
                  reference: Optional[str],
                  language: str) -> Tuple[bytes, int]:
            if voice.person is None:  # her lines are done: withdraw now
                profiles.people["ana"] = VoiceProfile("ana")
            return super().speak(text, voice, reference, language)

    with pytest.raises(DemoVoiceError, match="withdrawn during the render"):
        renderer(store, Late(), Gate(), profiles).render(
            CALL, {"cliente": "clone:ana", "agente": "piper:sharvard-f"})
    assert (store.files, store.manifests[CALL]) == before


def test_a_render_sweeps_files_no_manifest_names() -> None:
    store = Store()
    renderer(store).render(CALL, STOCK)
    store.files[(CALL, "t0.s0-0000000000.wav")] = b"RIFF"  # a killed run
    store.files[(CALL, ".part-x.wav")] = b"RI"
    renderer(store).render(CALL, {})
    assert len(store.files) == 27
    assert set(n for _, n in store.files) == {
        c.file for c in store.manifests[CALL].clips.values()}
