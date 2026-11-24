"""Use cases: render the scripted examples' voices and serve them (#254).

:class:`RenderDemoVoices` is the operator's command: it says every line
of a demo with the voice chosen for each scripted person, ahead of
time, so playing an example needs no model, GPU or session slot. It
renders only what changed (a line's text, voice or engine), takes the
GPU lease only when an engine needs the GPU, never while a live session
holds it, and replaces the clips only once every new one is written: a
refused lease or a failing engine leaves the previous clips as they
were.

:class:`DemoVoiceLibrary` is what the live page uses. It answers which
examples can be heard and their clips' lengths, gives a clip, and is
the demos' source for the sessions (:class:`~.ports.DemoScripts`): a
call whose clips are all there is replayed with its sentences as long
as the clips (:func:`~orchestrator.domain.demo_voices.retime`), so the
text types in while the voice says it. Each time it looks, the clips of
a person who withdrew their clone consent (or was forgotten) are
deleted, as the transcriber does when the consent is withdrawn.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

from orchestrator.application.ports import (
    DemoScripts,
    DemoSpeaker,
    DemoVoiceStore,
    Document,
    GpuGate,
    TtsUnavailable,
    VoiceProfiles,
)
from orchestrator.domain.demo import DemoScript
from orchestrator.domain.demo_voices import (
    DemoClip,
    DemoLine,
    DemoVoice,
    DemoVoiceManifest,
    check_voices,
    clip_file,
    demo_lines,
    page_view,
    parse_voice,
    playable,
    retime,
    stale_lines,
    withdrawn,
)
from orchestrator.domain.speech import wav_bytes


class DemoVoiceError(Exception):
    """The render cannot be done; nothing was changed."""


def utc_stamp() -> str:
    """The current time as ISO 8601, UTC, seconds."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class RenderResult:
    """What a render did: lines rendered, kept and clips removed."""

    demo: str
    rendered: List[str] = field(default_factory=list)
    kept: List[str] = field(default_factory=list)
    removed: List[str] = field(default_factory=list)
    voices: Dict[str, str] = field(default_factory=dict)
    lease: Optional[int] = None
    """MiB asked with the GPU lease (``None``: no lease taken)."""

    def as_document(self) -> Document:
        """The result as JSON data (what the command prints)."""
        return {"demo": self.demo, "rendered": self.rendered,
                "kept": len(self.kept), "removed": self.removed,
                "voices": self.voices, "gpu_lease_mib": self.lease}


class RenderDemoVoices:
    """Render a demo's clips with the voices the operator chose.

    Args:
        scripts: The demos (their files, as written).
        store: Where the clips and the manifest go.
        speaker: Says one line with one voice.
        profiles: The enrolled people (consent and reference clip).
        gate: The GPU lease (``None``: no GPU engine may be used).
        now: The time stamped on new clips.
    """

    def __init__(
        self,
        scripts: DemoScripts,
        store: DemoVoiceStore,
        speaker: DemoSpeaker,
        profiles: VoiceProfiles,
        gate: Optional[GpuGate] = None,
        now: Callable[[], str] = utc_stamp,
    ):
        """Wire the ports; nothing is rendered yet."""
        self.scripts = scripts
        self.store = store
        self.speaker = speaker
        self.profiles = profiles
        self.gate = gate
        self.now = now

    def voices(
        self, demo_id: str, chosen: Dict[str, str], clone_engine: str,
    ) -> Dict[str, DemoVoice]:
        """The voice of every person: ``chosen`` over the last render's.

        Raises:
            DemoVoiceError: A voice cannot be read.
        """
        manifest = self.store.manifest(demo_id)
        specs = dict(manifest.voices) if manifest is not None else {}
        specs.update(chosen)
        try:
            return {p: parse_voice(s, clone_engine)
                    for p, s in specs.items()}
        except ValueError as exc:
            raise DemoVoiceError(str(exc)) from exc

    def render(
        self, demo_id: str, chosen: Dict[str, str],
        clone_engine: str = "qwen3",
    ) -> RenderResult:
        """Render what changed; return what was done.

        ``chosen`` maps cast ids to voice specs; a person left out keeps
        the voice of the last render (an earlier explicit choice), and a
        person with neither is an error.

        Raises:
            DemoVoiceError: No such demo, a voice is not allowed, the
                GPU lease is refused or an engine fails; the clips kept
                before are untouched.
        """
        try:
            script = self.scripts.load(demo_id)
        except KeyError as exc:
            raise DemoVoiceError(f"no demo {demo_id!r}") from exc
        voices = self.voices(demo_id, chosen, clone_engine)
        problems = check_voices(script, voices, self.profiles.get)
        if problems:
            raise DemoVoiceError("; ".join(problems))
        lines = demo_lines(script)
        old = self.store.manifest(demo_id)
        stale = stale_lines(
            lines, voices, self.speaker.engine, old,
            lambda name: self.store.has_clip(demo_id, name),
        )
        result = RenderResult(demo=demo_id, voices={
            p: v.spec for p, v in sorted(voices.items())})
        result.kept = [line.key for line in lines if line not in stale]
        need = max([self.speaker.need_mib(voices[line.person])
                    for line in stale] or [0])
        taken = False
        try:
            if need:
                self._take(need)
                taken = True
                result.lease = need
            fresh = self._say(script, stale, voices)
        finally:
            if taken and self.gate is not None:
                self.gate.drop()
        result.rendered = [line.key for line in stale]
        result.removed = self._swap(script, lines, voices, old, fresh)
        return result

    def _take(self, need: int) -> None:
        """Hold the GPU lease for ``need`` MiB, or refuse the render."""
        if self.gate is None:
            raise DemoVoiceError("a voice needs the GPU and no lease is wired")
        # check() allows the GPU only under a live session's lease: a
        # session is running, and the render must not take its memory.
        if self.gate.check(0) is None:
            raise DemoVoiceError(
                "a live session holds the GPU: render when it has ended")
        refused = self.gate.take(need)
        if refused:
            self.gate.drop()
            raise DemoVoiceError(f"GPU lease refused: {refused}")

    def _say(
        self, script: DemoScript, lines: List[DemoLine],
        voices: Dict[str, DemoVoice],
    ) -> Dict[str, DemoClip]:
        """Write a new clip for every line; on failure, none is kept."""
        language = script.transcript.session.language or "es"
        written: List[str] = []
        fresh: Dict[str, DemoClip] = {}
        try:
            # One voice after another: a GPU engine loads once.
            for line in sorted(lines, key=lambda x: voices[x.person].spec):
                voice = voices[line.person]
                reference = None
                if voice.person is not None:
                    profile = self.profiles.get(voice.person)
                    if profile is None or not profile.clonable:
                        raise DemoVoiceError(
                            f"{voice.spec}: consent or clip gone")
                    reference = profile.reference
                try:
                    pcm, rate = self.speaker.speak(
                        line.text, voice, reference, language)
                except TtsUnavailable as exc:
                    raise DemoVoiceError(
                        f"{voice.spec} could not say {line.key}: {exc}"
                    ) from exc
                if not pcm:
                    raise DemoVoiceError(
                        f"{voice.spec} said nothing for {line.key}")
                engine = self.speaker.engine(voice)
                name = clip_file(line, voice, engine)
                self.store.write_clip(script.id, name, wav_bytes(pcm, rate))
                written.append(name)
                fresh[line.key] = DemoClip(
                    key=line.key, file=name, person=line.person,
                    voice=voice.spec, kind=voice.kind,
                    cloned_from=voice.person, engine=engine,
                    text_sha256=line.text_sha256,
                    seconds=round(len(pcm) / 2 / rate, 3),
                    sample_rate=rate, rendered_at=self.now(),
                )
        except BaseException:
            for name in written:
                self.store.delete_clip(script.id, name)
            raise
        return fresh

    def _swap(
        self, script: DemoScript, lines: List[DemoLine],
        voices: Dict[str, DemoVoice], old: Optional[DemoVoiceManifest],
        fresh: Dict[str, DemoClip],
    ) -> List[str]:
        """Write the new manifest, then delete the clips it dropped."""
        keys = {line.key for line in lines}
        before = old.clips if old is not None else {}
        specs = {p: v.spec for p, v in sorted(voices.items())}
        if (not fresh and old is not None and old.voices == specs
                and set(before) == keys):
            return []  # nothing changed: the manifest stays as it is
        clips = {k: c for k, c in before.items()
                 if k in keys and k not in fresh}
        clips.update(fresh)
        self.store.save_manifest(DemoVoiceManifest(
            demo=script.id, updated_at=self.now(),
            voices=specs, clips=dict(sorted(clips.items())),
        ))
        kept = {c.file for c in clips.values()}
        removed = sorted({c.file for c in before.values()} - kept)
        for name in removed:
            self.store.delete_clip(script.id, name)
        return removed


class DemoVoiceLibrary:
    """The rendered clips as the live page and the sessions see them.

    Args:
        scripts: The demos (their files).
        store: The clips and manifests.
        profiles: The enrolled people (is a clone still allowed?).
    """

    def __init__(
        self, scripts: DemoScripts, store: DemoVoiceStore,
        profiles: VoiceProfiles,
    ):
        """Wire the ports."""
        self.scripts = scripts
        self.store = store
        self.profiles = profiles

    def _manifest(self, script: DemoScript) -> Optional[DemoVoiceManifest]:
        """The manifest without the clips of withdrawn people.

        Those clips are deleted at once (files and entries).
        """
        manifest = self.store.manifest(script.id)
        if manifest is None:
            return None
        gone = withdrawn(manifest, self.profiles.get)
        if not gone:
            return manifest
        for clip in manifest.clips.values():
            if clip.cloned_from in gone:
                self.store.delete_clip(script.id, clip.file)
        kept = {k: c for k, c in manifest.clips.items()
                if c.cloned_from not in gone}
        manifest = manifest.model_copy(update={"clips": kept})
        self.store.save_manifest(manifest)
        return manifest

    def _clips(self, script: DemoScript) -> Dict[str, Dict[str, DemoClip]]:
        """The clips that may be played now, per example."""
        manifest = self._manifest(script)
        found = playable(
            demo_lines(script), manifest,
            lambda name: self.store.has_clip(script.id, name),
        )
        return {str(k): v for k, v in found.items()}

    def view(self, demo_id: str) -> Document:
        """What the page needs to play a demo's voices.

        Raises:
            KeyError: No such demo.
        """
        script = self.scripts.load(demo_id)
        return page_view(script, self._clips(script))

    def clip(self, demo_id: str, key: str) -> bytes:
        """One playable clip's WAV.

        Raises:
            KeyError: No such demo, line or playable clip.
        """
        script = self.scripts.load(demo_id)
        for clips in self._clips(script).values():
            if key in clips:
                return self.store.read_clip(demo_id, clips[key].file)
        raise KeyError(key)

    # ---- DemoScripts: the sessions replay the call at the clips' pace

    def list(self) -> List[Document]:
        """Every readable demo, as a page lists it (timed by its clips)."""
        out: List[Document] = []
        for listed in self.scripts.list():
            try:
                out.append(self.load(str(listed["id"])).summary())
            except KeyError:
                continue
        return out

    def load(self, demo_id: str) -> DemoScript:
        """One demo; a call that can be heard takes its clips' lengths.

        Raises:
            KeyError: No such demo.
        """
        script = self.scripts.load(demo_id)
        call = self._clips(script).get("call")
        if not call:
            return script
        seconds = {k: c.seconds for k, c in call.items()}
        return script.model_copy(update={
            "transcript": retime(script.transcript, seconds)})
