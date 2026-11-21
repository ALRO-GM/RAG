"""Pipeline settings (``settings.v1``): the allow-lists, the presets and
the use case that reads, stores and hands them to the jobs."""

from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple

import pytest

from src.application.knowledge.chosen_extractor import ChosenExtractor
from src.application.knowledge.ports import ExtractedDocument
from src.application.pipeline_settings import PipelineSettings
from src.domain.errors import BusyError, CorruptedDataError, InvalidInputError
from src.domain.pipeline_settings import (
    PRESETS,
    STEPS,
    RunChoices,
    SettingsDocument,
    catalogue,
    check_choices,
    defaults,
    preset_of,
    resolve,
    run_choices,
    step_models,
)

NOW = datetime(2026, 11, 21, 10, 0, tzinfo=timezone.utc)


class MemoryStore:
    """The settings store, in memory."""

    def __init__(self, document: Optional[SettingsDocument] = None) -> None:
        """Start with ``document`` (none: nothing saved)."""
        self.document = document
        self.broken = False

    def load(self) -> Optional[SettingsDocument]:
        """Return the document (or fail when broken)."""
        if self.broken:
            raise CorruptedDataError("unreadable")
        return self.document

    def update(
        self,
        change: Callable[[Optional[SettingsDocument]], SettingsDocument],
    ) -> SettingsDocument:
        """Apply ``change``."""
        self.document = change(self.document)
        return self.document


class FakeJudge:
    """The judge record of ``/analysis/judge``."""

    def __init__(self, mode: str = "claude_code") -> None:
        """Start on ``mode``."""
        self.current = mode
        self.chosen: List[str] = []

    def mode(self) -> str:
        """Return the judge."""
        return self.current

    def choose(self, mode: str) -> None:
        """Store the judge."""
        self.current = mode
        self.chosen.append(mode)


def service(
    store: Optional[MemoryStore] = None, judge: Optional[FakeJudge] = None
) -> PipelineSettings:
    """The use case over in-memory fakes."""
    return PipelineSettings(store or MemoryStore(), lambda: NOW, judge)


def test_the_first_option_of_every_step_is_today_s_configuration() -> None:
    assert defaults() == PRESETS["quality"]
    assert preset_of({}) == "quality"
    assert all(len(set(s.options)) == len(s.options) >= 2 for s in STEPS
               if s.range is None and s.pattern is None)
    assert [s.id for s in STEPS if s.applies == "restart"] == [
        "transcription.recogniser"
    ]


def test_the_presets_differ_where_documented() -> None:
    quality, balanced, cost = (PRESETS[n] for n in ("quality", "balanced",
                                                    "cost"))
    assert {k for k in quality if quality[k] != balanced[k]} == {
        "analysis.confirmation_model"
    }
    assert cost["analysis.judge"] == "local"
    assert cost["analysis.local_judge"] == "qwen3-4b"
    assert cost["transcription.diarizer"] == "nemotron"
    assert preset_of(cost) == "cost" and preset_of(balanced) == "balanced"
    assert preset_of({**cost, "knowledge.ocr": "off"}) == "custom"


def test_choices_outside_the_allow_lists_are_refused() -> None:
    assert check_choices({"knowledge.ocr": "off"}) == {"knowledge.ocr": "off"}
    with pytest.raises(InvalidInputError, match="Unknown setting"):
        check_choices({"rm -rf": "on"})
    with pytest.raises(InvalidInputError, match="Invalid option"):
        check_choices({"analysis.judge": "--dangerous"})
    with pytest.raises(InvalidInputError, match="Missing"):
        check_choices({"knowledge.ocr": "on"}, partial=False)
    with pytest.raises(ValueError):
        SettingsDocument(choices={"live.asr": "tpu"})
    with pytest.raises(ValueError):
        SettingsDocument(preset="cheap")


def test_voice_controls_accept_numbers_in_range_and_person_ids() -> None:
    checked = check_choices({
        "voicebot.length_scale": "1.10", "voicebot.noise_scale": "0",
        "voicebot.noise_w": "0.333", "voicebot.sentence_silence": "0.25",
        "voicebot.voice": "clone:alberto", "voicebot.match_rate": "ana-2",
    })
    assert checked == {
        "voicebot.length_scale": "1.1", "voicebot.noise_scale": "0",
        "voicebot.noise_w": "0.333", "voicebot.sentence_silence": "0.25",
        "voicebot.voice": "clone:alberto", "voicebot.match_rate": "ana-2",
    }
    for step, bad in (("voicebot.length_scale", "0.5"),
                      ("voicebot.length_scale", "1.6"),
                      ("voicebot.noise_scale", "-0.1"),
                      ("voicebot.noise_w", "0.1234"),
                      ("voicebot.sentence_silence", "1e0"),
                      ("voicebot.voice", "clone:../x"),
                      ("voicebot.voice", "clone:"),
                      ("voicebot.match_rate", "--x")):
        with pytest.raises(InvalidInputError, match="Invalid option"):
            check_choices({step: bad})
    assert resolve({"voicebot.length_scale": "9"})[
        "voicebot.length_scale"] == "1"
    assert defaults()["voicebot.match_rate"] == "off"
    steps = {s["id"]: s for s in catalogue()["steps"]}  # type: ignore
    assert steps["voicebot.noise_w"]["range"] == {
        "low": 0.0, "high": 1.0, "step": 0.05}
    assert steps["voicebot.voice"]["free"] == "speaker"


def test_a_reader_ignores_unknown_steps_and_options() -> None:
    chosen = resolve({"knowledge.ocr": "off", "x": "y", "live.asr": "tpu"})
    assert chosen["knowledge.ocr"] == "off" and chosen["live.asr"] == "gpu"
    assert "x" not in chosen


def test_the_run_choices_and_models_of_a_preset() -> None:
    assert run_choices(PRESETS["quality"]) == RunChoices()
    cost = run_choices(PRESETS["cost"])
    assert cost.local_preference == ("Qwen/Qwen3-4B",)
    assert run_choices({"analysis.act_tagger": "rules"}).act_tagger == "rules"
    assert step_models(PRESETS["balanced"]) == {
        "confirmation": "haiku", "assessment": "opus", "synthesis": "opus",
    }


def test_the_catalogue_lists_steps_and_presets() -> None:
    listed = catalogue()
    steps = listed["steps"]
    assert isinstance(steps, list) and len(steps) == len(STEPS)
    assert listed["presets"] == {n: dict(c) for n, c in PRESETS.items()}


def test_nothing_saved_reads_as_none_and_shows_quality() -> None:
    settings = service()
    assert settings.current() is None
    shown = settings.show()
    assert shown["stored"] is False and shown["preset"] == "quality"
    assert shown["revision"] == 0 and shown["updated_at"] is None


def test_a_preset_is_stored_with_a_new_revision_and_the_judge() -> None:
    store, judge = MemoryStore(), FakeJudge()
    settings = service(store, judge)
    shown = settings.change({"preset": "cost"})
    assert shown["stored"] and shown["revision"] == 1
    assert shown["preset"] == "cost" and judge.chosen == ["local"]
    assert shown["updated_at"] == NOW.isoformat()
    assert settings.current() == PRESETS["cost"]


def test_choices_change_over_the_current_ones_and_become_custom() -> None:
    settings = service(MemoryStore(), FakeJudge())
    settings.change({"preset": "balanced"})
    shown = settings.change({"choices": {"knowledge.captions": "claude"},
                             "revision": 1})
    assert shown["preset"] == "custom" and shown["revision"] == 2
    assert shown["choices"]["analysis.confirmation_model"] == "haiku"
    assert shown["choices"]["knowledge.captions"] == "claude"


def test_a_stale_revision_and_bad_bodies_are_refused() -> None:
    settings = service(MemoryStore(), FakeJudge())
    settings.change({"preset": "quality"})
    with pytest.raises(BusyError, match="revision 1"):
        settings.change({"choices": {"live.asr": "cpu"}, "revision": 0})
    for body in ({"preset": "cheap"}, {"choices": ["x"]},
                 {"revision": "1"}, {"revision": True},
                 {"choices": {"live.asr": "tpu"}}):
        with pytest.raises(InvalidInputError):
            settings.change(body)
    assert settings.show()["revision"] == 1


def test_the_judge_record_wins_and_its_failure_is_not_fatal() -> None:
    store, judge = MemoryStore(), FakeJudge()
    settings = service(store, judge)
    settings.change({"preset": "quality"})
    judge.current = "both"
    assert settings.show()["choices"]["analysis.judge"] == "both"

    class Broken(FakeJudge):
        def mode(self) -> str:
            raise CorruptedDataError("no file")

    shown = service(store, Broken()).show()
    assert shown["choices"]["analysis.judge"] == "claude_code"
    store.broken = True
    assert settings.current() is None


class Reader:
    """A document reader that says which combination it is."""

    def __init__(self, ocr: bool, captions: str) -> None:
        """Keep the combination."""
        self.key = (ocr, captions)

    def verify(self, data: bytes, document_type: str) -> None:
        """Accept everything."""

    def extract(self, data: bytes, document_type: str) -> ExtractedDocument:
        """Return the combination as text."""
        return ExtractedDocument(text=str(self.key))


def test_uploads_follow_the_ocr_and_captions_saved_now() -> None:
    saved: Dict[str, str] = {}
    built: List[Tuple[bool, str]] = []

    def build(ocr: bool, captions: str) -> Reader:
        built.append((ocr, captions))
        return Reader(ocr, captions)

    reader = ChosenExtractor(lambda: saved or None, build, (True, "florence"))
    assert reader.extract(b"x", "pdf").text == "(True, 'florence')"
    saved.update({"knowledge.ocr": "off", "knowledge.captions": "claude"})
    reader.verify(b"x", "pdf")
    assert reader.extract(b"x", "pdf").text == "(False, 'claude')"
    saved["knowledge.captions"] = "weird"
    assert reader.choice() == (False, "florence")
    reader.extract(b"x", "pdf")
    reader.extract(b"x", "pdf")
    assert built == [(True, "florence"), (False, "claude"),
                     (False, "florence")], "each combination built once"
