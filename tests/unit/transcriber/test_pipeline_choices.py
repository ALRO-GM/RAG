"""What the transcriber takes from the pipeline settings (``settings.v1``):
the allow-listed mapping, the body check, and the jobs that read the
choices when they start (diarizer, name finder)."""

from typing import Any, Dict, List, Optional

import pytest

from tests.unit.transcriber.test_diarizer_choice import run, setup
from transcriber.application.pipeline_settings import (
    PipelineSettingsAccess,
    checked_body,
)
from transcriber.application.transcribe import TranscriptionOptions
from transcriber.domain.diarization import (
    BUILTIN_RULE,
    DEFAULT_DIARIZER,
    TWO_SPEAKER_DIARIZER,
    DiarizerRule,
)
from transcriber.domain.errors import (
    AnalysisUnavailableError,
    InvalidInputError,
)
from transcriber.domain.pipeline_choices import (
    diarizer_rule,
    live_cpu_only,
    name_models,
    recogniser,
)

CONFIGURED = DiarizerRule()


def test_the_first_option_or_nothing_keeps_the_configuration() -> None:
    for choices in (None, {}, {"transcription.diarizer": "auto"},
                    {"transcription.diarizer": "--evil"}):
        assert diarizer_rule(choices, CONFIGURED) is CONFIGURED
    assert recogniser(None, "my/model") == "my/model"
    assert recogniser({"transcription.recogniser": "whisper-large-v3-turbo"},
                      "my/model") == "my/model"
    assert name_models({"transcription.names": "qwen3-8b"}, ("A", "B")) == (
        "A", "B"
    )
    assert live_cpu_only(None) is False


def test_each_option_maps_to_its_model() -> None:
    assert recogniser({"transcription.recogniser": "whisper-large-v3"},
                      "x") == "openai/whisper-large-v3"
    rule = diarizer_rule({"transcription.diarizer": "nemotron"}, CONFIGURED)
    assert rule.model_for(4) == rule.model_for(2) == TWO_SPEAKER_DIARIZER
    builtin = diarizer_rule({"transcription.diarizer": "builtin"},
                            CONFIGURED)
    assert builtin.model_for(2) == BUILTIN_RULE.model
    pyannote = diarizer_rule({"transcription.diarizer": "pyannote"},
                             CONFIGURED)
    assert pyannote.model == "pyannote-community-1"
    assert name_models({"transcription.names": "qwen3-4b"}, ("A",)) == (
        "Qwen/Qwen3-4B",
    )
    assert name_models({"transcription.names": "rules"}, ("A",)) == ()
    assert live_cpu_only({"live.asr": "cpu"}) is True


def test_a_job_uses_the_diarizer_saved_when_it_runs() -> None:
    saved: Dict[str, str] = {}
    s = setup()
    s.transcribe._choices = lambda: saved
    run(s, TranscriptionOptions(num_speakers=2))
    assert s.diarizer.calls[-1] == (TWO_SPEAKER_DIARIZER, 2)
    saved["transcription.diarizer"] = "diarizen"
    s2 = setup()
    s2.transcribe._choices = lambda: saved
    run(s2, TranscriptionOptions(num_speakers=2))
    assert s2.diarizer.calls == [(DEFAULT_DIARIZER, 2)]
    saved["transcription.diarizer"] = "builtin"
    s3 = setup()
    s3.transcribe._choices = lambda: saved
    run(s3)
    assert s3.diarizer.calls == [], "the built-in grouping runs no model"


def test_the_body_is_checked_before_it_is_forwarded() -> None:
    assert checked_body({"preset": "cost"}) == {"preset": "cost"}
    assert checked_body({"choices": {"live.asr": "cpu"}, "revision": 2}) == {
        "choices": {"live.asr": "cpu"}, "revision": 2,
    }
    voices = {"voicebot.voice": "pocket:lola",
              "voicebot.clone_engine": "qwen3",
              "voicebot.pocket_device": "cuda"}
    assert checked_body({"choices": voices}) == {"choices": voices}
    for body in ({"preset": "Cost now"}, {"preset": 1}, {"x": 1},
                 {"choices": {"live.asr": "c p u"}},
                 {"choices": {"live asr": "cpu"}},
                 {"choices": {"live.asr": 1}}, {"choices": "x"},
                 {"choices": {"voicebot.voice": "pocket:lola voz"}},
                 {"choices": {"voicebot.voice": "pocket:../lola"}},
                 {"choices": {f"s{i}": "a" for i in range(65)}},
                 {"revision": -1}, {"revision": True}, {"revision": "1"}):
        with pytest.raises(InvalidInputError):
            checked_body(body)


class FakePort:
    """The RAG's settings, in memory."""

    def __init__(self) -> None:
        """Nothing saved."""
        self.sent: List[Dict[str, Any]] = []
        self.saved: Optional[Dict[str, str]] = None

    def show(self) -> Dict[str, Any]:
        """Return what is saved."""
        return {"stored": self.saved is not None,
                "choices": self.saved or {}}

    def change(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Keep the choices sent."""
        self.sent.append(body)
        self.saved = dict(body.get("choices", {}))
        return self.show()

    def current(self) -> Optional[Dict[str, str]]:
        """Return the saved choices."""
        return self.saved


def test_the_use_case_forwards_and_reads_the_saved_choices() -> None:
    port = FakePort()
    access = PipelineSettingsAccess(port)
    assert access.current() is None
    assert access.change({"choices": {"live.asr": "cpu"}})["stored"]
    assert access.current() == {"live.asr": "cpu"}
    with pytest.raises(InvalidInputError):
        access.change({"choices": {"live.asr": "c;pu"}})
    assert len(port.sent) == 1
    off = PipelineSettingsAccess(None)
    assert off.current() is None
    with pytest.raises(AnalysisUnavailableError):
        off.show()
    with pytest.raises(AnalysisUnavailableError):
        off.change({})


def test_the_name_finder_asks_for_its_model_at_every_search() -> None:
    from tests.unit.transcriber.test_speaker_names import Kit

    kit = Kit()
    chosen: Dict[str, Any] = {"model": None}
    kit.find._choose_model = lambda: chosen["model"]
    rules_only = kit.find("s1")
    assert rules_only.auto == [] and rules_only.model is None
    assert kit.model is not None and kit.model.prompts == []
    chosen["model"] = kit.model
    found = kit.find("s1")
    assert found.model == "fake-qwen" and kit.model.prompts
