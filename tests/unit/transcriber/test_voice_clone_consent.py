"""Consent to voice cloning at enrolment (#207): separate, revocable, and
without it no clip of the voice is kept; the speaking rate is measured."""

import numpy as np
import pytest

from tests.unit.transcriber.fakes import (
    NOW,
    FakeVoiceprintRepository,
    FixedClock,
    reading,
)
from tests.unit.transcriber.test_enrol_speaker import SAMPLE, enrol_use_case
from transcriber.application.browse import ListSpeakers
from transcriber.application.enrol import WithdrawVoiceClone
from transcriber.domain.enrolment import clone_span, speaking_rate
from transcriber.domain.errors import InvalidInputError, NotFoundError
from transcriber.domain.models import ConsentRecord


def test_the_rate_and_the_clip_s_span_come_from_the_speech() -> None:
    assert speaking_rate(30, [(0.0, 5.0), (6.0, 11.0)]) == 3.0
    assert speaking_rate(0, [(0.0, 5.0)]) is None
    assert speaking_rate(4, []) is None
    assert clone_span([(1.0, 10.0), (12.0, 50.0)], 60.0) == (1.0, 31.0)
    assert clone_span([(2.0, 8.0)], 60.0) == (2.0, 8.0)
    assert clone_span([(70.0, 80.0)], 60.0) is None, "outside the audio"
    assert clone_span([], 60.0) is None


def test_without_the_box_ticked_no_clip_is_kept() -> None:
    voiceprints = FakeVoiceprintRepository()
    enrol = enrol_use_case(reading(), voiceprints)
    voiceprint = enrol("ana", "Ana", "es", SAMPLE, True, "consent-1")
    consent = voiceprints.consent("ana")
    assert consent is not None and consent.voice_clone is False
    assert voiceprints.clips == {}
    assert voiceprint.words_per_second is not None
    assert voiceprint.words_per_second > 0
    for not_true in ("true", 1, "yes"):
        enrol("ana", "Ana", "es", SAMPLE, True, "consent-1",
              clone_consent=not_true)  # type: ignore[arg-type]
        assert voiceprints.clips == {}, not_true


def test_with_the_box_ticked_a_clip_is_kept_until_withdrawn() -> None:
    voiceprints = FakeVoiceprintRepository()
    enrol = enrol_use_case(reading(), voiceprints)
    enrol("ana", "Ana", "es", SAMPLE, True, "consent-1", clone_consent=True)
    consent = voiceprints.consent("ana")
    assert consent is not None and consent.voice_clone is True
    samples, rate = voiceprints.clips["ana"]
    assert rate > 0 and 0 < len(samples) <= 30 * rate
    listed = ListSpeakers(voiceprints)()
    assert listed[0].voice_clone is True
    assert listed[0].words_per_second == voiceprints.voiceprints[
        "ana"].words_per_second

    WithdrawVoiceClone(voiceprints, FixedClock())("ana")
    consent = voiceprints.consent("ana")
    assert consent is not None and consent.voice_clone is False
    assert consent.voice_clone_withdrawn_at == NOW
    assert "ana" not in voiceprints.clips
    assert voiceprints.get("ana") is not None, "the voiceprint stays"
    assert ListSpeakers(voiceprints)()[0].voice_clone is False


def test_enrolling_again_without_the_box_deletes_the_old_clip() -> None:
    voiceprints = FakeVoiceprintRepository()
    enrol = enrol_use_case(reading(), voiceprints)
    enrol("ana", "Ana", "es", SAMPLE, True, "consent-1", clone_consent=True)
    enrol("ana", "Ana", "es", SAMPLE, True, "consent-1")
    assert voiceprints.clips == {}


def test_withdrawing_needs_an_enrolled_person_and_a_valid_id() -> None:
    withdraw = WithdrawVoiceClone(FakeVoiceprintRepository(), FixedClock())
    with pytest.raises(NotFoundError):
        withdraw("nobody")
    with pytest.raises(InvalidInputError):
        withdraw("../x")
    voiceprints = FakeVoiceprintRepository()
    voiceprints.enrol("bea", np.array([1.0, 0.0, 0.0, 0.0]))
    voiceprints.consents["bea"] = ConsentRecord(
        speaker_id="bea", consent_version="c", text_version="t",
        given_at=NOW)
    WithdrawVoiceClone(voiceprints, FixedClock())("bea")
    consent = voiceprints.consent("bea")
    assert consent is not None and consent.voice_clone_withdrawn_at is None
