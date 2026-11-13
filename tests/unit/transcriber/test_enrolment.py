"""Transcriber enrolment: reading texts and checks on a voice sample."""

from datetime import datetime, timezone

import numpy as np
import pytest

from transcriber.domain.enrolment import (
    ENROLMENT_TEXTS,
    NAME_PLACEHOLDER,
    Rejection,
    SampleRules,
    build_voiceprint,
    enrolment_text,
    enrolment_vector,
)
from transcriber.domain.errors import InvalidInputError, SampleRejectedError
from transcriber.domain.voices import FloatArray

WHEN = datetime(2026, 11, 13, 10, 0, tzinfo=timezone.utc)


class TestEnrolmentTexts:
    @pytest.mark.parametrize("language", ["es", "en"])
    def test_text_starts_with_the_name(self, language: str) -> None:
        text = enrolment_text(language)
        assert text.template.split(NAME_PLACEHOLDER)[0].count(" ") <= 3
        read = text.text("  Ana   María ")
        assert "Ana María" in read
        assert NAME_PLACEHOLDER not in read
        assert read.index("Ana") < 15

    def test_texts_are_versioned_and_long_enough(self) -> None:
        keys = [(t.language, t.version) for t in ENROLMENT_TEXTS]
        assert len(set(keys)) == len(keys)
        for text in ENROLMENT_TEXTS:
            assert text.template.count(NAME_PLACEHOLDER) == 1
            # Calm reading is roughly 2 to 3 words a second.
            assert 70 <= len(text.template.split()) <= 100

    def test_language_variants_and_versions(self) -> None:
        assert enrolment_text("es-ES").language == "es"
        assert enrolment_text("EN_gb").language == "en"
        assert enrolment_text("en", version="en-1").version == "en-1"
        with pytest.raises(InvalidInputError, match="fr"):
            enrolment_text("fr")
        with pytest.raises(InvalidInputError, match="es-9"):
            enrolment_text("es", version="es-9")

    def test_the_name_is_required(self) -> None:
        with pytest.raises(InvalidInputError):
            enrolment_text("es").text("   ")


def one_voice(count: int = 20) -> FloatArray:
    """Window embeddings of one voice with a little noise, not unit."""
    rng = np.random.default_rng(0)
    base = np.array([3.0, 1.0, 0.0, 0.0], dtype=np.float32)
    out: FloatArray = (
        base + rng.normal(0.0, 0.05, size=(count, 4))
    ).astype(np.float32)
    return out


def rejection(error: pytest.ExceptionInfo[SampleRejectedError]) -> str:
    """The reason of a rejected sample."""
    return error.value.reason


class TestEnrolmentVector:
    def test_one_voice_gives_a_unit_mean(self) -> None:
        vector = enrolment_vector(one_voice(), [(0.0, 28.0)], 30.0)
        assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-6)
        expected = np.array([3.0, 1.0, 0.0, 0.0]) / np.sqrt(10.0)
        assert float(np.dot(vector, expected)) > 0.999

    def test_mean_of_unit_vectors_not_of_raw_ones(self) -> None:
        """A loud window must not outweigh a quiet one."""
        vectors = np.array([[10.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        rules = SampleRules(same_voice_threshold=-1.0)
        vector = enrolment_vector(vectors, [(0.0, 30.0)], 30.0, rules)
        assert vector == pytest.approx([0.7071, 0.7071], abs=1e-4)

    def test_too_short(self) -> None:
        with pytest.raises(SampleRejectedError) as error:
            enrolment_vector(one_voice(), [(0.0, 9.0)], 10.0)
        assert rejection(error) == Rejection.TOO_SHORT.value
        assert isinstance(error.value, InvalidInputError)

    def test_mostly_silent(self) -> None:
        with pytest.raises(SampleRejectedError) as error:
            enrolment_vector(one_voice(), [(0.0, 5.0), (20.0, 25.0)], 30.0)
        assert rejection(error) == Rejection.MOSTLY_SILENT.value

    def test_no_windows_is_silent(self) -> None:
        with pytest.raises(SampleRejectedError) as error:
            enrolment_vector(
                np.zeros((0, 4), dtype=np.float32), [(0.0, 30.0)], 30.0
            )
        assert rejection(error) == Rejection.MOSTLY_SILENT.value

    def test_a_zero_window_carries_no_voice(self) -> None:
        with pytest.raises(SampleRejectedError) as error:
            enrolment_vector(
                np.zeros((1, 4), dtype=np.float32), [(0.0, 30.0)], 30.0
            )
        assert rejection(error) == Rejection.MOSTLY_SILENT.value

    def test_two_voices_are_rejected(self) -> None:
        other = one_voice(8)[:, [2, 3, 0, 1]]
        vectors = np.vstack([one_voice(12), other]).astype(np.float32)
        with pytest.raises(SampleRejectedError) as error:
            enrolment_vector(vectors, [(0.0, 30.0)], 30.0)
        assert rejection(error) == Rejection.SEVERAL_VOICES.value

    def test_a_few_stray_windows_are_tolerated_and_left_out(self) -> None:
        stray = np.array([[0.0, 0.0, 1.0, 0.0]], dtype=np.float32)
        vectors = np.vstack([one_voice(19), stray]).astype(np.float32)
        vector = enrolment_vector(vectors, [(0.0, 30.0)], 30.0)
        assert abs(vector[2]) < 0.05
        strict = SampleRules(max_other_voice_share=0.0)
        with pytest.raises(SampleRejectedError):
            enrolment_vector(vectors, [(0.0, 30.0)], 30.0, strict)

    def test_rules_are_parameters(self) -> None:
        rules = SampleRules(min_duration=5.0, min_speech_ratio=0.1)
        vector = enrolment_vector(one_voice(), [(0.0, 1.0)], 6.0, rules)
        assert len(vector) == 4


class TestBuildVoiceprint:
    def test_voiceprint_keeps_only_vector_and_consent(self) -> None:
        voiceprint = build_voiceprint(
            "ana", "wavlm", "consent-1", WHEN, one_voice(),
            [(0.0, 29.0)], 30.0,
        )
        assert voiceprint.speaker_id == "ana"
        assert voiceprint.model == "wavlm"
        assert voiceprint.consent_version == "consent-1"
        assert voiceprint.enrolled_at == WHEN
        assert len(voiceprint.vector) == 4

    def test_rejection_propagates(self) -> None:
        with pytest.raises(SampleRejectedError):
            build_voiceprint("ana", "wavlm", "c", WHEN, one_voice(),
                             [(0.0, 1.0)], 3.0)
