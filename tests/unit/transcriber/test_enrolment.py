"""Transcriber enrolment: reading texts and checks on a voice sample."""

from datetime import datetime, timezone
from typing import List, Tuple

import numpy as np
import pytest

from transcriber.domain.enrolment import (
    ENROLMENT_TEXTS,
    NAME_PLACEHOLDER,
    Rejection,
    SampleRules,
    build_voiceprint,
    clip_spans,
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


def spread(count: int, duration: float = 30.0) -> List[Tuple[float, float]]:
    """``count`` back-to-back windows covering ``[0, duration]``."""
    size = duration / count
    return [(i * size, (i + 1) * size) for i in range(count)]


def sample(vectors: FloatArray, duration: float = 30.0) -> List[float]:
    """The voiceprint of ``vectors`` spread over a fully spoken sample."""
    return enrolment_vector(
        vectors, spread(int(vectors.shape[0]), duration),
        [(0.0, duration)], duration,
    )


def two_real_voices(per_voice: int = 40) -> FloatArray:
    """Windows of two voices whose cosine is about 0.36 (512 dims)."""
    rng = np.random.default_rng(1)
    a = np.zeros(512)
    a[0] = 1.0
    b = np.zeros(512)
    b[0], b[1] = 0.36, float(np.sqrt(1 - 0.36 ** 2))
    noise = rng.normal(0.0, 0.01, size=(2 * per_voice, 512))
    out: FloatArray = (
        np.vstack([np.tile(a, (per_voice, 1)), np.tile(b, (per_voice, 1))])
        + noise
    ).astype(np.float32)
    return out


def rejection(error: pytest.ExceptionInfo[SampleRejectedError]) -> str:
    """The reason of a rejected sample."""
    return error.value.reason


class TestEnrolmentVector:
    def test_one_voice_gives_a_unit_mean(self) -> None:
        vector = sample(one_voice())
        assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-6)
        expected = np.array([3.0, 1.0, 0.0, 0.0]) / np.sqrt(10.0)
        assert float(np.dot(vector, expected)) > 0.999

    def test_mean_of_unit_vectors_not_of_raw_ones(self) -> None:
        """A loud window must not outweigh a quiet one."""
        vectors = np.array([[10.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        rules = SampleRules(same_voice_threshold=-1.0, min_windows=2)
        vector = enrolment_vector(
            vectors, spread(2), [(0.0, 30.0)], 30.0, rules
        )
        assert vector == pytest.approx([0.7071, 0.7071], abs=1e-4)

    def test_too_short(self) -> None:
        with pytest.raises(SampleRejectedError) as error:
            enrolment_vector(one_voice(), spread(20, 10.0), [(0.0, 9.0)],
                             10.0)
        assert rejection(error) == Rejection.TOO_SHORT.value
        assert isinstance(error.value, InvalidInputError)

    def test_mostly_silent(self) -> None:
        with pytest.raises(SampleRejectedError) as error:
            enrolment_vector(one_voice(), spread(20),
                             [(0.0, 5.0), (20.0, 25.0)], 30.0)
        assert rejection(error) == Rejection.MOSTLY_SILENT.value

    def test_no_windows_is_silent(self) -> None:
        with pytest.raises(SampleRejectedError) as error:
            enrolment_vector(
                np.zeros((0, 4), dtype=np.float32), [], [(0.0, 30.0)], 30.0
            )
        assert rejection(error) == Rejection.MOSTLY_SILENT.value

    def test_zero_windows_carry_no_voice(self) -> None:
        with pytest.raises(SampleRejectedError) as error:
            sample(np.zeros((20, 4), dtype=np.float32))
        assert rejection(error) == Rejection.MOSTLY_SILENT.value

    def test_two_voices_are_rejected(self) -> None:
        other = one_voice(8)[:, [2, 3, 0, 1]]
        vectors = np.vstack([one_voice(12), other]).astype(np.float32)
        with pytest.raises(SampleRejectedError) as error:
            sample(vectors)
        assert rejection(error) == Rejection.SEVERAL_VOICES.value

    def test_two_voices_alone_are_rejected(self) -> None:
        with pytest.raises(SampleRejectedError) as error:
            sample(two_real_voices())
        assert rejection(error) == Rejection.SEVERAL_VOICES.value

    @pytest.mark.parametrize("stray", ["random", "zero"])
    def test_stray_windows_do_not_merge_two_voices(self, stray: str) -> None:
        """More clusters than a cap must not merge the two real voices.

        Nine stray windows plus two voices are eleven clusters; a cap of
        ten merged the closest pair, the two voices, into one "main"
        voice holding 90% of the windows, and the blend was accepted.
        """
        rng = np.random.default_rng(2)
        extra = (rng.normal(0.0, 1.0, size=(9, 512)) if stray == "random"
                 else np.zeros((9, 512)))
        vectors = np.vstack([two_real_voices(), extra]).astype(np.float32)
        with pytest.raises(SampleRejectedError) as error:
            sample(vectors)
        assert rejection(error) == Rejection.SEVERAL_VOICES.value

    def test_many_zero_windows_do_not_hide_a_second_voice(self) -> None:
        voices = two_real_voices(10)
        vectors = np.vstack([voices, np.zeros((10, 512))]).astype(np.float32)
        with pytest.raises(SampleRejectedError) as error:
            sample(vectors)
        assert rejection(error) == Rejection.SEVERAL_VOICES.value

    def test_zero_windows_are_left_out_not_counted_as_a_voice(self) -> None:
        """Silence is neither the main voice nor another voice."""
        vectors = np.vstack(
            [one_voice(20), np.zeros((10, 4))]
        ).astype(np.float32)
        vector = sample(vectors)
        expected = np.array([3.0, 1.0, 0.0, 0.0]) / np.sqrt(10.0)
        assert float(np.dot(vector, expected)) > 0.999

    def test_a_few_stray_windows_are_tolerated_and_left_out(self) -> None:
        stray = np.array([[0.0, 0.0, 1.0, 0.0]], dtype=np.float32)
        vectors = np.vstack([one_voice(19), stray]).astype(np.float32)
        vector = sample(vectors)
        assert abs(vector[2]) < 0.05
        strict = SampleRules(max_other_voice_share=0.0)
        with pytest.raises(SampleRejectedError):
            enrolment_vector(vectors, spread(20), [(0.0, 30.0)], 30.0,
                             strict)

    def test_defaults_reject_a_close_second_voice_on_a_fifth(self) -> None:
        """The case TR-05 measured: two readers, raw cosine near 0.7.

        Raw WavLM embeddings of different readers have a median cosine
        of about 0.69, so the old clustering threshold of 0.5 merged
        them; and a second reader on 20% of the sample stayed under the
        old 20% share. The measured defaults reject both.
        """
        rules = SampleRules()
        assert (rules.same_voice_threshold,
                rules.max_other_voice_share) == (0.85, 0.1)
        a = np.zeros(512)
        a[0] = 1.0
        b = np.zeros(512)
        b[0], b[1] = 0.7, float(np.sqrt(1 - 0.7 ** 2))
        noise = np.random.default_rng(2).normal(0.0, 0.01, (20, 512))
        vectors = (np.vstack([np.tile(a, (16, 1)), np.tile(b, (4, 1))])
                   + noise).astype(np.float32)
        with pytest.raises(SampleRejectedError) as error:
            sample(vectors)
        assert rejection(error) == Rejection.SEVERAL_VOICES.value
        old = SampleRules(same_voice_threshold=0.5, max_other_voice_share=0.2)
        assert len(enrolment_vector(
            vectors, spread(20), [(0.0, 30.0)], 30.0, old
        )) == 512

    def test_one_window_is_not_a_voiceprint(self) -> None:
        """Regions may claim speech; the voiceprint needs the windows."""
        with pytest.raises(SampleRejectedError) as error:
            enrolment_vector(one_voice(1), [(0.0, 30.0)], [(0.0, 30.0)],
                             30.0)
        assert rejection(error) == Rejection.MOSTLY_SILENT.value
        assert "1 windows" in str(error.value)

    def test_windows_must_cover_enough_speech(self) -> None:
        short = [(i * 0.5, i * 0.5 + 0.5) for i in range(20)]  # 10 s
        rules = SampleRules(min_window_seconds=12.0)
        with pytest.raises(SampleRejectedError) as error:
            enrolment_vector(one_voice(), short, [(0.0, 30.0)], 30.0, rules)
        assert rejection(error) == Rejection.MOSTLY_SILENT.value
        lenient = SampleRules(min_window_seconds=10.0)
        assert len(enrolment_vector(
            one_voice(), short, [(0.0, 30.0)], 30.0, lenient
        )) == 4

    def test_zero_windows_do_not_count_towards_coverage(self) -> None:
        vectors = np.vstack(
            [one_voice(5), np.zeros((15, 4))]
        ).astype(np.float32)
        with pytest.raises(SampleRejectedError) as error:
            sample(vectors)   # 5 voiced windows of 1.5 s: 7.5 s < 10 s
        assert rejection(error) == Rejection.MOSTLY_SILENT.value

    def test_rules_are_parameters(self) -> None:
        rules = SampleRules(min_duration=5.0, min_speech_ratio=0.1,
                            min_windows=1, min_window_seconds=0.5)
        vector = enrolment_vector(one_voice(1), [(0.0, 1.0)], [(0.0, 1.0)],
                                  6.0, rules)
        assert len(vector) == 4


class TestSampleInput:
    @pytest.mark.parametrize("duration", [float("nan"), float("inf"), -1.0])
    def test_duration_must_be_finite_and_non_negative(
        self, duration: float
    ) -> None:
        with pytest.raises(InvalidInputError, match="duration"):
            enrolment_vector(one_voice(), spread(20), [(0.0, 30.0)],
                             duration)

    @pytest.mark.parametrize("region", [
        (0.0, float("nan")), (float("nan"), 1.0), (0.0, float("inf")),
    ])
    def test_regions_must_be_finite(
        self, region: Tuple[float, float]
    ) -> None:
        with pytest.raises(InvalidInputError, match="finite"):
            enrolment_vector(one_voice(), spread(20), [region], 30.0)

    def test_a_reversed_region_is_an_error(self) -> None:
        with pytest.raises(InvalidInputError, match="before it starts"):
            enrolment_vector(one_voice(), spread(20),
                             [(0.0, 30.0), (20.0, 5.0)], 30.0)

    def test_a_reversed_window_is_an_error(self) -> None:
        windows = spread(20)
        windows[3] = (5.0, 4.0)
        with pytest.raises(InvalidInputError, match="window"):
            enrolment_vector(one_voice(), windows, [(0.0, 30.0)], 30.0)

    def test_regions_are_clipped_to_the_recording(self) -> None:
        """31 s of claimed speech in a 25 s sample is 1 s of speech."""
        rules = SampleRules(min_duration=20.0, min_speech_ratio=0.5)
        with pytest.raises(SampleRejectedError) as error:
            enrolment_vector(one_voice(), spread(20, 25.0),
                             [(0.0, 1.0), (100.0, 130.0)], 25.0, rules)
        assert rejection(error) == Rejection.MOSTLY_SILENT.value
        assert "1.0 s of 25.0 s" in str(error.value)
        assert clip_spans([(-2.0, 3.0), (24.0, 30.0), (40.0, 50.0)], 25.0,
                          "region") == [(0.0, 3.0), (24.0, 25.0),
                                        (25.0, 25.0)]

    def test_windows_are_clipped_to_the_recording(self) -> None:
        late = [(25.0 + i, 26.0 + i) for i in range(20)]
        with pytest.raises(SampleRejectedError) as error:
            enrolment_vector(one_voice(), late, [(0.0, 25.0)], 25.0)
        assert rejection(error) == Rejection.MOSTLY_SILENT.value

    def test_vectors_must_match_the_windows(self) -> None:
        with pytest.raises(InvalidInputError, match="per analysis window"):
            enrolment_vector(one_voice(), spread(19), [(0.0, 30.0)], 30.0)
        with pytest.raises(InvalidInputError, match="per analysis window"):
            enrolment_vector(np.ones(4, dtype=np.float32), spread(4),
                             [(0.0, 30.0)], 30.0)

    def test_vectors_must_be_finite(self) -> None:
        vectors = one_voice()
        vectors[2, 1] = np.nan
        with pytest.raises(InvalidInputError, match="finite"):
            sample(vectors)


class TestBuildVoiceprint:
    def test_voiceprint_keeps_only_vector_and_consent(self) -> None:
        voiceprint = build_voiceprint(
            "ana", "wavlm", "consent-1", WHEN, one_voice(), spread(20),
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
                             spread(20, 3.0), [(0.0, 1.0)], 3.0)
