"""The certainty of each attribution, from the voice windows' cosines."""

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pytest

from transcriber.domain.certainty import (
    DEFAULT_DOUBT_MARGIN,
    VoiceEvidence,
    assess_utterances,
    keep_sentences_whole,
    support,
)
from transcriber.domain.models import Utterance, Word

# Short windows of 1 s every 0.5 s from 0 to 10 s.
WINDOWS = [(t / 2, t / 2 + 1.0) for t in range(19)]


def evidence(
    prefer: Sequence[Tuple[float, float, float]],
    voices: Sequence[Optional[str]] = ("ana", "luis"),
) -> VoiceEvidence:
    """Windows that score ana 0.5 and luis 0.0, unless ``prefer`` says.

    ``prefer`` lists ``(start, end, luis)``: windows starting inside
    ``[start, end)`` give luis that cosine (ana keeps 0.5).
    """
    rows = []
    for left, _ in WINDOWS:
        luis = 0.0
        for start, end, value in prefer:
            if start <= left < end:
                luis = value
        rows.append([0.5, luis] + [0.0] * (len(voices) - 2))
    return VoiceEvidence(WINDOWS, np.array(rows, dtype=np.float32), voices)


def utterance(
    speaker: str, start: float, end: float, n: int = 0
) -> Utterance:
    """A one-sentence turn ``n`` of ``speaker``."""
    return Utterance(
        id=f"t{n}.s0", turn=n, index=0, speaker=speaker, start=start,
        end=end, text="Algo.", previous_speaker=None,
        previous_utterance=None,
    )


def spoken(*items: Tuple[str, float, float, str]) -> List[Word]:
    """Words from ``(text, start, end, speaker)``."""
    return [Word(text=t, start=s, end=e, speaker=w) for t, s, e, w in items]


class TestSupport:
    def test_margin_is_own_cosine_minus_the_closest_other(self) -> None:
        held = support(evidence([]), 1.0, 3.0, "ana")
        assert held is not None
        assert held.margin == pytest.approx(0.5)
        assert held.runner_up == "luis"

    def test_a_speaker_the_windows_do_not_prefer_has_a_negative_margin(
        self,
    ) -> None:
        held = support(evidence([]), 1.0, 3.0, "luis")
        assert held is not None and held.margin == pytest.approx(-0.5)

    def test_windows_are_weighted_by_the_seconds_they_share(self) -> None:
        # Windows from 4 s on prefer luis (0.9 against ana's 0.5).
        held = support(evidence([(4.0, 10.0, 0.9)]), 3.0, 5.0, "ana")
        assert held is not None
        assert -0.4 < held.margin < 0.5

    def test_nothing_to_compare_gives_none(self) -> None:
        assert support(evidence([]), 1.0, 2.0, "eva") is None
        alone = VoiceEvidence(
            WINDOWS, np.ones((len(WINDOWS), 1), np.float32), ["ana"]
        )
        assert support(alone, 1.0, 2.0, "ana") is None
        assert support(evidence([]), 50.0, 60.0, "ana") is None

    def test_a_voice_nobody_is_is_left_out(self) -> None:
        held = support(
            evidence([], voices=("ana", "luis", None)), 1.0, 2.0, "ana"
        )
        assert held is not None and held.runner_up == "luis"

    def test_an_instant_takes_the_windows_that_contain_it(self) -> None:
        held = support(evidence([]), 2.2, 2.2, "ana")
        assert held is not None and held.margin == pytest.approx(0.5)


class TestAssessUtterances:
    def test_a_clear_sentence(self) -> None:
        [clear] = assess_utterances([utterance("ana", 1.0, 3.0)],
                                    evidence([]))
        assert clear.speaker_certainty == pytest.approx(0.5)
        assert clear.doubtful is False and clear.overlap is False
        assert clear.alternative_speaker is None

    def test_a_doubtful_sentence_names_the_alternative(self) -> None:
        close = evidence([(0.0, 10.0, 0.45)])
        [doubt] = assess_utterances([utterance("ana", 1.0, 3.0)], close)
        assert doubt.speaker_certainty == pytest.approx(0.05)
        assert doubt.doubtful is True
        assert doubt.alternative_speaker == "luis"

    def test_the_cut_is_a_parameter(self) -> None:
        close = evidence([(0.0, 10.0, 0.45)])
        [sure] = assess_utterances(
            [utterance("ana", 1.0, 3.0)], close, cut=0.01
        )
        assert sure.doubtful is False

    def test_another_voice_inside_the_sentence_is_an_overlap(self) -> None:
        # One window wholly inside 1-6 s prefers luis.
        [mixed] = assess_utterances(
            [utterance("ana", 1.0, 6.0)], evidence([(3.0, 3.5, 0.9)])
        )
        assert mixed.overlap is True
        assert mixed.doubtful is False
        assert mixed.alternative_speaker == "luis"

    def test_a_window_across_the_edge_is_no_overlap(self) -> None:
        # The window starting at 5.5 s reaches past the sentence's end.
        [edge] = assess_utterances(
            [utterance("ana", 1.0, 6.0)], evidence([(5.5, 6.0, 0.9)])
        )
        assert edge.overlap is False

    def test_without_evidence_nothing_is_said(self) -> None:
        [plain] = assess_utterances([utterance("ana", 1.0, 3.0)], None)
        assert plain.speaker_certainty is None
        assert plain.doubtful is None and plain.overlap is None

    def test_a_speaker_without_windows_is_not_assessed(self) -> None:
        [plain] = assess_utterances(
            [utterance("eva", 1.0, 3.0)], evidence([])
        )
        assert plain.speaker_certainty is None and plain.overlap is None

    def test_the_default_cut(self) -> None:
        assert DEFAULT_DOUBT_MARGIN == 0.1


class TestKeepSentencesWhole:
    def test_a_stray_window_does_not_cut_a_sentence(self) -> None:
        """The first word of a reply heard as the previous speaker.

        Luis's sentence starts at 4.0 s; the one window over its first
        word barely prefers ana, so the word stays with the sentence.
        """
        ev = evidence([(4.5, 10.0, 0.9), (3.5, 4.5, 0.55)])
        found = keep_sentences_whole(spoken(
            ("Hasta", 1.0, 1.5, "ana"), ("luego.", 1.5, 2.5, "ana"),
            ("No", 4.0, 4.3, "ana"), ("sé", 4.3, 4.8, "luis"),
            ("de", 4.8, 5.0, "luis"), ("qué", 5.0, 5.5, "luis"),
            ("hablas.", 5.5, 6.5, "luis"),
        ), ev)
        assert [w.speaker for w in found] == [
            "ana", "ana", "luis", "luis", "luis", "luis", "luis",
        ]

    def test_a_well_supported_change_inside_a_sentence_stays(self) -> None:
        ev = evidence([(3.0, 10.0, 0.9)])
        given = spoken(
            ("Pero", 1.0, 1.5, "ana"), ("yo", 1.5, 2.0, "ana"),
            ("creo", 2.0, 3.4, "ana"), ("que", 3.5, 4.5, "luis"),
            ("no", 4.5, 5.5, "luis"), ("es", 5.5, 6.0, "luis"),
            ("así.", 6.0, 7.0, "luis"),
        )
        assert keep_sentences_whole(given, ev) == given

    def test_a_pause_ends_a_sentence_without_punctuation(self) -> None:
        ev = evidence([(4.0, 10.0, 0.52)])
        given = spoken(
            ("vale", 1.0, 2.0, "ana"), ("bien", 2.0, 2.5, "ana"),
            ("ya", 4.0, 4.5, "luis"),
        )
        assert keep_sentences_whole(given, ev, gap=1.0) == given

    def test_without_evidence_nothing_changes(self) -> None:
        given = spoken(("a", 1.0, 1.2, "ana"), ("b", 1.2, 1.4, "luis"))
        assert keep_sentences_whole(given, None) == given

    def test_without_windows_the_longer_neighbour_wins(self) -> None:
        far = VoiceEvidence([(50.0, 51.0)],
                            np.array([[0.5, 0.0]], np.float32),
                            ["ana", "luis"])
        found = keep_sentences_whole(spoken(
            ("uno", 1.0, 1.2, "ana"), ("dos", 1.2, 1.4, "ana"),
            ("tres", 1.4, 1.6, "luis"),
        ), far)
        assert [w.speaker for w in found] == ["ana", "ana", "ana"]

    def test_a_word_without_speaker_is_left_alone(self) -> None:
        given = [Word(text="eh", start=1.0, end=1.2),
                 Word(text="ya", start=1.2, end=1.4, speaker="ana")]
        found = keep_sentences_whole(given, evidence([]))
        assert found[0].speaker is None
