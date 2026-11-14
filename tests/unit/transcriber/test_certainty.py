"""The certainty of each attribution, from the voice windows' cosines."""

import time
from typing import Callable, List, Optional, Sequence, Tuple

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
from transcriber.domain.sentences import build_utterances
from transcriber.domain.speakers import turns_from_words

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
        ), ev).words
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
        assert keep_sentences_whole(given, ev).words == given

    def test_a_pause_ends_a_sentence_without_punctuation(self) -> None:
        ev = evidence([(4.0, 10.0, 0.52)])
        given = spoken(
            ("vale", 1.0, 2.0, "ana"), ("bien", 2.0, 2.5, "ana"),
            ("ya", 4.0, 4.5, "luis"),
        )
        assert keep_sentences_whole(given, ev, gap=1.0).words == given

    def test_without_evidence_nothing_changes(self) -> None:
        given = spoken(("a", 1.0, 1.2, "ana"), ("b", 1.2, 1.4, "luis"))
        assert keep_sentences_whole(given, None).words == given

    def test_without_windows_the_longer_neighbour_wins(self) -> None:
        far = VoiceEvidence([(50.0, 51.0)],
                            np.array([[0.5, 0.0]], np.float32),
                            ["ana", "luis"])
        found = keep_sentences_whole(spoken(
            ("uno", 1.0, 1.2, "ana"), ("dos", 1.2, 1.4, "ana"),
            ("tres", 1.4, 1.6, "luis"),
        ), far).words
        assert [w.speaker for w in found] == ["ana", "ana", "ana"]

    def test_a_word_without_speaker_is_left_alone(self) -> None:
        given = [Word(text="eh", start=1.0, end=1.2),
                 Word(text="ya", start=1.2, end=1.4, speaker="ana")]
        found = keep_sentences_whole(given, evidence([])).words
        assert found[0].speaker is None


def scored(
    end: float,
    score: Callable[[float], Sequence[float]],
    voices: Sequence[Optional[str]] = ("ana", "luis"),
    length: float = 1.0,
    step: float = 0.5,
) -> VoiceEvidence:
    """Windows of ``length`` s every ``step`` s up to ``end``.

    ``score(start)`` gives the row of cosines of the window at ``start``.
    """
    count = int(round((end - length) / step)) + 1
    windows = [(n * step, n * step + length) for n in range(count)]
    rows = [list(score(left)) for left, _ in windows]
    return VoiceEvidence(windows, np.array(rows, dtype=np.float32), voices)


def one_sentence(
    *runs: Tuple[float, float, str], word: float = 0.5
) -> List[Word]:
    """Words of ``word`` s over each ``(start, end, speaker)`` run.

    One sentence: only the last word ends with a full stop.
    """
    words: List[Word] = []
    for start, end, speaker in runs:
        t = start
        while t < end - 1e-9:
            words.append(Word(text="la", start=t, end=min(t + word, end),
                              speaker=speaker))
            t += word
    last = words[-1]
    words[-1] = last.model_copy(update={"text": "la."})
    return words


class TestReassignmentNeedsEvidence:
    """The review of #24: a run must not go where the windows disagree."""

    @staticmethod
    def reviewer() -> Tuple[List[Word], VoiceEvidence]:
        """One 29 s sentence: ana 0-20 s ([0.9, 0.5]), luis 20-29 s.

        Over luis's 9 s the windows give ana 0.5 and luis 0.59: they
        prefer luis, by less than the cut.
        """
        def score(left: float) -> Sequence[float]:
            return [0.9, 0.5] if left < 20.0 else [0.5, 0.59]

        words = one_sentence((0.0, 20.0, "ana"), (20.0, 29.0, "luis"))
        return words, scored(29.0, score)

    def test_a_long_weak_run_the_windows_prefer_stays(self) -> None:
        words, ev = self.reviewer()
        kept = keep_sentences_whole(words, ev, max_move=float("inf"))
        assert kept.words == words and kept.moved == []

    def test_the_sentence_is_then_cut_at_the_change(self) -> None:
        words, ev = self.reviewer()
        kept = keep_sentences_whole(words, ev).words
        utterances = build_utterances(turns_from_words(kept, 1.0))
        assert [(u.speaker, u.start, u.end) for u in utterances] == [
            ("ana", 0.0, 20.0), ("luis", 20.0, 29.0),
        ]
        assessed = assess_utterances(utterances, ev)
        assert assessed[1].doubtful is True
        certainty = assessed[1].speaker_certainty
        assert certainty is not None and 0 < certainty < 0.1

    def test_a_run_longer_than_the_bound_never_moves(self) -> None:
        """Even when its windows prefer the neighbour."""
        def score(left: float) -> Sequence[float]:
            return [0.9, 0.5] if left < 20.0 else [0.6, 0.5]

        words = one_sentence((0.0, 20.0, "ana"), (20.0, 24.0, "luis"))
        ev = scored(24.0, score)
        assert keep_sentences_whole(words, ev, max_move=3.0).moved == []
        moved = keep_sentences_whole(words, ev, max_move=4.0)
        assert moved.moved == [(20.0, 24.0)]
        assert {w.speaker for w in moved.words} == {"ana"}

    def test_a_short_run_its_windows_reject_moves(self) -> None:
        def score(left: float) -> Sequence[float]:
            return [0.4, 0.6] if left < 9.0 else [0.45, 0.6]

        words = one_sentence((0.0, 9.0, "luis"), (9.0, 10.5, "ana"))
        kept = keep_sentences_whole(words, scored(10.5, score))
        assert {w.speaker for w in kept.words} == {"luis"}
        assert kept.moved == [(9.0, 10.5)]

    def test_a_moved_run_is_never_silently_confident(self) -> None:
        """Moved to ana although a third voice is the one heard.

        Over luis's 1.5 s the windows prefer eva; luis's own margin is
        negative, so the run goes to ana, its only neighbour. The
        sentence's mean margin stays high (ana's 9 s), but the sentence
        carries the moved stretch's margin and is doubtful, with eva as
        the alternative.
        """
        voices = ("ana", "luis", "eva")

        def score(left: float) -> Sequence[float]:
            return [0.9, 0.1, 0.1] if left < 8.5 else [0.3, 0.4, 0.9]

        words = one_sentence((0.0, 9.0, "ana"), (9.0, 10.5, "luis"))
        ev = scored(10.5, score, voices)
        kept = keep_sentences_whole(words, ev)
        assert {w.speaker for w in kept.words} == {"ana"}
        [whole] = build_utterances(turns_from_words(kept.words, 1.0))
        [plain] = assess_utterances([whole], ev)
        [marked] = assess_utterances([whole], ev, moved=kept.moved)
        assert plain.doubtful is False
        assert marked.doubtful is True
        assert marked.speaker_certainty is not None
        assert marked.speaker_certainty < 0
        assert marked.alternative_speaker == "eva"

    def test_a_moved_stretch_without_windows_is_doubtful(self) -> None:
        ev = scored(5.0, lambda left: [0.9, 0.1])
        [plain] = assess_utterances([utterance("ana", 1.0, 30.0)], ev)
        [marked] = assess_utterances(
            [utterance("ana", 1.0, 30.0)], ev, moved=[(20.0, 21.0)]
        )
        assert plain.doubtful is False and marked.doubtful is True

    def test_moves_outside_the_sentence_do_not_count(self) -> None:
        ev = scored(10.0, lambda left: [0.9, 0.1])
        [plain] = assess_utterances(
            [utterance("ana", 1.0, 3.0)], ev, moved=[(5.0, 6.0)]
        )
        assert plain.doubtful is False


class TestAlternativeOfAnOverlap:
    def test_with_three_voices_it_is_the_voice_that_flagged(self) -> None:
        """Luis is close over the whole sentence; one window is eva's."""
        voices = ("ana", "luis", "eva")

        def score(left: float) -> Sequence[float]:
            if 4.0 <= left < 4.5:
                return [0.5, 0.3, 0.9]
            return [0.8, 0.5, 0.0]

        [mixed] = assess_utterances(
            [utterance("ana", 1.0, 8.0)], scored(10.0, score, voices)
        )
        assert mixed.doubtful is False and mixed.overlap is True
        assert mixed.alternative_speaker == "eva"


class TestScale:
    def test_many_windows_and_sentences_take_bounded_time(self) -> None:
        """10^5 windows, 10^4 sentences: no pass over every window each.

        Scanning all windows for each sentence would compare 10^9
        pairs (minutes); the index takes well under the bound.
        """
        count = 100_000
        windows = [(n * 0.5, n * 0.5 + 1.0) for n in range(count)]
        rng = np.random.default_rng(0)
        ev = VoiceEvidence(
            windows, rng.random((count, 2)).astype(np.float32),
            ("ana", "luis"),
        )
        sentences = [
            utterance("ana" if n % 2 else "luis", n * 5.0, n * 5.0 + 4.0, n)
            for n in range(10_000)
        ]
        words = [
            Word(text="la." if k == 3 else "la", start=n * 5.0 + k,
                 end=n * 5.0 + k + 1.0,
                 speaker="ana" if (n + k) % 3 else "luis")
            for n in range(10_000) for k in range(4)
        ]
        began = time.perf_counter()
        assessed = assess_utterances(sentences, ev)
        keep_sentences_whole(words, ev)
        assert time.perf_counter() - began < 20.0
        assert len(assessed) == 10_000
