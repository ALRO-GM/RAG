"""Names on multi-party recordings (#116): variants, moderators, votes."""

from typing import Dict, Tuple

import pytest

from transcriber.domain.name_variants import (
    cluster_names,
    edit_distance,
    similar_names,
    similar_words,
    sound_key,
    strip_honorifics,
)
from transcriber.domain.name_votes import (
    CUE_WEIGHT,
    best_matching,
    find_moderators,
    margin,
    votes_for,
)


class TestVariants:
    @pytest.mark.parametrize("name, expected", [
        ("señor Ruiz", "Ruiz"),
        ("Señora Gil Pons", "Gil Pons"),
        ("Mr. Smith", "Smith"),
        ("Dr Watson", "Watson"),
        ("doña Doña Inés", "Inés"),
        ("Señor", ""),
        ("Ana", "Ana"),
    ])
    def test_honorifics_are_left_out(self, name: str, expected: str) -> None:
        assert strip_honorifics(name) == expected

    def test_edit_distance(self) -> None:
        assert edit_distance("kitten", "sitting") == 3
        assert edit_distance("", "abc") == 3
        assert edit_distance("same", "same") == 0

    def test_sound_key(self) -> None:
        assert sound_key("itziar") == sound_key("ichiar")
        assert sound_key("valdivieso") == sound_key("baldiviezo")
        assert sound_key("helena") == sound_key("elena")
        assert sound_key("chema") == "chema"

    @pytest.mark.parametrize("a, b", [
        ("Bermejo", "Bermeyo"),  # one letter changed
        ("Bermejo", "Ber Mejo"),  # split in two
        ("Berme", "Bermejo"),  # cut short
        ("Itziar", "Ichiar"),  # sounds alike
        ("Ruiz", "Ana Ruiz"),  # the surname of a full name
        ("Pablo Bermejo", "Ber Mejo"),  # a first name before it
        ("Ana García", "ana garcia"),
        ("señor Ruiz", "Ruiz"),
    ])
    def test_variants_of_one_name(self, a: str, b: str) -> None:
        assert similar_names(a, b) and similar_names(b, a)

    @pytest.mark.parametrize("a, b", [
        ("Mario", "María"),  # gender pairs are two people
        ("Juan", "Juana"),
        ("Marta", "Martín"),
        ("Ana", "Eva"),
        ("Lucía", "Lucas"),
        ("Juan Ruiz", "Ana Ruiz"),
        ("Señor", "Ruiz"),
    ])
    def test_different_names(self, a: str, b: str) -> None:
        assert not similar_names(a, b)

    def test_similar_words_edges(self) -> None:
        assert not similar_words("", "ana")
        assert similar_words("ana", "ana")
        assert not similar_words("eva", "ava")

    def test_clusters_show_the_most_frequent_spelling(self) -> None:
        shown = cluster_names([
            "Bermejo", "señor Bermejo", "Bermeyo", "Ber Mejo",
            "Pablo Bermejo", "Ruiz", "Ruiz", "Ruis", "Ana",
        ])
        assert shown["Ber Mejo"] == shown["Pablo Bermejo"] == "Bermejo"
        assert shown["señor Bermejo"] == "Bermejo"
        assert shown["Ruis"] == "Ruiz"
        assert shown["Ana"] == "Ana"

    def test_a_first_name_does_not_chain_two_people(self) -> None:
        shown = cluster_names(["Juan Ruiz", "Juan Gil", "Juan"])
        assert shown["Juan Ruiz"] == "Juan Ruiz"
        assert shown["Juan Gil"] == "Juan Gil"
        assert shown["Juan"] == "Juan"
        assert cluster_names(["Señor", ""]) == {}


DEBATE = [
    ("M", "Bienvenidos al debate."),
    ("M", "Tiene la palabra la señora Ruiz."),
    ("A", "Gracias."),
    ("M", "Turno para el señor Gil."),
    ("B", "Yo opino otra cosa."),
    ("M", "Le pregunto a la señora Vera."),
    ("C", "Bien."),
    ("A", "Una réplica."),
]


class TestModerators:
    def test_the_voice_that_hands_the_floor(self) -> None:
        assert find_moderators(DEBATE) == frozenset({"M"})

    def test_two_people_have_no_moderator(self) -> None:
        lines = [("A", "Turno para ti."), ("B", "Bienvenido.")] * 5
        assert find_moderators(lines) == frozenset()

    def test_two_presenters(self) -> None:
        lines = DEBATE + [
            ("N", "Bienvenidos de nuevo."), ("N", "Turno para Gil."),
            ("N", "Le pregunto a Vera."), ("B", "Claro."),
        ]
        assert find_moderators(lines) == frozenset({"M", "N"})


SPEAKERS = ["M", "A", "M", "B", "A", "M", "C"]


class TestVotes:
    def test_self_next_and_previous(self) -> None:
        assert votes_for(SPEAKERS, 1, "self") == [("A", CUE_WEIGHT)]
        assert votes_for(SPEAKERS, 2, "next", frozenset({"M"})) == [
            ("B", CUE_WEIGHT),
        ]
        assert votes_for(SPEAKERS, 3, "previous", frozenset({"M"})) == [
            ("M", CUE_WEIGHT),
        ]

    def test_the_other_side_when_nobody_is_on_one(self) -> None:
        assert votes_for(["A", "B"], 1, "next") == [("A", CUE_WEIGHT)]
        assert votes_for(["A", "B"], 0, "previous") == [("B", CUE_WEIGHT)]
        assert votes_for(["A"], 0, "next") == []
        assert votes_for(["A"], 0, "around") == []

    def test_a_rebuttal_skips_the_moderator(self) -> None:
        votes = dict(votes_for(SPEAKERS, 4, "around", frozenset({"M"}), 1.0))
        assert votes == {"B": 0.5, "C": 0.25, "A": 0.0} or votes == {
            "B": pytest.approx(0.5 / 0.75), "C": pytest.approx(0.25 / 0.75),
        }
        assert "M" not in votes

    def test_a_moderator_mention_is_split(self) -> None:
        votes = dict(votes_for(SPEAKERS, 2, "around", frozenset({"M"}), 1.0))
        assert votes == {"A": 0.5, "B": 0.5}

    def test_shares_add_up_and_merge(self) -> None:
        votes = dict(votes_for(["A", "B", "A", "C"], 2, "around", weight=1.0))
        assert sum(votes.values()) == pytest.approx(1.0)
        assert votes["B"] == pytest.approx(0.5 / 0.75)
        votes = dict(votes_for(["B", "C", "A", "B"], 2, "around", weight=1.0))
        assert sum(votes.values()) == pytest.approx(1.0)


class TestMatching:
    def test_one_to_one_with_the_highest_total(self) -> None:
        scores: Dict[Tuple[str, str], float] = {
            ("S1", "Ruiz"): 5.0, ("S1", "Gil"): 4.0,
            ("S2", "Ruiz"): 4.5, ("S2", "Gil"): 0.5,
        }
        assert sorted(best_matching(scores)) == [
            ("S1", "Gil"), ("S2", "Ruiz"),
        ]

    def test_only_positive_pairs_and_rectangular(self) -> None:
        scores = {
            ("S1", "Ruiz"): 3.0, ("S2", "Ruiz"): -1.0, ("S3", "Gil"): 0.0,
            ("S1", "Vera"): 1.0,
        }
        assert best_matching(scores) == [("S1", "Ruiz")]
        assert best_matching({}) == []

    def test_margin_over_the_best_rival(self) -> None:
        scores = {
            ("S1", "Ruiz"): 6.0, ("S1", "Gil"): 1.0, ("S2", "Ruiz"): 2.5,
            ("S2", "Gil"): 9.0,
        }
        assert margin(scores, "S1", "Ruiz") == 3.5
        assert margin({("S1", "Ruiz"): 2.0}, "S1", "Ruiz") == 2.0
