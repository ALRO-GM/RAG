"""Reference data: forms, validation, annotator agreement, consensus."""

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import pytest

from src.application.skills.gold import (
    AnnotatedEvidence,
    Annotation,
    AnnotationSet,
    agreement,
    blank_annotations,
    check_on_scale,
    consensus,
    median_low,
    validate_annotations,
)
from src.domain.errors import InvalidInputError, RagError
from src.domain.skills.agreement import icc_two_way
from src.domain.skills.rubric import Scale
from tests.fakes import make_catalogue, make_transcript

CATALOGUE = make_catalogue()
TRANSCRIPT = make_transcript()
SCALE = Scale()
Marks = Sequence[Tuple[int, str]]
Cell = Tuple[Optional[int], Marks]
Cells = Mapping[Tuple[str, str], Cell]
N = None

KRIPPENDORFF: List[List[Optional[int]]] = [
    [1, 2, 3, 3, 2, 1, 4, 1, 2, N, N, N],
    [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, N, 3],
    [N, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, N],
    [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, N],
]


def annotation(
    participant: str, competency: str, rating: Optional[int] = None,
    marks: Marks = (),
) -> Annotation:
    """Return one annotated cell with (turn, indicator) evidence."""
    return Annotation(
        participant=participant, competency_id=competency, rating=rating,
        evidence=[
            AnnotatedEvidence(turn=t, indicator_id=i) for t, i in marks
        ],
    )


def aset(
    annotator: str, cells: Cells, session: str = "s1", version: str = "1.0"
) -> AnnotationSet:
    """Return an annotation set from ``(participant, id) → cell``."""
    return AnnotationSet(
        session_id=session, annotator=annotator, catalogue="core",
        catalogue_version=version,
        annotations=[
            annotation(p, c, rating, marks)
            for (p, c), (rating, marks) in cells.items()
        ],
    )


def rated(annotator: str, ratings: Sequence[Optional[int]]) -> AnnotationSet:
    """Return a set rating one competency of participants p00, p01..."""
    return aset(annotator, {
        (f"p{i:02d}", "listening"): (rating, [(i, "L1")] if rating else [])
        for i, rating in enumerate(ratings)
    })


def merged(sets: Sequence[AnnotationSet]) -> Dict[Tuple[str, str], Annotation]:
    """Return the single-session consensus keyed by (participant, id)."""
    result, = consensus(sets)
    return {(a.participant, a.competency_id): a for a in result.annotations}


VALID: Cells = {
    ("Maya", "listening"): (4, [(0, "L1"), (3, "L2")]),
    ("Maya", "planning"): (None, []),
    ("Tom", "listening"): (2, [(1, "L3")]),
    ("Tom", "planning"): (3, [(4, "P2"), (7, "P1")]),
    ("Priya", "listening"): (None, []),
    ("Priya", "planning"): (4, [(5, "P1")]),
}


def problems(cells: Cells, **fields: str) -> List[str]:
    """Validate ``cells`` (as a set by ``ann``) against the sample."""
    form = aset("ann", cells).model_copy(update=fields)
    return validate_annotations(form, TRANSCRIPT, CATALOGUE)


def changed(key: Tuple[str, str], cell: Cell) -> Cells:
    """Return the valid form with one cell replaced."""
    return {**VALID, key: cell}


# ----------------------------------------------------------- blank form


def test_blank_form_has_one_empty_cell_per_speaker_and_competency() -> None:
    """The form lists speakers × competencies, unrated, without evidence."""
    form = blank_annotations(TRANSCRIPT, CATALOGUE, "  ann  ")
    assert (form.session_id, form.annotator) == ("s1", "ann")
    assert (form.catalogue, form.catalogue_version) == ("core", "1.0")
    assert [(a.participant, a.competency_id) for a in form.annotations] == [
        ("Maya", "listening"), ("Maya", "planning"),
        ("Tom", "listening"), ("Tom", "planning"),
        ("Priya", "listening"), ("Priya", "planning"),
    ]
    assert all(
        a.rating is None and a.evidence == [] and a.notes == ""
        for a in form.annotations
    )


def test_blank_form_is_valid_as_it_is() -> None:
    """An untouched form (all 'insufficient evidence') has no problem."""
    form = blank_annotations(TRANSCRIPT, CATALOGUE, "ann")
    assert validate_annotations(form, TRANSCRIPT, CATALOGUE) == []


@pytest.mark.parametrize("name", ["", "   ", "\t"])
def test_blank_form_needs_an_annotator_name(name: str) -> None:
    """An empty annotator name is rejected."""
    with pytest.raises(InvalidInputError, match="annotator name is empty"):
        blank_annotations(TRANSCRIPT, CATALOGUE, name)


def test_annotation_set_by_unit_and_turns() -> None:
    """Cells are keyed by (session, participant, competency)."""
    form = aset("ann", VALID)
    units = form.by_unit()
    assert len(units) == 6
    assert units[("s1", "Maya", "listening")].turns() == {0, 3}
    assert units[("s1", "Maya", "planning")].turns() == set()


def test_annotated_evidence_rejects_negative_turns() -> None:
    """A turn number below zero does not validate."""
    with pytest.raises(ValueError):
        AnnotatedEvidence(turn=-1, indicator_id="L1")


# ----------------------------------------------------------- validation


def test_valid_annotations_have_no_problem() -> None:
    """A correctly filled form validates."""
    assert problems(VALID) == []


def test_validation_session_mismatch() -> None:
    """A form for another session is reported."""
    assert problems(VALID, session_id="other") == [
        "session is 'other', transcript is 's1'"
    ]


@pytest.mark.parametrize("fields", [
    {"catalogue": "extra"}, {"catalogue_version": "0.9"},
])
def test_validation_catalogue_mismatch(fields: Dict[str, str]) -> None:
    """Another catalogue or version is reported."""
    found = problems(VALID, **fields)
    assert len(found) == 1
    assert found[0].startswith("annotated with catalogue")
    assert found[0].endswith("expected core 1.0")


def test_validation_cell_annotated_twice() -> None:
    """The same participant and competency twice is a problem."""
    form = aset("ann", VALID)
    form.annotations.append(annotation("Tom", "planning"))
    assert validate_annotations(form, TRANSCRIPT, CATALOGUE) == [
        "Tom/planning: annotated twice"
    ]


def test_validation_unknown_participant_and_competency() -> None:
    """Names that are not in the transcript or the catalogue."""
    found = problems({
        **VALID, ("Zoe", "listening"): (None, []),
        ("Tom", "charisma"): (None, []),
    })
    assert found == [
        "Zoe/listening: unknown participant",
        "Tom/charisma: unknown competency",
    ]


@pytest.mark.parametrize("rating", [0, 6, -2])
def test_validation_rating_off_the_scale(rating: int) -> None:
    """A rating outside 1-5 is a problem."""
    found = problems(changed(("Tom", "listening"), (rating, [(1, "L3")])))
    assert found == [f"Tom/listening: rating {rating} off the scale"]


def test_validation_rating_without_evidence() -> None:
    """A rating must rest on at least one marked turn."""
    assert problems(changed(("Tom", "listening"), (2, []))) == [
        "Tom/listening: rating without evidence"
    ]


def test_validation_evidence_without_rating_is_allowed() -> None:
    """Marking evidence but abstaining from rating is valid."""
    assert problems(changed(("Tom", "listening"), (None, [(1, "L3")]))) == []


def test_validation_turn_not_found() -> None:
    """A turn past the end of the transcript is a problem."""
    assert problems(changed(("Tom", "listening"), (2, [(8, "L3")]))) == [
        "Tom/listening: turn 8 not found"
    ]


def test_validation_turn_of_somebody_else() -> None:
    """Evidence must be a turn of the annotated participant."""
    assert problems(changed(("Tom", "listening"), (2, [(0, "L3")]))) == [
        "Tom/listening: turn 0 is not theirs"
    ]


def test_validation_indicator_of_another_competency() -> None:
    """The indicator must belong to the annotated competency."""
    assert problems(changed(("Tom", "listening"), (2, [(1, "P2")]))) == [
        "Tom/listening: indicator 'P2' is not of listening"
    ]


def test_validation_reports_turn_and_indicator_together() -> None:
    """One bad evidence item can raise two problems."""
    found = problems(changed(("Tom", "listening"), (2, [(99, "nope")])))
    assert found == [
        "Tom/listening: turn 99 not found",
        "Tom/listening: indicator 'nope' is not of listening",
    ]


def test_validation_missing_cells() -> None:
    """Cells left out of the form are listed in one problem."""
    cells = {k: v for k, v in VALID.items() if k[0] != "Priya"}
    assert problems(cells) == [
        "not annotated: Priya/listening, Priya/planning"
    ]


def test_validation_collects_every_problem() -> None:
    """Problems of different cells are all returned."""
    found = problems({
        **VALID, ("Tom", "listening"): (9, []),
        ("Maya", "listening"): (4, [(1, "L1")]),
    }, session_id="x")
    assert len(found) == 4


# ------------------------------------------------------------ agreement


def test_agreement_needs_two_annotators() -> None:
    """One annotator (even with two files) cannot agree with anyone."""
    with pytest.raises(InvalidInputError, match="at least two annotators"):
        agreement([aset("ann", VALID)], SCALE)
    with pytest.raises(InvalidInputError, match="at least two annotators"):
        agreement([aset("ann", VALID), aset("ann", VALID, "s2")], SCALE)
    with pytest.raises(InvalidInputError, match="at least two annotators"):
        agreement([], SCALE)


def test_agreement_of_identical_annotators_is_perfect() -> None:
    """Two copies of one form agree on everything."""
    report = agreement([aset("a", VALID), aset("b", VALID)], SCALE)
    assert report.annotators == ["a", "b"]
    assert (report.sessions, report.units) == (1, 6)
    assert report.alpha == pytest.approx(1.0)
    assert report.band == "too few units"
    assert (report.icc, report.icc_average) == pytest.approx((1.0, 1.0))
    assert report.within_one == 1.0
    assert report.abstention_kappa == pytest.approx(1.0)
    assert report.evidence_f1 == pytest.approx(1.0)
    assert report.pairwise_kappa == {"a|b": pytest.approx(1.0)}
    assert [c.competency_id for c in report.per_competency] == [
        "listening", "planning",
    ]
    assert [c.units for c in report.per_competency] == [3, 3]


def test_agreement_reproduces_krippendorff_with_missing_cells() -> None:
    """Four annotators with unrated cells give the published alpha."""
    sets = [rated(f"r{i}", row) for i, row in enumerate(KRIPPENDORFF)]
    report = agreement(sets, SCALE)
    assert report.annotators == ["r0", "r1", "r2", "r3"]
    assert report.units == 12
    assert report.alpha == pytest.approx(0.815, abs=0.0005)
    assert report.band == "reliable"
    assert len(report.pairwise_kappa) == 6
    complete = [i for i in range(12) if all(r[i] for r in KRIPPENDORFF)]
    expected = icc_two_way(
        [[float(r[i] or 0) for i in complete] for r in KRIPPENDORFF]
    )
    assert expected is not None
    assert (report.icc, report.icc_average) == pytest.approx(expected)


def test_agreement_competency_band_needs_enough_co_rated_units() -> None:
    """Under ten co-rated units the per-competency alpha has no band."""
    few = agreement([rated("a", [1, 2, 3, N]), rated("b", [1, 2, 4, 5])],
                    SCALE).per_competency[0]
    assert (few.units, few.rated_units) == (4, 3)
    assert few.alpha is not None and few.band == "too few units"
    enough = agreement(
        [rated(f"r{i}", row) for i, row in enumerate(KRIPPENDORFF)], SCALE
    ).per_competency[0]
    assert (enough.units, enough.rated_units) == (12, 11)
    assert enough.band == "reliable"
    assert enough.alpha == pytest.approx(0.815, abs=0.0005)


def test_agreement_absent_cells_equal_unrated_cells_for_alpha() -> None:
    """A cell left out and a cell left unrated weigh the same."""
    full = [rated(f"r{i}", row) for i, row in enumerate(KRIPPENDORFF)]
    sparse = [
        item.model_copy(update={"annotations": [
            a for a in item.annotations if a.rating is not None
        ]})
        for item in full
    ]
    assert agreement(sparse, SCALE).alpha == pytest.approx(
        agreement(full, SCALE).alpha or 0.0
    )
    assert agreement(sparse, SCALE).units == 12


def test_agreement_three_annotators_within_one_and_pairs() -> None:
    """Pairwise figures are averaged over the three pairs."""
    report = agreement([
        rated("a", [1, 2, 3, 4]), rated("b", [1, 2, 3, 4]),
        rated("c", [1, 2, 5, 2]),
    ], SCALE)
    assert sorted(report.pairwise_kappa) == ["a|b", "a|c", "b|c"]
    assert report.pairwise_kappa["a|b"] == pytest.approx(1.0)
    assert report.pairwise_kappa["a|c"] == report.pairwise_kappa["b|c"]
    assert report.within_one == pytest.approx((1.0 + 0.5 + 0.5) / 3)
    assert report.alpha is not None and 0.0 < report.alpha < 1.0
    assert report.icc is not None and report.icc_average is not None
    assert report.icc < report.icc_average < 1.0


def test_agreement_abstention_kappa() -> None:
    """Rated-or-not agreement: identical 1, independent 0."""
    same = agreement([
        rated("a", [3, 3, N, N]), rated("b", [4, 2, N, N]),
    ], SCALE)
    independent = agreement([
        rated("a", [3, 3, N, N]), rated("b", [3, N, 3, N]),
    ], SCALE)
    assert same.abstention_kappa == pytest.approx(1.0)
    assert independent.abstention_kappa == pytest.approx(0.0)


def test_agreement_abstention_ignores_cells_an_annotator_lacks() -> None:
    """A cell missing from one form is not counted as an abstention."""
    a = rated("a", [3, 3, N, N, 5])
    b = rated("b", [3, 3, N, N])
    report = agreement([a, b], SCALE)
    assert report.units == 5
    assert report.abstention_kappa == pytest.approx(1.0)


def test_agreement_evidence_f1_is_the_mean_pairwise_overlap() -> None:
    """Turn overlap is averaged over the units with any evidence."""
    a = aset("a", {
        ("Maya", "listening"): (4, [(0, "L1"), (3, "L2")]),
        ("Tom", "listening"): (2, [(1, "L3")]),
        ("Priya", "listening"): (None, []),
    })
    b = aset("b", {
        ("Maya", "listening"): (4, [(3, "L1"), (6, "L2")]),
        ("Tom", "listening"): (2, [(1, "L1")]),
        ("Priya", "listening"): (None, []),
    })
    report = agreement([a, b], SCALE)
    assert report.evidence_f1 == pytest.approx((0.5 + 1.0) / 2)
    assert report.per_competency[0].evidence_f1 == pytest.approx(0.75)


def test_agreement_without_any_evidence_or_common_rating() -> None:
    """Nothing to compare gives ``None`` and 'not computable'."""
    report = agreement([
        rated("a", [N, N, 3]), rated("b", [N, 2, N]),
    ], SCALE)
    assert report.alpha is None and report.band == "too few units"
    assert report.icc is None and report.icc_average is None
    assert report.within_one is None
    assert report.pairwise_kappa == {"a|b": None}
    blank = agreement([rated("a", [N, N]), rated("b", [N, N])], SCALE)
    assert blank.evidence_f1 is None
    assert blank.abstention_kappa == pytest.approx(1.0)


def test_agreement_over_several_sessions() -> None:
    """Units of every session are pooled; sessions are counted."""
    sets = [
        aset("a", VALID), aset("b", VALID),
        aset("a", VALID, "s2"),
        aset("b", changed(("Tom", "planning"), (5, [(4, "P2")])), "s2"),
    ]
    report = agreement(sets, SCALE)
    assert (report.sessions, report.units) == (2, 12)
    assert report.alpha is not None and 0.5 < report.alpha < 1.0
    planning = report.per_competency[1]
    assert planning.units == 6
    assert planning.alpha is not None and planning.alpha < 1.0
    assert report.per_competency[0].alpha == pytest.approx(1.0)


def test_agreement_rejects_a_unit_annotated_twice_by_one_annotator() -> None:
    """Two files of one annotator for one session are an error."""
    with pytest.raises(InvalidInputError, match="a annotated s1/Maya/"):
        agreement([aset("a", VALID), aset("a", VALID), aset("b", VALID)],
                  SCALE)


def test_agreement_off_scale_rating_is_a_clean_error() -> None:
    """An unvalidated form with a rating of 9 gives a domain error."""
    with pytest.raises(RagError, match="Rating.s. 9 are off the 1-5 scale"):
        agreement([rated("a", [9, 2]), rated("b", [1, 2])], SCALE)


def test_check_on_scale() -> None:
    """Unrated cells pass; every off-scale value is named once, sorted."""
    check_on_scale([1, 5, None, 3], SCALE)
    check_on_scale([], SCALE)
    with pytest.raises(InvalidInputError, match="Rating.s. 0, 7 are off"):
        check_on_scale([7, 2, 0, 7, None], SCALE)


# ------------------------------------------------------------ consensus


@pytest.mark.parametrize("values, expected", [
    ([3], 3), ([2, 5], 2), ([5, 2], 2), ([3, 4, 5], 4), ([1, 2, 4, 5], 2),
    ([4, 4, 1], 4),
])
def test_median_low(values: List[int], expected: int) -> None:
    """The median, or the lower middle value of an even count."""
    assert median_low(values) == expected


def test_consensus_needs_annotations() -> None:
    """No set at all is an error."""
    with pytest.raises(InvalidInputError, match="No annotation set"):
        consensus([])


def test_consensus_of_one_annotator_is_their_own_form() -> None:
    """A single annotator is their own majority."""
    result, = consensus([aset("a", VALID)])
    assert result.annotator == "consensus"
    assert (result.session_id, result.catalogue_version) == ("s1", "1.0")
    original = aset("a", VALID).by_unit()
    for key, item in result.by_unit().items():
        assert item.rating == original[key].rating
        assert item.turns() == original[key].turns()


def test_consensus_rates_when_a_majority_rated() -> None:
    """More than half must rate; the rating is their low median."""
    cells = merged([
        rated("a", [4, 4, 2, N, 3]), rated("b", [5, N, 5, N, 4]),
        rated("c", [2, N, N, N, 5]),
    ])
    ratings = [cells[(f"p{i:02d}", "listening")].rating for i in range(5)]
    assert ratings == [4, None, 2, None, 4]


def test_consensus_two_annotators_one_abstains() -> None:
    """One of two is not a majority: the unit stays unrated."""
    cells = merged([rated("a", [4, 2]), rated("b", [N, 5])])
    assert cells[("p00", "listening")].rating is None
    assert cells[("p01", "listening")].rating == 2


def test_consensus_evidence_needs_a_majority_of_annotators() -> None:
    """A turn is kept when more than half marked it."""
    key = ("Maya", "listening")
    cells = merged([
        aset("a", {key: (4, [(0, "L1"), (3, "L2")])}),
        aset("b", {key: (4, [(0, "L1"), (6, "L2")])}),
        aset("c", {key: (4, [(0, "L2"), (3, "L2")])}),
    ])
    assert [(e.turn, e.indicator_id) for e in cells[key].evidence] == [
        (0, "L1"), (3, "L2"),
    ]


def test_consensus_evidence_of_two_annotators_must_be_shared() -> None:
    """With two annotators only the turns both marked survive."""
    key = ("Maya", "listening")
    cells = merged([
        aset("a", {key: (4, [(0, "L1"), (3, "L2")])}),
        aset("b", {key: (4, [(3, "L2")])}),
    ])
    assert cells[key].turns() == {3}


def test_consensus_indicator_tie_takes_the_first_id() -> None:
    """When annotators split on the indicator the lower id is kept."""
    key = ("Maya", "listening")
    cells = merged([
        aset("a", {key: (4, [(0, "L2")])}), aset("b", {key: (4, [(0, "L1")])}),
    ])
    assert [(e.turn, e.indicator_id) for e in cells[key].evidence] == [
        (0, "L1"),
    ]


def test_consensus_counts_a_turn_once_per_annotator() -> None:
    """One annotator marking a turn under two indicators is one vote."""
    key = ("Maya", "listening")
    cells = merged([
        aset("a", {key: (4, [(0, "L1"), (0, "L2")])}),
        aset("b", {key: (4, [(3, "L2")])}),
        aset("c", {key: (4, [(3, "L2")])}),
    ])
    assert cells[key].turns() == {3}


def test_consensus_majority_is_over_those_who_have_the_cell() -> None:
    """An annotator without the cell does not vote on it."""
    key = ("Maya", "listening")
    cells = merged([
        aset("a", {key: (4, [(0, "L1")])}),
        aset("b", {("Tom", "listening"): (2, [(1, "L3")])}),
    ])
    assert cells[key].rating == 4
    assert cells[key].turns() == {0}
    assert cells[("Tom", "listening")].rating == 2


def test_consensus_rejects_mixed_catalogue_versions() -> None:
    """One session annotated with two catalogue versions is an error."""
    with pytest.raises(InvalidInputError, match="different catalogue"):
        consensus([aset("a", VALID), aset("b", VALID, version="2.0")])


def test_consensus_versions_may_differ_between_sessions() -> None:
    """Each session keeps the catalogue version it was annotated with."""
    first, second = consensus([
        aset("a", VALID), aset("b", VALID),
        aset("a", VALID, "s2", "2.0"), aset("b", VALID, "s2", "2.0"),
    ], annotator="gold")
    assert (first.session_id, first.catalogue_version) == ("s1", "1.0")
    assert (second.session_id, second.catalogue_version) == ("s2", "2.0")
    assert first.annotator == second.annotator == "gold"
    assert len(first.annotations) == len(second.annotations) == 6


def test_consensus_rejects_the_same_annotator_twice() -> None:
    """A duplicated file must not count as two independent votes."""
    key = ("Maya", "listening")
    with pytest.raises(InvalidInputError, match="same annotator"):
        consensus([
            aset("a", {key: (5, [(0, "L1")])}),
            aset("a", {key: (5, [(0, "L1")])}),
            aset("b", {key: (None, [])}),
        ])


def test_evidence_may_point_at_a_sentence_of_its_turn() -> None:
    """A sentence id narrows the evidence; it must belong to the turn."""
    form = blank_annotations(TRANSCRIPT, CATALOGUE, "A1")
    cell = next(
        a for a in form.annotations
        if (a.participant, a.competency_id) == ("Tom", "listening")
    )
    cell.rating = 2
    cell.evidence = [
        AnnotatedEvidence(turn=1, indicator_id="L3", sentence="t1.s0"),
    ]
    assert validate_annotations(form, TRANSCRIPT, CATALOGUE) == []
    assert cell.turns() == {1}
    cell.evidence = [
        AnnotatedEvidence(turn=1, indicator_id="L3", sentence="t4.s0"),
        AnnotatedEvidence(turn=1, indicator_id="L3", sentence="t1.s9"),
        AnnotatedEvidence(turn=1, indicator_id="L3", sentence="nonsense"),
    ]
    assert validate_annotations(form, TRANSCRIPT, CATALOGUE) == [
        "Tom/listening: sentence 't4.s0' is not a sentence of turn 1",
        "Tom/listening: sentence 't1.s9' is not a sentence of turn 1",
        "Tom/listening: sentence 'nonsense' is not a sentence of turn 1",
    ]


def test_forms_without_sentences_stay_valid_and_serialise_as_before() -> None:
    """The field is optional: existing forms are unchanged."""
    evidence = AnnotatedEvidence(turn=1, indicator_id="L3")
    assert evidence.sentence is None
    assert AnnotatedEvidence.model_validate(
        {"turn": 1, "indicator_id": "L3"}
    ) == evidence
