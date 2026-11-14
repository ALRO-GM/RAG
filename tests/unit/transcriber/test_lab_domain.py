"""The lab's plain data: catalogue, requests, runs, keys, labels, scores."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pytest
from pydantic import ValidationError

from transcriber.domain.errors import InvalidInputError
from transcriber.domain.lab import (
    Catalogue,
    CatalogueEntry,
    LabRequest,
    LabRun,
    PipelineRef,
    ReferenceLabel,
    Segment,
    SpeakerMapping,
    Step,
    check_labels,
    check_override,
    merged_params,
    named_segments,
    stage_key,
)
from transcriber.domain.lab_scoring import rank_scores, score_pipeline
from transcriber.domain.models import Word

NOW = datetime(2026, 11, 14, tzinfo=timezone.utc)


def entry(model_id: str, stage: str = "diarize", **extra: Any) -> Dict[
    str, Any
]:
    """A catalogue entry as JSON."""
    return {"id": model_id, "stage": stage, "family": "fam",
            "display_name": model_id, **extra}


class TestCatalogue:
    def test_lookup_by_stage(self) -> None:
        catalogue = Catalogue.model_validate({
            "schema": "lab.catalogue.v1",
            "entries": [entry("a"), entry("b", "recognise")],
        })
        assert catalogue.require("a", "diarize").id == "a"
        assert catalogue.get("zz") is None
        with pytest.raises(InvalidInputError):
            catalogue.require("b", "diarize")

    @pytest.mark.parametrize("document", [
        {"schema": "lab.catalogue.v2", "entries": []},
        {"schema": "lab.catalogue.v1", "entries": [entry("a"), entry("a")]},
        {"schema": "lab.catalogue.v1", "entries": [entry("A b")]},
    ])
    def test_invalid_catalogues(self, document: Dict[str, Any]) -> None:
        with pytest.raises(ValidationError):
            Catalogue.model_validate(document)

    def test_vote_group_defaults_to_the_id(self) -> None:
        assert CatalogueEntry.model_validate(entry("a")).vote_group == "a"
        assert CatalogueEntry.model_validate(
            entry("a", lineage="pyannote")
        ).vote_group == "pyannote"

    def test_parameters(self) -> None:
        model = CatalogueEntry.model_validate(entry("a", parameters={
            "num_speakers": None, "threshold": 0.5,
        }, types={"num_speakers": "integer"}))
        assert merged_params(model, {"threshold": 0.7}, {
            "num_speakers": 2, "language": "es",
        }) == {"num_speakers": 2, "threshold": 0.7}
        assert merged_params(model, {}, {"num_speakers": None}) == {
            "num_speakers": None, "threshold": 0.5,
        }
        with pytest.raises(InvalidInputError, match="no parameter"):
            merged_params(model, {"other": 1}, {})
        with pytest.raises(InvalidInputError, match="greater than 0"):
            merged_params(model, {"threshold": 0}, {})


TYPED = {
    "piece_seconds": 10.0, "batch_size": 8, "vad_filter": False,
    "mode": "fast", "num_speakers": None, "hop": 0.75, "shift": 0.0,
    "cuts": [1.0], "speakers": None, "gain": None, "lang": None,
    "flag": None,
}
#: The declared types of the null defaults of :data:`TYPED`.
NULL_TYPES = {
    "num_speakers": "integer", "speakers": "string_list",
    "gain": "number", "lang": "string", "flag": "boolean",
}


class TestOverrides:
    """An override has the default's type; a number is positive."""

    @pytest.fixture()
    def model(self) -> CatalogueEntry:
        """A model with one parameter of each kind; ``hop`` >= 0.01."""
        return CatalogueEntry.model_validate(entry(
            "m", parameters=TYPED, types=NULL_TYPES,
            minimums={"hop": 0.01, "shift": -3.0},
        ))

    @pytest.mark.parametrize("name, value", [
        ("piece_seconds", 2.5), ("piece_seconds", 3), ("batch_size", 1),
        ("vad_filter", True), ("mode", "slow"), ("num_speakers", 2),
        ("num_speakers", None), ("speakers", ["ana", "luis"]),
        ("speakers", []), ("speakers", None), ("gain", 1.5), ("gain", 2),
        ("lang", "es"), ("flag", True), ("flag", None), ("hop", 0.01),
        ("shift", 0.0), ("shift", -3.0), ("shift", 2),
    ])
    def test_accepted(
        self, model: CatalogueEntry, name: str, value: Any
    ) -> None:
        assert check_override(model, name, value) == value
        assert merged_params(model, {name: value}, {})[name] == value

    @pytest.mark.parametrize("name, value, error", [
        ("piece_seconds", 0, "greater than 0"),
        ("piece_seconds", 0.0, "greater than 0"),
        ("piece_seconds", -1.0, "greater than 0"),
        ("piece_seconds", float("inf"), "finite"),
        ("piece_seconds", float("nan"), "finite"),
        ("piece_seconds", True, "must be a number, not a boolean"),
        ("piece_seconds", "10", "must be a number, not a string"),
        ("piece_seconds", None, "must be a number, not null"),
        ("batch_size", 8.0, "must be an integer, not a number"),
        ("batch_size", False, "must be an integer, not a boolean"),
        ("batch_size", 0, "greater than 0"),
        ("vad_filter", 1, "must be a boolean, not an integer"),
        ("vad_filter", "yes", "must be a boolean"),
        ("mode", 1, "must be a string, not an integer"),
        ("mode", ["fast"], "must be a string, not a list"),
        ("num_speakers", 0, "greater than 0"),
        ("num_speakers", -2, "greater than 0"),
        ("num_speakers", "auto", "must be null or an integer, not a string"),
        ("num_speakers", 1.5, "must be null or an integer, not a number"),
        ("num_speakers", True, "must be null or an integer, not a boolean"),
        ("num_speakers", {"a": 1}, "not an object"),
        ("num_speakers", [1, 2], "not a list"),
        ("speakers", [1, 2], "must be null or a list of strings, not a list"),
        ("speakers", "ana", "must be null or a list of strings"),
        ("gain", 0, "greater than 0"),
        ("gain", float("nan"), "finite"),
        ("gain", "loud", "must be null or a number, not a string"),
        ("lang", 1, "must be null or a string, not an integer"),
        ("flag", 1, "must be null or a boolean, not an integer"),
        ("hop", 0.005, "at least 0.01"),
        ("hop", 0, "at least 0.01"),
        ("shift", -3.5, "at least -3"),
        ("cuts", [2.0], "must be a list"),
        ("other", 1, "no parameter"),
    ])
    def test_refused(
        self, model: CatalogueEntry, name: str, value: Any, error: str
    ) -> None:
        with pytest.raises(InvalidInputError, match=error):
            check_override(model, name, value)
        if name != "other":
            with pytest.raises(InvalidInputError, match=error):
                merged_params(model, {name: value}, {})

    def test_standard_values_are_not_overrides(
        self, model: CatalogueEntry
    ) -> None:
        """The service's own values pass through as they are."""
        assert merged_params(model, {}, {"num_speakers": 3})[
            "num_speakers"
        ] == 3

    @pytest.mark.parametrize("minimums", [
        {"other": 1.0}, {"hop": float("inf")},
    ])
    def test_invalid_minimums(self, minimums: Dict[str, float]) -> None:
        with pytest.raises(ValidationError):
            CatalogueEntry.model_validate(entry(
                "m", parameters=TYPED, types=NULL_TYPES, minimums=minimums,
            ))

    @pytest.mark.parametrize("types, error", [
        ({k: v for k, v in NULL_TYPES.items() if k != "gain"},
         "no declared type for null parameter gain"),
        ({**NULL_TYPES, "hop": "number"}, "without a null default: hop"),
        ({**NULL_TYPES, "other": "number"},
         "without a null default: other"),
        ({**NULL_TYPES, "gain": "decibels"}, "gain"),
    ])
    def test_null_defaults_need_a_declared_type(
        self, types: Dict[str, str], error: str
    ) -> None:
        with pytest.raises(ValidationError, match=error):
            CatalogueEntry.model_validate(entry(
                "m", parameters=TYPED, types=types,
            ))

    def test_the_shipped_catalogue_types_its_null_defaults(self) -> None:
        """Every null default of the shipped catalogue has a type."""
        catalogue = Catalogue.model_validate_json(
            (Path(__file__).parents[3] / "lab" / "catalogue.json")
            .read_text(encoding="utf-8")
        )
        pyannote = catalogue.require("pyannote-3.1", "diarize")
        with pytest.raises(InvalidInputError, match="an integer"):
            check_override(pyannote, "max_speakers", "many")
        with pytest.raises(InvalidInputError, match="an integer"):
            check_override(pyannote, "num_speakers", 2.5)
        deepfilter = catalogue.require("deepfilternet3", "denoise")
        with pytest.raises(InvalidInputError, match="a number"):
            check_override(deepfilter, "atten_lim_db", [1])
        assert check_override(deepfilter, "atten_lim_db", 12) == 12
        builtin = catalogue.require("builtin", "diarize")
        assert check_override(builtin, "expected_speakers", ["ana"]) == [
            "ana"
        ]

    def test_the_catalogue_bounds_the_lab_diarizer_steps(self) -> None:
        """The shipped catalogue refuses a hop below one 10 ms frame."""
        catalogue = Catalogue.model_validate_json(
            (Path(__file__).parents[3] / "lab" / "catalogue.json")
            .read_text(encoding="utf-8")
        )
        ecapa = catalogue.require("ecapa-spectral", "diarize")
        with pytest.raises(InvalidInputError, match="at least 0.01"):
            check_override(ecapa, "hop", 0.005)
        sepformer = catalogue.require("sepformer-dns4", "denoise")
        with pytest.raises(InvalidInputError, match="greater than 0"):
            check_override(sepformer, "piece_seconds", 0)


class TestRequestsAndRuns:
    def test_defaults_and_validation(self) -> None:
        request = LabRequest()
        assert request.denoisers == ["none"]
        assert request.diarizers == ["builtin"]
        cases: List[Dict[str, Any]] = [
            {"diarizers": ["a", "a"]}, {"diarizers": [], "recognisers": []},
            {"diarizers": ["Bad id"]}, {"params": {"Bad": {}}},
            {"params": {"a": {"x": "y" * 3000}}}, {"other": 1},
        ]
        for bad in cases:
            with pytest.raises(ValidationError):
                LabRequest.model_validate(bad)

    def test_pipelines_and_their_ids(self) -> None:
        run = LabRun(
            id="0" * 32, session_id="s", created_at=NOW,
            request=LabRequest(denoisers=["none", "dn"],
                               diarizers=["z"], recognisers=["r1", "r2"]),
        )
        pipelines = run.pipelines()
        assert [p.id for p in pipelines] == [
            "none+z+r1", "none+z+r2", "dn+z+r1", "dn+z+r2",
        ]
        assert pipelines[2].track == "dn+z"
        assert pipelines[2].transcript == "dn+r1"
        alone = PipelineRef(denoiser="none", recogniser="r")
        assert (alone.id, alone.track) == ("none+-+r", None)
        assert PipelineRef(denoiser="none").transcript is None

    def test_steps_by_name(self) -> None:
        steps = [Step(stage="denoise", model="dn", denoiser="dn"),
                 Step(stage="diarize", model="z", denoiser="dn")]
        run = LabRun(id="0" * 32, session_id="s", created_at=NOW,
                     request=LabRequest(), steps=steps)
        assert [s.name for s in steps] == ["denoise:dn", "diarize:dn+z"]
        changed = run.with_step(steps[1].model_copy(update={"state": "done"}))
        assert changed.step("diarize:dn+z") is not None
        assert changed.steps[1].state == "done"
        assert run.step("nothing") is None

    def test_stage_keys_depend_on_every_input(self) -> None:
        key = stage_key("a" * 64, "diarize", "z", {"n": 2})
        assert len(key) == 40
        assert key == stage_key("a" * 64, "diarize", "z", {"n": 2})
        others = {
            stage_key("b" * 64, "diarize", "z", {"n": 2}),
            stage_key("a" * 64, "recognise", "z", {"n": 2}),
            stage_key("a" * 64, "diarize", "y", {"n": 2}),
            stage_key("a" * 64, "diarize", "z", {"n": 3}),
            stage_key("a" * 64, "diarize", "z", {"n": 2}, {"v": 1}),
        }
        assert key not in others and len(others) == 5

    def test_named_segments(self) -> None:
        segments = [Segment(start=0, end=1, speaker="S0"),
                    Segment(start=1, end=2, speaker="S1"),
                    Segment(start=2, end=3, speaker="S9")]
        mapping = [SpeakerMapping(label="S0", speaker="ana",
                                  method="voiceprint"),
                   SpeakerMapping(label="S1")]
        assert [s.speaker for s in named_segments(segments, mapping)] == [
            "ana", "~S1", "~S9",
        ]
        with pytest.raises(ValidationError):
            Segment(start=2, end=1, speaker="x")


def label(start: float, end: float, kind: str, *speakers: str,
          text: Any = None) -> ReferenceLabel:
    """A reference label."""
    return ReferenceLabel(start=start, end=end, kind=kind,  # type: ignore
                          speakers=list(speakers), text=text)


class TestLabels:
    @pytest.mark.parametrize("values", [
        {"start": 2, "end": 1, "kind": "nobody"},
        {"start": 0, "end": 1, "kind": "person"},
        {"start": 0, "end": 1, "kind": "person", "speakers": ["a", "b"]},
        {"start": 0, "end": 1, "kind": "both", "speakers": ["a"]},
        {"start": 0, "end": 1, "kind": "nobody", "speakers": ["a"]},
        {"start": 0, "end": 1, "kind": "both", "speakers": ["a", "a"]},
        {"start": 0, "end": 1, "kind": "person", "speakers": ["../x"]},
        {"start": 0, "end": 1, "kind": "maybe"},
    ])
    def test_invalid_labels(self, values: Dict[str, Any]) -> None:
        with pytest.raises(ValidationError):
            ReferenceLabel.model_validate(values)

    def test_a_set_of_labels(self) -> None:
        labels = [label(5, 6, "nobody"), label(0, 5, "person", "ana")]
        assert [lb.start for lb in check_labels(labels, 10, ["ana"])] == [
            0, 5,
        ]
        assert label(0, 1, "unsure").reference() is None
        assert label(0, 1, "both", "luis", "ana").reference() == (
            "ana", "luis",
        )
        with pytest.raises(InvalidInputError, match="overlap"):
            check_labels([label(0, 5, "nobody"), label(4, 6, "nobody")],
                         10, [])
        with pytest.raises(InvalidInputError, match="after the recording"):
            check_labels([label(0, 12, "nobody")], 10, [])
        with pytest.raises(InvalidInputError, match="Not enrolled"):
            check_labels([label(0, 1, "person", "eve")], 10, ["ana"])
        with pytest.raises(InvalidInputError, match="At most"):
            check_labels([label(n, n + 0.5, "nobody")
                          for n in range(5001)], 1e9, [])


def word(text: str, start: float, end: float,
         speaker: Any = None) -> Word:
    """A word."""
    return Word(text=text, start=start, end=end, speaker=speaker)


class TestScoring:
    LABELS = [
        label(0, 4, "person", "ana", text="hola qué tal"),
        label(4, 6, "both", "ana", "luis"),
        label(6, 8, "nobody"),
        label(8, 10, "unsure"),
    ]

    def test_a_perfect_pipeline(self) -> None:
        segments = [Segment(start=0, end=6, speaker="ana"),
                    Segment(start=4, end=6, speaker="luis")]
        heard = [word("Hola,", 0.5, 1, "ana"), word("qué", 1, 1.5, "ana"),
                 word("tal", 1.5, 2, "ana"), word("sí", 4.2, 4.5, "luis")]
        score = score_pipeline("p", segments, heard, self.LABELS, 10.0)
        assert score.labelled_seconds == pytest.approx(8.0)
        assert score.reference_speech == pytest.approx(8.0)
        assert score.error_rate == pytest.approx(0.0)
        assert score.overlap_detected == pytest.approx(1.0)
        assert score.overlap_exact == pytest.approx(1.0)
        assert (score.words_counted, score.words_right) == (4, 4)
        assert score.wer == pytest.approx(0.0)
        assert score.text_labels == 1

    def test_mistakes_of_every_kind(self) -> None:
        # luis named where ana spoke (0-2), nobody 2-4, only ana in the
        # overlap, speech in the silence (6-7); unsure time ignored.
        segments = [Segment(start=0, end=2, speaker="luis"),
                    Segment(start=4, end=7, speaker="ana"),
                    Segment(start=8, end=10, speaker="ana")]
        heard = [word("hola", 0.5, 1, "luis"), word("que", 4.2, 4.5, "ana"),
                 word("eh", 6.2, 6.5, "ana")]
        score = score_pipeline("p", segments, heard, self.LABELS, 10.0)
        assert score.confusion == pytest.approx(2.0)
        assert score.missed == pytest.approx(2.0 + 2.0)
        assert score.false_alarm == pytest.approx(1.0)
        assert score.error_rate == pytest.approx(7.0 / 8.0)
        assert score.overlap_detected == pytest.approx(0.0)
        assert (score.words_counted, score.words_right) == (2, 1)
        assert score.words_in_silence == 1
        assert score.wer == pytest.approx(2 / 3)

    def test_without_labels_nothing_is_measured(self) -> None:
        score = score_pipeline("p", [], [], [], 10.0)
        assert score.error_rate is None and score.wer is None
        assert score.words_right_share is None
        assert score.overlap_detected is None

    def test_ranking(self) -> None:
        segments = [Segment(start=0, end=6, speaker="ana"),
                    Segment(start=4, end=6, speaker="luis")]
        good = score_pipeline("good", segments, [], self.LABELS, 10.0)
        bad = score_pipeline("bad", [], [], self.LABELS, 10.0)
        same = score_pipeline("same", segments, [], self.LABELS, 10.0)
        unknown = score_pipeline("none", [], [], [], 10.0)
        ranked = rank_scores([bad, unknown, good, same])
        assert [(s.pipeline, s.rank) for s in ranked] == [
            ("good", 1), ("same", 1), ("bad", 3), ("none", 4),
        ]
