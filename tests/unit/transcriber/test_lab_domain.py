"""The lab's plain data: catalogue, requests, runs, keys, labels, scores."""

from datetime import datetime, timezone
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
        }))
        assert merged_params(model, {"threshold": 0.7}, {
            "num_speakers": 2, "language": "es",
        }) == {"num_speakers": 2, "threshold": 0.7}
        assert merged_params(model, {}, {"num_speakers": None}) == {
            "num_speakers": None, "threshold": 0.5,
        }
        with pytest.raises(InvalidInputError, match="no parameter"):
            merged_params(model, {"other": 1}, {})


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
