"""The study page's data builder: pure parts, on small hand-made inputs."""

from datetime import date
from typing import Any, Dict, List

import pytest

from transcriber.adapters.inbound.study_data import (
    DOCUMENTED, SOURCES, Documented, StudyDataError, StudyInputs,
    build_study, corpus_rows, denoiser_effect, documented_figures,
    events_block, find_section, gain, headline, judges, licences, slug,
    switch_block, tagger_block,
)


def metric(mean: float) -> Dict[str, Any]:
    """One aggregated metric as the reference summary writes it."""
    return {"n": 4, "mean": mean, "sd": 0.01, "min": mean, "max": mean}


def model(
    model_id: str, der: float, kind: str = "diarization",
) -> Dict[str, Any]:
    """One summary row with only a DER and its parts."""
    return {
        "kind": kind, "id": model_id, "recordings": 4,
        "metrics": {
            "der": metric(der), "missed_rate": metric(der / 2),
            "false_alarm_rate": metric(der / 4),
            "confusion_rate": metric(der / 4), "jer": metric(der),
            "speaker_count_error": metric(1.0),
        },
    }


SUMMARY = {
    "non_commercial": False,
    "recordings": [{"licence": "CC BY 4.0", "duration": 600.0},
                   {"licence": "CC BY 4.0", "duration": 300.0}],
    "excluded": [{"reason": "no speech in the audio"}],
    "models": [
        model("none+builtin", 0.5),
        model("none+fancy", 0.1),
        model("noisy+fancy", 0.3),
        model("noisy+builtin", 0.6),
        model("none+fancy+asr", 0.1, kind="pipeline"),
    ],
}
COMMERCIAL = {"fancy": False, "builtin": True, "noisy": True, "none": True}


def test_github_anchors() -> None:
    assert slug("13.8 Results of #44 (2026-11-15)") == \
        "138-results-of-44-2026-11-15"
    assert slug("Full run with `whisper` (run `e7`)") == \
        "full-run-with-whisper-run-e7"
    assert slug("4.2 Local judge") == "42-local-judge"


def test_a_section_runs_to_the_next_heading_of_its_level() -> None:
    text = ("# Top\n## 1. One\nalpha\n### 1.1 Sub\nbeta\n```\n# not a "
            "heading\n```\ngamma\n## 2. Two\ndelta\n")
    title, body = find_section(text, "1. One")
    assert title == "1. One"
    assert "beta" in body and "gamma" in body and "delta" not in body
    assert find_section(text, "1.1")[1].strip().startswith("beta")
    with pytest.raises(StudyDataError, match="No heading"):
        find_section(text, "3. Three")


def test_documented_figures_must_still_be_in_their_section() -> None:
    bodies = {key: "" for key in SOURCES}
    with pytest.raises(StudyDataError, match="no longer in"):
        documented_figures(bodies)
    item = Documented("x", 1, "lab_r1", "the  answer\nis 1")
    assert item.quote  # whitespace is folded before comparing
    assert all(d.source in SOURCES for d in DOCUMENTED)


def test_gain_in_points_and_error_reduction() -> None:
    assert gain(0.1, 0.5) == {"points": 0.4, "error_reduction": 0.8}
    assert gain(0.0, 0.0) == {"points": 0.0, "error_reduction": None}


def test_rows_sorted_flagged_and_compared_with_the_best() -> None:
    rows = corpus_rows(SUMMARY, COMMERCIAL)
    assert [r["id"] for r in rows] == [
        "none+fancy", "noisy+fancy", "none+builtin", "noisy+builtin"]
    best = rows[0]
    assert best["best"] and best["non_commercial"] and best["rank"] == 1
    assert best["success"] == 0.9 and best["gain"]["points"] == 0
    builtin = [r for r in rows if r["builtin"]]
    assert [r["id"] for r in builtin] == ["none+builtin"]
    assert builtin[0]["gain"] == {"points": 0.4, "error_reduction": 0.8}
    assert rows[1]["missed"] == 0.15
    with pytest.raises(StudyDataError, match="no diarization"):
        corpus_rows({"models": []}, COMMERCIAL)


def test_headline_names_the_built_in_next_and_best_commercial() -> None:
    rows = corpus_rows(SUMMARY, COMMERCIAL)
    head = headline(rows)
    assert head["best"] == "none+fancy"
    assert head["builtin"]["id"] == "none+builtin"
    assert head["next"]["id"] == "noisy+fancy"
    assert head["best_commercial"]["id"] == "none+builtin"
    commercial_best = headline(corpus_rows(SUMMARY, {}))
    assert commercial_best["best_commercial"] is None
    single = headline(rows[:1])
    assert single["next"] is None and single["builtin"] is None


def test_the_denoiser_effect_is_the_mean_over_diarizers() -> None:
    effect = denoiser_effect(corpus_rows(SUMMARY, COMMERCIAL))
    assert [(e["denoiser"], e["success"]) for e in effect] == [
        ("none", 0.7), ("noisy", 0.55)]


def test_licences_from_the_catalogue() -> None:
    catalogue = {"entries": [{"id": "a", "licence_commercial": False},
                             {"id": "b"}]}
    assert licences(catalogue) == {"a": False, "b": True}


def test_pending_work_until_its_results_exist() -> None:
    assert switch_block(None) == {"issue": 68, "status": "pending",
                                  "corpora": {}}
    done = switch_block({"corpora": {"ami": {"der": 0.1, "n": 4}}})
    assert done["status"] == "done"
    assert done["corpora"]["ami"] == {"der": 0.1, "success": 0.9, "n": 4}
    assert tagger_block([])["status"] == "pending"
    tagger = tagger_block([
        {"approach": "rules", "items": 10, "top1_accuracy": 0.4},
        {"approach": "encoder", "items": 10, "top1_accuracy": 0.6},
    ])
    assert tagger["status"] == "done"
    assert [a["approach"] for a in tagger["approaches"]] == [
        "encoder", "rules"]


FIGURES = {
    "judge.acts_items": 200.0, "judge.qwen3-8b.accuracy": 0.62,
    "judge.qwen3-1.7b.accuracy": 0.365, "judge.qwen3-1.7b.confirmed": 40,
    "judge.qwen3-1.7b.rejected": 23, "judge.qwen3-4b.accuracy": 0.545,
    "judge.qwen3-4b.confirmed": 63, "judge.qwen3-4b.rejected": 1,
    "judge.claude.right": 14, "judge.claude.items": 20,
}
QWEN8B = {"bits": 4, "acts_items": 200, "acts_accuracy": 0.62,
          "tags_confirmed": 58, "tags_rejected": 6}


def test_judges_in_order_with_the_small_sample_flagged() -> None:
    out = judges(FIGURES, QWEN8B)
    assert [j["model"] for j in out] == [
        "Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B", "Claude Code (opus)"]
    assert out[-1]["accuracy"] == 0.7 and out[-1]["small_sample"]
    assert not any(j["small_sample"] for j in out[:-1])


@pytest.mark.parametrize("change", [{"acts_items": 100},
                                    {"acts_accuracy": 0.5}])
def test_the_result_file_must_agree_with_the_document(
    change: Dict[str, Any],
) -> None:
    with pytest.raises(StudyDataError, match="disagree"):
        judges(FIGURES, {**QWEN8B, **change})


def test_event_totals_and_micro_precision_recall() -> None:
    rows: List[Dict[str, Any]] = [
        {"type": "a", "expected": 4, "derived": 2, "matched": 2,
         "precision": 1.0, "recall": 0.5},
        {"type": "b", "expected": 0, "derived": 2, "matched": 0,
         "precision": 0.0, "recall": None},
    ]
    block = events_block({"dialogues": ["x", "y"], "given_links": rows,
                          "inferred_links": []})
    assert block["dialogues"] == 2
    given = block["given_links"]
    assert (given["expected"], given["derived"]) == (4, 4)
    assert (given["precision"], given["recall"]) == (0.5, 0.5)
    assert block["inferred_links"]["precision"] is None


def test_missing_inputs_are_refused() -> None:
    with pytest.raises(StudyDataError, match="Missing document"):
        build_study(StudyInputs({}, {}, {}, {}, {}), date(2026, 11, 15))
