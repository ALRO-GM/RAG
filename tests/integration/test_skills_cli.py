"""The soft-skills commands run as ``python -m src skills ...``.

Only commands that need no model run here; ``assess``, ``evidence``
and ``lab experiment`` are exercised in-process with fakes in
``test_skills_container.py``.
"""

import json
import subprocess
from pathlib import Path
from typing import List

import pytest

from src.application.skills.gold import AnnotationSet
from src.domain.models import AnsweredQuestion, RagDataset
from src.domain.skills.assessment import (
    CompetencyAssessment,
    Evidence,
    ParticipantScorecard,
    SessionAssessment,
)
from src.domain.skills.interaction import describe
from tests.fakes import make_transcript
from tests.integration.test_cli import run_cli
from tests.skills_samples import (
    EN,
    ES,
    GOLD,
    RUBRICS,
    write_form,
    write_rubrics,
    write_transcripts,
)

Result = "subprocess.CompletedProcess[str]"


def skills(args: List[str], cwd: Path) -> "subprocess.CompletedProcess[str]":
    """Run ``python -m src skills`` with the tmp catalogue and outputs."""
    return run_cli(
        ["skills", "--rubric_dir", RUBRICS, "--output_dir", "out",
         "--lab_dir", "lab", *args],
        cwd,
    )


def assert_error(
    result: "subprocess.CompletedProcess[str]", *fragments: str,
    lines: int = 1,
) -> None:
    """Check a clean failure: ``Error: ...`` on stderr, exit code 1."""
    assert result.returncode == 1, result.stderr
    assert result.stderr.startswith("Error: "), result.stderr
    assert "Traceback" not in result.stderr
    assert len(result.stderr.strip().splitlines()) == lines, result.stderr
    assert result.stdout == ""
    for fragment in fragments:
        assert fragment in result.stderr, result.stderr


def system_assessment() -> SessionAssessment:
    """Return an assessment of the sample meeting, as ``assess`` saves."""
    transcript = make_transcript()
    team = describe(transcript)

    def cell(
        competency: str, rating: int, turns: List[int]
    ) -> CompetencyAssessment:
        """Return a rated competency with evidence on ``turns``."""
        return CompetencyAssessment(
            competency_id=competency, status="rated", rating=rating,
            evidence=[Evidence(turn=t, quote="q", indicator_id="L1",
                               polarity="positive") for t in turns],
            candidate_turns=turns, proposed=len(turns) + 1, rejected=1,
        )

    return SessionAssessment(
        session_id="s1", language="en", catalogue="core",
        catalogue_version="1.0", judge_model="fake", embedding_model="",
        min_incidents=2, team=team,
        scorecards=[
            ParticipantScorecard(
                participant="Maya", interaction=team.of("Maya"),
                assessments=[cell("listening", 4, [0, 3])],
            ),
            ParticipantScorecard(
                participant="Tom", interaction=team.of("Tom"),
                assessments=[cell("planning", 4, [4])],
            ),
        ],
    )


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write catalogues, transcripts, annotations and broken files once."""
    root = tmp_path_factory.mktemp("skills")
    write_rubrics(root)
    write_transcripts(root)
    write_form(root / "ann/a.json", "a")
    write_form(root / "ann/b.json", "b", {
        **GOLD, ("Tom", "planning"): (4, [(4, "P2")]),
    })
    write_form(root / "ann/a_s2.json", "a", session="s2")
    write_form(root / "ann/v2.json", "c", version="2.0")
    write_form(root / "ann/off.json", "c", {
        **GOLD, ("Tom", "planning"): (9, [(4, "P2")]),
    })
    write_form(root / "ann/wrong.json", "c", {
        ("Maya", "listening"): (7, []),
        ("Zoe", "listening"): (None, []),
        ("Tom", "charisma"): (None, []),
        ("Tom", "listening"): (2, [(0, "P1"), (99, "L3")]),
    })
    write_form(root / "ann/blank.json", "c", {
        key: (None, []) for key in GOLD
    })
    write_form(root / "ann/other.json", "c", session="s9")
    (root / "empty.txt").write_text("")
    (root / "empty.json").write_text("")
    (root / "bad.json").write_text("{not valid json")
    (root / "shape.json").write_text(json.dumps({"hello": [1, 2]}))
    (root / "shape_pretty.json").write_text(
        json.dumps({"hello": [1, 2]}, indent=2)
    )
    (root / "binary.bin").write_bytes(b"\xff\xfe\x00\x01\x80\x81binary")
    (root / "notes.txt").write_text("nothing but notes\n")
    (root / "system.json").write_text(
        system_assessment().model_dump_json(), encoding="utf-8"
    )
    broken = root / "broken"
    broken.mkdir()
    (broken / "core.en.json").write_text('{"name": "core"}')
    return root


# ------------------------------------------------------------ catalogue


def test_catalogue_lists_competencies_and_indicators(project: Path) -> None:
    """The catalogue is printed with polarity signs and exclusions."""
    result = skills(["catalogue", "--language", "en"], project)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[0] == "core 1.0 (en), scale 1-5"
    assert "listening  Active listening" in lines
    assert "  L1 (+) Asks a clarifying question" in lines
    assert "  P2 (-) Refuses to plan" in lines
    assert "  Empathy: needs the voice" in lines
    assert result.stderr == ""


def test_catalogue_in_spanish_and_by_default_in_english(
    project: Path,
) -> None:
    """``--language es`` shows the translated catalogue."""
    spanish = skills(["catalogue", "--language", "es"], project)
    assert spanish.returncode == 0, spanish.stderr
    assert "core 1.0 (es)" in spanish.stdout
    assert "  L3 (-) Descalifica a los demás" in spanish.stdout
    default = skills(["catalogue"], project)
    assert "core 1.0 (en)" in default.stdout


def test_catalogue_errors(project: Path) -> None:
    """Unknown language, missing directory and invalid file."""
    assert_error(
        skills(["catalogue", "--language", "fr"], project),
        "not available in 'fr'", "core (en), core (es)",
    )
    assert_error(
        run_cli(["skills", "--rubric_dir", "nowhere", "catalogue"], project),
        "Available: none",
    )
    assert_error(
        run_cli(["skills", "--rubric_dir", "broken", "catalogue"], project),
        "Invalid catalogue",
    )
    assert_error(
        skills(["--catalogue", "other", "catalogue"], project), "'other'",
    )


@pytest.mark.parametrize("option, value, message", [
    ("--candidates", "0", "candidates"),
    ("--min_incidents", "0", "min_incidents"),
    ("--judge", "cloud", "judge"),
    ("--min_similarity", "7", "min_similarity"),
    ("--min_similarity", "high", "min_similarity"),
    ("--embedding_model", "5", "embedding_model"),
])
def test_invalid_global_options(
    project: Path, option: str, value: str, message: str
) -> None:
    """A bad option value is reported before any work."""
    result = skills([option, value, "catalogue"], project)
    assert_error(result, "Invalid option", message)


def test_renamed_options_reach_the_skills_settings(project: Path) -> None:
    """``--dense_weight`` and ``--local_model`` are validated by skills."""
    result = skills(["--dense_weight", "-1", "catalogue"], project)
    assert_error(result, "Invalid option", "semantic_weight")
    assert_error(skills(["--local_model", "7", "catalogue"], project),
                 "Invalid option", "llm")
    valid = skills(["--dense_weight", "0.5", "--local_model", "a/b",
                    "--min_episodes", "1", "--min_participant_words", "0",
                    "catalogue"], project)
    assert valid.returncode == 0, valid.stderr


def test_boolean_global_option_before_the_command(project: Path) -> None:
    """The global ``--no_cache=True`` is accepted around ``skills``."""
    result = skills(["--no_cache=True", "catalogue"], project)
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("core 1.0 (en)")


# ---------------------------------------------------------- interaction


def test_interaction_of_a_timed_transcript(project: Path) -> None:
    """Counts, shares, questions, seconds and overlaps per speaker."""
    result = skills(["interaction", EN], project)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[0].startswith("3 speakers, 8 turns, 82 words; "
                               "participation equality 0.")
    assert lines[1].split()[:3] == ["Maya", "turns", "3"]
    assert "questions 1" in lines[1] and "overlaps 0" in lines[1]
    assert len(lines) == 4


def test_interaction_of_an_untimed_spanish_transcript(project: Path) -> None:
    """Without timestamps no duration is printed."""
    result = skills(["interaction", ES], project)
    assert result.returncode == 0, result.stderr
    assert "3 speakers, 8 turns" in result.stdout
    assert "overlaps" not in result.stdout
    assert "Ana" in result.stdout and "questions 1" in result.stdout


@pytest.mark.parametrize("path, message", [
    ("missing.txt", "Transcript not found: missing.txt"),
    ("transcripts", "Transcript not found"),
    ("empty.txt", "is empty"),
    ("empty.json", "is empty"),
    ("bad.json", "not valid JSON"),
    ("shape_pretty.json", "must contain a list of turns"),
    ("binary.bin", "not a UTF-8 text file"),
    ("notes.txt", "no speaker turn"),
])
def test_interaction_degenerate_inputs(
    project: Path, path: str, message: str
) -> None:
    """Every unreadable transcript is a one-line error."""
    assert_error(skills(["interaction", path], project), message)


def test_interaction_rejects_a_one_line_json_object(project: Path) -> None:
    """A compact JSON document of the wrong shape is an error."""
    assert_error(skills(["interaction", "shape.json"], project),
                 "must contain a list of turns")


# -------------------------------------------------- annotation template


def test_annotation_template_default_and_explicit_output(
    project: Path,
) -> None:
    """The form goes to <output_dir>/annotations or to ``--output``."""
    result = skills(["annotation_template", EN, "ann1"], project)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == (
        "Wrote 6 empty cells to out/annotations/s1.ann1.json"
    )
    form = AnnotationSet.model_validate_json(
        (project / "out/annotations/s1.ann1.json").read_text("utf-8")
    )
    assert (form.session_id, form.annotator) == ("s1", "ann1")
    assert (form.catalogue, form.catalogue_version) == ("core", "1.0")
    assert len(form.annotations) == 6
    assert all(a.rating is None and not a.evidence for a in form.annotations)
    explicit = skills(
        ["annotation_template", ES, "7", "--output", "forms/es.json"],
        project,
    )
    assert explicit.returncode == 0, explicit.stderr
    spanish = AnnotationSet.model_validate_json(
        (project / "forms/es.json").read_text("utf-8")
    )
    assert (spanish.session_id, spanish.annotator) == ("es1", "7")
    assert {a.participant for a in spanish.annotations} == {
        "Ana", "Luis", "Marta",
    }


def test_annotation_template_is_valid_as_written(project: Path) -> None:
    """A fresh form passes ``validate_annotations``."""
    skills(["annotation_template", EN, "fresh", "--output", "f.json"],
           project)
    result = skills(["validate_annotations", "f.json", EN], project)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "f.json: valid (6 cells)"


def test_annotation_template_errors(project: Path) -> None:
    """Missing transcript, empty annotator, unwritable output."""
    assert_error(
        skills(["annotation_template", "missing.txt", "ann"], project),
        "Transcript not found",
    )
    assert_error(
        skills(["annotation_template", EN, " "], project),
        "annotator name is empty",
    )
    assert_error(
        skills(["annotation_template", EN, "ann", "--output", RUBRICS],
               project),
        "Cannot write",
    )
    assert_error(
        skills(["annotation_template", "empty.txt", "ann"], project),
        "is empty",
    )


# ------------------------------------------------- validate_annotations


def test_validate_annotations_accepts_a_filled_form(project: Path) -> None:
    """A correct form is reported valid."""
    result = skills(["validate_annotations", "ann/a.json", EN], project)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ann/a.json: valid (6 cells)"


def test_validate_annotations_lists_every_problem(project: Path) -> None:
    """Problems are listed one per line after the ``Error:`` summary."""
    result = skills(["validate_annotations", "ann/wrong.json", EN], project)
    assert_error(
        result, "8 problem(s) in ann/wrong.json:",
        "Maya/listening: rating 7 off the scale",
        "Maya/listening: rating without evidence",
        "Zoe/listening: unknown participant",
        "Tom/charisma: unknown competency",
        "Tom/listening: turn 0 is not theirs",
        "Tom/listening: indicator 'P1' is not of listening",
        "Tom/listening: turn 99 not found",
        "not annotated: Maya/planning",
        lines=9,
    )


def test_validate_annotations_against_another_session(project: Path) -> None:
    """A form checked against the wrong transcript does not validate."""
    result = skills(["validate_annotations", "ann/a.json", ES], project)
    assert result.returncode == 1
    assert result.stderr.startswith("Error: ")
    assert "session is 's1', transcript is 'es1'" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("form, transcript, message", [
    ("missing.json", EN, "File not found: missing.json"),
    ("ann/a.json", "missing.txt", "Transcript not found"),
    ("bad.json", EN, "Malformed JSON in bad.json"),
    ("empty.json", EN, "Malformed JSON in empty.json"),
    ("shape.json", EN, "does not match the AnnotationSet format"),
    ("binary.bin", EN, "not a UTF-8 text file"),
    ("transcripts", EN, "File not found"),
])
def test_validate_annotations_degenerate_inputs(
    project: Path, form: str, transcript: str, message: str
) -> None:
    """Unreadable forms or transcripts are one-line errors."""
    assert_error(
        skills(["validate_annotations", form, transcript], project), message
    )


# ------------------------------------------------------------ agreement


def test_agreement_between_two_annotators(project: Path) -> None:
    """Alpha, ICC, within-one, abstention kappa and evidence F1."""
    result = skills(["agreement", "ann/a.json,ann/b.json"], project)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[0] == "Annotators: a, b; 6 units in 1 session(s)"
    assert "Krippendorff alpha (ordinal) 0.750" in lines[1]
    assert "within one point 1.000" in lines[1]
    assert "rated-or-not kappa 1.000; evidence turns F1 0.917" in lines[2]
    assert lines[3].split()[:3] == ["listening", "alpha", "1.000"]
    assert lines[4].split()[:3] == ["planning", "alpha", "0.000"]
    assert len(lines) == 5


def test_agreement_uses_the_scale_of_the_chosen_language(
    project: Path,
) -> None:
    """``--language es`` reads the Spanish catalogue (same scale)."""
    result = skills(
        ["agreement", "ann/a.json,ann/b.json", "--language", "es"], project
    )
    assert result.returncode == 0, result.stderr
    assert "alpha (ordinal) 0.750" in result.stdout


@pytest.mark.parametrize("files, message", [
    ("ann/a.json", "at least two annotators"),
    ("ann/a.json,ann/a_s2.json", "at least two annotators"),
    ("ann/a.json,ann/a.json,ann/b.json", "a annotated s1/Maya/listening"),
    ("ann/a.json,ann/b.json,ann/off.json", "off the 1-5 scale"),
    ("ann/a.json,missing.json", "File not found: missing.json"),
    ("ann/a.json,bad.json", "Malformed JSON in bad.json"),
    ("ann/a.json,empty.json", "Malformed JSON in empty.json"),
    ("ann/a.json,shape.json", "does not match the AnnotationSet format"),
    ("ann/a.json,binary.bin", "not a UTF-8 text file"),
    ("ann/a.json,system.json", "does not match the AnnotationSet format"),
])
def test_agreement_degenerate_inputs(
    project: Path, files: str, message: str
) -> None:
    """One annotator, duplicates and unreadable files are clean errors."""
    assert_error(skills(["agreement", files], project), message)


def test_agreement_unknown_language(project: Path) -> None:
    """The catalogue language must exist."""
    assert_error(
        skills(["agreement", "ann/a.json,ann/b.json", "--language", "fr"],
               project),
        "not available in 'fr'",
    )


def test_agreement_off_scale_rating_is_a_clean_error(project: Path) -> None:
    """A rating of 9 on a 1-5 scale is reported as a user error."""
    assert_error(skills(["agreement", "ann/a.json,ann/off.json"], project),
                 "Rating(s) 9 are off the 1-5 scale.")
    assert_error(skills(["evaluate", "system.json", "ann/off.json"], project),
                 "Rating(s) 9 are off the 1-5 scale.")


def test_empty_file_lists_name_the_right_option(project: Path) -> None:
    """The error for an empty list names the option of the command."""
    assert_error(skills(["agreement", ","], project),
                 "--annotations must list at least one file.")
    assert_error(skills(["consensus", ","], project),
                 "--annotations must list at least one file.")
    assert_error(skills(["evaluate", ",", "ann/a.json"], project),
                 "--assessments must list at least one file.")
    assert_error(skills(["evaluate", "system.json", ","], project),
                 "--reference must list at least one file.")
    assert_error(skills(["lab_dataset", ",", "lab/x.json"], project),
                 "--reference must list at least one file.")


# ------------------------------------------------------------ consensus


def test_consensus_writes_one_reference_per_session(project: Path) -> None:
    """Annotators are merged per session into <session>.consensus.json."""
    result = skills(
        ["consensus", "ann/a.json,ann/b.json,ann/a_s2.json",
         "--save_directory", "ref"],
        project,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "ref/s1.consensus.json: 4/6 units rated",
        "ref/s2.consensus.json: 4/6 units rated",
    ]
    merged = AnnotationSet.model_validate_json(
        (project / "ref/s1.consensus.json").read_text("utf-8")
    )
    assert merged.annotator == "consensus"
    cells = merged.by_unit()
    tom = cells[("s1", "Tom", "planning")]
    assert tom.rating == 3
    assert [(e.turn, e.indicator_id) for e in tom.evidence] == [(4, "P2")]
    assert cells[("s1", "Maya", "planning")].rating is None


def test_consensus_default_directory_and_single_annotator(
    project: Path,
) -> None:
    """One annotator alone is their own consensus, under output_dir."""
    result = skills(["consensus", "ann/blank.json"], project)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == (
        "out/reference/s1.consensus.json: 0/6 units rated"
    )


@pytest.mark.parametrize("files, message", [
    ("ann/a.json,ann/v2.json", "different catalogue versions"),
    ("ann/a.json,ann/a.json,ann/b.json", "same annotator"),
    ("ann/a.json,missing.json", "File not found"),
    ("bad.json", "Malformed JSON"),
    ("binary.bin", "not a UTF-8 text file"),
])
def test_consensus_degenerate_inputs(
    project: Path, files: str, message: str
) -> None:
    """Mixed versions and unreadable files are clean errors."""
    assert_error(skills(["consensus", files], project), message)


def test_consensus_unwritable_directory(project: Path) -> None:
    """A save directory that is a file is a clean error."""
    assert_error(
        skills(["consensus", "ann/a.json", "--save_directory", "bad.json"],
               project),
        "Cannot write",
    )


# ------------------------------------------------------------- evaluate


def test_evaluate_assessment_against_reference(project: Path) -> None:
    """Evidence, rating and abstention scores are printed and saved."""
    result = skills(
        ["evaluate", "system.json", "ann/a.json", "--output",
         "out/evaluation.json"],
        project,
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[0] == "2 units in 1 session(s)"
    assert "retrieval recall 0.750, precision 1.000, recall 0.750" in lines[1]
    assert "ratings (2 rated by both): exact 0.500, within one 1.000, MAE " \
        "0.500" in lines[2]
    assert "rated-or-not: both 2, neither 0, system only 0, reference " \
        "only 0" in lines[3]
    assert "quotes verified: 0.600; dropped as unsupported: 0; flags: {}" \
        in lines[4]
    saved = json.loads(
        (project / "out/evaluation.json").read_text("utf-8")
    )
    assert saved["units"] == 2
    assert saved["evidence"]["reference_turns"] == 4
    assert sorted(saved["per_competency"]) == ["listening", "planning"]


@pytest.mark.parametrize("assessments, reference, message", [
    ("missing.json", "ann/a.json", "File not found: missing.json"),
    ("system.json", "missing.json", "File not found: missing.json"),
    ("ann/a.json", "ann/a.json", "does not match the SessionAssessment"),
    ("system.json", "system.json", "does not match the AnnotationSet"),
    ("bad.json", "ann/a.json", "Malformed JSON in bad.json"),
    ("system.json", "empty.json", "Malformed JSON in empty.json"),
    ("binary.bin", "ann/a.json", "not a UTF-8 text file"),
    ("system.json", "ann/other.json", "share no"),
])
def test_evaluate_degenerate_inputs(
    project: Path, assessments: str, reference: str, message: str
) -> None:
    """Swapped, missing, broken or unrelated files are clean errors."""
    assert_error(
        skills(["evaluate", assessments, reference], project), message
    )


def test_evaluate_unknown_language(project: Path) -> None:
    """The catalogue language must exist."""
    assert_error(
        skills(["evaluate", "system.json", "ann/a.json", "--language", "fr"],
               project),
        "not available in 'fr'",
    )


# ---------------------------------------------------------- lab_dataset


def test_lab_dataset_from_reference_annotations(project: Path) -> None:
    """Each unit with evidence becomes a question with turn sources."""
    result = skills(
        ["lab_dataset", "ann/a.json,ann/a_s2.json", "lab/evidence.json"],
        project,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == (
        "Wrote 8 questions to lab/evidence.json"
    )
    dataset = RagDataset.model_validate_json(
        (project / "lab/evidence.json").read_text("utf-8")
    )
    first = dataset.rag_questions[0]
    assert isinstance(first, AnsweredQuestion)
    assert first.question == "s1\tMaya\tlistening"
    assert first.question_id == "s1/Maya/listening"
    assert [s.first_character_index for s in first.sources] == [0, 3]
    assert {s.file_path for s in first.sources} == {"s1"}
    assert first.answer == "4"


@pytest.mark.parametrize("reference, message", [
    ("ann/blank.json", "no evidence turn"),
    ("missing.json", "File not found"),
    ("bad.json", "Malformed JSON"),
    ("system.json", "does not match the AnnotationSet"),
])
def test_lab_dataset_degenerate_inputs(
    project: Path, reference: str, message: str
) -> None:
    """A reference without evidence or an unreadable file is an error."""
    assert_error(
        skills(["lab_dataset", reference, "lab/x.json"], project), message
    )
    assert not (project / "lab/x.json").exists()


# ------------------------------------------- model commands, safe paths


def test_model_commands_fail_cleanly_before_loading_anything(
    project: Path,
) -> None:
    """A missing transcript stops ``assess`` and ``evidence`` at once."""
    assert_error(skills(["assess", "missing.txt"], project),
                 "Transcript not found")
    assert_error(skills(["assess", "binary.bin"], project),
                 "not a UTF-8 text file")
    assert_error(skills(["evidence", "missing.txt", "Tom", "planning"],
                        project), "Transcript not found")
    assert_error(skills(["evidence", EN, "Zoe", "planning"], project),
                 "Unknown participant 'Zoe'", "Maya, Tom, Priya")
    assert_error(skills(["evidence", "empty.txt", "Tom", "planning"],
                        project), "is empty")


# -------------------------------------------------------- the skills lab


def test_skills_lab_has_no_router_to_train(project: Path) -> None:
    """``skills lab train_router`` is a clean error, with or without data."""
    assert_error(skills(["lab", "train_router"], project),
                 "This lab has no query router to train.")
    assert_error(
        skills(["lab", "train_router", "--datasets", "ann/a.json"], project),
        "no query router",
    )


def test_skills_lab_errors_without_suite_or_results(project: Path) -> None:
    """Unknown suites and an empty trial store are clean errors."""
    assert_error(skills(["lab", "experiment", "nosuite"], project),
                 "Suite not found: nosuite")
    assert_error(skills(["lab", "tune", "nosuite"], project),
                 "Suite not found: nosuite")
    assert_error(
        run_cli(["skills", "--lab_dir", "emptylab", "lab", "analyze"],
                project),
        "No experiment results yet",
    )


def test_skills_lab_needs_an_explicit_dataset(project: Path) -> None:
    """The skills lab has no default dataset: ``--datasets`` is required."""
    suite = project / "lab/suites/lex.json"
    suite.parent.mkdir(parents=True, exist_ok=True)
    suite.write_text(json.dumps({"name": "lex", "base": {"nope": 1}}))
    assert_error(skills(["lab", "experiment", "lex"], project),
                 "--datasets must list at least one file.")
    skills(["lab_dataset", "ann/a.json", "lab/one.json"], project)
    unknown = skills(
        ["lab", "experiment", "lex", "--datasets", "lab/one.json"], project
    )
    assert unknown.returncode == 1
    assert "Error: Unknown parameter(s): nope." in unknown.stderr
    assert "Traceback" not in unknown.stderr
    assert_error(
        skills(["lab", "experiment", "lex", "--datasets", "missing.json"],
               project),
        "File not found",
    )
