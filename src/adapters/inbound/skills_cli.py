"""Inbound adapter: soft-skills commands (``python -m src skills ...``)."""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from pydantic import ValidationError

from src.adapters.inbound.lab_cli import LabCLI, parse_paths, write_text
from src.adapters.outbound.json_dataset_repository import (
    read_model,
    write_model,
)
from src.application.skills.evaluation import evaluate
from src.application.skills.gold import (
    AnnotationSet,
    agreement,
    blank_annotations,
    consensus,
    validate_annotations,
)
from src.application.skills.lab import lab_dataset
from src.application.skills.report import render_scorecards
from src.config import SkillsSettings
from src.domain.errors import InvalidInputError, short_reason
from src.domain.skills.assessment import SessionAssessment
from src.domain.skills.interaction import describe
from src.skills_bootstrap import SkillsContainer

Guard = Callable[[Callable[..., Any]], Callable[..., Any]]


def parse_names(value: Any) -> Optional[List[str]]:
    """Parse a comma-separated list (or a Fire tuple) of names."""
    if value in (None, "", ()):
        return None
    raw = value if isinstance(value, (tuple, list)) else str(value).split(",")
    names = [str(v).strip() for v in raw if str(v).strip()]
    return names or None


def fmt(value: Optional[float]) -> str:
    """Format an optional statistic."""
    return "n/a" if value is None else f"{value:.3f}"


class SkillsCLI:
    """Soft-skills analyzer: evidence-backed competency scorecards."""

    def __init__(self, guard: Guard, options: Dict[str, Any]):
        """Keep the shared options; the container is built on demand."""
        self._guard = guard
        self._options = options
        self._container: Optional[SkillsContainer] = None
        for name in (
            "catalogue", "interaction", "evidence", "assess",
            "annotation_template", "validate_annotations", "agreement",
            "consensus", "evaluate", "lab_dataset",
        ):
            setattr(self, name, guard(getattr(self, f"_{name}")))

    def _app(self) -> SkillsContainer:
        """Validate the shared options and build the container."""
        if self._container is None:
            values = {
                k: v for k, v in self._options.items() if v is not None
            }
            if "no_cache" in values:
                values["use_cache"] = not bool(values.pop("no_cache"))
            unknown = sorted(set(values) - set(SkillsSettings.model_fields))
            if unknown:
                raise InvalidInputError(
                    f"Unknown option(s): {', '.join(unknown)}."
                )
            try:
                settings = SkillsSettings.model_validate(values)
            except ValueError as exc:
                raise InvalidInputError(
                    f"Invalid option: {short_reason(exc)}"
                )
            self._container = SkillsContainer(settings)
        return self._container

    def lab(self) -> LabCLI:
        """Experiment lab over evidence retrieval: experiment, tune, analyze.

        Datasets come from ``lab_dataset``; transcripts are read from
        ``--transcripts_dir``.
        """
        return LabCLI(self._app, self._guard, default_datasets="")

    def _annotations(
        self, paths: Any, option: str = "annotations"
    ) -> List[AnnotationSet]:
        """Read annotation files."""
        return [
            read_model(path, AnnotationSet)
            for path in parse_paths(paths or "", option)
        ]

    def _catalogue(self, language: str = "en") -> None:
        """List the competencies and indicators of the rubric catalogue.

        Args:
            language: Language of the catalogue to show (en, es).
        """
        app = self._app()
        rubric = app.catalogues.load(app.settings.catalogue, str(language))
        print(f"{rubric.name} {rubric.version} ({rubric.language}), scale "
              f"{rubric.scale.minimum}-{rubric.scale.maximum}")
        for competency in rubric.competencies:
            print(f"\n{competency.id}  {competency.name}")
            print(f"  {competency.definition}")
            for indicator in competency.indicators:
                sign = "+" if indicator.polarity == "positive" else "-"
                print(f"  {indicator.id} ({sign}) {indicator.text}")
        if rubric.excluded:
            print("\nNot assessed from a transcript:")
            for item in rubric.excluded:
                print(f"  {item.name}: {item.reason}")

    def _interaction(self, transcript: str) -> None:
        """Print who spoke, how much and how evenly (no model needed).

        Args:
            transcript: Transcript file (JSON or ``Speaker: text``).
        """
        team = describe(self._app().transcripts.read(str(transcript)))
        print(f"{team.speakers} speakers, {team.turns} turns, "
              f"{team.words} words; participation equality "
              f"{team.participation_equality:.2f}")
        for s in team.per_speaker:
            extra = (
                "" if s.seconds is None
                else f"  {s.seconds:.0f}s  overlaps {s.overlapping_starts}"
            )
            print(f"  {s.speaker:<16} turns {s.turns:>3}  words "
                  f"{s.words:>5} ({s.word_share:.0%})  questions "
                  f"{s.questions}{extra}")

    def _evidence(
        self, transcript: str, participant: str, competency: str, k: int = 8
    ) -> None:
        """Show the turns of a participant closest to a competency.

        This is retrieval only: no language model is called.

        Args:
            transcript: Transcript file.
            participant: Speaker name as written in the transcript.
            competency: Competency id of the catalogue.
            k: Number of turns to show.
        """
        app = self._app()
        document = app.transcripts.read(str(transcript))
        if str(participant) not in document.speakers():
            raise InvalidInputError(
                f"Unknown participant {participant!r}. Speakers: "
                f"{', '.join(document.speakers())}."
            )
        index = app.retriever.index(document)
        rubric = app.rubric(index.language)
        found = app.retriever.candidates(
            index, str(participant), rubric.competency(str(competency)),
            int(k), rubric.language,
        )
        for candidate in found:
            turn = document.turns[candidate.turn]
            print(f"[{turn.index}] {candidate.score:.3f} "
                  f"{candidate.indicator_id}: {turn.text}")
        if not found:
            print("No eligible turn.")

    def _assess(
        self,
        transcript: str,
        participants: str = "",
        competencies: str = "",
        save_directory: str = "",
    ) -> None:
        """Produce the scorecards of a session (JSON and Markdown).

        Args:
            transcript: Transcript file.
            participants: Comma-separated speakers (default: all).
            competencies: Comma-separated competency ids (default: all).
            save_directory: Where to write the files (default: the
                configured output directory).
        """
        app = self._app()
        document = app.transcripts.read(str(transcript))
        language = app.retriever.language_of(document)
        rubric = app.rubric(language)
        result = app.assessment.assess(
            document, rubric, parse_names(participants),
            parse_names(competencies),
        )
        directory = str(save_directory) or app.settings.output_dir
        name = document.session_id
        path = write_model(directory, f"{name}.assessment.json", result)
        report = str(Path(directory) / f"{name}.scorecard.md")
        write_text(report, render_scorecards(result, rubric, document))
        for card in result.scorecards:
            cells = "  ".join(
                f"{a.competency_id}="
                f"{'-' if a.rating is None else a.rating}"
                for a in card.assessments
            )
            print(f"{card.participant:<16} {cells}")
        print(f"Saved {path} and {report} ({result.seconds:.1f}s)")

    def _annotation_template(
        self, transcript: str, annotator: str, output: str = ""
    ) -> None:
        """Write an empty annotation form for a human annotator.

        Args:
            transcript: Transcript file.
            annotator: Name or code of the annotator.
            output: File to write (default:
                <output_dir>/annotations/<session>.<annotator>.json).
        """
        app = self._app()
        document = app.transcripts.read(str(transcript))
        rubric = app.rubric(app.retriever.language_of(document))
        form = blank_annotations(document, rubric, str(annotator))
        target = Path(str(output)) if output else Path(
            app.settings.output_dir, "annotations",
            f"{document.session_id}.{form.annotator}.json",
        )
        path = write_model(str(target.parent), target.name, form)
        print(f"Wrote {len(form.annotations)} empty cells to {path}")

    def _validate_annotations(self, annotations: str, transcript: str) -> None:
        """Check a filled annotation form against its transcript.

        Args:
            annotations: Annotation file.
            transcript: Transcript file it refers to.
        """
        app = self._app()
        document = app.transcripts.read(str(transcript))
        rubric = app.rubric(app.retriever.language_of(document))
        form = read_model(str(annotations), AnnotationSet)
        problems = validate_annotations(form, document, rubric)
        if problems:
            raise InvalidInputError(
                f"{len(problems)} problem(s) in {annotations}:\n  "
                + "\n  ".join(problems)
            )
        print(f"{annotations}: valid ({len(form.annotations)} cells)")

    def _agreement(self, annotations: str, language: str = "en") -> None:
        """Measure the agreement between annotators.

        Args:
            annotations: Comma-separated annotation files, from at
                least two annotators.
            language: Catalogue language (only its scale is used).
        """
        app = self._app()
        rubric = app.catalogues.load(app.settings.catalogue, str(language))
        report = agreement(self._annotations(annotations), rubric.scale)
        print(f"Annotators: {', '.join(report.annotators)}; "
              f"{report.units} units in {report.sessions} session(s)")
        print(f"  ratings: Krippendorff alpha (ordinal) {fmt(report.alpha)} "
              f"→ {report.band}; ICC(2,1) {fmt(report.icc)}, ICC(2,k) "
              f"{fmt(report.icc_average)}; within one point "
              f"{fmt(report.within_one)}")
        print(f"  rated-or-not kappa {fmt(report.abstention_kappa)}; "
              f"evidence turns F1 {fmt(report.evidence_f1)}")
        for item in report.per_competency:
            print(f"  {item.competency_id:<8} alpha {fmt(item.alpha)} "
                  f"({item.band}, {item.rated_units} rated), within "
                  f"one {fmt(item.within_one)}, evidence F1 "
                  f"{fmt(item.evidence_f1)}")

    def _consensus(self, annotations: str, save_directory: str = "") -> None:
        """Merge annotators into one reference set per session.

        Args:
            annotations: Comma-separated annotation files.
            save_directory: Where to write <session>.consensus.json.
        """
        app = self._app()
        directory = str(save_directory) or str(
            Path(app.settings.output_dir, "reference")
        )
        for merged in consensus(self._annotations(annotations)):
            path = write_model(
                directory, f"{merged.session_id}.consensus.json", merged
            )
            rated = sum(1 for a in merged.annotations if a.rating is not None)
            print(f"{path}: {rated}/{len(merged.annotations)} units rated")

    def _evaluate(
        self, assessments: str, reference: str, language: str = "en",
        output: str = "",
    ) -> None:
        """Score assessments against reference annotations.

        Args:
            assessments: Comma-separated assessment JSON files.
            reference: Comma-separated reference annotation files.
            language: Catalogue language (only its scale is used).
            output: Optional JSON file for the full result.
        """
        app = self._app()
        rubric = app.catalogues.load(app.settings.catalogue, str(language))
        sessions = [
            read_model(path, SessionAssessment)
            for path in parse_paths(assessments or "", "assessments")
        ]
        result = evaluate(
            sessions, self._annotations(reference, "reference"),
            rubric.scale,
        )
        e, r, a = result.evidence, result.ratings, result.abstention
        print(f"{result.units} units in {result.sessions} session(s)")
        print(f"  evidence: retrieval recall {fmt(e.retrieval_recall)}, "
              f"precision {fmt(e.precision)}, recall {fmt(e.recall)}, "
              f"F1 {fmt(e.f1)}")
        print(f"  ratings ({r.rated_by_both} rated by both): exact "
              f"{fmt(r.exact)}, within one {fmt(r.within_one)}, MAE "
              f"{fmt(r.mean_absolute_error)}, weighted kappa "
              f"{fmt(r.weighted_kappa)}, alpha {fmt(r.alpha)}")
        print(f"  rated-or-not: both {a.both_rated}, neither "
              f"{a.both_abstained}, system only {a.system_only_rated}, "
              f"reference only {a.reference_only_rated}")
        print(f"  quotes verified: {fmt(result.quote_verification_rate)}; "
              f"dropped as unsupported: {result.unsupported_quotes}; "
              f"flags: {json.dumps(result.flagged)}")
        if output:
            write_text(str(output), result.model_dump_json(indent=2))

    def _lab_dataset(self, reference: str, output: str) -> None:
        """Convert reference annotations into a dataset for ``skills lab``.

        Args:
            reference: Comma-separated reference annotation files.
            output: Dataset file to write.
        """
        try:
            dataset = lab_dataset(self._annotations(reference, "reference"))
        except ValidationError as exc:
            raise InvalidInputError(short_reason(exc))
        target = Path(str(output))
        path = write_model(str(target.parent), target.name, dataset)
        print(f"Wrote {len(dataset.rag_questions)} questions to {path}")
