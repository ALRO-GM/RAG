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
from src.application.skills.tagging import render_timeline
from src.config import SkillsSettings
from src.domain.errors import InvalidInputError, short_reason
from src.domain.skills.assessment import SessionAssessment
from src.domain.skills.interaction import describe
from src.domain.skills.speech import clock, transcript_sentences
from src.skills_bootstrap import SkillsContainer

Guard = Callable[[Callable[..., Any]], Callable[..., Any]]


def parse_names(value: Any) -> Optional[List[str]]:
    """Parse a comma-separated list (or a Fire tuple) of names."""
    if value in (None, "", ()):
        return None
    raw = value if isinstance(value, (tuple, list)) else str(value).split(",")
    names = [str(v).strip() for v in raw if str(v).strip()]
    return names or None


def parse_speaker_names(value: Any) -> Dict[str, str]:
    """Parse ``SPEAKER_01=Ana,SPEAKER_02=Luis`` into a mapping.

    A bare list (``Ana,Luis``) names the speakers in order of first
    appearance: ``SPEAKER_01`` is the first voice heard.
    """
    names: Dict[str, str] = {}
    for position, item in enumerate(parse_names(value) or []):
        label, sep, name = item.partition("=")
        if not sep:
            label, name = f"SPEAKER_{position + 1:02d}", item
        if not label.strip() or not name.strip():
            raise InvalidInputError(
                f"Invalid speaker name {item!r}; use LABEL=Name."
            )
        names[label.strip()] = name.strip()
    return names


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
            "transcribe", "sentences", "tag",
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

    def _transcribe(
        self,
        audio: str,
        speakers: int = 0,
        names: str = "",
        language: str = "",
        session: str = "",
        output: str = "",
    ) -> None:
        """Transcribe a recording: who said what, and when.

        The recording is processed on this machine; nothing is sent
        anywhere. The transcript is saved as JSON (turns with times and
        words, and the list of sentences) where ``assess``, ``tag`` and
        the lab find it.

        Args:
            audio: Recording (WAV, FLAC, OGG, MP3; others need ffmpeg).
            speakers: Number of people in the recording (0 = estimate
                it; giving it is much more reliable).
            names: ``SPEAKER_01=Ana,SPEAKER_02=Luis``, or ``Ana,Luis``
                in order of first appearance.
            language: Spoken language code, e.g. es (default: detect).
            session: Session identifier (default: the file name).
            output: Transcript file (default:
                <transcripts_dir>/<session>.json).
        """
        app = self._app()
        count = int(speakers)
        if count < 0:
            raise InvalidInputError("--speakers must not be negative.")
        document = app.transcription.transcribe(
            str(audio),
            session_id=str(session) or None,
            language=str(language) or None,
            num_speakers=count or None,
            names=parse_speaker_names(names) or None,
        )
        path = app.transcripts.save(document, str(output))
        for turn in document.turns:
            print(f"[{clock(turn.start)}] {turn.speaker}: {turn.text}")
        print(f"{len(document.turns)} turns, "
              f"{len(document.speakers())} speakers "
              f"({', '.join(document.speakers())}), "
              f"language {document.language or 'unknown'}")
        print(f"Saved {path}")

    def _sentences(self, transcript: str) -> None:
        """List the sentences of a transcript with speaker and time.

        Args:
            transcript: Transcript file.
        """
        document = self._app().transcripts.read(str(transcript))
        for s in transcript_sentences(document):
            print(f"[{clock(s.start)}-{clock(s.end)}] {s.id} "
                  f"{s.speaker}: {s.text}")

    def _tag(
        self, transcript: str, save_directory: str = "", show: bool = True
    ) -> None:
        """Tag every sentence with the rubric indicators it may show.

        Tags are candidates found by text similarity (no language model
        is called): a short list for the judge or an annotator, not a
        verdict.

        Args:
            transcript: Transcript file.
            save_directory: Where to write <session>.tags.json
                (default: the configured output directory).
            show: Print the tagged timeline.
        """
        app = self._app()
        document = app.transcripts.read(str(transcript))
        rubric = app.rubric(app.retriever.language_of(document))
        tagged = app.tagger.tag(document, rubric)
        directory = str(save_directory) or app.settings.output_dir
        path = str(Path(directory) / f"{document.session_id}.tags.json")
        write_text(path, json.dumps({
            "session_id": document.session_id,
            "catalogue": rubric.name,
            "catalogue_language": rubric.language,
            "sentences": [t.model_dump(exclude_none=True) for t in tagged],
        }, ensure_ascii=False, indent=1))
        if show:
            print(render_timeline(tagged))
        with_tags = sum(1 for t in tagged if t.tags)
        print(f"{len(tagged)} sentences, {with_tags} with candidate tags")
        print(f"Saved {path}")

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
