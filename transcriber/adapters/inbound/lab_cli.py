"""Inbound adapter: ``python -m transcriber lab ...`` (the model lab).

* ``lab catalogue`` lists the candidate models with their state.
* ``lab run SESSION`` runs a suite in this process and prints a summary
  of the comparison: aggregate numbers only (times, memory, speakers
  found, seconds of speech and overlap, agreement), never text. Models
  are chosen with ``--denoise``/``--diarize``/``--recognise`` (ids,
  comma separated) or ``--all`` (every ready model of the catalogue
  whose weights are not known to be missing: ``--all`` never starts a
  download of weights the catalogue knows are absent).
* ``lab runs SESSION`` lists the runs; ``lab summary SESSION RUN``
  prints a run's summary again; ``lab resume SESSION RUN`` resumes it.
* ``lab import-reference CORPUS [--limit N] [--split S]`` imports
  recordings of a labelled public corpus as sessions with a reference,
  skipping those whose audio holds no speech (``--check`` runs that
  check again on the sessions already imported);
  ``lab reference-summary [--corpus C]`` gives the mean and spread of
  each reference metric per model over them.

The run uses the GPU in this process: like ``import --wait``, do not
start it while ``serve`` is working on a job. It writes only under
``<data_dir>/lab``.
"""

import argparse
import json
import time
from statistics import mean
from typing import Any, Callable, Dict, List, Optional

from transcriber.adapters.inbound.cli import Output, UsageError
from transcriber.application.lab import ACTIVE
from transcriber.application.lab_reference import (
    CorpusSummary,
    ImportedRecording,
)
from transcriber.bootstrap import LabUseCases
from transcriber.domain.lab import NO_DENOISER, LabRequest, LabRun
from transcriber.domain.lab_reference import CORPORA, MetricSpread


def _ids(text: Optional[str]) -> Optional[List[str]]:
    """Return ``a,b`` as ``["a", "b"]`` (``None`` when not given)."""
    if text is None:
        return None
    return [part.strip() for part in text.split(",") if part.strip()]


def _number(value: Optional[float], digits: int = 3) -> str:
    """Return a number for a table (``-`` when missing)."""
    return "-" if value is None else f"{value:.{digits}f}"


class LabCommands:
    """The ``lab`` sub-commands over the lab's use cases."""

    def __init__(
        self, lab: LabUseCases, output: Output, poll: float = 0.5,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Keep the use cases and where to write."""
        self.lab = lab
        self.output = output
        self.poll = poll
        self.sleep = sleep

    def run(self, args: argparse.Namespace) -> int:
        """Dispatch to the chosen sub-command."""
        action = args.lab_action
        if action == "catalogue":
            return self.catalogue(args.json)
        if action == "runs":
            return self.runs(args.session_id, args.json)
        if action == "summary":
            return self.summary(args.session_id, args.run_id, args.json)
        if action == "resume":
            run = self.lab.resume(args.session_id, args.run_id)
            return self.wait(run, args.json)
        if action == "import-reference" and args.check:
            return self.check_reference(args.corpus, args.json)
        if action == "import-reference":
            return self.import_reference(
                args.corpus, args.split, args.limit, args.json
            )
        if action == "reference-summary":
            return self.reference_summary(args.corpus, args.json)
        return self.start(args)

    def import_reference(
        self, corpus: str, split: Optional[str], limit: Optional[int],
        as_json: bool,
    ) -> int:
        """Import recordings of a public corpus as lab sessions."""
        def report(item: ImportedRecording) -> None:
            """Say what happened to one recording."""
            if as_json:
                return
            if item.reason is not None:
                self.output.line(_excluded_line(item))
                return
            if item.state == "exists":
                self.output.line(f"{item.session_id}: already imported")
                return
            if item.state == "repaired":
                self.output.line(
                    f"{item.session_id}: reference stored again"
                )
                return
            words = "no transcript" if item.words is None else (
                f"{item.words} reference words"
            )
            self.output.line(
                f"{item.session_id}: {_number(item.duration, 1)} s,"
                f" {item.speakers} speakers, {item.turns} turns, {words}"
            )

        done = self.lab.import_reference(corpus, split, limit, report)
        spec = CORPORA[corpus]
        if as_json:
            self.output.json({
                "corpus": spec.id, "licence": spec.licence,
                "licence_commercial": spec.licence_commercial,
                "source_url": spec.source_url,
                "recordings": [d.model_dump(mode="json") for d in done],
            })
            return 0
        usable = sum(d.usable for d in done)
        invalid = sum(not d.usable and d.speech is None for d in done)
        broken = f" {invalid} with an invalid reference;" if invalid else ""
        self.output.line(
            f"{usable} usable recordings of {spec.display_name}"
            f" ({spec.licence}, {spec.source_url});{broken}"
            f" {len(done) - usable - invalid} without speech."
        )
        if not spec.licence_commercial:
            self.output.note(
                "Non-commercial corpus: internal evaluation only."
            )
        return 0

    def check_reference(self, corpus: str, as_json: bool) -> int:
        """Run the speech check on the sessions already imported."""
        def report(item: ImportedRecording) -> None:
            """Say what the check found on one session."""
            if as_json:
                return
            if item.reason is not None:
                self.output.line(_excluded_line(item))
                return
            figures = "" if item.speech is None else (
                f" ({item.speech.describe()})"
            )
            self.output.line(f"{item.session_id}: speech{figures}")

        done = self.lab.check_reference(corpus, report)
        if as_json:
            self.output.json({
                "corpus": corpus,
                "recordings": [d.model_dump(mode="json") for d in done],
            })
            return 0
        failed = sum(not d.usable for d in done)
        self.output.line(
            f"{len(done)} sessions of {CORPORA[corpus].display_name}"
            f" checked; {failed} marked unusable (no speech in the audio)."
        )
        return 0

    def reference_summary(self, corpus: Optional[str], as_json: bool) -> int:
        """Print the mean and spread of each metric per model."""
        summary = self.lab.reference_summary(corpus)
        if as_json:
            self.output.json(summary.model_dump(mode="json"))
            return 0
        for line in reference_summary_lines(summary):
            self.output.line(line)
        return 0

    def catalogue(self, as_json: bool) -> int:
        """Print the models with their state."""
        view = self.lab.catalogue()
        if as_json:
            self.output.json({
                "token_present": view.token_present,
                "models": [
                    {"id": i.entry.id, "stage": i.entry.stage,
                     "family": i.entry.family, "state": i.state,
                     "reason": i.reason,
                     "licence": i.entry.licence,
                     "licence_commercial": i.entry.licence_commercial,
                     "gated": i.entry.gated}
                    for i in view.items
                ],
            })
            return 0
        self.output.line(
            f"Hugging Face token: {'yes' if view.token_present else 'no'}"
        )
        for item in view.items:
            entry = item.entry
            flags = []
            if entry.gated:
                flags.append("gated")
            if not entry.licence_commercial:
                flags.append("non-commercial")
            extra = f" [{', '.join(flags)}]" if flags else ""
            reason = f" ({item.reason})" if item.reason else ""
            self.output.line(
                f"{entry.stage:9} {entry.id:32} {item.state}{reason}{extra}"
            )
        return 0

    def _chosen(self, args: argparse.Namespace) -> LabRequest:
        """Return the request the options describe."""
        params: Dict[str, Dict[str, Any]] = {}
        if args.params:
            loaded = json.loads(args.params)
            if not isinstance(loaded, dict):
                raise UsageError("--params must be a JSON object.")
            params = loaded
        denoise = _ids(args.denoise)
        diarize = _ids(args.diarize)
        recognise = _ids(args.recognise)
        if args.all:
            ready: Dict[str, List[str]] = {
                "denoise": [], "diarize": [], "recognise": [],
            }
            for item in self.lab.catalogue().items:
                if item.state == "ready" and item.weights_cached is not False:
                    ready[item.entry.stage].append(item.entry.id)
                elif item.state == "ready":
                    self.output.note(
                        f"{item.entry.id}: weights not downloaded; left out"
                        " of --all (name it to download them)."
                    )
            denoise = denoise or ready["denoise"]
            diarize = diarize or ready["diarize"]
            recognise = recognise or ready["recognise"]
        if denoise is not None and NO_DENOISER not in denoise:
            denoise = [NO_DENOISER, *denoise]
        values: Dict[str, Any] = {"params": params}
        if denoise is not None:
            values["denoisers"] = denoise
        if diarize is not None:
            values["diarizers"] = diarize
        if recognise is not None:
            values["recognisers"] = recognise
        if args.language:
            values["language"] = args.language
        if args.speakers:
            values["num_speakers"] = args.speakers
        try:
            return LabRequest.model_validate(values)
        except ValueError as exc:
            raise UsageError(str(exc)) from exc

    def start(self, args: argparse.Namespace) -> int:
        """Start a run and wait for it."""
        run = self.lab.start(args.session_id, self._chosen(args))
        if not args.json:
            self.output.line(
                f"Run {run.id}: {len(run.steps)} steps,"
                f" {len(run.pipelines())} pipelines."
            )
        return self.wait(run, args.json)

    def wait(self, run: LabRun, as_json: bool) -> int:
        """Show the run's progress until it ends, then its summary."""
        last = None
        while True:
            view = self.lab.status(run.session_id, run.id)
            if not as_json:
                line = (
                    f"{view.state} {view.done_steps}/{view.total_steps}"
                    f" {view.job.stage if view.job else ''}"
                )
                if line != last:
                    self.output.note(line)
                    last = line
            if view.state not in ACTIVE:
                break
            self.sleep(self.poll)
        return self.summary(run.session_id, run.id, as_json)

    def runs(self, session_id: str, as_json: bool) -> int:
        """List the session's runs."""
        found = self.lab.runs(session_id)
        if as_json:
            self.output.json({"runs": [
                {"id": r.id, "state": state,
                 "created_at": r.created_at.isoformat(),
                 "steps": len(r.steps),
                 "pipelines": len(r.pipelines())}
                for r, state in found
            ]})
            return 0
        if not found:
            self.output.line("No lab runs.")
        for run, state in found:
            self.output.line(
                f"{run.id}  {state}  {run.created_at.isoformat()}"
                f"  {len(run.steps)} steps, {len(run.pipelines())}"
                " pipelines"
            )
        return 0

    def summary(self, session_id: str, run_id: str, as_json: bool) -> int:
        """Print the aggregate comparison of a run."""
        view = self.lab.status(session_id, run_id)
        failed = [
            {"step": s.name, "code": s.error.code if s.error else None,
             "message": s.error.message if s.error else None}
            for s in view.run.steps if s.state == "failed"
        ]
        try:
            results = self.lab.results(session_id, run_id)
        except Exception as exc:  # no results: a failed or cancelled run
            results = None
            if not as_json:
                self.output.line(f"No results: {exc}")
        data = summarise(results) if results is not None else {}
        data.update(state=view.state, failures=failed)
        if as_json:
            self.output.json(data)
        else:
            for line in summary_lines(data):
                self.output.line(line)
        return 0 if view.state in ("done", "partial") else 1


def summarise(results: Dict[str, Any]) -> Dict[str, Any]:
    """Return the aggregate numbers of a run's results (no text)."""
    pairs = [
        d for d in results["diarization"] if d["agreement"] is not None
    ]
    recognition = [
        r for r in results["recognition"] if r["wer"] is not None
    ]
    return {
        "duration": results["duration"],
        "consensus_coverage": results["consensus"]["coverage"],
        "tracks": [
            {k: t[k] for k in (
                "id", "speakers", "speech_seconds", "overlap_seconds",
                "labels", "agreement_with_consensus", "seconds",
                "peak_gpu_mb", "device",
            )} | {"mapping": [
                {k: m[k] for k in ("label", "speaker", "method",
                                   "similarity", "speech_seconds")}
                for m in t["mapping"]
            ]}
            for t in results["tracks"]
        ],
        "transcripts": [
            {k: t[k] for k in ("id", "words", "language", "seconds",
                               "peak_gpu_mb", "device")}
            for t in results["transcripts"]
        ],
        "pipelines": [
            {k: p[k] for k in (
                "id", "speech_seconds", "overlap_seconds", "words",
                "seconds", "peak_gpu_mb", "agreement_with_consensus",
                "words_on_consensus", "licence_commercial",
            )} | {"speakers": len(p["speakers"])}
            for p in results["pipelines"]
        ],
        "diarization_pairs": [
            {k: d[k] for k in ("reference", "hypothesis", "agreement",
                               "name_agreement", "error_rate")}
            | {"change_f1": d["change_points"]["f1"],
               "overlap_iou": d["overlap"]["iou"]}
            for d in pairs
        ],
        "mean_track_agreement": round(mean(
            d["agreement"] for d in pairs
        ), 4) if pairs else None,
        "recognition_pairs": [
            {k: r[k] for k in ("reference", "hypothesis", "wer",
                               "time_agreement")}
            for r in recognition
        ],
        "denoiser_effect": results["denoiser_effect"],
        "fragments": results.get("fragment_count"),
        "reference": _reference_figures(results.get("reference")),
    }


def _reference_figures(
    block: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Return a run's reference scores without text (``None``: none)."""
    if not block:
        return None
    return {
        k: block[k] for k in (
            "corpus", "recording", "licence", "licence_commercial",
            "source_url", "collar", "speech_seconds", "overlap_seconds",
            "reference_tokens",
        )
    } | {
        "speakers": len(block["speakers"]),
        "pipelines": block["pipelines"],
    }


def summary_lines(data: Dict[str, Any]) -> List[str]:
    """Return the summary as lines of text."""
    lines = [f"State: {data['state']}"]
    if "duration" in data:
        lines.append(
            f"Audio {data['duration']:.1f} s; consensus on"
            f" {_number(data['consensus_coverage'])} of the frames;"
            f" {data['fragments']} fragments."
        )
        lines.append("Pipelines (agreement, not accuracy):")
        lines.append(
            "  id | s | peak MB | speakers | speech s | overlap s |"
            " words | agree consensus | words on consensus"
        )
        for p in data["pipelines"]:
            lines.append(
                f"  {p['id']} | {_number(p['seconds'], 1)} |"
                f" {_number(p['peak_gpu_mb'], 0)} | {p['speakers']} |"
                f" {_number(p['speech_seconds'], 1)} |"
                f" {_number(p['overlap_seconds'], 1)} | {p['words']} |"
                f" {_number(p['agreement_with_consensus'])} |"
                f" {_number(p['words_on_consensus'])}"
                + ("" if p["licence_commercial"] else " [non-commercial]")
            )
        lines.append(
            "Mean pairwise track agreement:"
            f" {_number(data['mean_track_agreement'])}"
        )
        for r in data["recognition_pairs"]:
            lines.append(
                f"  WER {r['hypothesis']} against {r['reference']}:"
                f" {_number(r['wer'])}"
            )
        lines.extend(_reference_lines(data.get("reference")))
    for failure in data.get("failures", []):
        lines.append(
            f"Failed {failure['step']}: {failure['code']}:"
            f" {failure['message']}"
        )
    return lines


def _reference_lines(block: Optional[Dict[str, Any]]) -> List[str]:
    """Return a run's reference scores as lines (none without one)."""
    if not block:
        return []
    flag = "" if block["licence_commercial"] else (
        " [non-commercial corpus: internal evaluation only]"
    )
    lines = [
        f"Against the {block['corpus']} reference {block['recording']}"
        f" ({block['licence']}){flag}: {block['speakers']} speakers,"
        f" {_number(block['speech_seconds'], 1)} s of speech,"
        f" {_number(block['overlap_seconds'], 1)} s of overlap,"
        f" collar {block['collar']} s.",
        "  pipeline | DER | JER | overlap P | overlap R |"
        " speaker count error | WER",
    ]
    for p in block["pipelines"]:
        count = p["speaker_count_error"]
        shown = "-" if count is None else f"{count:+d}"
        lines.append(
            f"  {p['id']} | {_number(p['der'])} | {_number(p['jer'])} |"
            f" {_number(p['overlap_precision'])} |"
            f" {_number(p['overlap_recall'])} | {shown} |"
            f" {_number(p['wer'])}"
        )
    return lines


def _spread(value: Optional[MetricSpread]) -> str:
    """Return ``mean ± sd`` of a metric (``-`` when missing)."""
    if value is None or value.mean is None:
        return "-"
    sd = "" if value.sd is None else f" ± {value.sd:.3f}"
    return f"{value.mean:.3f}{sd}"


SUMMARY_TABLES = (
    ("diarization",
     "Diarization (mean ± SD over recordings): track | n | DER | missed"
     " | false alarm | confusion | JER | overlap P | overlap R |"
     " speaker count error",
     ("der", "missed_rate", "false_alarm_rate", "confusion_rate", "jer",
      "overlap_precision", "overlap_recall", "speaker_count_error")),
    ("recognition", "Recognition: transcript | n | WER", ("wer",)),
    ("pipeline", "Pipelines: id | n | DER | JER | WER",
     ("der", "jer", "wer")),
)


def _excluded_line(item: ImportedRecording) -> str:
    """Return the line of a recording that is not used, with figures."""
    verb = "skipped" if item.state == "skipped" else "unusable"
    figures = "" if item.speech is None else f" ({item.speech.describe()})"
    return f"{item.session_id}: {verb}: {item.reason}{figures}"


def reference_summary_lines(summary: CorpusSummary) -> List[str]:
    """Return the corpus summary as lines (aggregates only)."""
    lines = [
        f"{len(summary.recordings)} imported recordings"
        f" ({', '.join(summary.corpora) or 'none'});"
        f" {len(summary.unscored)} without a scored run."
    ]
    reasons: Dict[str, int] = {}
    for item in summary.excluded:
        reasons[item.reason] = reasons.get(item.reason, 0) + 1
    for reason, count in sorted(reasons.items()):
        lines.append(f"{count} excluded: {reason}.")
    if summary.non_commercial:
        lines.append(
            "Non-commercial corpus included: internal evaluation only."
        )
    for kind, title, names in SUMMARY_TABLES:
        rows = [m for m in summary.models if m.kind == kind]
        if not rows:
            continue
        lines.append(title)
        for model in rows:
            cells = " | ".join(_spread(model.metrics.get(n)) for n in names)
            lines.append(f"  {model.id} | {model.recordings} | {cells}")
    return lines


def add_lab_parser(
    commands: "argparse._SubParsersAction[argparse.ArgumentParser]",
    json_flag: Callable[[argparse.ArgumentParser], None],
) -> None:
    """Add the ``lab`` command and its sub-commands."""
    lab = commands.add_parser(
        "lab", help="model lab: compare denoisers, diarizers, recognisers",
        description=(
            "Run several models on a session and compare them (agreement,"
            " not accuracy). 'run' uses the GPU in this process: do not"
            " start it while serve is working on a job."
        ),
    )
    actions = lab.add_subparsers(dest="lab_action", metavar="ACTION")
    actions.required = True
    listed = actions.add_parser("catalogue", help="list the models")
    json_flag(listed)
    run = actions.add_parser("run", help="run a suite on a session")
    run.add_argument("session_id", metavar="SESSION")
    run.add_argument("--denoise", help="denoiser ids, comma separated"
                     " (none is always included)")
    run.add_argument("--diarize", help="diarizer ids, comma separated")
    run.add_argument("--recognise", help="recogniser ids, comma separated")
    run.add_argument("--all", action="store_true",
                     help="every ready model for the stages not given")
    run.add_argument("--language", help="language for the recognisers")
    run.add_argument("--speakers", type=int,
                     help="number of speakers for the diarizers")
    run.add_argument("--params", help="JSON: {model id: {param: value}}")
    json_flag(run)
    runs = actions.add_parser("runs", help="list a session's runs")
    runs.add_argument("session_id", metavar="SESSION")
    json_flag(runs)
    for name, text in (("summary", "print a run's summary"),
                       ("resume", "resume a run and wait for it")):
        sub = actions.add_parser(name, help=text)
        sub.add_argument("session_id", metavar="SESSION")
        sub.add_argument("run_id", metavar="RUN")
        json_flag(sub)
    imported = actions.add_parser(
        "import-reference",
        help="import recordings of a labelled public corpus as sessions",
        description=(
            "Download the shards needed into <data_dir>/lab-corpora and"
            " create one session per recording with its reference"
            " (speaker turns, licence, source). A recording whose audio"
            " holds no speech, or whose reference is invalid, is"
            " skipped. Runs on those sessions are"
            " also scored against the reference."
        ),
    )
    imported.add_argument("corpus", choices=sorted(CORPORA))
    imported.add_argument("--limit", type=int,
                          help="at most this many usable recordings")
    imported.add_argument("--check", action="store_true",
                          help="check the sessions already imported for"
                          " speech instead (no download); failing ones"
                          " are left out of the summary")
    imported.add_argument("--split", help="the corpus split (default: test"
                          " for ami, data for callhome)")
    json_flag(imported)
    summary = actions.add_parser(
        "reference-summary",
        help="mean and spread of the reference scores per model",
    )
    summary.add_argument("--corpus", choices=sorted(CORPORA),
                         help="only this corpus")
    json_flag(summary)
