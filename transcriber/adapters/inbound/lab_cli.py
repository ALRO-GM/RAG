"""Inbound adapter: ``python -m transcriber lab ...`` (the model lab).

* ``lab catalogue`` lists the candidate models with their state.
* ``lab run SESSION`` runs a suite in this process and prints a summary
  of the comparison: aggregate numbers only (times, memory, speakers
  found, seconds of speech and overlap, agreement), never text. Models
  are chosen with ``--denoise``/``--diarize``/``--recognise`` (ids,
  comma separated) or ``--all`` (every ready model of the catalogue).
* ``lab runs SESSION`` lists the runs; ``lab summary SESSION RUN``
  prints a run's summary again; ``lab resume SESSION RUN`` resumes it.

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
from transcriber.bootstrap import LabUseCases
from transcriber.domain.lab import NO_DENOISER, LabRequest, LabRun


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
        return self.start(args)

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
                if item.state == "ready":
                    ready[item.entry.stage].append(item.entry.id)
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
    for failure in data.get("failures", []):
        lines.append(
            f"Failed {failure['step']}: {failure['code']}:"
            f" {failure['message']}"
        )
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
