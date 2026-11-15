"""Measure naming speakers from what they say (TR-20, issue #102).

Runs the search of ``transcriber/domain/speaker_names.py`` over the
synthetic Spanish set ``experiments/transcriber/speaker_names/set.json``
(and the simulated discussion of #88, made anonymous), in one of two
modes:

``rules``
    the rules alone: their own evidence and confidence, decided with
    the same precision-first policy (automatic names allowed), the
    baseline;
``model``
    the rules pick the candidates and the local model
    (:class:`~transcriber.adapters.outbound.local_name_model.
    LocalNameModel`, pinned to one model with ``--model``) reads at
    most ``max_candidates`` of them, as the service does (#116: the
    other mentions of a name it confirmed count as the rules saw them).

Either way the evidence is spread over the voices of the session and
decided as the service does (``attribute`` then ``decide``, #116).

For every session it counts, per speaker, the names given
automatically (right when it is the speaker's true name) and the
suggestions, and writes one JSON row per run to ``--out`` (default
``experiments/transcriber/speaker_names/results.jsonl``):

- automatic names: precision (right / given) and recall (right / the
  speakers who clearly introduce themselves; since #116 names said by
  others can be given too, so ``auto_recall_named`` divides by the
  speakers who introduce themselves or whose name is said);
- suggestions: precision (right / given) and recall (speakers whose
  name is said anywhere who end up with the right name automatically
  or among their suggestions / those speakers);
- seconds per session (the search, the model already loaded) and the
  seconds to load the model; the GPU memory peak of this process.

Usage::

    PYTHONPATH=. python experiments/transcriber/speaker_names_eval.py \\
        --mode rules
    HF_HUB_OFFLINE=1 PYTHONPATH=. python \\
        experiments/transcriber/speaker_names_eval.py --mode model \\
        --model Qwen/Qwen3-8B --device cuda

The script may import :mod:`transcriber`; it never imports ``src``.
"""

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from transcriber.adapters.outbound.local_name_model import LocalNameModel
from transcriber.domain.models import (
    Pipeline,
    Session,
    SessionState,
    Speaker,
    Transcript,
    Utterance,
)
from transcriber.domain.sentences import regroup_utterances
from transcriber.domain.speaker_names import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    Decision,
    NameCandidate,
    NameEvidence,
    NameSearchParameters,
    attribute,
    contains_name,
    decide,
    find_candidates,
    gated_evidence,
    model_prompt,
    parse_reply,
    pick_for_model,
    rule_evidence,
)
from transcriber.domain.name_variants import similar_names

ROOT = Path(__file__).resolve().parents[2]
SET = ROOT / "experiments/transcriber/speaker_names/set.json"
OUT = ROOT / "experiments/transcriber/speaker_names/results.jsonl"

Gold = Dict[str, Dict[str, Any]]


def build(lines: Sequence[Tuple[str, str]], session_id: str) -> Transcript:
    """Return a finished transcript of anonymous voices saying ``lines``."""
    rough = [
        Utterance(
            id="t0.s0", turn=0, index=0, speaker=speaker, start=3.0 * n,
            end=3.0 * n + 2.5, text=text, previous_speaker=None,
            previous_utterance=None,
        )
        for n, (speaker, text) in enumerate(lines)
    ]
    speakers = [
        Speaker(id=s, name=None, enrolled=False)
        for s in dict.fromkeys(speaker for speaker, _ in lines)
    ]
    return Transcript(
        session=Session(
            id=session_id, language="es", duration=3.0 * len(lines) + 1,
            state=SessionState.DONE,
        ),
        speakers=speakers, utterances=regroup_utterances(rough),
        pipeline=Pipeline(recogniser="r", voice_model="v", transcriber="t"),
    )


def from_contract(path: Path) -> Tuple[List[Tuple[str, str]], Gold]:
    """Return the lines of a ``transcript.v1`` file made anonymous.

    Speakers become ``S1``, ``S2``… in order of first appearance; the
    true name is the file's; nobody introduces themselves, and a name
    is ``said`` when another speaker says it.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    order = list(dict.fromkeys(u["speaker"] for u in data["utterances"]))
    label = {who: f"S{n + 1}" for n, who in enumerate(order)}
    names = {s["id"]: s["name"] or s["id"] for s in data["speakers"]}
    lines = [(label[u["speaker"]], u["text"]) for u in data["utterances"]]
    gold: Gold = {}
    for who, anon in label.items():
        said = any(
            contains_name(u["text"], names[who])
            for u in data["utterances"] if u["speaker"] != who
        )
        gold[anon] = {"name": names[who], "introduces": False, "said": said}
    return lines, gold


def sessions(path: Path) -> List[Tuple[str, str, Transcript, Gold]]:
    """Return ``(id, case, transcript, gold)`` for every session."""
    out = []
    for item in json.loads(path.read_text(encoding="utf-8"))["sessions"]:
        if "from" in item:
            lines, gold = from_contract(ROOT / item["from"])
        else:
            lines = [(s, t) for s, t in item["lines"]]
            gold = item["speakers"]
        out.append((item["id"], item["case"], build(lines, item["id"]), gold))
    return out


def evidence_of(
    transcript: Transcript,
    params: NameSearchParameters,
    model: Optional[LocalNameModel],
) -> Tuple[List[NameEvidence], int]:
    """Return the evidence of the mode and how many candidates were read."""
    candidates = find_candidates(transcript, params.context, limit=None)
    if model is None:
        found = [rule_evidence(transcript, c) for c in candidates]
        return [e for e in found if e is not None], len(candidates)
    read: List[Tuple[NameCandidate, Optional[NameEvidence]]] = []
    for candidate in pick_for_model(candidates, params.max_candidates):
        reply = model.complete(
            SYSTEM_PROMPT, model_prompt(candidate), params.max_tokens
        )
        read.append((candidate, parse_reply(transcript, candidate, reply)))
    return gated_evidence(transcript, candidates, read), len(read)


def right(name: str, truth: Optional[str]) -> bool:
    """Tell whether ``name`` is ``truth`` (any spelling of it).

    A voice whose name is never said (``truth`` empty) has no right
    name.
    """
    return bool(truth) and similar_names(name, str(truth))


def score(decision: Decision, gold: Gold) -> Dict[str, Any]:
    """Count right and wrong names of one session."""
    auto_right = [
        a for a in decision.auto if right(a.name, gold[a.speaker]["name"])
    ]
    auto_wrong = [
        f"{a.speaker}={a.name} (is {gold[a.speaker]['name']})"
        for a in decision.auto if a not in auto_right
    ]
    sugg_right = [
        s for s in decision.suggestions
        if right(s.name, gold[s.speaker]["name"])
    ]
    found = {a.speaker for a in auto_right} | {s.speaker for s in sugg_right}
    said = [s for s, g in gold.items() if g["said"]]
    return {
        "auto_given": len(decision.auto),
        "auto_right": len(auto_right),
        "auto_target": sum(1 for g in gold.values() if g["introduces"]),
        "named_target": sum(
            1 for g in gold.values() if g["introduces"] or g["said"]
        ),
        "auto_wrong": auto_wrong,
        "suggestions": len(decision.suggestions),
        "suggestions_right": len(sugg_right),
        "said": len(said),
        "said_found": sum(1 for s in said if s in found),
        "auto": [f"{a.speaker}={a.name}" for a in decision.auto],
        "suggested": [
            f"{s.speaker}={s.name} ({s.reason})" for s in decision.suggestions
        ],
    }


def ratio(right: int, total: int) -> Optional[float]:
    """Return ``right / total`` rounded, ``None`` when nothing to count."""
    return round(right / total, 3) if total else None


def gpu_peak_gb() -> Optional[float]:
    """Return this process's peak GPU memory in GB, if on CUDA."""
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 1e9, 2)
    except ImportError:
        pass
    return None


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run one mode over the set and append its row to ``--out``."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--mode", choices=("rules", "model"), required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"),
                        default="auto")
    parser.add_argument("--wait", type=float, default=600.0)
    parser.add_argument("--set", type=Path, default=SET)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    params = NameSearchParameters()
    model = None
    load_seconds = None
    if args.mode == "model":
        model = LocalNameModel(
            (args.model,), device=args.device, wait_seconds=args.wait
        )
        began = time.monotonic()
        model.complete(SYSTEM_PROMPT, "S1: Hola.", 4)
        load_seconds = round(time.monotonic() - began, 1)
    rows = []
    for session_id, case, transcript, gold in sessions(args.set):
        began = time.monotonic()
        evidence, read = evidence_of(transcript, params, model)
        decision = decide(
            transcript.speakers, attribute(transcript, evidence), (),
            params.auto_confidence, min_votes=params.min_votes,
            min_margin=params.margin,
        )
        seconds = round(time.monotonic() - began, 2)
        rows.append({
            "session": session_id, "case": case, "candidates": read,
            "seconds": seconds, **score(decision, gold),
        })
        print(session_id, rows[-1]["auto"], rows[-1]["suggested"],
              f"{seconds} s", flush=True)
    total = {key: sum(r[key] for r in rows) for key in (
        "auto_given", "auto_right", "auto_target", "named_target",
        "suggestions",
        "suggestions_right", "said", "said_found", "candidates",
    )}
    summary = {
        "mode": args.mode,
        "prompt": PROMPT_VERSION if model is not None else None,
        "model": model.model_name if model is not None else None,
        "device": args.device if model is not None else None,
        "sessions": len(rows),
        **total,
        "auto_precision": ratio(total["auto_right"], total["auto_given"]),
        "auto_recall": ratio(total["auto_right"], total["auto_target"]),
        "auto_recall_named": ratio(
            total["auto_right"], total["named_target"]
        ),
        "auto_wrong": [w for r in rows for w in r["auto_wrong"]],
        "suggestion_precision": ratio(
            total["suggestions_right"], total["suggestions"]
        ),
        "named_or_suggested_recall": ratio(
            total["said_found"], total["said"]
        ),
        "seconds_per_session": round(
            sum(r["seconds"] for r in rows) / len(rows), 2
        ),
        "seconds_max": max(r["seconds"] for r in rows),
        "load_seconds": load_seconds,
        "gpu_peak_gb": gpu_peak_gb() if model is not None else None,
        "parameters": params.model_dump(),
        "machine": platform.processor() or platform.machine(),
        "rows": rows,
    }
    if model is not None:
        model.release()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(summary, ensure_ascii=False) + "\n")
    shown = {k: v for k, v in summary.items() if k != "rows"}
    print(json.dumps(shown, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
