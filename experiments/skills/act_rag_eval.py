"""Measure the retrieval act taggers of #94 (R and RG) as #82 did.

Same Spanish set, folds and measures as
``experiments/skills/act_tagger_eval.py`` (5 folds by dialogue, top-1,
top-3, macro-F1, per-category F1, log-loss, ECE), plus:

- **retrieval configurations**: the theory cards alone, or the cards
  plus the labelled sentences of the other folds (never the test
  dialogue); two multilingual embedders; dense or hybrid (BM25 +
  dense, the RAG core's fusion);
- **view ablation**: single, pair, window and their fusions;
- **the answerer labelled pregunta**: sentences right after another
  speaker's pregunta that are not themselves pregunta, predicted
  pregunta;
- **combinations** with the Spanish encoder (``--es-folds``): R only
  for the encoder's "dudoso" labels; a weighted ensemble;
- **RG** (retrieval + judge): ``--run-judge claude_code`` asks Claude
  Code about the same 140 sentences as #82's run (``--ids``),
  ``--run-judge local`` asks the local judge about every sentence;
  verdicts are saved and scored on the next run.

Every fusion weight, temperature, mix and outcome table applied to a
fold is fitted on the other four.

Usage::

    HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=. .venv/bin/python \\
        experiments/skills/act_rag_eval.py \\
        --es-folds data/models/act-tagger/xlmr-es-folds
    # RG with Claude Code on #82's 140 sentences (140 calls)
    ... act_rag_eval.py --run-judge claude_code \\
        --ids experiments/skills/results/act_tagger/llm_claude_code.json
"""

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from experiments.skills.act_tagger_eval import (
    FOLDS,
    Corpus,
    compact,
    llm_predictions,
    rules_predictions,
    summary_files,
)
from src.adapters.outbound.bm25_index import BM25Builder
from src.adapters.outbound.encoder_act_tagger import write_json
from src.adapters.outbound.json_act_cards import JsonActCards
from src.adapters.outbound.minilm_embedder import SentenceTransformerEmbedder
from src.application.ports import Embedder, FloatArray
from src.application.skills.act_tagger import llm_outcome
from src.application.skills.rag_act_tagger import (
    ActCardIndex,
    Evidence,
    JudgeVerdict,
    RetrievalActTagger,
    RetrievalJudgeTagger,
    RetrievalSettings,
    answerer_as_question,
    mix_judged,
)
from src.domain.skills.act_calibration import (
    as_probabilities,
    cross_fit_outcomes,
    fit_temperature,
    fold_of,
    reference_rates,
    softmax,
)
from src.domain.skills.act_rag_prompting import JudgedActs
from src.domain.skills.act_retrieval import (
    VIEWS,
    CardChunk,
    FusionWeights,
    TheoryCorpus,
    closest_pairs,
    corpus_chunks,
    dialogue_chunks,
)
from src.domain.skills.act_scoring import score
from src.domain.skills.act_tagging import (
    Probabilities,
    Reference,
    top_category,
)
from src.domain.skills.dialogue_acts import CATEGORIES

EMBEDDERS = {
    "minilm-multi": ("sentence-transformers/"
                     "paraphrase-multilingual-MiniLM-L12-v2", ""),
    "e5-small": ("intfloat/multilingual-e5-small", "query: "),
}
"""Multilingual embedders in the local cache: name → (model, prefix)."""

GRID = [0.0, 0.25, 0.5, 1.0, 2.0]
"""Candidate weights of the pair and window views (single = 1)."""

MIXES = [round(0.1 * i, 1) for i in range(11)]
"""Candidate shares of the retrieval distribution in a mix."""

FLOOR = 0.02
HYBRID = 0.3
"""Share of BM25 in the hybrid similarity (fixed before measuring)."""

ABLATION = {
    "single": {"single": 1.0},
    "pair": {"pair": 1.0},
    "window": {"window": 1.0},
    "single+pair": None,
    "single+window": None,
    "single+pair+window": None,
}
"""View sets; ``None``: the pair/window weights are fitted."""

CONTEXT_ACTS = ("respuesta", "aceptacion", "rechazo", "compromiso",
                "pregunta")


class MemoEmbedder:
    """An embedder with an in-memory cache and an optional prefix."""

    def __init__(self, inner: Embedder, prefix: str = ""):
        """Wrap ``inner``; ``prefix`` goes before every text (e5)."""
        self.inner = inner
        self.prefix = prefix
        self.memo: Dict[str, FloatArray] = {}

    @property
    def model_name(self) -> str:
        """Return the wrapped model's name."""
        return self.inner.model_name

    def encode(self, texts: Sequence[str],
               progress: bool = False) -> FloatArray:
        """Return cached rows, encoding only new texts."""
        new = sorted({t for t in texts if t not in self.memo})
        if new:
            rows = self.inner.encode([self.prefix + t for t in new])
            self.memo.update(zip(new, rows))
        dim = next(iter(self.memo.values())).shape[0] if self.memo else 1
        if not texts:
            return np.zeros((0, dim), dtype=np.float32)
        return np.stack([self.memo[t] for t in texts]).astype(np.float32)


class Setup:
    """The Spanish set with what the extra measures need."""

    def __init__(self, data: str):
        """Read the set; note each sentence's previous gold and turn."""
        self.corpus = Corpus("own", f"{data}/own.jsonl", test_only=False)
        self.records = self.corpus.test
        self.refs = self.corpus.references
        self.folds = fold_of(self.corpus.groups, FOLDS)
        index = {(r.dialogue, r.order): n for n, r in enumerate(self.records)}
        self.gold = [r.categories[0] for r in self.records]
        self.previous: List[Optional[str]] = []
        self.changed: List[bool] = []
        for r in self.records:
            before = index.get((r.dialogue, r.order - 1))
            self.previous.append(None if before is None
                                 else self.gold[before])
            self.changed.append(
                before is not None
                and self.records[before].speaker != r.speaker)

    def split(self, fold: int) -> Tuple[List[int], List[int]]:
        """Return (train, test) indices of ``fold``."""
        return ([n for n, f in enumerate(self.folds) if f != fold],
                [n for n, f in enumerate(self.folds) if f == fold])

    def answerer(self, predictions: Sequence[Probabilities],
                 subset: Optional[Sequence[int]] = None) -> Dict[str, int]:
        """Count answerers labelled pregunta (on ``subset`` if given)."""
        rows = list(subset) if subset is not None else list(
            range(len(self.records)))
        wrong, total = answerer_as_question(
            [self.previous[n] for n in rows], [self.gold[n] for n in rows],
            [self.changed[n] for n in rows], predictions)
        return {"answerer_as_pregunta": wrong, "after_other_pregunta": total}


def chunks_for(setup: Setup, theory: List[CardChunk], with_folds: bool,
               fold: int) -> List[CardChunk]:
    """Return the index chunks for testing ``fold``."""
    if not with_folds:
        return theory
    train, _ = setup.split(fold)
    return theory + dialogue_chunks([setup.records[n] for n in train])


def evidence(setup: Setup, theory: List[CardChunk], embedder: MemoEmbedder,
             settings: RetrievalSettings, with_folds: bool
             ) -> Tuple[List[Evidence], float]:
    """Return every sentence's held-out evidence and seconds spent."""
    out: List[Optional[Evidence]] = [None] * len(setup.records)
    start = time.perf_counter()
    lexical = BM25Builder() if settings.lexical_weight > 0 else None
    shared: Optional[ActCardIndex] = None
    for fold in range(FOLDS):
        _, test = setup.split(fold)
        if with_folds or shared is None:
            index = ActCardIndex(chunks_for(setup, theory, with_folds, fold),
                                 embedder, settings, lexical)
            shared = index
        found = shared.evidence([setup.corpus.inputs[n] for n in test])
        for n, e in zip(test, found):
            out[n] = e
    assert all(e is not None for e in out)
    return [e for e in out if e is not None], time.perf_counter() - start


def view_logits(ev: Sequence[Evidence], views: Dict[str, float]
                ) -> FloatArray:
    """Return log fused votes (temperature 1) for every sentence."""
    weights = FusionWeights(views=views, floor=FLOOR)
    rows = []
    for e in ev:
        mixed = {c: 0.0 for c in CATEGORIES}
        norm = 0.0
        for view, p in e.votes.items():
            w = weights.views.get(view, 0.0)
            if p and w > 0:
                norm += w
                for c in CATEGORIES:
                    mixed[c] += w * p.get(c, 0.0)
        if norm <= 0:
            single = e.votes.get("single") or {}
            mixed = {c: single.get(c, 1.0 / 13) for c in CATEGORIES}
            norm = 1.0
        rows.append([math.log((1 - FLOOR) * mixed[c] / norm + FLOOR / 13)
                     for c in CATEGORIES])
    out: FloatArray = np.asarray(rows, dtype=np.float32)
    return out


def set_nll(p: FloatArray, refs: Sequence[Reference]) -> float:
    """Return the mean −ln P(reference set)."""
    total = 0.0
    for row, ref in zip(p, refs):
        mass = sum(row[CATEGORIES.index(c)] for c in ref.categories)
        total -= math.log(max(float(mass), 1e-15))
    return total / max(len(refs), 1)


def fit_fusion(setup: Setup, ev: Sequence[Evidence], rows: Sequence[int],
               fixed: Optional[Dict[str, float]], views: Sequence[str]
               ) -> FusionWeights:
    """Fit view weights (grid) and temperature on ``rows``."""
    refs = [setup.refs[n] for n in rows]
    sub = [ev[n] for n in rows]
    candidates: List[Dict[str, float]] = []
    if fixed is not None:
        candidates = [fixed]
    else:
        for pw in GRID if "pair" in views else [0.0]:
            for ww in GRID if "window" in views else [0.0]:
                candidates.append({"single": 1.0, "pair": pw,
                                   "window": ww})
    best: Optional[Tuple[float, FusionWeights]] = None
    for views_w in candidates:
        logits = view_logits(sub, views_w)
        t = fit_temperature(logits.astype(np.float64), refs)
        nll = set_nll(softmax(logits, t), refs)
        if best is None or nll < best[0]:
            best = (nll, FusionWeights(views=views_w, floor=FLOOR,
                                       temperature=t))
    assert best is not None
    return best[1]


def cross_fit_fusion(setup: Setup, ev: Sequence[Evidence],
                     fixed: Optional[Dict[str, float]], views: Sequence[str]
                     ) -> Tuple[List[Probabilities], List[FusionWeights]]:
    """Return held-out fused predictions and each fold's weights."""
    out: List[Probabilities] = [{} for _ in ev]
    fitted = []
    for fold in range(FOLDS):
        train, test = setup.split(fold)
        weights = fit_fusion(setup, ev, train, fixed, views)
        fitted.append(weights)
        logits = view_logits([ev[n] for n in test], weights.views)
        for n, p in zip(test, as_probabilities(
                softmax(logits.astype(np.float64), weights.temperature))):
            out[n] = p
    return out, fitted


def headline(s: Dict[str, Any]) -> Dict[str, Any]:
    """Return the figures a table shows for one approach."""
    keep = {k: s[k] for k in ("sentences", "top1", "top3", "macro_f1",
                              "log_loss", "ece")}
    keep["f1"] = {c: s["per_category"][c]["f1"] for c in CATEGORIES}
    return keep


def fit_mix(first: Sequence[Probabilities], second: Sequence[Probabilities],
            refs: Sequence[Reference]) -> float:
    """Return the share of ``second`` that minimises the log-loss."""
    def nll(a: float) -> float:
        rows = np.asarray([[(1 - a) * p.get(c, 0.0) + a * q.get(c, 0.0)
                            for c in CATEGORIES]
                           for p, q in zip(first, second)])
        return set_nll(rows, refs)
    return min(MIXES, key=nll)


def cross_fit_mix(setup: Setup, first: Sequence[Probabilities],
                  second: Sequence[Probabilities],
                  rows: Optional[Sequence[int]] = None
                  ) -> Tuple[List[Probabilities], List[float]]:
    """Return held-out mixes (share of ``second`` fitted on other folds).

    ``rows`` restricts everything to those sentences (``first`` and
    ``second`` are then aligned with ``rows``).
    """
    idx = list(rows) if rows is not None else list(range(len(first)))
    folds = [setup.folds[n] for n in idx]
    out: List[Probabilities] = [{} for _ in idx]
    shares = []
    for fold in range(FOLDS):
        train = [i for i, f in enumerate(folds) if f != fold]
        test = [i for i, f in enumerate(folds) if f == fold]
        if not test:
            continue
        a = fit_mix([first[i] for i in train], [second[i] for i in train],
                    [setup.refs[idx[i]] for i in train]) if train else 0.0
        shares.append(a)
        for i in test:
            out[i] = {c: (1 - a) * first[i].get(c, 0.0)
                      + a * second[i].get(c, 0.0) for c in CATEGORIES}
    return out, shares


def doubtful_swap(setup: Setup, encoder: Sequence[Probabilities],
                  other: Sequence[Probabilities], margin: float,
                  average: bool) -> Tuple[List[Probabilities], int]:
    """Use ``other`` where the encoder's top label is "dudoso".

    "Dudoso": the top probability is below its category's base rate
    (fitted on the other folds) plus ``margin``, as in the card of #82.
    ``average`` averages the two instead of replacing.
    """
    out: List[Probabilities] = list(encoder)
    swapped = 0
    for fold in range(FOLDS):
        train, test = setup.split(fold)
        rates = reference_rates([setup.refs[n] for n in train])
        for n in test:
            top = top_category(encoder[n])
            if encoder[n][top] < rates[top] + margin:
                swapped += 1
                out[n] = ({c: (encoder[n][c] + other[n].get(c, 0.0)) / 2
                           for c in CATEGORIES} if average else other[n])
    return out, swapped


def encoder_predictions(directory: str, setup: Setup, device: str,
                        cache: Path) -> List[Probabilities]:
    """Return held-out encoder-es predictions (cached as JSON)."""
    path = cache / "encoder_es_folds.json"
    if path.is_file():
        data: List[Probabilities] = json.loads(path.read_text())
        if len(data) == len(setup.records):
            return data
    from experiments.skills.act_tagger_eval import es_fold_predictions
    data = es_fold_predictions(directory, setup.corpus, device)
    cache.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return data


def overlap(corpus: TheoryCorpus, embedder: MemoEmbedder) -> Dict[str, Any]:
    """Return the closest pairs of category definitions."""
    names = [c.category for c in corpus.cards]
    vectors = embedder.encode([f"{c.definition} {c.function}"
                               for c in corpus.cards])
    pairs = closest_pairs(names, vectors, top=8)
    return {"embedder": embedder.model_name,
            "closest": [{"a": a, "b": b, "cosine": round(s, 4)}
                        for a, b, s in pairs]}


def load_verdicts(path: Path) -> Dict[str, JudgeVerdict]:
    """Return saved RG verdicts by sentence id."""
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {v["id"]: JudgeVerdict.model_validate(v) for v in data["verdicts"]}


def run_judge(args: argparse.Namespace, setup: Setup, theory: List[CardChunk],
              corpus: TheoryCorpus, embedder: MemoEmbedder,
              settings: RetrievalSettings, with_folds: bool,
              ev: Sequence[Evidence], out: Path) -> None:
    """Ask a judge (RG) about the chosen sentences and save the verdicts."""
    from src.adapters.outbound.sqlite_cache import SqliteCache
    from src.application.skills.judges import release
    from src.config import SkillsSettings
    from src.skills_bootstrap import SkillsContainer
    rows = list(range(len(setup.records)))
    if args.ids:
        wanted = {v["id"] for v in json.loads(
            Path(args.ids).read_text(encoding="utf-8"))["first"]}
        rows = [n for n, i in enumerate(setup.corpus.inputs)
                if i.id in wanted]
    if args.limit:
        rows = rows[:args.limit]
    app = SkillsContainer(SkillsSettings.model_validate({
        "judge": args.run_judge, "progress": False, "use_cache": False,
        "device": "cuda" if args.run_judge == "local" else "cpu",
        **({"llm": args.local_model, "local_bits": 4,
            "local_policy": "fixed"} if args.run_judge == "local" else {}),
    }))
    judge = app.claude_judge if args.run_judge == "claude_code" else (
        app.local_judge)
    cache = SqliteCache(args.judge_cache)
    path = out / f"rg_{args.run_judge}.json"
    done = load_verdicts(path)
    verdicts: List[JudgeVerdict] = []
    start = time.perf_counter()
    calls = 0
    for fold in range(FOLDS):
        _, test = setup.split(fold)
        mine = [n for n in rows if n in set(test)]
        if not mine:
            continue
        train, _ = setup.split(fold)
        weights = fit_fusion(setup, ev, train, None, VIEWS)
        index = ActCardIndex(chunks_for(setup, theory, with_folds, fold),
                             embedder, settings,
                             BM25Builder() if settings.lexical_weight else
                             None)
        tagger = RetrievalJudgeTagger(
            RetrievalActTagger(index, weights), corpus, judge, cache=cache)
        for n in mine:
            item = setup.corpus.inputs[n]
            if item.id in done:
                verdicts.append(done[item.id])
                continue
            got, _ = tagger.verdicts([item])
            calls += 1
            verdicts.append(got[0])
            if calls % 10 == 0:
                print(f"{calls} calls, {time.perf_counter() - start:.0f} s",
                      flush=True)
                write_json(str(path), {"judge": args.run_judge,
                                       "verdicts": [v.model_dump()
                                                    for v in verdicts]})
    release(judge)
    write_json(str(path), {
        "judge": args.run_judge, "calls": calls,
        "seconds": round(time.perf_counter() - start, 1),
        "verdicts": [v.model_dump() for v in verdicts]})
    print(f"Saved {path}: {len(verdicts)} verdicts, {calls} new calls")


def judge_scores(setup: Setup, verdicts: Dict[str, JudgeVerdict],
                 retrieval: Sequence[Probabilities]) -> Dict[str, Any]:
    """Score saved RG verdicts three ways, on the judged sentences."""
    rows = [n for n, i in enumerate(setup.corpus.inputs) if i.id in verdicts]
    if not rows:
        return {}
    judged: List[Optional[JudgedActs]] = [
        verdicts[setup.corpus.inputs[n].id].judged for n in rows]
    refs = [setup.refs[n] for n in rows]
    raw = [mix_judged(j, retrieval[n], 0.0) for j, n in zip(judged, rows)]
    mixed, shares = cross_fit_mix(
        setup, raw, [retrieval[n] for n in rows], rows)
    outcomes = [llm_outcome(j.labels[0][0] if j else None) for j in judged]
    table = cross_fit_outcomes(outcomes, refs,
                               [setup.corpus.groups[n] for n in rows], FOLDS)
    unparsed = sum(1 for j in judged if j is None)
    models = sorted({verdicts[setup.corpus.inputs[n].id].model
                     for n in rows})
    return {
        "rows": rows, "models": models, "unparsed": unparsed,
        "mix_shares": shares,
        "weights+R": (mixed, compact(score(refs, mixed))),
        "outcome table": (table, compact(score(refs, table))),
    }


def main() -> None:  # noqa: C901 - one linear experiment
    """Run the retrieval taggers and write the results."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default="data/dialogue_acts")
    parser.add_argument("--theory", default="knowledge/dialogue_acts/theory")
    parser.add_argument("--es-folds", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output",
                        default="experiments/skills/results/act_rag")
    parser.add_argument("--cache", default="data/models/act-tagger/cache")
    parser.add_argument("--llm", action="append", default=[
        "experiments/skills/results/act_tagger/llm_claude_code.json",
        "experiments/skills/results/act_tagger/llm_qwen3-8b_4bit.json"])
    parser.add_argument("--run-judge", choices=["claude_code", "local"],
                        default="")
    parser.add_argument("--ids", default="",
                        help="judge only the sentences of this verdict file")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--local-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--judge-cache", default="data/skills_cache/act_rag")
    parser.add_argument("--configs", default="",
                        help="comma list of configs to run (default all)")
    args = parser.parse_args()
    setup = Setup(args.data)
    cards = JsonActCards(args.theory)
    corpus = cards.load()
    theory = corpus_chunks(corpus)
    out = Path(args.output)
    embedders = {name: MemoEmbedder(SentenceTransformerEmbedder(model),
                                    prefix)
                 for name, (model, prefix) in EMBEDDERS.items()}
    configs = [(f"{source}/{name}/{kind}", source == "cards+folds", name,
                RetrievalSettings(lexical_weight=HYBRID if kind == "hybrid"
                                  else 0.0))
               for source in ("cards", "cards+folds")
               for name in EMBEDDERS for kind in ("dense", "hybrid")]
    if args.configs:
        configs = [c for c in configs if c[0] in args.configs.split(",")]
    refs = setup.refs
    results: Dict[str, Any] = {
        "theory_version": cards.version(),
        "chunks": {v: sum(1 for c in theory if c.view == v) for v in VIEWS},
        "card_examples": len(corpus.example_texts()),
        "overlap": {n: overlap(corpus, e) for n, e in embedders.items()},
        "configs": {}}
    predictions: Dict[str, List[Probabilities]] = {}
    evidences: Dict[str, List[Evidence]] = {}
    for name, with_folds, emb, settings in configs:
        ev, seconds = evidence(setup, theory, embedders[emb], settings,
                               with_folds)
        evidences[name] = ev
        fused, fitted = cross_fit_fusion(setup, ev, None, VIEWS)
        predictions[name] = fused
        s = compact(score(refs, fused))
        results["configs"][name] = {
            **headline(s), **setup.answerer(fused),
            "seconds": round(seconds, 1),
            "fold_weights": [w.model_dump() for w in fitted]}
        print(name, s["top1"], s["macro_f1"], flush=True)
    best = max(results["configs"], key=lambda n: (
        results["configs"][n]["top1"], -results["configs"][n]["log_loss"]))
    results["best_config"] = best
    ev = evidences[best]
    ablation = {}
    for label, fixed in ABLATION.items():
        views = label.split("+")
        preds, fitted = cross_fit_fusion(setup, ev, fixed, views)
        s = compact(score(refs, preds))
        ablation[label] = {
            **headline(s), **setup.answerer(preds),
            "fold_weights": [w.model_dump() for w in fitted]}
        predictions[f"R views {label}"] = preds
    results["ablation"] = ablation
    cards_best = max((n for n in results["configs"] if n.startswith("cards/")),
                     key=lambda n: results["configs"][n]["top1"])
    examples_only = corpus_chunks(corpus, theory=False)
    theory_ablation = {}
    for name in dict.fromkeys([best, cards_best]):
        _, wf, emb, settings = next(c for c in configs if c[0] == name)
        ev_x, _ = evidence(setup, examples_only, embedders[emb], settings, wf)
        preds, _ = cross_fit_fusion(setup, ev_x, None, VIEWS)
        theory_ablation[f"{name} without definitions/cues/boundaries"] = {
            **headline(compact(score(refs, preds))), **setup.answerer(preds)}
    results["theory_ablation"] = theory_ablation
    _, wf, emb, settings = next(c for c in configs if c[0] == best)
    if args.run_judge:
        run_judge(args, setup, theory, corpus, embedders[emb], settings, wf,
                  ev, out)
        return
    rag = predictions[best]
    rules, _ = rules_predictions(setup.corpus)
    baselines: Dict[str, Any] = {
        "rules": {**headline(compact(score(refs, rules))),
                  **setup.answerer(rules)}}
    if args.es_folds:
        enc = encoder_predictions(args.es_folds, setup, args.device,
                                  Path(args.cache))
        baselines["encoder-es"] = {**headline(compact(score(refs, enc))),
                                   **setup.answerer(enc)}
        combos: Dict[str, Any] = {}
        for avg in (False, True):
            swapped, count = doubtful_swap(setup, enc, rag, 0.45, avg)
            key = ("encoder-es, R averaged in on dudoso" if avg
                   else "encoder-es, R on dudoso")
            combos[key] = {**headline(compact(score(refs, swapped))),
                           **setup.answerer(swapped), "dudoso": count}
            predictions[key] = swapped
        ens, shares = cross_fit_mix(setup, enc, rag)
        combos["encoder-es + R ensemble"] = {
            **headline(compact(score(refs, ens))), **setup.answerer(ens),
            "r_share_per_fold": shares}
        predictions["encoder-es + R ensemble"] = ens
        results["combinations"] = combos
        predictions["encoder-es"] = enc
    for path in args.llm:
        if not Path(path).is_file():
            continue
        model, judged, preds = llm_predictions(path, setup.corpus)
        sub = [refs[n] for n in judged]
        baselines[f"llm:{model} ({len(judged)})"] = {
            **headline(compact(score(sub, preds))),
            **setup.answerer(preds, judged)}
    results["baselines"] = baselines
    rg: Dict[str, Any] = {}
    for judge in ("claude_code", "local"):
        verdicts = load_verdicts(out / f"rg_{judge}.json")
        if not verdicts:
            continue
        got = judge_scores(setup, verdicts, rag)
        rows = got["rows"]
        meta = json.loads((out / f"rg_{judge}.json").read_text())
        entry: Dict[str, Any] = {
            "models": got["models"], "sentences": len(rows),
            "unparsed": got["unparsed"], "mix_shares": got["mix_shares"],
            "calls": meta.get("calls"), "seconds": meta.get("seconds")}
        for variant in ("weights+R", "outcome table"):
            preds, s = got[variant]
            entry[variant] = {**headline(s), **setup.answerer(preds, rows)}
            if judge == "local" and len(rows) == len(setup.records):
                predictions[f"RG {judge} ({variant})"] = preds
        same = {"R": rag, "rules": rules}
        if "encoder-es" in predictions:
            same["encoder-es"] = predictions["encoder-es"]
        entry["same_sentences"] = {
            k: {**headline(compact(score([refs[n] for n in rows],
                                         [v[n] for n in rows]))),
                **setup.answerer([v[n] for n in rows], rows)}
            for k, v in same.items()}
        rg[judge] = entry
    results["rg"] = rg
    write_json(str(out / "scores.json"), results)
    own = {f"rag {best}": compact(score(refs, rag))}
    for name in ("encoder-es + R ensemble", "encoder-es, R on dudoso"):
        if name in predictions:
            own[name] = compact(score(refs, predictions[name]))
    for name, preds in predictions.items():
        if name.startswith("RG "):
            own[name] = compact(score(refs, preds))
    for name, s in list(own.items()):
        s["sentences"] = len(refs)
    summary_files(own, str(out.parent))
    print(json.dumps({k: v for k, v in results.items()
                      if k not in ("configs", "ablation")}, indent=1,
                     ensure_ascii=False)[:4000])


if __name__ == "__main__":
    main()
