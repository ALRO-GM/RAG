# Seed domains and their retrieval (PL-9, #146)

Two synthetic authored domains in `knowledge/domains/`, Spanish, with
manifest vocabularies and per-record metadata. All content is invented
(the company "Lumbre Hogar" does not exist); no figure or policy in it
is real.

| Domain | Content | Chunks |
|---|---|---|
| `interview.generic` | job-interview guide: script phases `opening`, `experience`, `technical`, `behavioural`, `closing`; follow-up questions; what not to ask legally. No candidate scoring and no state or emotion signals about the candidate | 65 |
| `support.demo` | returns, billing, delivery, escalation policy, plus the six expressed customer signals of `docs/platform/ARCHITECTURE.md` §8.1 (`complaint`, `stated_frustration`, `urgency`, `churn_risk`, `confusion`, `satisfaction`) as `intent` values, with `tags: [signal]` | 67 |

Each domain is a `manifest.yaml`, one Markdown guide with front matter
and `fragmentos.jsonl` (one record = one chunk, with `intent`,
`script_phase`, `tags`, `category`).

## Measurement

`experiments/platform/seed_eval.py` indexes both domains (`force`) with
the knowledge embedder on CPU and sends 20 labelled queries per domain
(`experiments/platform/queries/<domain>.jsonl`, 1-2 relevant chunk ids
each) through `KnowledgeQueryService`, in process, hybrid mode,
`top_k` 5. The queries were written before the first run and not edited
after seeing results. Three conditions: no filter; `script_phase`; and
`intent` + `script_phase`. The filters are the labels of the relevant
chunks, i.e. an oracle (a perfect classifier): the filtered rows are an
upper bound, not what a live suggester will get. recall@5 is the mean
share of a query's relevant chunks in the top 5; hit@5 the share of
queries with at least one. Raw output: `seed_eval.json`.

| Domain | Filters | recall@5 | hit@5 |
|---|---|---|---|
| interview.generic | none | 0.850 | 0.950 |
| interview.generic | script_phase | 0.875 | 1.000 |
| interview.generic | intent + script_phase | 1.000 | 1.000 |
| support.demo | none | 0.725 | 0.850 |
| support.demo | script_phase | 0.800 | 0.850 |
| support.demo | intent + script_phase | 0.975 | 1.000 |

Limits: 20 queries per domain (one query is 5 points of hit@5), written
by the same author as the content and in paraphrase of it, so the
no-filter numbers are probably optimistic for real calls; the chunks
are short and the domains small (about 65 chunks), where the filter
mostly removes distractors. Nothing here validates the content itself.

Run: `HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=. .venv/bin/python
experiments/platform/seed_eval.py`.
