# «Competencia» page (#291)

`competition.html` shows a non-technical audience how the system works,
which model does what, what retrieval (RAG) saves, how the system
compares with companies that help during job interviews and in call
centres, how the customer's content is kept up to date, and what is
planned. It lives under «Administración», after «Investigación», on every
page (sidebar, «Más» on a phone, the «Administración» overview cards and
the Ctrl/Cmd + K switcher, which lists the sidebar).

| File | What it holds |
|---|---|
| `transcriber/adapters/inbound/web/competition.html` | The shell (same as `study.html`), empty `<main>` |
| `…/web/competition/data.json` | **Every figure and every sentence about a company or about our system**, in Spanish and English, with its source |
| `…/web/js/competition.js` | Entry point: loads the data, draws, redraws on a language change, installs «Compartir pantalla» (#284) |
| `…/web/js/lib/competition.js` | Pure logic: schema check, language pick, grouping, the RAG estimate |
| `…/web/js/views/competition.js` | The DOM (no canvas: the shared screen mirrors it) |
| `…/web/js/i18n/es.js`, `en.js` | Headings and labels (`comp.*`, `nav.competition`, `admin.competition`) |
| `tests/web/competition.test.mjs` | Data checks (sources, dates, both languages, nothing private), the estimate's numbers, navigation on every page |

The data file is kept by hand (there is no build script: its inputs are
research notes outside the repository and repository documents, not a
generated artefact). Change it, run `node --test tests/web/`, and the page
follows; the page holds no figure of its own.

## 1. Sections

1. **Cómo funciona** — four steps (`steps`): documents in → kept by topic
   (a job offer becomes an interview plan, manuals become call material)
   → help during the conversation → report. Steps 3 and 4 carry the «RAG»
   mark; below, what uses retrieval and what does not (`rag_uses`).
2. **Con RAG y sin RAG** — the estimate of §3, marked «Estimación», with
   its assumptions in one line and what is not measured.
3. **Qué modelo hace cada cosa** — per phase, the model and whether it
   runs on the customer's processing server or at an outside service
   (`models`), with the privacy sentence (`privacy`): the voice is turned
   into text locally; the **text** goes to Claude (Anthropic).
4. **La competencia** — one table (cards on a phone): company, price,
   users and contract, where it runs, grouped by purpose, each row with
   the link to the page where the price was seen and the date; our row
   last. The only link to repository documents is «Fuentes y método» at
   the end (this document). Then «Dónde destacamos» (`strengths`) and «Dónde vamos por
   detrás» (`gaps`).
5. **Tu contenido, siempre al día** — what can be changed today
   (`content_today`; each item's source is in the data file and in §4, not linked on the page), the
   sentence «Hoy nada aprende solo de las llamadas» (`content_automatic`)
   and what is planned (`content_planned`).
6. **Próximas mejoras** — `roadmap`, introduced as plans with no date.

## 2. External figures: where each comes from

All prices are US list prices in USD, read on the vendor's own pricing
page (`official`) or on its AWS Marketplace listing (`marketplace`) on
**2026-11-27** (research notes of that date, not in the repository); they
were not checked in euros or for Spain, and the page says so.
«Sin precio público» (`basis: none`) means the vendor's page shows no
figure. «Precio no comprobado» is used where the notes only say the
reading tool saw no price (Metaview's Notetaker and Application Review
tabs): that is not proof that none is published.

The column «Cómo se ofrece (según su página)» and its badge «Servicio en
la nube» repeat what the notes found: every company checked is sold as a
cloud service (SaaS) according to its own documentation; the specific
evidence re-checked on 27-11 is Cresta («deployed on AWS») and
Observe.AI (SaaS listings on AWS Marketplace); the rest is from the
earlier report of 24-11 (the vendors' documentation, column «Despliegue:
Nube»). No vendor was found offering live assistance run on the
customer's hardware, which does not prove none exists; the page does not
claim it. The page makes no claim about the competitors' Spanish (the
earlier report's remark was not re-verified).

| Company | Figure on the page | Basis | Source |
|---|---|---|---|
| Talkdesk | from 85 (Digital Essentials, no voice) to 225 USD per user per month; with voice from 105 (Voice Essentials); Copilot without price; Express: 25 licences, no commitment | official | https://www.talkdesk.com/pricing/ |
| NICE CXone | 110–249 USD per agent per month, +0.25 USD per session on the top plan; copilots by usage, no rate | official | https://www.nice.com/pricing |
| Five9 | 119–159 USD per concurrent user per month, «usage-based pricing may apply, with a minimum of 50 seats»; 3,000 AI minutes per seat + usage | official | https://www.five9.com/pricing |
| Twilio | Agent Copilot (beta; «Public Beta prices are subject to change») 0.035 USD per voice minute; Flex from 1 USD per active hour or 150 USD per user per month | official | https://www.twilio.com/en-us/flex/pricing |
| Cresta | 150,000 USD for 12 months up to 100,000 calls; 1.50 USD per extra call; not cancellable or refundable | marketplace | https://aws.amazon.com/marketplace/pp/prodview-lmdwown6xoo6q |
| Observe.AI | 828 USD per agent; the listing says «per Agent per month» but shows the amount as «Cost/12 months» (unclear); 12/24/36-month contracts, no refunds | marketplace | https://aws.amazon.com/marketplace/pp/prodview-lgvu5s7gueesi |
| BrightHire | no public price; quoted by applicants, interviews or company size | none | https://brighthire.com/pricing/ |
| HireVue | no public price | none | https://www.hirevue.com/pricing |
| Metaview | AI interviewer (Screening): no public price; Notetaker and Application Review: price not checked (the tabs showed no price to the reading tool); sourcing agent 0, 100 or 300 USD per user per month | official | https://www.metaview.ai/pricing |
| Retell AI | 0.07–0.31 USD per minute; 20 concurrent calls included, 8 USD per extra line per month | official | https://www.retellai.com/pricing |
| Vapi | 0.05 USD per minute plus models at cost; Pro from 999 USD a month; 4–30 concurrent calls | official | https://vapi.ai/pricing |
| PolyAI | API 0.02 USD per connected minute plus text tokens, telephony not included; 10 concurrent calls; managed platform by quote | official | https://poly.ai/pricing/ |
| Claude Haiku 5.5 (for §3) | 0.10 USD per million input tokens, 0.50 output; ×5 (0.50 / 2.50) when the prompt exceeds 100,000 tokens | official | https://platform.claude.com/docs/en/about-claude/pricing |

**Left out on purpose** (accuracy rules of #291): Gong and Balto (only
aggregator figures, Vendr medians seen in search snippets), Genesys
(its figures were seen only in a search snippet; the page could not be
read), every third-party estimate (blogs, G2, Vendr). They can come back
when their own page shows a price. Names are written as the companies
write them; no logos.

## 3. «Con RAG y sin RAG»: the arithmetic

**What is compared.** How much text the model must read for one live
suggestion, what that costs, and how long the parts that are measured
take. **Answer quality with and without the knowledge base is not
measured** for suggestions or the voice bot, so the page does not claim
better answers; it says so in its assumptions line.

**Measured inputs (all synthetic data):**

- The latency test's suggestion prompt is about **1,750 input tokens**
  (system ~700 + 9 transcript lines + 3 retrieved pieces) and got
  **44–90** output tokens typically, with outliers of 384 and 797 —
  `docs/live/HAIKU_LATENCY.md`. This is the latency experiment's prompt,
  not the production templates, which retrieve `top_k: 4`
  (`config/products/live_assist.*.yaml`); the page says «la petición
  medida en la prueba de latencia (3 trozos cortos)». Output: the page
  uses «unos 90»; `docs/live/ASSIST_MODES.md` measured medians of 79 to
  212.5 output tokens with a cap of 240, so 90 is not an upper limit.
- The 3 pieces in that measured prompt are 984 characters
  (`experiments/live/haiku_latency.py`, `CHUNKS`) ≈ **250 tokens**; so the
  rest of the prompt is 1,750 − 250 = **1,500 tokens** (`promptBase`).
- Search time in live calls **9–27 ms**; model time to a full suggestion
  with retrieval, p50 **1.26–2.18 s** across the three runs —
  `docs/live/LIVE_MEASURES.md` §1. Search on a 1,030-piece library:
  p50 10.6 ms — `experiments/knowledge/QUERY_LATENCY.md`.
- Size of the libraries, counted on the committed files (characters of
  the `text` fields of the `.jsonl` files plus `guia-general.md`):
  `support.demo` 11,955 + 413 = 12,368 characters; `interview.generic`
  10,519 + 38,252 + 528 = 49,299 characters. Counted with
  `python3 -c` over `knowledge/domains/*/` on 2026-11-27.

**Assumptions:** 4 characters per token (the rule the cost notes use; the
real Haiku 5.5 tokenizer is not available offline); list price, no prompt
cache, synchronous calls; without retrieval the same request would carry
the whole library in place of the retrieved pieces.

**Sample libraries** (`ragRows`, one chart on one scale): the measured
prompt against the same prompt with the whole library.

| Row | Tokens per suggestion | Ratio | Cost of 1,000 suggestions |
|---|---|---|---|
| With RAG (latency-test prompt, 3 short pieces) | 1,750 | 1 | (1,750 × 0.10 + 90 × 0.50) / 10⁶ × 1,000 = **0.22 USD** |
| Without RAG, `support.demo` (12,368 / 4 ≈ 3,100) | 1,500 + 3,100 = 4,600 | 2.6 | (4,600 × 0.10 + 45) / 1,000 = **0.505 USD** |
| Without RAG, `interview.generic` (49,299 / 4 ≈ 12,300) | 1,500 + 12,300 = 13,800 | 7.9 | (13,800 × 0.10 + 45) / 1,000 = **1.425 USD** |

**Large library, a maximum** (`ragMaxRows`, its own chart and scale,
labelled «Como máximo: si cada trozo fuese de 1.000 caracteres»): the
1,030 pieces of the search test, every one assumed full (1,000
characters, the chunk limit), so 1,030 × 1,000 / 4 = 257,500 tokens.
Both sides use that piece size: with retrieval the products' 4 pieces,
4 × 1,000 / 4 = 1,000 tokens.

| Row | Tokens per suggestion | Ratio | Cost of 1,000 suggestions |
|---|---|---|---|
| With RAG, 4 full pieces | 1,500 + 1,000 = 2,500 | 1 | (2,500 × 0.10 + 45) / 1,000 = **0.295 USD** |
| Without RAG, the whole library | 1,500 + 257,500 = 259,000 | 104 (103.6) | above 100,000 tokens: (259,000 × 0.50 + 90 × 2.50) / 1,000 = **129.7 USD** |

`tests/web/competition.test.mjs` checks every number of both tables.

**Time without RAG: not measured.** The repository has no measurement of
how the model's time grows with the prompt, so the page shows «Sin medir»
instead of a number.

**With a prompt cache** the large case would cost less: cached reads above
100,000 tokens are 0.05 USD per million (Anthropic pricing, same page),
so 259,000 × 0.05 / 10⁶ + 90 × 2.50 / 10⁶ ≈ 0.0132 USD per suggestion,
about 13 USD per 1,000 — still about 45 times the 0.295 USD with
retrieval, and ignoring cache writes. Not shown on the page.

**Quality with and without retrieval** was measured only on a different
task, tagging the type of each sentence (dialogue acts): Claude alone
74.3 %, retrieval + Claude 86.4 %, on the same 140 synthetic sentences
(`docs/skills/DIALOGUE_ACT_TAGGER.md` §0, §9); Qwen3-8B 55.3 % → 69.8 %
with retrieval on 1,534 sentences (§9.7). The system does not use
retrieval there today: the default tagger is the fine-tuned encoder
(73.1 %, no retrieval). The page says this.

## 4. Our system: what each statement rests on

| Statement | Source |
|---|---|
| «La voz no va a ningún servicio de IA externo»: voice to text on the processing server (Whisper large-v3-turbo) | `docs/live/LIVE_ASR.md`, `transcriber/config.py` |
| Who speaks: one channel per person, Sortformer with one microphone; DiariZen for recordings | `docs/live/MONO_DIARIZATION.md`, `transcriber/config.py` |
| Documents: Docling, RapidOCR, Florence-2; PDF up to 500 MB, video up to 1 GB, a video searched by what is said | `docs/KNOWLEDGE.md` §4.1, §6–8 |
| Search: multilingual MiniLM + BM25 | `docs/KNOWLEDGE.md`, `src/config.py` |
| Live help, voice bot, plan and report: Claude Haiku through `claude -p` | `docs/live/ASSIST_MODES.md`, `src/adapters/outbound/claude_code_judge.py` (`LIVE_CLAUDE_MODEL`) |
| Post-call analysis: Opus by default, local Qwen3 8B/4B as an option | `src/config.py` (`DEFAULT_CLAUDE_CODE_MODEL`), `docs/SETTINGS.md` |
| Sentence type: fine-tuned XLM-R, 73.1 % vs Qwen3-8B 55.3 % (1,534 synthetic sentences) | `docs/skills/DIALOGUE_ACT_TAGGER.md` §0 |
| Voice: Piper or Pocket TTS on the CPU | `docs/live/VOICEBOT.md`, `orchestrator/domain/config.py` |
| One live session at a time | `docs/live/LIVE_SESSION.md` |
| Suggestion on screen 1.5 s (interview) / 2.7 s (support) after speech ends, p50, synthetic | `docs/live/LIVE_MEASURES.md` §1 |
| «Mejor evitar» hints with the reason and an alternative; no mark for the candidate during the interview | `orchestrator/application/avoid.py`, `docs/live/ASSIST_MODES.md` |
| Uses retrieval: live help (support and interview), voice bot and AI interviewer, interview plan (method corpus), interview report. Does not: «lo que hay que decir» (similarity to example sentences), questions to avoid (rules file), post-call analysis | `docs/live/ASSIST_MODES.md`, `docs/live/VOICEBOT_INTERVIEWER.md` §2–§6, `docs/live/MANDATORY_ITEMS.md`, `docs/live/INTERVIEW_KNOWLEDGE.md` |

**«Tu contenido» today:** documents uploaded, their metadata changed
(re-indexed) or deleted (`docs/KNOWLEDGE.md` §4); interview plans edited
with a revision check and removed points never given again
(`docs/live/VOICEBOT_INTERVIEWER.md` §3, §6.4, `retired_points`);
confirmed answers carried between rounds (`docs/live/CANDIDATE_PROFILE.md`);
mandatory items versioned in the domain's manifest
(`docs/live/MANDATORY_ITEMS.md`). **Nothing learns from calls by itself
today**; the two «Previsto» items and the roadmap are plans with no date.

**Where we are behind** comes from the same research notes (no phone
system input, one session, synthetic figures, shared access code, no
CRM/ATS link, no certification or legal review, installed by hand), each
checked against the repository documents above.

## 5. How to update

- A price changes: edit the company's `price` / `users` (both languages)
  and the source's `seen` date in `sources`; if the page no longer shows a
  figure, set `basis: "none"` and «Sin precio público» / «No public
  price». Never publish an aggregator or third-party figure as a price.
- A new company: add its source (`label`, `url`, `seen`) and a row with
  `kind` (`callcentre`, `interviews`, `voicebot`), `basis`, `runs`.
- A new measurement of tokens, latency or price: change the inputs under
  `rag`, then this document's §3 and the expected numbers in
  `tests/web/competition.test.mjs`.
- Something becomes true of our system (e.g. learning from calls, a
  phone connection): move it from `content_planned` / `roadmap` /
  `gaps` to where it belongs, with its `source`.
- Update `updated` (shown at the bottom of the page).
