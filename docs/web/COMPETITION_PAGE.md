# «Info» page (#291, #295)

`info.html` («Competencia» until #295) shows a non-technical audience,
in this order: what our system does and where it stands out, what the
same service costs against each company, one comparison chart per task
with every company, us against one company at a time, the published
prices and where we are behind, and then how it works (steps, what
retrieval saves, which model does what, how the content is kept up to
date, what is planned). It lives under «Administración», after
«Investigación», on every page (sidebar, «Más» on a phone, the
«Administración» overview cards and the Ctrl/Cmd + K switcher, which
lists the sidebar), named «Info» in both languages. The old address
`competition.html` only forwards to `info.html` keeping the section
(`#coste`, `#tareas`…): a `<meta http-equiv="refresh">` and
`js/info-redirect.js` (no inline script), plus a plain link.

| File | What it holds |
|---|---|
| `transcriber/adapters/inbound/web/info.html` | The shell (same as `study.html`), empty `<main>` |
| `…/web/competition.html`, `…/web/js/info-redirect.js` | The old address, forwarding to `info.html` |
| `…/web/competition/data.json` | **Every figure and every sentence about a company or about our system**, in Spanish and English, with its source |
| `…/web/js/competition.js` | Entry point: loads the data, draws, redraws on a language change, installs «Compartir pantalla» (#284) |
| `…/web/js/lib/competition.js` | Pure logic: schema check, language pick, grouping, the RAG estimate, the cost per scenario, the task cells, us against one company |
| `…/web/js/views/competition.js` | The DOM (no canvas: the shared screen mirrors it); every chart is a table |
| `…/web/js/i18n/es.js`, `en.js` | Headings and labels (`comp.*`, `nav.competition`, `admin.competition`) |
| `tests/web/competition.test.mjs` | Data checks (sources, dates, both languages, nothing private, «no» only where backed), every number of both estimates and of the cost chart, navigation on every page, the old address |

The data file is kept by hand (there is no build script: its inputs are
research notes outside the repository and repository documents, not a
generated artefact). Change it, run `node --test tests/web/`, and the page
follows; the page holds no figure of its own.

## 1. Sections, in order

1. **Lo que hacemos y dónde destacamos** — `strengths`, as cards. The
   first is «La voz no va a ningún servicio de IA externo» (exact
   wording); the second, «Tu contenido mejora con cada equipo», says
   that each team uploads and changes its documents and its interview
   plan, every change is kept, and nothing learns from the calls by
   itself (`content_today`, `content_automatic`).
2. **Cuánto cuesta el mismo servicio** — one chart per scenario, a
   switch between «Equipo pequeño: 5 personas» and «Equipo grande: 50
   personas» (§3c). Us first: a range marked «Estimación de coste, no es
   un precio»; then each company's published price applied to the same
   scenario, or «Precio no publicado» / «No comparable» with the
   service it sells instead. Under the chart: how to read the bar, our
   assumptions in one line, the same with Opus, and that their prices
   include their phone system or whole service while ours is only the
   running cost.
3. **Tarea por tarea** — a legend (Sí / En parte / No / No publicado,
   each a shape and a word; «No publicado» does not mean the company
   lacks it), then one chart per task (`tasks`, §3b): every company in
   the same order, us first with an accent edge and «Nosotros», a mark
   and a short sentence per cell, «Según su web, lo tienen N de 12
   empresas», and the cell sources under each chart. Our strengths come
   first, «Varias conversaciones a la vez» (where we are behind) last.
4. **Nosotros frente a cada una** — a company picker; for the company
   chosen: its price, users and where it runs, then «Nosotros sí; X no o
   solo en parte», «Nosotros sí; X no lo publica», «X sí; nosotros no o
   solo en parte» (the task cells where it is ahead plus its `offers`)
   and «Los dos» (`versus` in the lib), with the sources.
5. **Precios publicados** — the price table (cards on a phone): company,
   price, users and contract, where it runs, grouped by purpose, each
   row with the link to the page where the price was seen and the date;
   our row last. Then «Dónde vamos por detrás» (`gaps`).
6. **Cómo funciona** — four steps (`steps`): documents in → kept by topic
   (a job offer becomes an interview plan, manuals become call material)
   → help during the conversation → report. Steps 3 and 4 carry the «RAG»
   mark; below, what uses retrieval and what does not (`rag_uses`).
7. **Con RAG y sin RAG** — the estimate of §3, marked «Estimación», with
   its assumptions in one line and what is not measured.
8. **Qué modelo hace cada cosa** — per phase, the model and whether it
   runs on the customer's processing server or at an outside service
   (`models`), with the privacy sentence (`privacy`): the voice is turned
   into text locally; the **text** goes to Claude (Anthropic).
9. **Tu contenido, siempre al día** — what can be changed today
   (`content_today`; each item's source is in the data file and in §4, not linked on the page), the
   sentence «Hoy nada aprende solo de las llamadas» (`content_automatic`)
   and what is planned (`content_planned`).
10. **Próximas mejoras** — `roadmap`, introduced as plans with no date.

Every chart is an HTML table (`role` attributes kept so that the phone
layout, where each row becomes a card, does not hide it from screen
readers): the rows and cells are the chart's text equivalent, the bars
and mark shapes are `aria-hidden` and always sit next to their words.
Light and dark come from the design tokens; nothing rests on colour
alone (our row has «Nosotros» and an edge, a level has a shape and a
word, the chosen scenario a check mark). The chosen scenario and company survive
a change of language.

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

## 3b. «Tarea por tarea»: every cell and where it comes from

Levels: `yes` (Sí), `partial` (En parte), `no` (No), `unknown` (No
publicado). A company the file does not fill for a task is `unknown`;
the page says once that «No publicado» means not seen on its website or
not checked, **not** that the company lacks it. «No» about a company is
used only in «Dónde se procesa la voz», backed by its own page being
the cloud service of §2 (the test checks it); everything else that the
notes did not confirm stays `unknown`, or `partial` when the function is
confirmed but the part the task asks for is not (said in the cell).

The feature pages were read on **2026-11-24** (research notes of that
date, outside the repository: only pages the notes mark as read by the
author or as the vendor's own documentation; search-engine snippets were
not used for cells); prices and the cloud column on 2026-11-27 (§2).
Our cells cite repository documents.

| Task | Company | Level | Cell | Source |
|---|---|---|---|---|
| Job interviews | **us** | yes | Plan from the job offer, what is still to ask and questions to avoid; and a voice interviewer | docs/live/ASSIST_MODES.md |
| Job interviews | BrightHire | yes | Live guide for the interviewer and an AI interviewer that scores | https://brighthire.com/ |
| Job interviews | HireVue | yes | Video interviews and an AI interviewer | https://www.hirevue.com/platform |
| Job interviews | Metaview | yes | AI interviewer | https://www.metaview.ai/pricing |
| Where the voice is processed | **us** | yes | On your server; the text does go to Claude (Anthropic) | docs/live/LIVE_ASR.md |
| Where the voice is processed | Talkdesk | no | In its cloud, according to its website | https://www.talkdesk.com/pricing/ |
| Where the voice is processed | NICE CXone | no | In its cloud, according to its website | https://www.nice.com/pricing |
| Where the voice is processed | Five9 | no | In its cloud, according to its website | https://www.five9.com/pricing |
| Where the voice is processed | Twilio | no | In its cloud, according to its website | https://www.twilio.com/en-us/flex/pricing |
| Where the voice is processed | Cresta | no | In its cloud, according to its website | https://aws.amazon.com/marketplace/pp/prodview-lmdwown6xoo6q |
| Where the voice is processed | Observe.AI | no | In its cloud, according to its website | https://aws.amazon.com/marketplace/pp/prodview-lgvu5s7gueesi |
| Where the voice is processed | BrightHire | no | In its cloud, according to its website | https://brighthire.com/pricing/ |
| Where the voice is processed | HireVue | no | In its cloud, according to its website | https://www.hirevue.com/pricing |
| Where the voice is processed | Metaview | no | In its cloud, according to its website | https://www.metaview.ai/pricing |
| Where the voice is processed | Retell AI | no | In its cloud, according to its website | https://www.retellai.com/pricing |
| Where the voice is processed | Vapi | no | In its cloud, according to its website | https://vapi.ai/pricing |
| Where the voice is processed | PolyAI | no | In its cloud, according to its website | https://poly.ai/pricing/ |
| What must be said | **us** | yes | A list in red until it is said; the record says what was missing | docs/live/MANDATORY_ITEMS.md |
| What must be said | Talkdesk | partial | Step-by-step scripts; it does not say whether it checks what was said | https://www.talkdesk.com/agent-assist/ |
| What must be said | Five9 | yes | Live checklists | https://www.five9.com/products/capabilities/agent-assist |
| What must be said | Twilio | yes | Flags script violations during the call | https://www.twilio.com/en-us/products/conversational-ai/conversational-intelligence |
| What must be said | Cresta | yes | A list of steps during the call | https://cresta.com/product/agent-assist |
| Live help with your documents | **us** | yes | Searches your documents and suggests what to say in 1.5–2.7 s (synthetic recordings) | docs/live/LIVE_MEASURES.md §1 |
| Live help with your documents | Talkdesk | yes | Answers from your knowledge base (add-on) | https://www.talkdesk.com/agent-assist/ |
| Live help with your documents | NICE CXone | yes | Knowledge and next steps during the call | https://www.nice.com/products/copilot-for-agents |
| Live help with your documents | Five9 | partial | Live guidance cards; it does not say which documents they come from | https://www.five9.com/products/capabilities/agent-assist |
| Live help with your documents | Twilio | partial | Suggested answers from your sources; live only in English and in beta | https://www.twilio.com/docs/flex/admin-guide/setup/copilot |
| Live help with your documents | Cresta | yes | Answers with their sources cited | https://cresta.com/product/agent-assist |
| Live help with your documents | Observe.AI | partial | Live help; with your documents, not checked | https://aws.amazon.com/marketplace/pp/prodview-lgvu5s7gueesi |
| Your content, up to date | **us** | yes | PDFs up to 500 MB and videos up to 1 GB; the interview plan keeps every version | docs/KNOWLEDGE.md |
| Your content, up to date | Talkdesk | unknown | Uses your knowledge base; how it is updated, not checked | https://www.talkdesk.com/agent-assist/ |
| Your content, up to date | NICE CXone | unknown | Works best with its own knowledge manager | https://www.nice.com/products/copilot-for-agents |
| Your content, up to date | Twilio | yes | You upload your sources for the answers | https://www.twilio.com/docs/flex/admin-guide/setup/copilot |
| Your content, up to date | Cresta | unknown | Answers from your sources; how they are uploaded, not checked | https://cresta.com/product/agent-assist |
| Your content, up to date | Retell AI | yes | Knowledge bases: 10 free, 8 USD a month for each extra one | https://www.retellai.com/pricing |
| Turning the conversation into text | **us** | yes | Live and from recordings; one channel per person or a single microphone | docs/live/MONO_DIARIZATION.md |
| Turning the conversation into text | Talkdesk | yes | Live transcription (add-on) | https://www.talkdesk.com/agent-assist/ |
| Turning the conversation into text | Five9 | yes | Real-time transcription | https://www.five9.com/products/capabilities/agent-assist |
| Turning the conversation into text | Twilio | yes | Live transcription, billed separately | https://www.twilio.com/docs/flex/admin-guide/setup/copilot |
| Turning the conversation into text | Cresta | yes | Live transcription | https://cresta.com/product/agent-assist |
| Turning the conversation into text | BrightHire | partial | AI notes of the interview | https://brighthire.com/ |
| Turning the conversation into text | Metaview | partial | Interview notes | https://www.metaview.ai/pricing |
| Analysis after the call | **us** | yes | A report and each person's skills analysis | docs/SETTINGS.md |
| Analysis after the call | Talkdesk | yes | AI call evaluation (add-on) | https://support.talkdesk.com/hc/en-us/articles/50371682083739-Talkdesk-QM-Using-Per-Question-Evaluation-Modes-and-AI-Rewrite |
| Analysis after the call | NICE CXone | yes | Quality checks and automatic summary | https://www.nice.com/pricing |
| Analysis after the call | Five9 | yes | Call summary | https://www.five9.com/products/capabilities/agent-assist |
| Analysis after the call | Twilio | yes | Wrap-up notes and summary | https://www.twilio.com/docs/flex/admin-guide/setup/copilot |
| Analysis after the call | Cresta | yes | Summary with reason, resolution and outcome | https://cresta.com/product/agent-assist |
| Analysis after the call | Observe.AI | yes | Automatic quality review | https://observe.ai/platform/auto-qa |
| Analysis after the call | BrightHire | yes | Notes and a filled-in scorecard | https://brighthire.com/ |
| Analysis after the call | Metaview | yes | Notes and reports | https://www.metaview.ai/pricing |
| Voice assistant | **us** | partial | Talks through the browser's microphone; no phone yet | docs/live/VOICEBOT.md |
| Voice assistant | Twilio | yes | ConversationRelay, per minute | https://www.twilio.com/en-us/products/conversational-ai/pricing |
| Voice assistant | Observe.AI | partial | AI agents for customers; by voice, not checked | https://observe.ai/llm-info |
| Voice assistant | Retell AI | yes | By phone | https://www.retellai.com/pricing |
| Voice assistant | Vapi | yes | By phone, SIP or web | https://vapi.ai/pricing |
| Voice assistant | PolyAI | yes | Per connected minute; telephony separate | https://poly.ai/pricing/ |
| Several conversations at once | **us** | no | One live conversation at a time per server | docs/live/LIVE_SESSION.md |
| Several conversations at once | Talkdesk | yes | Per user; Express plan of 25 licences | https://www.talkdesk.com/pricing/ |
| Several conversations at once | NICE CXone | yes | Per agent | https://www.nice.com/pricing |
| Several conversations at once | Five9 | yes | Per concurrent user; minimum 50 seats | https://www.five9.com/pricing |
| Several conversations at once | Twilio | yes | Per use or per user | https://www.twilio.com/en-us/flex/pricing |
| Several conversations at once | Cresta | yes | Up to 100,000 calls a year | https://aws.amazon.com/marketplace/pp/prodview-lmdwown6xoo6q |
| Several conversations at once | Observe.AI | yes | Per agent | https://aws.amazon.com/marketplace/pp/prodview-lgvu5s7gueesi |
| Several conversations at once | BrightHire | unknown | Quote by applicants or interviews | https://brighthire.com/pricing/ |
| Several conversations at once | Metaview | yes | Per user | https://www.metaview.ai/pricing |
| Several conversations at once | Retell AI | yes | 20 calls at once included | https://www.retellai.com/pricing |
| Several conversations at once | Vapi | yes | 4 to 30 calls at once | https://vapi.ai/pricing |
| Several conversations at once | PolyAI | yes | 10 calls at once (API) | https://poly.ai/pricing/ |

**What each company offers that we do not** (`offers`, in «Nosotros
frente a cada una»), each from the source named in the file: Talkdesk a
phone system with voice and digital channels (pricing page); NICE a
cloud phone system with quality checks and automatic summaries (pricing
page); Five9 a cloud phone system with AI minutes included (pricing
page); Twilio phone, SMS, WhatsApp and chat (Conversational
Intelligence page); Cresta chat with the supervisor during the call
(Agent Assist page); Observe.AI quality review calibrated by people
(product summary); BrightHire more than 30 languages (pricing page) and
links to recruiting software (home page); HireVue video interviews and
coding tests (platform page); Metaview links to Ashby, Greenhouse,
Lever and others and SOC 2 Type II (pricing page); Retell its own phone
line or Twilio / Telnyx (pricing page); Vapi phone, SIP and web
(pricing page); PolyAI a tool to design the conversations (home page).

**Declined** (no source in the notes that a reader could open, or only a
search snippet or a competitor's blog): what the interview tools show
the interviewer live (BrightHire only says «real-time guides»); Metaview live prompts to the
interviewer (a review site only); Bland's on-premises option (not in
the list); Genesys, Gong, Balto (§2); any claim about the companies'
Spanish; any latency of theirs.

## 3c. «Cuánto cuesta el mismo servicio»: the arithmetic

**Scenarios** (`cost.scenarios`): 5 and 50 people, each with
80 hours of calls a month (an assumption: about 4 hours a day on 20
days), calls of 10 minutes, live help during the call and analysis
after it. Small: 400 hours, 24,000 minutes, 2,400 calls a month. Large:
4,000 hours, 240,000 minutes, 24,000 calls.

**Ours: an estimate of running cost, not a price** (we have none). Per
hour of conversation:

- Live help: **13 Haiku calls in the 108 s** support call of
  `docs/live/LIVE_MEASURES.md` §1 → 433.3 calls an hour; each at the
  measured ~1,750 input and ~90 output tokens (§3) =
  (1,750 × 0.10 + 90 × 0.50) / 10⁶ = 0.00022 USD → **0.0953 USD/h**.
- Analysis after the call: the Haiku session analysis of
  `docs/skills/MODEL_COMPARISON.md` (150 calls; prompt tokens 300 +
  287,729 cache write + 4,585 cache read = **292,614**; output
  **88,498**) of the 881.8 s test meeting (`docs/skills/SIMULATION.md`
  §3), scaled by duration (× 3,600 / 881.8 = 4.083): 1,194,615 input
  and 361,302 output tokens → 0.1195 + 0.1807 = **0.3001 USD/h**. The
  meeting had four people; a call has two, so scaling by duration
  overstates the cost.
- Total **0.3954 USD/h** at the Haiku 5.5 list price, no cache, no
  batch. The faint part of our bar adds **30 %** of tokens: Anthropic
  says Haiku 5.5's tokenizer counts about 30 % more tokens for the same
  text than Haiku 4.5, and the repository does not record which Haiku
  the measures used (https://platform.claude.com/docs/en/models/haiku-5-5/overview, seen 2026-11-27).
- Small: 400 × 0.3954 = **158 USD** (to 206). Large: **1,582 USD** (to
  2,056).
- **With Opus for the analysis** (the default today, `src/config.py`;
  chosen per step on the «Análisis» page): Opus 5.5 at 4 / 20 USD per
  million (Anthropic pricing page, seen 2026-11-27): 1,194,615 × 4 +
  361,302 × 20 = 12.00 USD/h plus the live help → 4,840 USD (small),
  48,399 USD (large). The page says so under the chart.
- **Not included** and said so: the server with its graphics card and
  electricity (no sourced figure), any second server (one live
  conversation at a time per server today, `docs/live/LIVE_SESSION.md`).

**Theirs: the published list price applied to the scenario**
(`cost.competitors`, `planCost`; low and high are the cheapest and the
dearest plan):

| Company | Low | High | Small (USD/month) | Large (USD/month) | Left out, said on the row |
|---|---|---|---|---|---|
| Talkdesk | 105 per user (cheapest plan with voice) | 225 per user | 525–1,125 | 5,250–11,250 | the AI copilot (no published price) |
| NICE CXone | 110 per agent | 249 per agent + 0.25 per session (one session per call assumed) | 550–1,845 | 5,500–18,450 | the copilots (by usage, no rate) |
| Five9 | 119 per concurrent user, minimum 50 | 159 | 5,950–7,950 | 5,950–7,950 | usage charges that may apply |
| Twilio | Copilot 0.035 per minute + Flex 1 per active hour (call hours as the least active time) | Copilot + Flex 150 per user | 1,240–1,590 | 12,400–15,900 | the streaming transcription Copilot requires (docs) |
| Cresta | 150,000 a year up to 100,000 calls, 1.50 per extra call | same | 12,500 | 36,000 (288,000 calls a year) | — |
| Observe.AI | 828 per agent read as a year | 828 per agent read as a month | 345–4,140 | 3,450–41,400 | the listing is unclear, said on the row |

BrightHire, HireVue and Metaview sell job-interview services («Precio no
publicado» for the AI interviewers); Retell AI, Vapi and PolyAI sell a
voice assistant charged per minute: their prices do not apply to this
case and the chart says «No comparable», without a figure. Their prices
are in «Precios publicados».

**Not symmetric, and the page says it:** their prices include their
phone system or whole service, support and margin; ours is only what it
costs to run the language model. `tests/web/competition.test.mjs` checks
every number above.


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
- A task cell changes (a company publishes a function, or a page no
  longer says it): edit the cell under `tasks` with its `source` and
  that source's `seen` date; a cell with nothing to back it is removed
  (it shows «No publicado»), never set to `no`.
- A price model changes: edit `cost.competitors` and the expected
  numbers in the test and §3c; a new measurement of tokens per hour,
  `cost.ours` and §3c.
- Update `updated` (shown at the bottom of the page).
