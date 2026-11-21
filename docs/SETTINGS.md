# Pipeline settings: the «Ajustes» page

The web page **Ajustes** (`#/settings`, group *Sistema* of the sidebar)
chooses between the highest-quality and the lowest-cost way to run the
pipeline, and the model of every step that has measured alternatives.
Each option shows what was measured with it; the figures are copied
from the documents listed at the end, never computed for this page.

## Where the settings live and who reads them

| | |
|---|---|
| Document | `settings.v1` (`contracts/settings.v1.schema.json`), one choice per step from a closed list |
| Owner | the RAG: `data/settings.json` (`SkillsSettings.settings_path`), `GET/PUT /settings` behind its bearer token |
| Web | the transcriber forwards `GET/PUT /api/settings` (behind the access code) to the RAG, like `/api/analysis/judge` |
| Orchestrator | reads `GET /settings` of the RAG (`--knowledge-url`, the RAG token) when a live session starts |

The RAG owns it because most choices are consumed there (analysis,
knowledge, the live reasoner) and it already owned the analysis judge
(`data/skills/judge.json`). The judge stays in that record: saving
`analysis.judge` here writes it there too, and the page shows what that
record holds.

Every value is checked against the allow-lists of
`src/domain/pipeline_settings.py` (RAG), the transcriber checks the
shape again before forwarding, and every consumer maps the option names
through its own allow-list; no free string ever reaches a command line.
`PUT` carries the `revision` it was based on: an older one is refused
(409) so two people do not overwrite each other silently.

**Nothing saved, or the RAG not answering, changes nothing.** The first
option of every step is the service's own configuration (what it did
before this page), and a consumer that cannot read the settings keeps
it.

## When a change applies

| Step | Read by | Applies |
|---|---|---|
| `transcription.recogniser` | transcriber, at start-up (the model stays loaded) | **after restarting the transcriber** |
| `transcription.diarizer` | transcriber, when a job reaches the diarizer | next transcription |
| `transcription.names` | transcriber, when the name search starts | next search |
| `live.asr` | transcriber, when a live session starts | next session |
| `live.diarization`, `live.suggester_model`, `live.deadline`, `live.assist_mode` | orchestrator, when a session starts | next session (the mode can also be switched on the page) |
| `live.max_suggestions`, `live.show_employee`, `live.show_other` | the live page, when it opens (what this browser chose last wins) | next time the page opens |
| `live.suggester_effort` | RAG, at every `/reason` call (one warm process per model and effort) | next call |
| `analysis.judge`, `analysis.*_model`, `analysis.local_judge`, `analysis.act_tagger` | RAG, when an analysis starts (passed to the `skills analyze` child as flags) | next analysis |
| `knowledge.ocr`, `knowledge.captions` | RAG, at every upload | next upload |
| `voicebot.*` | orchestrator, when a voice-bot session starts | next session |
| `interview.slot_filler`, `interview.max_probes`, `interview.post_call` | orchestrator, when an interview session starts | next interview |
| `interview.minutes` | orchestrator, when a plan is drafted | next plan |

The «Análisis» start form keeps its per-run Claude model choice; it is
now prefilled from the saved settings and still applies to that run
only (it wins over the saved choice).

## Steps and options

The first option is the default. Figures are as written in the cited
documents; almost all come from synthetic data and a single run.

| Step | Options |
|---|---|
| Speech recogniser | Whisper large-v3 turbo (WER 0.088 against the text Piper read; 13.6 s for 388 s; MIT) · Whisper large-v3 (WER 0.148 against turbo, not the truth; 29–41 s per step) |
| Separate voices | Automatic: DiariZen, Nemotron-3 for 2 people · DiariZen (AMI DER 0.089, CALLHOME 0.163, 20.2 s, 3.7 GB, CC BY-NC 4.0) · Nemotron-3 (CALLHOME 0.106, AMI 0.297, 10.3 s, 0.6 GB, OpenMDW) · pyannote community-1 (AMI 0.134, CALLHOME 0.195, 6.6 s, 2.2 GB) · built-in grouping (AMI 0.498, CALLHOME 0.487) |
| Find names | Qwen3-8B, 4B if it does not fit (13/13 right, recall 0.722, 6.32 GB) · Qwen3-4B (11/11, recall 0.611, 2.84 GB) · rules only (suggestions only) |
| Live recogniser | large-v3-turbo int8 on the GPU (end of speech to final 674 ms p50, WER 0.123) · small int8 on the CPU (not measured) |
| Voices on one microphone | Sortformer v2 (right speaker 98.3 % at first emission, lag p50 823 vs 680 ms, +1,140 MiB) · a single person |
| Suggestion model | Haiku (first token 688 ms, answer 1,063 ms p50 at low effort) · Sonnet, Opus (not measured live) |
| Suggestion effort | low (11 of 11 within 2.8 s) · medium, high (not measured) |
| Suggestion deadline | the product's (1,800 ms interview, 2,800 ms support) · 1.8 s · 2.8 s |
| Kind of live help (`live.assist_mode`) | pincelada: a hint of ≤ 8 words (support, 6 synthetic calls: answer p50 1,319 ms, max 1,710 ms; 2 of 5 cut to 8 words) · guiado: the exact sentence to say (p50 985 ms, max 1,730 ms; 9–21 words); `docs/live/ASSIST_MODES.md` |
| Suggestions in view (`live.max_suggestions`) | 3 · 1 · 2 · 5 (older ones fold) |
| Employee's lines / other party's lines (`live.show_employee`, `live.show_other`) | on · off (both off: suggestions only); `docs/live/EMPLOYEE_UX.md` |
| Analysis judge | Claude by subscription (local after the quota) · local model · both, compared · opencode |
| Claude per judged step | Opus (91.7 % of indicators) · Sonnet (not measured) · Haiku (77.5 %; agrees with Opus on 67.6 % of confirmations and 59.1 % of ratings; the synthesis alone not measured) |
| Local judge | Qwen3-8B, 4B if it does not fit (34 % useful quotes, 6.31 GB) · Qwen3-4B (23 %, 2.87 GB) |
| Dialogue-act tagger | fine-tuned encoder (top-1 0.731, macro F1 0.693) · rules + encoder (0.732 / 0.699) · rules (0.519 / 0.449) |
| OCR | on (29.3 scanned pages/min on the CPU, up to 4.4 GB) · off (text layer only, images refused) |
| Interviewer slot filler | Haiku (one call per answer, before the next question) · rules (no model); neither measured on real interviews (`docs/live/VOICEBOT_INTERVIEWER.md`) |
| Interviewer follow-ups per slot | 1 (the owner's rule) · 0 · 2 |
| Interviewer default length | 20 · 15 · 30 · 45 minutes (each plan can change it) |
| Analyse the interview at the end | off · on (the analysis's calls) |
| Figure captions | Florence-2 on this PC (5/5 texts read, English, 2/5 invent) · Claude Haiku (2.0–2.2 s per image, better; **the image leaves the PC**) · none |

Encoder + RG (Claude reviewing the doubtful sentences, 0.864 top-1 on
140 sentences) is not wired into the analysis yet, so it is not an
option; the page says so under the tagger.

## Presets

| Preset | Choices | Why |
|---|---|---|
| **Máxima calidad** | every step at its default | The defaults already are the best measured option of each step (Opus judges, DiariZen / Nemotron by speakers, Qwen3-8B, encoder). It is what the system did before this page, so «nothing saved» reads as this preset. |
| **Equilibrado** | as above, but Haiku confirms the indicators | Confirmation is where Claude is called most (about 102 of 150 calls of a session); Haiku agrees with Opus on 67.6 % of them. Assessment (59.1 % agreement) and synthesis keep Opus. |
| **Menor coste** | judge `local` with Qwen3-4B; Haiku for every Claude step (used only if the judge is switched back to Claude); names with Qwen3-4B; Nemotron-3 to separate voices | No Claude quota in the analysis and the least GPU memory measured for each step (2.87 GB judge, 2.84 GB names, 0.6 GB diarizer). The measured quality drops: 23 % useful quotes against 93 %, DER 0.297 on 4-person meetings against 0.089. |
| **Personalizado** | any other combination | Shown as soon as one step differs from a preset. |

What the presets do **not** change, and why:

- The live suggester stays Haiku at low effort with the product's
  deadline in every preset: it is the only model measured live, and the
  measured deadlines (1.8 s, 2.8 s) leave no room for an unmeasured one.
- The live recogniser stays on the GPU: the CPU model is not measured.
- Captions stay with Florence-2 (local): Claude would send the image off
  the machine, so it is never chosen by a preset, only by hand.
- The recogniser stays turbo: large-v3 is slower with no measured gain.

## Voice bot

Merged on main while this page was built (#203), so it has its own two
steps, read by the orchestrator when a voice-bot session starts; the
suggester's live choices do not apply to it.

| Step | Options |
|---|---|
| Reply model (`voicebot.reasoner_model`) | Haiku (end of the user's sentence to first audio p50 1,584 ms, p90 1,731 ms; 14 of 15 turns answered, one hit the 1.8 s deadline; `docs/live/VOICEBOT.md`) · Sonnet, Opus (not measured in the bot) |
| Voice (`voicebot.voice`) | Piper `es_ES-davefx-medium` (the measured one: first sentence 93 ms p50 on the CPU; warmed at start) · `es_ES-sharvard-medium` (not measured as the bot's voice; not warmed) |

| Voice, cloned (`voicebot.voice = clone:<person>`) | OpenVoice v2 over Piper davefx: 118.5 ms per sentence p50 on the GPU, WavLM similarity 0.917 to the reference voice (davefx alone 0.598); only for people who agreed at enrolment; Piper if a sentence takes over 800 ms (`docs/live/VOICE_CLONING.md`) |
| Speed, intonation, cadence, pause (`voicebot.length_scale`, `noise_scale`, `noise_w`, `sentence_silence`) | Piper's own controls, numbers within 0.7–1.5, 0–1, 0–1, 0–1 s; the defaults (1, 0.5, 0.333, 0) are the voices' own |
| Match an operator's speed (`voicebot.match_rate`) | off · a person's measured rate at enrolment (words/s) sets the length scale: Piper davefx says 4.065 words/s at 1 |

Voice options map to Piper voice names in
`orchestrator/domain/live_choices.py` (`VOICES`); adding a voice means
adding it there, to `STEPS` and to the contract. Number steps carry a
`range` (`StepSpec.range`) and person steps a `pattern`; the page draws
a slider for the first and lists the enrolled people for the second.
«Escuchar ejemplo» says a fixed sentence with the unsaved choices
(`POST /api/assist/voicebot/sample`). Every preset keeps Haiku, davefx
and Piper's defaults: they are the only ones measured as the bot.

## Sources of the figures

`docs/skills/MODEL_COMPARISON.md`, `docs/skills/LOCAL_JUDGE_QUOTES.md`,
`docs/skills/DESIGN.md`, `docs/skills/MODEL_CARD_JUDGES.md`,
`docs/skills/DIALOGUE_ACT_TAGGER.md`, `docs/live/LIVE_ASR.md`,
`docs/live/MONO_DIARIZATION.md`, `docs/live/LIVE_MEASURES.md`,
`docs/transcriber/LAB.md` (§13.8), `docs/transcriber/ARCHITECTURE.md`,
`docs/transcriber/NAMES.md`, `docs/KNOWLEDGE.md`,
`docs/live/ASSIST_MODES.md`,
`experiments/knowledge/OCR.md`, `experiments/platform/STAGE1_E2E.md`,
`docs/live/VOICEBOT.md`, `docs/live/VOICE_CLONING.md`.
