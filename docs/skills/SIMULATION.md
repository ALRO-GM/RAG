# Simulated discussion: testing the whole flow without real people

Ticket #88. A discussion written for the project lets the owner run the
whole flow (sentences per person, retrieval, judge, reports) and the
real audio path (enrolment, transcription, diarization, publication,
analysis) before anyone is recorded. Everything here is **synthetic**:
it shows that the pieces work together, not how well they would do on
real people.

## 1. The script

`examples/skills/simulated/reunion_verano.txt`: an invented bakery
(Obrador Valdecruz) decides how to serve restaurants in July and August
without missing deliveries. Four invented people: Lucía (manager, runs
the meeting), Andrés (head baker, sceptical, cuts in), Nerea (sales,
brings data and proposals), Joaquín (deliveries, asks questions, takes
on tasks). There are disagreements (hiring two bakers vs. one, a summer
surcharge), an apology, proposals accepted, rejected and postponed,
an open question (Sunday orders), and a round of commitments with dates.
Original Spanish text written for this; 117 turns, 238 sentences,
2,045 words, six turns that cut in (`+` at the start of the line).

`python -m src skills simulate --script <file>` turns it into a
`transcript.v1` document (`domain/skills/simulation.py`): each sentence
lasts its words at 2.4 words per second (at least 0.6 s), 0.45 s
between turns, 0.25 s between sentences of a turn, and a turn marked
`+` starts 0.6 s before the previous one ends. These times are an
estimate, not a recording: 15.5 minutes for this script. The committed
`sim-reunion-verano.transcript.v1.json` is that output (a test checks
it matches the script).

`make demo-analysis` writes it to `data/transcripts/` and runs
`skills analyze` on it (`TUTORIAL.md` §7, `DESIGN.md` §12).

## 2. The audio

`experiments/skills/render_simulation.py` renders the same discussion
with [Piper](https://github.com/OHF-Voice/piper1-gpl), a local
text-to-speech engine, one voice per person, in a virtual environment
of its own (Piper is not a dependency of the project and nothing of
the project is imported by the script):

```sh
python3 -m venv ~/.local/share/ragputin/tts-venv
~/.local/share/ragputin/tts-venv/bin/pip install piper-tts==1.8.0
~/.local/share/ragputin/tts-venv/bin/python -m piper.download_voices \
    --download-dir ~/.local/share/ragputin/voices \
    es_ES-davefx-medium es_ES-sharvard-medium \
    es_ES-mls_10246-low es_ES-mls_9972-low
~/.local/share/ragputin/tts-venv/bin/python experiments/skills/render_simulation.py \
    examples/skills/simulated/sim-reunion-verano.transcript.v1.json \
    ~/.local/share/ragputin/voices data/simulated
```

Nothing is installed in the system: the environment and the voices
(about 270 MB) live under the user's home. It writes
`sim-reunion-verano.wav` (16 kHz mono), `sim-reunion-verano.reference.json`
(the rendered start, end and speaker of every sentence: the reference to
score the transcriber against) and `enrol-<id>.wav`, each voice reading
an original enrolment paragraph (31 to 45 s; the transcriber asks for
at least 20 s). Sentences are laid out one after the other with the
same pauses as the text level; a `+` turn starts 0.6 s before the
previous sentence ends and is mixed over it. Rendering took 43 s of
wall time on the workstation's CPU (20 threads), and 31.5 s on
2026-11-16. Piper adds random
noise to durations, so two renderings differ slightly in length
(14.6 and 14.8 minutes were seen); the reference file always matches
its own WAV.

| Person | Piper voice | Licence of the voice's data (model card) |
|---|---|---|
| Lucía | `es_ES-sharvard-medium`, speaker 1 | CC BY 3.0 (Sharvard corpus, University of Edinburgh DataShare) |
| Andrés | `es_ES-davefx-medium` | CC0 |
| Nerea | `es_ES-mls_10246-low` | CC BY 4.0 (Multilingual LibriSpeech, OpenSLR 94) |
| Joaquín | `es_ES-sharvard-medium`, speaker 0 | CC BY 3.0 (Sharvard corpus) |

`es_ES-mls_9972-low` (CC BY 4.0) is downloaded as a spare voice. Piper
itself (`piper-tts` 1.8.0) is GPL-3.0-or-later; it runs as a separate
program, so its licence does not extend to this repository. The
licences were read from each voice's `MODEL_CARD` in
`rhasspy/piper-voices` on Hugging Face on 2026-11-15. CC BY voices need
attribution if the audio is published; the audio is not committed.

### Testing the real path with it

1. Start the transcriber and the RAG as usual (`TUTORIAL.md` §1).
2. Enrol the four voices from the web («Registrar voz», uploading
   `enrol-<id>.wav`) or with
   `python -m transcriber enrol --consent --id lucia Lucía enrol-lucia.wav`
   (and the same for `andres`, `nerea`, `joaquin`). They are synthetic
   voices: the consent is a formality here, never for a person.
3. Upload `sim-reunion-verano.wav` in the web (type «Reunión para
   decidir»), or `python -m transcriber import --expect
   lucia,andres,nerea,joaquin --wait sim-reunion-verano.wav`.
4. Open the session, check the speakers, then **Análisis** →
   **Empezar el análisis** (it publishes the transcript to the RAG
   first).
5. Compare what the transcriber produced with
   `sim-reunion-verano.reference.json`:
   `python experiments/skills/score_audio_simulation.py
   sim-reunion-verano.reference.json <transcript.json>`; and the
   analysis with the one of the script:
   `python experiments/skills/compare_analyses.py
   data/output/skills/analysis/sim-reunion-verano
   data/output/skills/analysis/<session>`.

Use a data directory of its own for the synthetic voices
(`TRANSCRIBER_DATA_DIR=…`): enrolled in the live service's registry
they would sit next to real people's voices.

### What a synthetic recording shows, and what it does not

It does show that the whole path runs: audio in, voices enrolled and
recognised, sentences attributed, transcript published, analysis
started from the web, reports out. It exercises overlaps at known
places.

It does not show how the transcriber does on people. The voices are
clean, read at a steady pace from text the recogniser finds easy, with
no room noise, no hesitations, no laughter and only six short overlaps;
two of the four voices come from one model (same recording conditions).
Word error and speaker attribution measured on it are a best case, not
an estimate for real meetings (#19 is the real measurement).


## 3. Results (2026-11-16, #96)

Text level: one real run with Claude Code is in `DESIGN.md` §12 and
`docs/ESTADO.md` §4.7.

Audio level: run end to end once, on the workstation with the GPU free
(RTX 3070 Ti), on the code of `main` at 4d323c6. A first attempt on
2026-11-15 ran on the CPU and was stopped after 66 minutes of
transcription without a result (enrolment took 272 to 544 s per voice
there).

**What was run.** The real path minus the web upload: the commands of
§2 in a data directory of their own
(`TRANSCRIBER_DATA_DIR=~/.local/share/ragputin/sim96/data`), never the
live service's, so the synthetic voices never entered its registry.

| Step | Command | Wall time |
|---|---|---|
| Render (Piper, CPU) | `render_simulation.py` | 31.5 s; 881.8 s of audio (14.7 min), 238 sentences, 6 overlapping starts |
| Enrol 4 voices | `python -m transcriber enrol --consent --id …` | 5.5 to 5.8 s each, model load included |
| Transcribe | `python -m transcriber import --id sim-reunion-verano-audio --type reunion_decision --expect lucia,andres,nerea,joaquin --wait` | 93.4 s (9.4× real time); peak resident memory 18.6 GB |
| Publish | `python -m transcriber publish sim-reunion-verano-audio` (`TRANSCRIBER_LOCAL_RAG=1`) | to the live RAG's `POST /transcripts` |
| Analyse | `python -m src skills --judge=claude_code --max_confirmations=102 analyze --session=sim-reunion-verano-audio --synthetic --role=…` | 11 min 14 s |

The transcript records `openai/whisper-large-v3-turbo`, the voice model
`microsoft/wavlm-base-plus-sv` and the diarizer
`diarizen-wavlm-large-s80-md-v2`. The search for names
(`Qwen/Qwen3-8B@4bit`, 17.4 s) proposed none: the four voices were
already enrolled. The session is published as
`sim-reunion-verano-audio`, next to the text-level
`sim-reunion-verano`, so both analyses can be opened from the web.
The analysis ran from the terminal (not «Empezar el análisis») to cap
Claude's calls with `--max_confirmations`.

### The transcriber against the script

Scored with `experiments/skills/score_audio_simulation.py` (reference:
the rendered `sim-reunion-verano.reference.json`):

| Measure | Result |
|---|---|
| Word error over the whole text | **0.088** (85 substitutions, 56 deletions, 38 insertions; 2,045 words in the script, 2,027 transcribed) |
| Sentences: script 238, transcript | 227 utterances, the four enrolled ids, no unknown speaker |
| Best-overlap utterance has the right person | **227 of 238** (0.954); 6 wrong, 5 missed (no utterance over them) |
| Per person | Lucía 75/75, Andrés 61/64 (1 wrong, 2 missed), Nerea 51/56 (2 wrong, 3 missed), Joaquín 40/43 (3 wrong) |
| The 6 sentences that start over the previous one | 5 of 6 right; 1 merged into the sentence it cuts into |
| The 6 sentences they cut into | 6 of 6 right |
| Speech time covered by the right person | 0.892 |

Where the errors come from, read from the alignment and the audio:

- **The last 14.7 s of the recording are missing** from the transcript:
  the last word ends at 867.14 s; Nerea's «Gracias. Andrés, te mando la
  lista en cuanto la tenga.» and Andrés's «Vale. Me voy, que la masa no
  se hace sola.» (20 words) are not there, though the WAV has speech
  up to 880.8 s. These are 4 of the 5 missed sentences and 20 of the 56
  deleted words. Why the recogniser lost the end was not found in this
  run.
- **Numbers written as digits**: the transcript has 21 tokens written
  in digits («11», «12», «9 000», «300») where the script spells the
  number out; each counts as an error though the meaning is kept.
- **Nerea's voice** (`es_ES-mls_10246-low`) stretches short sentences:
  12 of her sentences last more than 1.2 s per word (one word,
  «Apuntado.», 6.5 s; «Es mi trabajo.», 8.9 s), with babble instead of
  speech. The worst stretches of the transcript sit there: «Doce, en
  realidad.» → «eren pide el río de a deidse jafre…», «Es un riesgo,
  sí.» → «rodriguesco es una sin», and after the overlap of t13 about
  12 s of her turn («Yo. Es mi trabajo. Y prefiero decirle eso en
  junio que pedirle…») are lost. This is the text-to-speech engine,
  not a property of the transcriber.
- Ordinary recognition errors on clean speech remain: «me quedo» →
  «me quiero», «Nerea» → «Andrea», «el viernes» → «elnes», «venden»
  → «cenden», «cierran» → «hierran», «retrasos» → «detrás».
- The overlapping start the transcriber missed is t13.s0 («Yo.», one
  word, 1 s, over Andrés); the other 5 got their own utterance with the
  right person, and the transcriber marked 6 utterances as overlaps.
  The other wrong attributions are short turns next to another
  person's: Andrés's «Eso ayudaría mucho, la verdad.» given to
  Joaquín, Joaquín's «Hecho. Mañana llamo al taller.» given to Andrés,
  and Joaquín's «¿Cuánto sería la subida?» joined to Nerea's answer.

### The analysis on the transcript against the analysis on the script

Same judge as the text-level demo (Claude Code `opus`, `--judge
claude_code`), same job description, same catalogue and settings;
`--max_confirmations=102` (the transcript had 102 sentences with
candidates, so nothing was left out). Compared with
`experiments/skills/compare_analyses.py`:

| | Script (2026-11-15) | Transcript (2026-11-16) |
|---|---|---|
| Sentences / with candidates / candidates | 238 / 101 / 150 | 227 / 102 / 150 |
| Dialogue acts: doubtful | 51 | 47 |
| Events: decisions, action items, unanswered questions | 6, 18, 5 | 8, 15, 5 |
| Events: proposals accepted, ignored, rejected | 6, 17, 0 | 8, 11, 1 |
| Confirmation: confirmed / rejected | 35 / 66 | 34 / 68 |
| Firm levels / provisional / evidence / observations | 21 / 15 / 111 / 26 | 22 / 11 / 104 / 22 |
| Profile points (synthesis) / dropped | 27 / 2 | 28 / 2 |
| Claude calls | 145 (101 confirmation, 44 assessment; the synthesis, added later, records 0 calls) | 135 (87 confirmation, 15 more answered from the cache; 44 assessment; 4 synthesis) |

Of the 44 cells (4 people × 11 competencies), **29 are the same** (15
firm with the same level, 7 provisional with the same level, 7 without
a level). Of the other 15: 4 firm levels moved by one (Lucía INIT 4→3,
SOLVE 3→4; Nerea CONFL 4→3, INFLU 4→3), one provisional level moved by
one (Lucía INCL 4→3), and 10 changed tier (firm→provisional 2,
provisional→firm 3, provisional→none 4, none→provisional 1). No level
moved by more than one point. Andrés has no change; Lucía 6, Nerea 5,
Joaquín 4.

The evidence changes more than the levels: 104 quotes against 111;
per person 15 of 27–28 (Andrés), 19 of 29 (Lucía), 11 of 21–24
(Joaquín) and 5 of 26–31 (Nerea) are the same indicator with the same
words. Of the 104 quotes of the transcript run, 25 are not word for
word in the script: by my reading about half only write numbers in
digits, the rest carry recognition errors into the report («en agosto
me quiero con dos panaderos menos», «Perdona, Andrea», «cerrar el
pedido del domingo elnes»); 1 is credited to the wrong person
(Joaquín's «Hecho. Mañana llamo al taller.» as evidence for Andrés).
Nerea, whose voice rendered worst, keeps the fewest of her quotes.

What this comparison cannot tell apart: the judge answers differently
from one run to the next on the same text, and no second run of the
script without the cache was made (it would cost about 145 more calls).
So the 15 changed cells mix the transcription's effect with the
judge's own variation; only the 15 confirmations answered from the
cache are known to be the judge's same answer to the same prompt.

### What this shows, and what it does not

It shows that the real path runs end to end on the GPU in minutes:
render, enrol, transcribe with the expected voices, publish to the
RAG, analyse with the same judge as the text level; that the enrolled
voices are recognised (95 % of sentences with the right person, all
four ids, no stranger); that short overlaps mostly get their own
utterance; and that a word error near 9 % and a few wrong attributions
leave most levels in place but put wrong words and one wrong person
into the evidence the reports quote.

It does not show how the transcriber does on people: §2's limits
hold (clean read speech, no noise, two voices from one model), and one
of the four voices is worse than a person would be (stretched short
sentences). Each figure is one run. Open points for follow-up: the
lost last 14.7 s, and the judge's run-to-run variation on the same
text.
