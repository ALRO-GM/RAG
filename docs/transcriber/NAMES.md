# Names from what people say (TR-20, issue #102)

After a transcription the voices nobody enrolled are `S1`, `S2`… People
often say who they are ("soy Ana", "me llamo Pedro", "hola, aquí Luis
de ventas") or call each other by name ("gracias, Marta", "os presento
a Ana"). The transcriber now reads the transcript for those names with
a small **local** model and names the voices itself when it is sure,
and otherwise suggests the name with its quote and time for one click.

Nothing leaves the PC: no Claude, no paid API (the text is real
people's speech). The model is Qwen3-8B at 4 bits, Qwen3-4B when the
8B does not fit the free GPU memory, never a smaller one.

## 1. When it runs

- As the **last step of every transcription job** (stage `names`),
  after the transcript is saved: the session is already `done`, so the
  page can open the transcript while names are searched; it reloads the
  speakers when the search ends.
- On request, with **«Buscar nombres» / «Find names»** on the transcript
  page (`POST /api/sessions/{id}/names`), for a session transcribed
  before this feature or after renaming by hand. While a job of the
  session (a search, or its transcription) is queued or running, the
  request queues nothing and answers that job's status.

**No publish meanwhile.** The session is `done` before its names are
given, so `POST /api/sessions/{id}/publish` answers **409 `busy`**
while the session's job (transcription or search) is queued or
running: otherwise the RAG would get `S1`/`S2` and nothing republishes
it afterwards. The analysis page shows «Buscando los nombres de quién
habla…» (also when `GET …/names` says `running`) and tries the publish
again every 3 s until it goes through; the transcript page shows the
same line beside «Análisis» while the search runs.

`names.enabled: false` in the settings removes the step from the job;
the button still works. A failure never fails the transcription: it is
logged and told as the job's note.

## 2. How it decides (`transcriber/domain/speaker_names.py`)

1. **Candidates, by rules.** Self-introduction patterns in Spanish and
   English (`me llamo X`, `mi nombre es X`, `soy X`, `aquí X`,
   `habla X`, `I'm X`, `my name is X`…) and names said to others
   (`gracias, X`, `X, ¿…?`, `¿y tú, X?`, `hola X`, `os presento a X`,
   `…, X.`). The name must be capitalised, and words that are never
   names (articles, greetings, numbers, months, interjections, adverbs
   in *-mente*) are refused, so "soy de Madrid", "me llamo la atención"
   or "Doce, en realidad" never reach the model. Each candidate keeps
   one sentence of context before and after; at most 60 per session.
2. **The model reads each candidate** (greedy decoding, no thinking)
   and answers JSON `{name, kind: self|addressed|none, quote,
   confidence}` about one word of one line. The answer is dropped when
   it is not that JSON or names a word the line does not contain, so
   the model cannot invent a name.
3. **Whose name it is** is fixed by the rules, not by the model: a
   self-introduction names the speaker of the sentence; a name said to
   someone else names a neighbour. Thanks and acknowledgements answer
   the person who **just spoke** ("Gracias, Nerea" follows Nerea in the
   simulated discussion of #88), questions, greetings and
   introductions call on the person who **speaks next**. (The issue
   said "the speaker who answers next"; the thanks case showed that
   the previous speaker is the right side for it.)
4. **Automatic only when sure** (precision first: a wrong name is
   worse than `S2`). A voice gets a name automatically only when it is
   anonymous and unnamed (or carries the name an earlier search gave),
   it introduced itself (`self`) with confidence ≥ 0.8, always with the
   same name, no other voice was given that name (by itself or by
   others: two people with the same name), no other speaker already has
   it, and nobody called it by a different name. Everything else is a
   **suggestion** with a reason (`addressed`, `unsure`,
   `several_names`, `same_name`, `name_taken`, `conflict`, `undone`,
   `rules_only`).
5. **Never overwritten:** enrolled people (voiceprints) and names set
   by hand are left alone. Names are given through `RenameSpeaker`,
   the same path as a rename by hand; an automatic name the search no
   longer supports is taken away (`UnnameSpeaker`).
6. **Undo:** each automatic name is kept in the session's `names.json`
   with its quote and time; «Deshacer» (`DELETE
   /api/sessions/{id}/names/{label}`) turns the voice back into `S#`
   and keeps the name as a suggestion. The undone name is also kept in
   `names.json` (`undone`, with the transcription run): later searches
   of that transcript never give it (or the same first name)
   automatically to that voice again, only as a suggestion (`undone`).
   A new transcription of the session forgets it, since the labels may
   then be other people. Accepting a suggestion is a normal rename
   (`PATCH …/speakers/{label}`).

When the model cannot run (no GPU memory after waiting, no 4-bit
loading installed, missing weights) the rules' findings are kept as
**suggestions only** (`rules_only`), names given earlier stay, and the
note says why. The RAG copy is not republished by this step: as for any
rename, it is republished when the session is published again.

## 3. The model on a shared GPU (`adapters/outbound/local_name_model.py`)

The transcriber's own copy of the loading rules of the RAG's local
judge (`src/adapters/outbound/local_judge.py`, policy `preferred`; the
two packages never import each other):

- Qwen3-8B at 4 bits when its measured peak (6.31 GB, from the judge's
  measurements on the RTX 3070 Ti) plus 0.4 GB fits the free memory,
  else Qwen3-4B (2.87 GB);
- when neither fits, it waits up to `names.wait_seconds` (600 s) for
  the 4B, looking every 15 s, then gives up; no other process is ever
  stopped and no smaller model is used (the settings refuse any model
  other than these two);
- the speech, voice and overlap models are freed before it loads, and
  it is freed after the search (and by the job runner after every
  job). A load that fails part-way (CUDA out of memory while the
  weights are copied) drops the partial weights and empties the CUDA
  cache before the error is reported.

On a machine without CUDA it loads the 4B in float32 (slow).

## 4. Measurement

Set: `experiments/transcriber/speaker_names/set.json`, 12 Spanish
conversations written for this test (no real person) plus the
simulated discussion of #88 (`examples/skills/simulated/
sim-reunion-verano.transcript.v1.json`, speakers made anonymous): a
round of introductions, look-alikes ("soy de Madrid", "me llamo la
atención", "soy Sevilla", "soy Géminis"), names only said to others,
an introduction on someone else's behalf ("os presento a Ana"), two
people called Ana, no names, people who are not present, a quoted
introduction ("mi jefa dice: soy Carmen…"), a phone call ("habla
Sergio"), a speaker who corrects their name, a nickname ("Francisco" /
"Paco") and an English introduction. 13 sessions, 34 speakers; 18
clearly introduce themselves (the target of automatic names) and 25
have their name said by someone (the target of suggestions).

Script: `experiments/transcriber/speaker_names_eval.py` (rows in
`experiments/transcriber/speaker_names/results.jsonl`). Automatic
names: precision = right / given, recall = right / the 18. Suggestions:
precision = right / given; "named or suggested" = of the 25, those that
end with the right name automatically or among their suggestions.
Seconds per session = the search with the model already loaded (mean
over the 13 sessions, one with no candidate). RTX 3070 Ti, 2026-11-15.

| Mode | Automatic: given / right / wrong | Precision | Recall | Suggestions: given / right | Precision | Named or suggested | s / session (max) | Load | GPU peak |
|---|---|---|---|---|---|---|---|---|---|
| Rules alone (same policy) | 15 / 12 / **3** | 0.80 | 0.667 | 16 / 13 | 0.812 | 25 / 25 | 0.00 (0.01) | – | – |
| Rules + Qwen3-8B, 4 bits | 13 / 13 / **0** | **1.00** | 0.722 | 15 / 11 | 0.733 | 24 / 25 | 3.59 (13.02) | 8.0 s | 6.32 GB |
| Rules + Qwen3-4B, 4 bits | 11 / 11 / **0** | **1.00** | 0.611 | 18 / 13 | 0.722 | 24 / 25 | 3.19 (12.24) | 15.0 s | 2.84 GB |

- **Target met: no wrong automatic name** with either model. The rules
  alone give three wrong ones (`Géminis`, `Sevilla`, the quoted
  `Carmen`): this is why the service never names automatically
  without the model (rules alone only suggest).
- The longest session is the #88 discussion (238 sentences, 12
  candidates, about 13 s with the 8B). Nobody introduces themselves in
  it; every name there is said to someone, so it only gets
  suggestions (one wrong: Andrés says "Perdona, Nerea." right after
  Lucía, so the thanks rule gives "Nerea" to Lucía).
- The 5 misses of the 8B, all on the safe side: Francisco (also
  called "Paco": conflict); Álvaro (the model took the quoted "soy
  Carmen" as said to him, which conflicts with his own "me llamo
  Álvaro"); the two Anas (no one gets "Ana" automatically, as the
  issue asks) and Diego, the third person of that session (the 8B
  took the second "me llamo Ana" as said to the next speaker, Diego,
  which then conflicts with his "soy Diego").
- The 7 misses of the 4B: the two Anas, Francisco, Pau and Inés (the
  correction "soy Pablo… soy Pau") and Mario and Kate (the English
  meeting). It also gave no wrong name.

How the set was used, honestly: it is the only labelled set, so it was
also used while building. Two changes were made after looking at
results on it, and the earlier rows are kept in
`experiments/transcriber/speaker_names/history.jsonl` (in order):

1. Interjections that open a sentence before a comma ("Ja", "Encantada",
   "Tranquilo", "Welcome", "Mejor"…) were added to the words that are
   never names after the first rules run. Row 1 is the rules with them.
2. Prompt v1 asked "what does the speaker do with a person's name";
   the 8B often answered about another name of the line or of the
   context ("habla Sergio" judged as said to someone). Row 2: 8 given,
   8 right, recall 0.444. Prompt v2 (now) asks about one word of one
   line and repeats the marked line. Row 3: 14 / 14, recall 0.778.
3. Then "two voices with the same proposed name" was made to count
   names given by others too, as the issue says (row 3 had given "Ana"
   to the first Ana because the 8B took the second one's "me llamo
   Ana" as addressed). That is the table above (13 / 13).

So the numbers are an optimistic estimate on a small synthetic set
written by the same person who wrote the rules. Real recordings add
recognition errors (a lower-case or misspelt name is not a
candidate), sentences given to the wrong voice by the diarizer (a
self-introduction on the wrong voice would be a wrong automatic name)
and other phrasings. Measuring on consented real sessions is pending.

## 5. Known limits

Besides the measurement's (above):

- **A name set by hand equal to the automatic one is treated as
  automatic.** The record does not say who set a name, only what the
  search gave: if a person renames `S2` to exactly the name the search
  had given it, a later search may take it away (when it no longer
  finds the introduction) or «Deshacer» may undo it.
- **A names job overwrites the transcription's note.** The job status
  keeps one note; the search's note (model not available, transcript
  changed) replaces the transcription job's note when the search runs
  as its last step, and a search on request replaces the status of the
  last transcription job.

## 6. Where things are

| Part | File |
|---|---|
| Rules, prompt, answer check, decision | `transcriber/domain/speaker_names.py` |
| Use cases: search, queue, show, undo | `transcriber/application/speaker_names.py` |
| Ports `ChatModel`, `SpeakerNameStore` | `transcriber/application/ports.py` |
| `UnnameSpeaker` | `transcriber/application/review.py` |
| Local model (8B → 4B → wait) | `transcriber/adapters/outbound/local_name_model.py` |
| Record `names.json` per session | `transcriber/adapters/outbound/fs_session_repository.py` |
| Routes `GET/POST /api/sessions/{id}/names`, `DELETE …/names/{label}` | `transcriber/adapters/inbound/http_api.py` |
| Page: names box, «Buscar nombres», «Deshacer», «Aceptar» | `web/js/views/transcript.js`, `web/js/lib/names.js`, `web/js/i18n/{es,en}.js` |
| Settings `names` | `transcriber/config.py` (`SpeakerNameSettings`) |
| Measurement | `experiments/transcriber/speaker_names_eval.py`, `experiments/transcriber/speaker_names/` |
