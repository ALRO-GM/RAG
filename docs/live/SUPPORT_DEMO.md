# The call-centre example (DM-2, #250)

One fictitious call, shown three ways, for people who want to see what
the product does for a customer-service team without a microphone or a
model. Company, agent, customer and every sentence are invented
(`Lumbre Hogar`, Mario, Andrés Ferrer); the pages say so.

**The example in two lines.** Andrés Ferrer calls to cancel Lumbre Plus
because the October renewal was charged after he cancelled in September;
after a hurtful moment (he is asked to repeat his data for the third
time) the agent checks the two conditions of the knowledge base (cancelled
before the renewal, free shipping not used that month), refunds the charge
and he stays on the monthly Lumbre Plus with a 10 % voucher. The agent says
the data notice late and without the customer's rights, never asks for the
express acceptance and leaves «¿me llegará algo por escrito?» unanswered.
Every offer is backed by the knowledge base (`support.demo` chunks s17,
s18, s20, s43, s46, s52, s53, s54): Lumbre Plus renews monthly and can be
cancelled any time, so no 12-month contract is offered.

## Where it is

| What | Route | Data |
|---|---|---|
| Live support call | `#/live/demo/support` (also from `#/live` and the home) | `examples/demo/llamada-lumbre.demo.json` |
| Call report | `#/call-report/llamada-lumbre` (home «Ver un informe de ejemplo», end of the live call) | same file, `report` |
| Voice assistant example | `#/voicebot/demo` (home «Escuchar un ejemplo», link on `#/voicebot`) | same file, `voicebot` |

`#/live/demo` still opens the interview. The file is `live.demo.v1`
with extra optional parts (`sources`, `mandatory`, `report`,
`voicebot`, `cast`); the orchestrator serves the report and the voice
example at `GET /demos/{id}/report` and `/voicebot`, and their voices at
`/voices` (proxied under `/api/assist/demos/...`). The call has 19 sentences and its last sentence ends at 173 s (2 min 53 s)
as scripted; when its voices are rendered it is replayed at their
length instead.

## What is scripted, what works on a real call, what is missing

| Part | In the demo | On a real call today |
|---|---|---|
| Sentences of the call | written by hand | transcribed live or from a recording |
| Customer signals quoted in the suggestion («El cliente dice…») | listed in the file per sentence | **works**: the support reasoner labels signals and the page quotes them (`orchestrator/domain/signals.py`) |
| One suggestion at a time with its source | prepared; the source text is copied from `knowledge/domains/support.demo` (a test compares it) | **works**: the RAG retrieves chunks and the reasoner cites them |
| Mandatory items pending, said, reminded | the file names the sentence that says each | **works**: matched by the RAG (`MANDATORY_ITEMS.md`), reminder every 45 s, hand-off record of said, ticked and missing |
| Hand-off record at the end | published by the demo | **works** (`<log-dir>/<session>.<product>.handoff.json`) |
| Report: sentence by sentence (helped / neutral / hurt by the customer's next reaction, better wording) | written by hand | **not built**: nothing judges a line by the reaction today |
| Report: reference script followed / skipped / reordered, and whether the script looks improvable | written by hand | **not built**: no reference script is stored for a call type and nothing aligns a call to one |
| Report: top-scorer wording | **illustrative text written for the demo**, labelled so on the page; it is not from real agents and not a measurement | **not built** |
| Report: mandatory items said / said late / never | consistent with the call (checked by a test); the data notice is marked late **and incomplete** (the item asks for who, what for and the customer's rights; the agent's sentence has no rights) | the said / missing part is real data of the hand-off record; «late» and «incomplete» are not computed |
| Report: customer questions left unanswered | written by hand | **not built** |
| Voice assistant example | the answers are written; sources copied from the same domain | the real bot (`#/voicebot`) answers with the reasoner and the RAG |

Nothing in the three demos calls the reasoner, the RAG, the voice
service or the GPU at play time. The live demo holds the single session
slot like the interview demo does and is stopped by a real session. The
voice example is played entirely in the browser and takes no session
slot. Both can be **heard** with voices rendered ahead of time (DM-3,
#254, next section); without them the call is silent and the voice
example's assistant is spoken by the browser's own speech synthesis
when it has a Spanish voice (otherwise only the text is shown).

## Voices (DM-3, #254)

Every line of the call and of the voice example can be heard, each
scripted person with their own synthetic voice. The clips are made
**ahead of time** by an operator; playing an example needs no model,
GPU or session slot.

**Who is who.** The demo file names the people once, in `cast`: their
speaker id, name and gender (`agente` Mario, male; `cliente` Andrés
Ferrer, male; `asistente`, any). The agent was Marta until the owner's
two consenting cloned voices, both male, were chosen for the call; he
was renamed with the command below (`agente=Mario --gender male`). The voice example's sides point at the
cast (`voicebot.user` = `cliente`, `voicebot.bot` = `asistente`), so the
customer has the same voice in both examples. A voice of known gender
reads only a person of that gender. To give a person a name that fits
another voice, rename them everywhere in the file at once:

```sh
uv run --no-sync python -m orchestrator demo-voices recast llamada-lumbre agente=<Name> --gender <female|male>
```

`recast` changes every text that says the old name (whole, first and
last word: the report, the notes, «le habla Mario») and refuses a new
name of a different word count, a word already in the file, or a gender
change for a person the script words by gender (`gendered_text`: the
customer is «señor Ferrer», «harto», «he»; changing that needs
rewording, not a rename). The agent's lines and the report about the
agent are neutral (checked in Spanish and English after the rename: no
«la agente», «ella», «she» or feminine adjective), so the agent can
change gender with a rename. The interview
demo has a cast too, but `recast` refuses it: its stored analysis names
the candidate as well.

**Rendering.** `python -m orchestrator demo-voices render <demo>
--voice <person>=<voice> ...`, one `--voice` per person:

| Voice | Engine | Gender |
|---|---|---|
| `pocket:lola` | Pocket TTS (CPU by default, `--pocket-device`) | female |
| `piper:davefx` | Piper | male |
| `piper:sharvard-m` / `piper:sharvard-f` | Piper, speaker 0 / 1 of `sharvard` | male / female |
| `clone:<person>[:female\|:male]` | Qwen3-TTS on the GPU (`--clone-engine qwen3`, default) or OpenVoice (`openvoice`) | as given, else unchecked |

The command refuses (and changes nothing) when a person who speaks has no
voice, when two people of the same example would share one, when a voice's
gender is not the person's, and when a clone's person has no clone
consent or no reference clip (`VOICE_CLONING.md` §2). A person left out
keeps the voice of the last render, an explicit earlier choice; one
never chosen is an error. Only lines whose text, voice or engine changed
(or whose clip is missing) are rendered again; a second run with nothing
changed renders nothing. Every clip is brought to the same loudness
(RMS 3000 of 32767, never clipping), so voices of different engines sit
together.

**One voice per person in a conversation (owner's decision).** The
owner has two consenting voices and asked for one cloned voice per
person in each conversation. The rule is therefore per example: the
agent and the customer of the call never share a voice, nor do the
customer and the assistant of the voice example; the agent and the
assistant never talk to each other, so they may share one.

**The GPU.** Piper and Pocket on the CPU need nothing. A clone (or Pocket
on `cuda`) takes the GPU lease for the run **only if nobody holds it**,
checked and taken in one step, so the render never joins a live
session's lease or takes its memory; it gives the lease back only after
its engine worker has stopped. A refused lease or an engine that fails
mid-way leaves the previous clips and manifest exactly as they were: new
clips are written beside the old ones and only swapped in once every
line is done. Clip files that no manifest names (a render killed
half-way, `.part-*` files) are deleted when the next render of that demo
starts.

On the workstation, run it through `systemd-run`, never next to a
running session:

```sh
systemd-run --user --wait --collect -p WorkingDirectory="$PWD" /bin/bash -lc \
  'uv run --no-sync python -m orchestrator demo-voices render llamada-lumbre \
     --voice agente=clone:<person-a> --voice cliente=clone:<person-b> \
     --voice asistente=clone:<person-a>'
```

`demo-voices show <demo>` prints who reads whom, the clips and their
length; `demo-voices clear <demo>` deletes them. `--engines tone` is a
dry run with tones.

**Where the audio lives.** Outside the repository (it is public):
`data/transcriber/demo-voices/<demo>/` (git-ignored, beside the
voiceprints, the folder the transcriber's consent hook cleans). Another
`--out` is refused unless `--outside-consent-hook` is given: a
withdrawal does not reach clips there, and deleting them is the
operator's job. The orchestrator's `--demo-voices-dir` changes the
folder it serves; both refuse a folder inside the checkout that is not
under `data/`. Folders are mode 700, files 600. One WAV per line,
`<line>-<hash>.wav`, and `manifest.json` (`demo.voices.v1`): who reads
whom (`voices`) and per line the file, person, voice, `kind` (`clone` or
`stock`), `cloned_from` (the enrolled person of a clone), engine, text
SHA-256, seconds, rate and date. Never committed (a test fails if an
audio file is tracked outside the engine study's synthetic samples),
never in the static site, never in a screenshot or video asset of the
repository.

**Serving.** `GET /demos/{id}/voices` (orchestrator; `/api/assist/demos/
{id}/voices` behind the access code) says, per example (`call`,
`voicebot`), whether it can be heard and each clip's length, and per
person whether the voice is cloned or stock; no person id or file name
leaves the server. `GET .../voices/{line}.wav` gives one clip
(`audio/wav`, `Cache-Control: private, no-store`); the page fetches it
with the access code and plays it through **Web Audio**
(`decodeAudioData`, one `AudioContext` resumed inside the viewer's
click), never from a `blob:` URL: the public site's
Content-Security-Policy (`media-src 'self'`, `infra/Caddyfile`) refuses
those (a web test fails if a view does it). An example is heard
**whole or not at all**: one line without a clip of its current text
(text hash), its person's chosen voice, its file and (for a clone) its
person's consent, and the example falls back to text.

**Consent.** Withdrawing the clone consent, enrolling again without it
or forgetting the person deletes the reference clip **and**, in the same
transcriber hook (`delete_clone_reference`), every demo clip whose
manifest says `cloned_from` that person, plus every clip file no
manifest names (its owner cannot be told: a render killed half-way, or
one still running whose clips are not published yet). The reference clip
is deleted first, so a failing clean-up never leaves the voice clonable;
the withdrawal is recorded and the error reported. A render running at
that moment checks the consent again before publishing and, when it was
withdrawn, deletes its new clips and fails, changing nothing; if the
withdrawal lands between that check and the manifest write, the sweep has
already removed the unpublished clips and the render fails asking to run
again. What is covered: the hook cleans `<transcriber data>/demo-voices`
(by default `data/transcriber/demo-voices`, the render's default); the
orchestrator also re-checks consent for the folder it serves each time
the page asks, deleting a withdrawn person's clips there. Clips rendered
into any other folder (`--outside-consent-hook`) are not covered.

**On the page.**
- Voice example (`#/voicebot/demo`): nothing plays before «Empezar el
  ejemplo». Customer and assistant are heard in turn; each line stays as
  long as its clip, and with «Con sonido» unticked the timing is the
  same. The browser's own voice is not used when there are clips.
- Live call (`#/live/demo/support`) and interview (`#/live/demo`): the
  orchestrator replays the demo at speed 1 with **each sentence as long
  as its clip** (the pauses between sentences kept), so the text types
  in at the voice's pace; the page plays a line's clip when that line's
  first partial arrives. The audio is unlocked by the click that starts
  the demo (the cards, «Reproducir de nuevo», «Con sonido»); arriving
  from the home's link relies on the browser's memory of that click
  (Chrome allows it; a stricter browser may need «Con sonido»). A page
  opened directly by its address (no click on it yet) does not play: it
  says «Marca «Con sonido» para oír las voces». The interview gets the same treatment at no extra cost
  (its own `cast`); its stored result is not retimed.
- «Con sonido» is remembered in this browser (`localStorage`, on by
  default). A line under the controls says, once and for screen readers
  (`role="status"`), that the voices are synthetic (cloned with
  permission and/or stock) and belong to no real customer, or that they
  are not available. A clip that fails to load or play, or does not end
  within its length plus 2.5 s, is given up: the example goes on with
  the text and the page says once that the voice could not be played.
- English pages show the English captions and play the Spanish voices,
  and say so.

**Checked.** With stock voices only (no clone, no GPU:
`CUDA_VISIBLE_DEVICES` empty), `agente=pocket:lola`,
`cliente=piper:davefx`, `asistente=piper:sharvard-f`, rendered through
`systemd-run` into a scratch folder: 27 clips (19 call lines, 8 voice
example lines), 193.1 s of audio, 28.3 s of wall time including Pocket's
load; the call's 19 sentences last 147.1 s as rendered against 156.4 s as
scripted (first render; the engines sample, so lengths change a little
from one render to the next: the voice example's first line was 5.07 s
in one render and 5.2 s in another). A second run rendered nothing. On a throwaway dev stack (fake
models, own ports, Chrome headless and muted) the voice example played
its clips one after another (each clip's `ended` fired), the support
call played each line's clip as the line started, the switch was
remembered, and with the clips removed the page said the voices are not
available. That check ran with the public site's CSP header added to
every page by the browser driver: an `<audio>` on a `blob:` URL was
refused there (`NotSupportedError`, violation `media-src blob`) while
the Web Audio clips played to their end. Screenshots:
`~/.local/share/ragputin/demo-shots/dm254/`
(outside the repository). The clips were deleted afterwards. The
rendered voices were not listened to by a person, and cloned voices were
not rendered here.

## What a real top-scorer comparison needs

- A set of calls of the same type (for example «baja por cobro erróneo»)
  scored by people: who scores, with which sheet, and how their
  agreement is checked (the same reliability work as the interview
  analysis).
- The reaction of the customer after each agent sentence, scored the
  same way, to derive which wordings help.
- Enough calls per moment for a wording to be called «bien valorada»;
  one or two calls are not evidence.
- A decision on privacy (agents see their own report; who else does) and
  on wording: it is coaching material, never a ranking of people or an
  input to hiring or firing.

## Notes

- The report's two strengths and two things to improve, and every row,
  point at lines of the call («Línea 5»); the chips open the full
  conversation at the foot of the page.
- **What is translated.** English: the report (all its texts and the
  names of the mandatory items), the voice example's questions and
  answers and the source titles, the live page's suggestions and every
  label. **What is not:** the call's own sentences (the call is in
  Spanish, as the interview demo); the text of the knowledge-base chunks
  quoted as sources (live card and voice example), which quote a Spanish
  knowledge base and carry the label «(source text, in Spanish)» on the
  English page; the mandatory-items checklist on the live page, which
  lists the domain's Spanish wording exactly as a real call does.
- The top-scorer section is headed «Otras formas de decirlo (ejemplo)» /
  «Other ways to say it (example)» and carries a note: the sentences are
  written for the demo, not taken from real colleagues.
- The demo video tool (`tools/demo_video`) keeps using the interview
  demo; it was not re-recorded. Its recording stays **silent** after
  #254: its orchestrator reads an empty `--demo-voices-dir` of its own
  temporary folder (so the scripted timing is unchanged), Chrome runs
  with `--mute-audio` and ffmpeg writes no audio track.
- Screenshots of a throwaway dev stack (fake models) are in
  `~/.local/share/ragputin/demo-shots/dm250/` (outside the repository).
