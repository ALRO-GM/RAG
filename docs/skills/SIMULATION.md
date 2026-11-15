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
`skills analyze` on it (`TUTORIAL.md` §7, `DESIGN.md` §11).

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
wall time on the workstation's CPU (20 threads). Piper adds random
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
   `sim-reunion-verano.reference.json`.

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

## 3. Results

See §4 for the run on the workstation.
