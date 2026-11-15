# Transcriber: a separate service from recording to attributed sentences

See also: [TUTORIAL.md](TUTORIAL.md) (in Spanish: how to enrol, record and read a transcript); [LAB.md](LAB.md) (the model lab: comparing denoisers, diarizers and recognisers).

Status: design, 2026-11-13. This document is the specification the
tickets of the *Transcriber* milestone implement. It replaces the
in-process audio commands of the analyzer (`docs/skills/AUDIO.md`),
whose measured building blocks it reuses.

## 1. What it is for

People sit around a table, a phone in the middle records them, and the
workstation turns the recording into a list the RAG and the analyzer
can use:

```
who spoke (a stable user id) · when (seconds in the audio) · what (one sentence) · who spoke before
```

Every sentence stays linked to its place in the audio, so a reader can
click it and hear it, starting a chosen number of seconds earlier.

## 2. Two services, one contract

```
 phone (browser) ──HTTPS (§5)──►  TRANSCRIBER (this workstation, GPU)
                                    domain · application · adapters
                                    owns: audio, voiceprints, sessions
                                        │
                                        │  transcript.v1 (JSON, text only)
                                        ▼  the workstation always initiates
                                  RAG / ANALYZER (workstation today, server later)
                                    domain · application · adapters
                                    owns: indexes, rubric, scorecards
```

| Decision | Reason |
|---|---|
| Two top-level packages in one repository: `src/` (RAG and analyzer) and `transcriber/` | Each is hexagonal on its own and can be tested, audited and reused alone; one repository keeps the contract, the CI and the history together. A test forbids imports between them in both directions. |
| They share only `contracts/transcript.v1.schema.json` | A versioned data contract is the whole coupling. Each side validates against it in its own tests (contract tests). |
| Asynchronous hand-over, started by the workstation | The workstation is behind NAT and often off; the server cannot call it. The transcriber *pushes* a finished transcript (`POST /transcripts` on the RAG) or writes it to a directory the RAG reads. No synchronous call from the RAG to the transcriber exists, so neither needs the other to be up. Deleting a session asks the RAG to delete its copy (`DELETE /transcripts/{id}`, best effort). Since TR-13 both run on the workstation and the hand-over goes over loopback (`docs/DEPLOYMENT.md`). |
| Audio and voiceprints are stored only on the workstation | They are personal (voice is biometric) data. The contract carries text, times and ids. The player gets the audio from the transcriber's API; through rag.peer4u.app it crosses the server in transit and is not kept there (§5). |
| One environment, dependency group `transcriber` | A second virtual environment would duplicate 5 GB of torch on a disk that is nearly full. The import ban keeps the packages independent anyway. |

Microservices were the author's proposal. The design keeps the
separation and the independent deployability and drops only the
synchronous request/response between services, which would fail
whenever the workstation is off.

## 3. Layers of `transcriber/`

```
transcriber/
  domain/          pure Python: no I/O, no torch
    models.py        Word, Utterance, Speaker, Voiceprint, Session, Transcript
    sessions.py      session states and moves, chunk acceptance, interruption
    parameters.py    TranscriptionParameters: every threshold of a transcription
    activity.py      speech activity from frame energy; drops words over silence
    seams.py         keeps once the passages the recogniser returned twice
                     at the seams of its windows; words back in time order
    languages.py     a session's language code or "auto"; validation
    sentences.py     sentence splitting, utterance ids
    speakers.py      word→speaker assignment, smoothing, turns, previous speaker
    certainty.py     per-sentence speaker certainty, doubt, overlap and
                     alternative speaker from the short windows' cosines;
                     keeps a sentence whole on a weak change of speaker
    voices.py        single-voice test, clustering, centroids, matching to voiceprints
    diarization.py   which lab diarizer a session gets (DiarizerRule); its
                     segments as spans, each label's own windows, label
                     voices and the short windows' cosines (TR-18)
    enrolment.py     the text a person reads, checks on an enrolment sample
    consent.py       the versioned consent notices (enrolment, recording)
    errors.py
  application/
    ports.py         AudioDecoder, SpeechRecogniser, VoiceEncoder,
                     OverlapDetector, Diarizer,
                     SessionRepository, VoiceprintRepository,
                     TranscriptPublisher, PublishedTranscripts, JobRunner,
                     Progress, SessionLocks, Clock
    common.py        id checks and lookups shared by the use cases
    enrol.py         EnrolSpeaker, ForgetSpeaker, ShowEnrolmentText
    consent.py       ShowConsentNotice
    record.py        OpenSession, AppendAudio, CloseSession, ImportRecording,
                     DeleteSession
    transcribe.py    TranscribeSession (recognise → seams → score words →
                     diarize (lab diarizer, or built-in grouping) →
                     sentences kept whole → identify →
                     sentences → certainty),
                     QueueTranscription, TranscriptionStatus
    recover.py       RecoverSessions (start-up: fail sessions left queued/running)
    publish.py       PublishTranscript
    review.py        RenameSpeaker, ReassignUtterance
    browse.py        ListSessions, ListSpeakers, ShowSession, ShowTranscript,
                     SessionAudio (read-only, for the pages)
  adapters/
    inbound/         http_api.py (FastAPI, all routes under /api),
                     access.py (access code, throttling of failed codes),
                     playback.py (byte ranges of the player's WAV, streamed),
                     cli.py (Fire), web/ (static pages)
    outbound/        av_audio_decoder.py, whisper_recogniser.py,
                     wavlm_voice_encoder.py, fake_models.py,
                     fs_session_repository.py (with the player's WAV
                     rendering, kept on disk), wav_playback.py,
                     fs_voiceprint_repository.py,
                     http_transcript_publisher.py, fs_transcript_publisher.py,
                     thread_job_runner.py, thread_session_locks.py,
                     lab_runner_diarizer.py (a lab runner as Diarizer)
  config.py        TranscriberSettings (pydantic)
  bootstrap.py     composition root (start_container also runs RecoverSessions)
  __main__.py
```

Session states: `recording → uploaded → queued → running → done |
failed`. A `failed` or `done` session can be queued again (a `done` one
keeps its transcript until the new one is saved); `queued` and
`running` sessions left by a stopped process are failed as
`interrupted` at start-up; a session can be deleted in any state but
`running`. Each queueing gives the session a new `run_id` that the job
carries, and every use case that reads then writes a session holds that
session's lock (`SessionLocks`), so a stale job never runs on, or saves
over, a session that was deleted, re-created or queued again.

Rules, same as the RAG: dependencies point inwards; the domain imports
nothing from the other layers; adapters are the only place that
imports torch, transformers, FastAPI, the file system or the network;
flake8 and `mypy --strict` clean; unit tests with fakes for every use
case, integration tests for every adapter, real models only in tests
marked `slow`.

## 4. The contract: `transcript.v1`

```json
{
  "schema": "transcript.v1",
  "session": {
    "id": "2026-11-14-equipo-a", "title": "Equipo A, reparto",
    "language": "es", "started_at": "2026-11-14T10:02:00+01:00",
    "duration": 1830.4, "audio_sha256": "…",
    "type": "reunion_decision", "goal": "Repartir el trabajo del sprint"
  },
  "speakers": [
    {"id": "ana", "name": "Ana", "enrolled": true},
    {"id": "S1", "name": null, "enrolled": false}
  ],
  "utterances": [
    {
      "id": "t3.s0", "turn": 3, "index": 0,
      "speaker": "ana", "start": 44.12, "end": 47.90,
      "text": "Eso no lo habíamos mirado.",
      "previous_speaker": "S1", "previous_utterance": "t2.s1",
      "confidence": 0.93, "speaker_confidence": 0.71,
      "speaker_certainty": 0.04, "doubtful": true, "overlap": false,
      "alternative_speaker": "S1"
    }
  ],
  "pipeline": {"recogniser": "openai/whisper-large-v3-turbo",
               "voice_model": "microsoft/wavlm-base-plus-sv",
               "transcriber": "0.1.0",
               "diarizer": "diarizen-wavlm-large-s80-md-v2",
               "diarizer_licence": "CC-BY-NC-4.0 (weights); MIT (code)",
               "diarizer_commercial": false}
}
```

- `session.type` and `session.goal` (optional, added compatibly, #65)
  are the kind of conversation and what it was for, declared by
  whoever records it (`docs/skills/CONVERSATION.md` §4.3: declared,
  never inferred). `type` is one of `reunion_decision`,
  `lluvia_ideas`, `resolucion_problemas`, `negociacion`, `entrevista`,
  `feedback`, `conflicto`, `coaching`, `venta`, `conversacion_libre`,
  `otro`; `goal` is free text of at most 500 characters. Both are
  written only when set. They are chosen when a session is opened or
  imported, and changed in any state with `PATCH /api/sessions/{id}`
  (`DescribeSession`: a field left out is kept, `null` clears it),
  which also updates the stored transcript; a job that was running
  meanwhile carries the stored values over when it saves
  (`carry_description`), so an edit is never lost. The analyzer and
  the person report show them as context; no rating uses them.
- An **utterance** is one sentence of one turn. Its id (`t<turn>.s<n>`)
  is the one the analyzer already uses for sentence tags.
- `start`/`end` are seconds from the start of the session audio: the
  link to the recording. They come from word times; only text with no
  usable word times (no words, or words that do not match the text) is
  given times interpolated from its turn, in proportion to its
  characters.
- `previous_speaker` is the speaker of the turn before this one (the
  same for every sentence of a turn; `null` in the first turn), and
  `previous_utterance` the last sentence of that turn. It lets a
  consumer frame a reply ("Luis, answering Marta") or build larger
  chunks that keep the exchange together.
- `speaker` is an enrolled user id when the voice was recognised, or
  an anonymous label (`S1`, `S2`, …) that is stable inside the session.
  `speaker_confidence` is the similarity of the speaker's whole voice
  group with their voiceprint (the same for every sentence of a
  speaker). `confidence` is the recogniser's mean word probability
  (Whisper re-reads its own words against the audio, at most 30 s at a
  time, and gives each token a probability; a word's is the mean over
  its tokens). The words are scored once they are final (seams
  resolved, words over silence dropped: `SpeechRecogniser.score_words`,
  a second port method the use case calls), so a doubled passage is
  never scored against audio that holds it once. A failure of that
  pass is logged and noted on the job, and the transcript is saved
  with `confidence` `null`: the scores are an extra, not a reason to
  fail the job.
- A session's `language` must be a code the recogniser knows (Whisper:
  the 100 codes it has a token for; `es-ES` is read as `es`) or `auto`.
  Opening, importing and queueing a session check it, so a typo is a
  `400` before any job runs, not a failed transcription.
- Per sentence, from the short voice windows over it (optional fields,
  added compatibly; absent when there is one voice or no window):
  `speaker_certainty` is the mean margin of the sentence's speaker over
  the closest other voice (a difference of cosines in [-2, 2], **not a
  probability**), `doubtful` a margin below `doubt_margin` (0.1),
  `overlap` a window inside the sentence that prefers another voice,
  and `alternative_speaker` that other voice when the sentence is
  doubtful (the other voice the windows over its weakest stretch
  prefer most) or an overlap (the voice of the window that flagged
  it). Definition, rationale and limits: `IDENTIFICATION.md`,
  appendix. A sentence moved by hand loses them, and so do the
  sentences of the target of a merge (their margins may have been
  taken against the voice merged into them).
- Utterances are strictly in time order. The Whisper pipeline reads
  30 s windows that start every 20 s; when its join of two windows
  fails, the overlap comes out twice and the word stream jumps back to
  the later window's start (seen at 20, 80, 180, 240, 260 and 360 s on
  the first real recordings). `domain/seams.py` keeps one copy. First,
  a word longer than it can plausibly last (3 s, or 0.4 s per letter
  for long words; a judgement) is shortened to that, keeping its end
  when its start jumped back and its start otherwise: one word with an
  impossible span must not look like a seam over a whole passage. Only
  a step back of more than 1 s is a seam; a smaller one is jitter of
  the word times (seen: a few hundredths of a second) and the word's
  start is clamped forward, in text order, unless the word is the same
  word as one it steps back over (then it is treated as a seam too).
  At a seam the two copies of the doubled stretch (plus the later
  copy's words up to 1 s past it) are aligned word by word (longest
  common subsequence of the words' letters, with compatible times); a
  word in both is kept once, from the window where it is farther from
  the window's edge (the earlier window before the stretch's middle,
  the later one after it). A word with no counterpart is kept, unless
  the other copy has different words at the same moment: then the copy
  with more words there wins, and with as many, the copy from the
  window farther from its edge. The step is idempotent; it drops a
  word said once only in favour of at least as many other words heard
  at the same moment, so one word never erases two.
- A change of speaker inside a sentence is undone only when the
  windows do not support it (`whole_sentences`, measured in
  `IDENTIFICATION.md`, appendix): a stretch of at most
  `whole_sentence_max_move` (6 s) goes to a neighbouring speaker of the
  same sentence when its own margin is unknown (no window), at most 0,
  or below `whole_sentence_margin` (0.1) with at most one window
  preferring its own speaker; and one sentence gives away at most
  `whole_sentence_max_moved` (6 s, a judgement) in all. A stretch whose
  windows prefer its own speaker by the cut, or with a positive margin
  that two windows prefer, stays, and the sentence is then cut at the change
  (each part is its own turn). A sentence holding a moved stretch
  takes that stretch's margin as its `speaker_certainty` when lower
  (doubtful when below the cut, or when no window covers the moved
  stretch). A sentence ends at final punctuation or at a pause longer
  than `turn_gap`.
- `overlap` combines two sources when an overlap detector is
  configured (`overlap_detector`, port `OverlapDetector`; used only to
  flag sentences, never to group voices): a sentence with at least
  `overlap_seconds` (0.5 s) on stretches the detector hears two voices
  in is an overlap, and `overlap_source` says which fired (`voices`,
  `detector`, `both`). The candidate measured in `OVERLAP.md` §3
  (NVIDIA's streaming Sortformer v2, CC BY 4.0) needs NeMo, which the
  project's environment cannot take without resolving pins (`OVERLAP.md`
  §9); the switch defaults to `off` and only a deterministic `fake`
  ships.
- `language` of a sentence is the language Whisper detects in the
  scoring window it was said in (one decoder step on the encoder output
  the scoring pass already has). With a session language of `auto`
  Whisper detects the language of every window when it transcribes and
  the session keeps `auto`; with a fixed one, a sentence detected in
  another language gets `language_differs: true`. The speaker's
  `identification` is described in §6.
- `pipeline.diarizer` names the model that separated the voices
  (`builtin`, or a lab catalogue id, §6.1), with the licence of its
  weights (`diarizer_licence`) and whether it allows commercial use
  (`diarizer_commercial`), copied from `lab/catalogue.json`. Optional
  fields added compatibly (TR-18): absent in older transcripts, and
  the licence fields absent for `builtin`.
- Words are kept by the transcriber (for the player and for
  re-splitting) but are not part of the contract.
- A change that removes or renames a field is `transcript.v2`.

## 5. Recording from the phone, through rag.peer4u.app

The app is used from **https://rag.peer4u.app** (decision of
2026-11-14, ticket TR-15). The server cannot run the models, so it is
only the front door; the work and every stored byte stay on the
workstation.

```
phone ──HTTPS──► Peer4U Caddy ──► ragputin-web (Caddy): static pages from /srv
                                    ├─ /api/*    ──► tunnel:9000 ──┐  chisel server, on a
                                    └─ /_tunnel  ──► tunnel:8080   │  private docker network
                                         ▲ (WebSocket)             │  (container `tunnel`)
                                         │                         ▼
          workstation: chisel client ────┘   R:0.0.0.0:9000 ──► API on 127.0.0.1,
                       (wss://rag.peer4u.app/_tunnel)             GPU, storage
```

`python -m transcriber serve` starts the API on 127.0.0.1, opens and
supervises the tunnel, and prints the join link and a QR code. The
phone opens the link in its browser; nothing is installed.

The tunnel is a chisel client (github.com/jpillora/chisel, the release
pinned in `infra/`) run by a supervisor thread
(`adapters/outbound/chisel_tunnel.py`, settings `tunnel.*`). It
connects over HTTPS to `wss://rag.peer4u.app/_tunnel` and opens one
reverse port, `R:0.0.0.0:9000`, in the server's `tunnel` container,
which leads back to the API; the server's Caddy sends `/api/*` there.
The credential (`user:password`, `make tunnel-secret`) is read from a
file of mode 600 and handed to chisel in its environment, never on the
command line. The client exits when its connection drops (with
`--max-retry-count 0`, tested with chisel 1.12.0: killing the server
under an established connection ends the client with code 1 after
`Connection error: websocket: close 1006`) and is restarted with capped
exponential back-off; it is reported `connected` only after chisel says
so and `https://rag.peer4u.app/api/health` answers through it. That
probe goes on every `probe_interval` seconds while connected; after 3
failures in a row (`PROBE_FAILURES`) the tunnel is reported
`reconnecting` and the client is ended and started again, for a path
that broke while chisel stayed up. Its pid, the exact arguments it was
started with and its start time are kept in `<data_dir>/tunnel.pid`,
so a client left by a crashed `serve` is ended at the next start, and
only when all three still match (a reused pid, even one of another
`serve`'s client, is left alone). The
access code is new at every start, or kept in
`data/transcriber/access_code` (mode 600) with `serve --keep-code` so a
printed QR keeps working; `--new-code` replaces the kept code. One
`serve` per data directory: it takes an exclusive lock on
`<data_dir>/serve.lock` and binds its port before recovering
interrupted jobs, so a second one stops (exit code 3, or 1 for a taken
port) without touching the first one's sessions. During a recording a
401 (the service restarted with a new code) pauses the uploads; the
page keeps capturing (at most 30 minutes in memory) and asks for the
new link or code, then resumes from the chunk the service holds.
Closing a session is idempotent. `--no-tunnel` prints an
`http://127.0.0.1:<port>/#k=<code>` link for local development. Settings
come from `TRANSCRIBER_CONFIG` (a JSON file), `TRANSCRIBER_DATA_DIR`,
`TRANSCRIBER_RUNTIME` (`fake` runs every command without torch) and
`TRANSCRIBER_DEVICE`.

| Concern | Decision |
|---|---|
| Microphone access needs HTTPS | The domain's real certificate. No self-signed certificate, no browser warning. |
| The site is on the internet | Every `/api/*` call carries an access code in a header. The join link is `https://rag.peer4u.app/#k=<code>`: a URL fragment is never sent to a server, so the code does not reach logs. Failed codes are throttled. Without the code the pages load and can do nothing. |
| Little on the shared server | Two small containers in `/opt/ragputin`: the Caddy that serves the pages, and a chisel server (64 MB, read-only, no capabilities, non-root) that the workstation reaches through that Caddy at `/_tunnel`. Only WebSocket upgrades on `/_tunnel` reach chisel. The chisel server is on a private docker network shared only with the Caddy: no other container of the host, and no published port, reaches it or the tunnel's port. No SSH access is involved: the tunnel's credential gives nothing on the host. Files and operations: `docs/DEPLOYMENT.md`. |
| What the tunnel credential allows | Only the reverse port 9000 (chisel's users file: `^R:0\.0\.0\.0:9000$`): no SOCKS, no forward tunnels, no other port, no shell. chisel 1.12.0 checks the list on every channel, not only on the remotes declared at connection; the docker proof's hand-made client with the credential is refused every other destination (`docs/DEPLOYMENT.md`). Whoever holds it can still serve `/api/*` for the domain while the workstation is not connected (and so impersonate the API to the phones): keep it on the workstation only; to rotate it, delete the credential file, `make tunnel-secret` and `make infra-up`. |
| Who is throttled | The inner Caddy sets `X-Client-IP` from the address the Peer4U Caddy saw, overwriting any value the caller sent; the API trusts that header only from loopback (the tunnel's local end) or a unix socket and never reads `X-Forwarded-For` (uvicorn runs with `proxy_headers=False`). A right code is never refused. |
| Where audio goes | Through the server in transit (TLS ends at the Peer4U Caddy; between the containers on the server it is plain HTTP, and from the `tunnel` container to the workstation it travels in chisel's encrypted SSH channel), stored only on the workstation. The server keeps static files and nothing else. |
| Third-party code on the internet | The chisel server is reachable from the internet behind the two Caddies (only for WebSocket upgrades on `/_tunnel`); a flaw in it before authentication would be exposed. It is pinned by version and digest, runs with no capabilities, read-only, as an unprivileged user, on a network only the Caddy shares. |
| Workstation off or tunnel down | `/api/*` answers 502; the pages say the workstation is not available. |
| A long recording must survive a dropped connection | The page captures PCM (16 kHz, mono, 16-bit) and uploads it in numbered chunks of a few seconds; the service appends them and tolerates a retried chunk. No codec is involved. A screen wake lock keeps the phone awake. |
| A recording made with another app | `transcriber import FILE` or the upload form; decoded with PyAV (bundles the FFmpeg libraries; nothing to install). |
| One GPU | Jobs run one at a time in a worker thread; the API reports `queued / running (stage, %) / done / failed`. Enrolment uses the same queue. |
| Choosing the analysis judge from the phone | `GET/PUT /api/analysis/judge` (access code) are forwarded by the transcriber to the RAG's `/analysis/judge` on loopback with the RAG's bearer token (`adapters/outbound/http_analysis_judge.py`, same URL and token as the HTTP publisher); the RAG stays out of the tunnel. Without the HTTP publisher they answer 503 `analysis_unavailable`. `docs/skills/DESIGN.md` §4. |
| Analysing a session from the phone | `POST /api/analysis/sessions/{id}`, `GET /api/analysis/sessions/{id}`, `GET …/steps/{step}`, `GET …/reports/{n}` (access code) are forwarded to the RAG's `/analysis/sessions/…` on loopback with the RAG's token (`adapters/outbound/http_session_analysis.py`); ids, step names and report numbers are checked first (`application/analysis.py`). A report is served as a download with `Content-Security-Policy: sandbox; default-src 'none'; style-src 'unsafe-inline'; img-src data:`. The «Análisis» page is `#/analysis/<id>` (`web/js/views/analysis.js`). `docs/skills/DESIGN.md` §12. |

The results site that used to live on this domain was retired on
2026-11-14; the benchmark documents remain in the repository
(`docs/benchmarks/`, `experiments/`). CI builds the pages with
`python -m transcriber.adapters.inbound.bundle` (plus `version.json`)
and deploys them on every merge to `main`.

## 6. Putting a name on a voice

Whisper writes the words; it does not know who speaks. Diarization
separates the voices of a recording into anonymous groups. To attach a
**user id** the service needs one sample of each person's voice
(*enrolment*): about 30 seconds, read from the screen, starting with
the person's name. The content does not matter to the voice model;
the fixed text guarantees enough varied speech, and the spoken name
lets a human check the sample.

- Enrolment keeps a **voiceprint** (one embedding vector), the consent
  given (date, text version) and nothing else; the sample audio is
  deleted unless configured otherwise. `speakers forget ID` deletes it.
- In a session, each voice group is matched to the enrolled
  voiceprints (best one-to-one assignment by cosine similarity, above
  a threshold). Unmatched groups stay `S1`, `S2`… and can be named by
  hand afterwards (`RenameSpeaker`), which never creates a voiceprint.
- Enrolling with the same phone, in the same room, right before the
  meeting gives the closest match. The tutorial says so.
- Known pitfall from `AUDIO.md`: embeddings were centred on the mean
  of the recording, which is meaningless for a one-voice enrolment
  sample. Matching across recordings needs its own normalisation and
  must be measured (ticket TR-05). Measured in `IDENTIFICATION.md`:
  raw unit vectors at cosine 0.95; enrolled voiceprints used as seeds
  of the grouping did better than cluster-then-match and are the
  recommended next step.
- **Expected speakers** (TR-17, from `OVERLAP.md` §2.5–2.6). When a
  session lists who is present and the number of speakers equals the
  list, the voices are clustered into that many groups and assigned to
  the listed voiceprints one to one (maximum total cosine), each pair
  accepted at `expected_floor` (0.8) or more. 0.8 is a judgement from
  three real sessions of two people (lowest true pair 0.899, highest
  cross pair 0.730), not a measured error rate; the list is what
  guards the names. A person named below the open threshold (0.95) is
  marked `identification: "expected_floor"` in the transcript and the
  player shows it as a less certain identification and recommends
  re-enrolling. Open sessions (no list, or a count other than the
  list) keep the measured seeding at 0.95, unchanged.

Without enrolment everything still works with anonymous labels.

### 6.1 Which diarizer separates the voices (TR-18)

The lab measured the built-in grouping far behind the published
diarizers on public recordings with a known truth (`LAB.md` §13.8: DER
0.49–0.50, five or six speakers too many). By the owner's decision of
2026-11-15 the service uses **the best measured diarizer, even with a
non-commercial licence**, to see the upper bound (the project is not
monetised):

| Session | Diarizer (setting `diarizer`) | Licence of the weights |
|---|---|---|
| default (meetings, unknown count) | `diarizen-wavlm-large-s80-md-v2` (`model`) | CC BY-NC 4.0: **non-commercial** |
| `num_speakers` = 2 (a call, two people) | `nemotron-3-diarization` (`by_speakers: {"2": ...}`) | OpenMDW-1.1, commercial use allowed |

- The model runs as in the lab: its runner (`lab/runners/<family>.py`)
  in the family's own environment (`make lab-env FAMILY=diarizen`,
  `FAMILY=transformers`), as a subprocess over a WAV the adapter writes
  under `<data_dir>/diarize/` and deletes afterwards
  (`adapters/outbound/lab_runner_diarizer.py`). The model is loaded for
  one job and its GPU memory is freed when the process ends; before it
  starts the service unloads its own recogniser (Whisper is not needed
  again in that job), so the card (8 GB, shared with the RAG and
  another project's worker) holds the diarizer and little else.
- The runner and its copy of the audio never outlive the job. The
  runner runs in its own process group, so the service kills it
  itself: when the session is deleted or queued again (the job polls
  the stored session every half second and stops quietly, saving
  nothing), and when the service stops (Ctrl-C, SIGTERM, or at exit:
  `TranscriberContainer.stop_runners`, which also refuses to start
  another runner; the job then fails and is recovered as interrupted
  at the next start). Deleting a session whose job is running is
  refused anyway (§3), so a running job's audio is never orphaned by a
  deletion. A `diarize-*` folder left by a service killed mid-run
  (`SIGKILL`, power loss: no `finally` runs) is deleted when the
  service starts (`LabRunnerDiarizer.sweep`, one service per data
  directory). A runner left by such a kill is not tracked across
  restarts: it ends when its model finishes or fails.
- A diarizer label too short for any voice window has no voice vector
  and is never matched with a voiceprint (it stays `S1`, `S2`…) in any
  normalisation (`domain/diarization.py`, `name_labels`).
- No denoiser runs first (`LAB.md` §13.8: cleaning did not help).
- Each word takes the diarizer segment it overlaps most. Each label's
  voice is embedded with the service's voice model over the stretches
  where it speaks alone (as the lab's `voiceprint` mapping does) and
  matched one to one to the enrolled voiceprints at `match_threshold`
  (or `expected_floor` under the expected-speakers rule); a label that
  matches nobody stays `S1`, `S2`… The voiceprints do not seed the
  grouping here: the diarizer already grouped the voices.
- The sentences keep their certainty, doubt, alternative speaker and
  "kept whole" rule: the short voice windows are scored against each
  label's voice (centred like the built-in grouping's). `overlap`
  joins the diarizer's own overlap stretches to the overlap detector's
  (`overlap_source: detector` when only those fired).
- Fallback: when the chosen model's environment is missing or not
  verified, the model is gated without a token, the runner fails, or
  it hears no one, the job uses the built-in grouping, logs a warning
  and notes it on the job (`pipeline.diarizer` is then `builtin`).
  `transcriber serve` warns at start-up for each model of the rule
  that cannot run. `{"diarizer": {"model": "builtin", "by_speakers":
  {}}}` turns the lab diarizers off; with `runtime: "fake"` they are
  never used.
- The lab's own `builtin` step keeps measuring the built-in grouping.

**Measured before the switch** (`experiments/transcriber/
service_diarizer_eval.py`, rows in `service_diarizer.jsonl`, summary in
`results/diarizer_switch.json`; the RTX 3070 Ti, 2026-11-15/16). The
whole transcription (Whisper, then the diarizer, then words on
speakers) on the original audio of the lab's corpus sessions, no one
enrolled; the turns of the attributed words scored against the human
reference with the lab's scoring (collar 0.25 s). CALLHOME calls were
started with 2 speakers (Nemotron-3), AMI meetings without a count
(DiariZen). Means over the sessions:

| Corpus | Built-in grouping | Lab diarizer | The diarizer's own segments | Speakers too many (built-in → lab) |
|---|---|---|---|---|
| AMI, 4 meetings, 4 people | DER 0.495 | **0.266** | 0.090 (#44: 0.089) | +5.5 → +0.75 |
| CALLHOME es, 10 calls | DER 0.359 | **0.223** | 0.109 (#44: 0.106) | −0.6 → +0.2 |

The transcription's DER is higher than the diarizer's own because it
scores the turns of the recognised words: speech Whisper does not
transcribe counts as missed, and a turn spans the pauses between its
words. Sentences flagged doubtful rose on the calls (10.7% → 33.1% of
sentences; AMI 57.4% → 44.1%), and overlaps now include the diarizer's
own (AMI 52%, calls 34% of sentences flagged): the certainty margins
were tuned on the built-in grouping and were not re-tuned here.

Words on the right id, synthetic meetings of `IDENTIFICATION.md` (20
per split, 4 readers, mean of the four enrolment conditions, match
threshold 0.95 unit; the default before recomputed on the same cache):

| Split | Before: voiceprints seed the built-in grouping, count given / estimated | DiariZen, then match (given / estimated) |
|---|---|---|
| dev | 94.0% / 88.9% | 99.1% / 99.1% |
| test | 91.4% / 81.7% | 98.1% / 98.1% |

DiariZen found the same voices with and without the count. The worst
meeting is 77.6% on `test` (before: 56.1% given). Read speech from
audiobooks, one voice at a time: not a phone on a table.

GPU (`nvidia-smi`, card otherwise idle at about 460 MiB): DiariZen's
runner peaks at 3.7 GB (torch), Nemotron's at 0.6 GB; the service's own
process peaks at 1.9–2.5 GB (Whisper, released before the runner
starts, then WavLM); the whole card peaked at 4.4–4.5 GB on AMI and
2.8–3.3 GB on the calls. One call (`callhome-data-0008`) was measured
on the CPU while another project held the card (76 minutes for 300 s
of audio, DER 0.168, `service_diarizer.jsonl`); its time and memory
are left out of the GPU figures. Time on the GPU: 31–57 s for a 5–10
min call, 131–212 s for a 25–39 min meeting.

## 7. Listening to a sentence

The transcript page lists the utterances (time, speaker, text). A
click sets the audio to `start − lead` and plays on from there; `lead`
(seconds of context before the sentence, default 3) is a number field
kept in the browser. The sentence being heard is highlighted while the
audio runs. The audio is served by the transcriber with HTTP range
requests, so seeking does not download the file. An `<audio>` element
cannot send the access-code header, so the page first calls
`POST /api/access/cookie` (with the header): it sets an `HttpOnly`,
`Secure`, `SameSite=Strict` cookie, limited to `/api`, that holds a
random playback token made when the service starts (not the access
code). Only the audio route accepts that cookie.

## 8. Privacy defaults

Consent is asked at enrolment and at the start of a session (the
recording page states what is recorded, what is derived and where it
stays). The notices are versioned data of the domain
(`domain/consent.py`, Spanish and English) served by the API
(`GET /api/consent`, and with the enrolment text); the pages show that
text, and the version shown is stored with the enrolment and with the
session (`Session.consent_version`). The transcription options chosen
when a recording starts (`num_speakers`, `expected_speakers`) are kept
on the session too, so it is queued the same from any device. Audio, voiceprints and transcripts are stored under
`data/transcriber/` (ignored by git). `sessions delete ID` and
`speakers forget ID` remove them. Pushing a transcript to another
machine is an explicit command. Rating people from recordings needs
the decisions listed in `AUDIO.md` §5 before real use.

## 9. What exists and what is new

| Piece | State |
|---|---|
| Whisper adapter, WavLM two-pass diarization, sentence splitting, WER and speaker metrics | Exist under `src/…/skills`; they **move** to `transcriber/` and the analyzer keeps only the transcript reader |
| Speaker identification by enrolment, sessions, phone recording, player, previous speaker, contract, publisher | New |
| Measured so far | Synthetic read speech only: WER 5.4%, 91.5% of words on the right speaker. A phone on a table (distance, overlap, noise) will be worse and must be measured on a consented real recording |
