# Transcriber: a separate service from recording to attributed sentences

See also: [TUTORIAL.md](TUTORIAL.md) (in Spanish: how to enrol, record and read a transcript).

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
| Asynchronous hand-over, started by the workstation | The workstation is behind NAT and often off; the server cannot call it. The transcriber *pushes* a finished transcript (`POST /transcripts` on the RAG) or writes it to a directory the RAG reads. No synchronous call from the RAG to the transcriber exists, so neither needs the other to be up. |
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
    enrolment.py     the text a person reads, checks on an enrolment sample
    consent.py       the versioned consent notices (enrolment, recording)
    errors.py
  application/
    ports.py         AudioDecoder, SpeechRecogniser, VoiceEncoder,
                     SessionRepository, VoiceprintRepository,
                     TranscriptPublisher, JobRunner, Progress, SessionLocks, Clock
    common.py        id checks and lookups shared by the use cases
    enrol.py         EnrolSpeaker, ForgetSpeaker, ShowEnrolmentText
    consent.py       ShowConsentNotice
    record.py        OpenSession, AppendAudio, CloseSession, ImportRecording,
                     DeleteSession
    transcribe.py    TranscribeSession (recognise → seams → score words →
                     diarize → sentences kept whole → identify →
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
                     thread_job_runner.py, thread_session_locks.py
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
    "duration": 1830.4, "audio_sha256": "…"
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
               "transcriber": "0.1.0"}
}
```

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
  the first real recordings). `domain/seams.py` keeps one copy. Only
  a step back of more than 1 s is a seam; a smaller one is jitter of
  the word times (seen: a few hundredths of a second) and the word's
  start is clamped forward, in text order. At a seam the two copies of
  the doubled stretch are aligned word by word (longest common
  subsequence of the words' letters, with compatible times); a word in
  both is kept once, from the window where it is farther from the
  window's edge (the earlier window before the stretch's middle, the
  later one after it). A word with no counterpart is kept, unless the
  other copy has different words at the same moment, in which case the
  copy from the window farther from its edge wins. The step is
  idempotent and never drops a word said once.
- A change of speaker inside a sentence is undone only when it is weak
  *and* the evidence favours the move (`whole_sentences`, measured in
  `IDENTIFICATION.md`, appendix): a stretch of at most
  `whole_sentence_max_move` (6 s) whose own speaker has fewer than two
  short windows or a margin below `whole_sentence_margin` (0.1) goes to
  a neighbouring speaker of the same sentence if at most one window
  prefers its own speaker, or its own margin is at most 0, or the
  neighbour's margin over it is higher. A stretch whose windows prefer
  its own speaker stays, and the sentence is then cut at the change
  (each part is its own turn). A sentence holding a moved stretch
  takes that stretch's margin as its `speaker_certainty` when lower
  (doubtful when below the cut, or when no window covers the moved
  stretch). A sentence ends at final punctuation or at a pause longer
  than `turn_gap`.
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

Without enrolment everything still works with anonymous labels.

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
