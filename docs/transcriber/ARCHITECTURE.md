# Transcriber: a separate service from recording to attributed sentences

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
 phone (browser) ──HTTPS, LAN──►  TRANSCRIBER (this workstation, GPU)
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
| Audio and voiceprints never leave the workstation | They are personal (voice is biometric) data. The contract carries text, times and ids. The player that needs the audio is served by the transcriber itself. |
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
    sentences.py     sentence splitting, utterance ids
    speakers.py      word→speaker assignment, smoothing, turns, previous speaker
    voices.py        clustering, centroids, matching clusters to voiceprints
    enrolment.py     the text a person reads, checks on an enrolment sample
    errors.py
  application/
    ports.py         AudioDecoder, SpeechRecogniser, VoiceEncoder,
                     SessionRepository, VoiceprintRepository,
                     TranscriptPublisher, JobRunner, Progress
    enrol.py         EnrolSpeaker, ForgetSpeaker
    record.py        OpenSession, AppendAudio, CloseSession, ImportRecording
    transcribe.py    TranscribeSession (recognise → diarize → identify → sentences)
    publish.py       PublishTranscript
    review.py        RenameSpeaker, ReassignUtterance
  adapters/
    inbound/         http_api.py (FastAPI), cli.py (Fire), web/ (static pages)
    outbound/        av_audio_decoder.py, whisper_recogniser.py,
                     wavlm_voice_encoder.py, fs_session_repository.py,
                     fs_voiceprint_repository.py, http_transcript_publisher.py,
                     fs_transcript_publisher.py, thread_job_runner.py
  config.py        TranscriberSettings (pydantic)
  bootstrap.py     composition root
  __main__.py
```

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
      "confidence": null, "speaker_confidence": 0.71
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
  link to the recording. They come from word times, not from
  interpolation.
- `previous_speaker` is the speaker of the turn before this one (the
  same for every sentence of a turn; `null` in the first turn), and
  `previous_utterance` the last sentence of that turn. It lets a
  consumer frame a reply ("Luis, answering Marta") or build larger
  chunks that keep the exchange together.
- `speaker` is an enrolled user id when the voice was recognised, or
  an anonymous label (`S1`, `S2`, …) that is stable inside the session.
  `speaker_confidence` is the similarity that supports the attribution.
- Words are kept by the transcriber (for the player and for
  re-splitting) but are not part of the contract.
- A change that removes or renames a field is `transcript.v2`.

## 5. Recording from the phone

`python -m transcriber serve` starts the service on the workstation,
prints the address and a QR code, and shows progress in the terminal.
The phone opens that address in its browser; nothing is installed.

| Concern | Decision |
|---|---|
| Microphone access needs HTTPS | A self-signed certificate is generated once (`openssl`), stored under `data/transcriber/tls/`. The tutorial explains the browser warning. |
| Only the people at the table may use it | The address carries a random token created at start-up; every request needs it. The service listens on the LAN interface only while `serve` runs. |
| A long recording must survive a dropped connection | The page captures PCM (16 kHz, mono, 16-bit) and uploads it in numbered chunks of a few seconds; the service appends them and tolerates a retried chunk. No codec is involved. A screen wake lock keeps the phone awake. |
| A recording made with another app | `transcriber import FILE` or the upload form; decoded with PyAV (bundles the FFmpeg libraries; nothing to install). |
| One GPU | Jobs run one at a time in a worker thread; the API reports `queued / running (stage, %) / done / failed`. |

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
  must be measured (ticket TR-05).

Without enrolment everything still works with anonymous labels.

## 7. Listening to a sentence

The transcript page lists the utterances (time, speaker, text). A
click sets the audio to `start − lead` and plays on from there; `lead`
(seconds of context before the sentence, default 3) is a number field
kept in the browser. The sentence being heard is highlighted while the
audio runs. The audio is served by the transcriber with HTTP range
requests, so seeking does not download the file.

## 8. Privacy defaults

Consent is asked at enrolment and at the start of a session (the
recording page states what is recorded, what is derived and where it
stays). Audio, voiceprints and transcripts are stored under
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
