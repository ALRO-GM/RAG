# Contracts

Data exchanged between the services of this repository. A contract is
the only thing two services share: neither imports the other's code.

| File | Producer | Consumers |
|---|---|---|
| `transcript.v1.schema.json` | `transcriber/` | `src/` (analyzer, RAG) |
| `knowledge.query.v1.schema.json` | orchestrator, analyzer, web (via the transcriber) | `src/` (`POST /knowledge/query`) |
| `knowledge.result.v1.schema.json` | `src/` | orchestrator, analyzer, web |
| `live.event.v1`, `gpu-lease.v1` | `transcriber/` (live) | orchestrator, live page, GPU workers |
| `reasoner.request.v1`, `reasoner.response.v1` | orchestrator | `src/` (`POST /reason`) |
| `tts.request.v1` | orchestrator | Piper adapter |
| `interview.plan.v1`, `interview.sheet.v1` | orchestrator (`data/interview/`, AI interviewer #208) | web (via the transcriber), orchestrator |
| `interview.board.v1` (no schema file: derived on read by `orchestrator/domain/offer_board.py`, `GET /interview/plans/{id}/board`, #271) | orchestrator | web «Ofertas» (via the transcriber) |
| `interview.candidate.v1` (no schema file: the pydantic model `orchestrator/domain/candidate.py` is its definition) | orchestrator (`data/interview/candidates/`, #235) | web (via the transcriber), orchestrator |
| `settings.v1` | `src/` (`GET/PUT /settings`, `data/settings.json`; the web saves it through the transcriber) | `transcriber/`, orchestrator, `src/` analysis and knowledge (read at job/session start) |

The platform contracts are specified in `docs/platform/CONTRACTS.md`.

Rules: fields are only added inside a version, and only as optional
fields (a document without them stays valid; the speaker-certainty
fields of `transcript.v1` were added that way, and so were the
speaker's `identification`, the utterance's `overlap_source`,
`language` and `language_differs`, and the session's `type` and
`goal`). Deploy order for an added field: update the reader (the RAG's
`ContractSession` forbids extra fields, so it rejects a document with a
field it does not know) before the writer (`transcriber/`) starts to
write it. Removing or renaming
one is a new version (`transcript.v2`) and both stay readable until
every consumer has moved. `examples/` holds one valid document per
contract, used by the contract tests of both sides.

Delivery of `transcript.v1`: `transcriber publish ID` writes it to a
directory, or with `publisher: "http"` posts it to the RAG's
`POST /transcripts` (bearer token; a re-post of a session replaces it).

## `transcript.v1`: the conversation's type and goal

Added compatibly on 2026-11-15 (#65), both optional and written only
when set. `session.type` is the kind of conversation, declared by
whoever records it, never inferred (`docs/skills/CONVERSATION.md`
§4.3): one of `reunion_decision`, `lluvia_ideas`,
`resolucion_problemas`, `negociacion`, `entrevista`, `feedback`,
`conflicto`, `coaching`, `venta`, `conversacion_libre`, `otro`.
`session.goal` is free text, at most 500 characters, saying what the
conversation was for. Both are context for the analysis, not a
measurement: the analyzer and the person report show them and do not
use them in a rating. They can be changed after the recording; the
next publish carries the new values. A document without them reads as
before (no type, no goal).

## `transcript.v1`: the speaker's role

Added compatibly on 2026-11-24 (#235), optional and written only when
known: `speakers[].role` is the role the live session gave the voice,
one of `interviewer`, `candidate`, `agent`, `customer`, `user`, `bot`
(the roles of `live.event.v1` without `unknown`). The live hand-off and
the AI interviewer write it; batch transcripts do not. The analyzer
keeps it (`sentences` step, `speakers[].role`) and reads only the
candidate against the job. **Deploy order:** the RAG's `ContractSpeaker`
forbids fields it does not know, so restart the RAG (reader) before the
transcriber and the orchestrator (writers).

## `live.event.v1`: the employee's voice check

Added compatibly on 2026-11-24 (#249) for two-party one-microphone
sessions, optional and written only there: the event type
`voice_check` with its object `voice_check` (`state`: `listening`,
`recognised`, `short`, `skipped`, `failed`; `speaker`, the employee's
id; `seconds` heard and `needed`), and the utterance's
`speaker_unsure` (written only when true). Every other session's
events are unchanged. Readers: the orchestrator's `LiveEvent` forbids
unknown fields and types, the live page reads them as plain objects.
Both appear only in a session started with the transcriber's new
`parties` field (`docs/live/LIVE_SESSION.md`), which only the new
orchestrator sends, and the old transcriber refuses it (422: its
request model forbids unknown fields). **Deploy order:** the
transcriber first, then the orchestrator. A new transcriber with the old
orchestrator works as before (no `parties`, so none of the new events);
the old transcriber with the new orchestrator would refuse every
one-microphone live session. With `ragputin-servicios restart` both
restart together, which is fine as long as no live session runs across
it.

## `knowledge.result.v1`: the times of a recording's chunk

Added compatibly on 2026-11-24 (#244), optional and written only when
set: `chunks[].metadata.start_s` and `end_s` are the seconds of the
part of a transcribed audio or video upload the chunk covers, so it can
be cited by time («min 03:12»). Every other chunk has neither. Readers:
the orchestrator and the web read chunks as plain objects, and the
orchestrator passes the metadata on as `reasoner.request.v1`'s free
`retrieved[].metadata`, so none refuses the new fields. **Deploy
order:** any reader that validates the schema strictly first, then the
RAG (writer); today that is only the contract tests.

The audio or video itself reaches the transcriber through its
`/internal/knowledge-media` routes, which answer with an ordinary
`transcript.v1` (no new contract; `docs/KNOWLEDGE.md` §8). Such a
transcript is never posted to `POST /transcripts`.

## `transcript.v1`: the diarizer and its licence

Added compatibly on 2026-11-15 (#68), all three optional and written
only when set: `pipeline.diarizer` is the model that separated the
voices (`builtin`, the transcriber's own grouping, or an id of
`lab/catalogue.json` such as `diarizen-wavlm-large-s80-md-v2`);
`pipeline.diarizer_licence` and `pipeline.diarizer_commercial` are the
licence of its weights and whether it allows commercial use, copied
from the catalogue (absent for `builtin`). DiariZen's weights are
CC BY-NC 4.0: `diarizer_commercial: false`. A document without them
reads as before. **Deploy order:** the RAG's `ContractPipeline` forbids
fields it does not know, so the RAG (reader) must run the version that
knows these fields before the transcriber (writer) starts to write
them; restart the RAG first.

## `transcript.v1`: rules the schema cannot express

JSON Schema checks each field on its own. Both readers
(`transcriber.domain.models.Transcript.from_contract` and
`src.domain.skills.transcript_contract.read_contract`) also refuse a
document that breaks any of these:

- speaker ids are unique, and every utterance's `speaker`,
  `previous_speaker` and `alternative_speaker` is one of them;
- a speaker's `identification` is set only when it is `enrolled`
  (an anonymous voice has no way of identification);
- an utterance's `alternative_speaker`, when set, is someone other than
  its `speaker`;
- an utterance's `id` is `t<turn>.s<index>`, and it does not end before
  it starts;
- turns come in order `0, 1, 2…`, sentences of a turn in order
  `0, 1, 2…`, all sentences of a turn share its speaker and its
  `previous_speaker`/`previous_utterance`, and every turn but the first
  points at the last sentence of the turn before and at that turn's
  speaker.

Not enforced by either reader (a producer should not write them):
`overlap_source` with `overlap` not `true`, and `language_differs`
without `language`.

**A name given through the expected-speakers floor.** A speaker with
`identification: "expected_floor"` was named only because the session
listed them as present, below the usual threshold: a person who spoke
without being listed is given a listed name this way. The contract says
it once, on the speaker; every utterance of that speaker inherits it.
Readers derive it per sentence instead of copying it: the player's
`sentenceNamedByFloor` (it marks those sentences and shows them among
the doubtful ones) and, in `src`, `ContractTranscript.named_through_floor`,
`Turn.named_through_floor()` and `Sentence.named_through_floor()`, so
an analysis can down-weight or leave out those attributions.
