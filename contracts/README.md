# Contracts

Data exchanged between the services of this repository. A contract is
the only thing two services share: neither imports the other's code.

| File | Producer | Consumers |
|---|---|---|
| `transcript.v1.schema.json` | `transcriber/` | `src/` (analyzer, RAG) |

Rules: fields are only added inside a version, and only as optional
fields (a document without them stays valid; the speaker-certainty
fields of `transcript.v1` were added that way, and so were the
speaker's `identification` and the utterance's `overlap_source`,
`language` and `language_differs`); removing or renaming
one is a new version (`transcript.v2`) and both stay readable until
every consumer has moved. `examples/` holds one valid document per
contract, used by the contract tests of both sides.

Delivery of `transcript.v1`: `transcriber publish ID` writes it to a
directory, or with `publisher: "http"` posts it to the RAG's
`POST /transcripts` (bearer token; a re-post of a session replaces it).

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
