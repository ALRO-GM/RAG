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
