# Contracts

Data exchanged between the services of this repository. A contract is
the only thing two services share: neither imports the other's code.

| File | Producer | Consumers |
|---|---|---|
| `transcript.v1.schema.json` | `transcriber/` | `src/` (analyzer, RAG) |

Rules: fields are only added inside a version; removing or renaming
one is a new version (`transcript.v2`) and both stay readable until
every consumer has moved. `examples/` holds one valid document per
contract, used by the contract tests of both sides.
