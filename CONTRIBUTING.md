# Contributing

How changes get into `main`, and the transcriber's pages onto
https://rag.peer4u.app.

## Branches

- `main` is protected: it only changes through pull requests whose CI
  checks pass (Lint, Tests, Build site). Every merge to `main` deploys
  the transcriber's pages (the results site was retired on 2026-11-14).
- Work happens on short-lived branches cut from `main`:

  | Prefix | For |
  |---|---|
  | `feat/` | a new capability (command, adapter, page) |
  | `fix/` | a bug fix |
  | `exp/` | new experiment results (lab runs, benchmarks) |
  | `docs/` | documentation only |
  | `chore/`, `ci/`, `refactor/`, `test/` | tooling, pipelines, structure, tests |

- Merge with **squash**, so `main` reads as one commit per change;
  the branch is deleted after the merge.

## Commit messages

[Conventional Commits](https://www.conventionalcommits.org/):
`<type>(<scope>): <summary>` in the imperative, for example

```
feat(transcriber): show the job's note on finished sessions
exp(lab): rerun the rerank suite on the GPU
fix(cli): keep explicit options over a profile
```

Types: `feat`, `fix`, `exp`, `docs`, `test`, `refactor`, `perf`, `ci`,
`chore`. The scope is the area: `cli`, `lab`, `skills`, `transcriber`,
`infra`, `bench`, …

## Before opening a pull request

```sh
make ci          # flake8 + mypy --strict, tests with coverage gate, page bundle
```

- New code comes with unit tests (domain and application, using the
  fakes in `tests/fakes.py`) and integration tests (real adapters in
  temporary directories). The coverage gate is in `pyproject.toml`
  (`[tool.coverage.report] fail_under`) and only goes up.
- Tests that load real models are marked `slow` and run with
  `make test-slow` on the workstation; CI runs offline and must not
  download models.

## Experiments: computed on the workstation, committed

The experiments need the workstation's GPU, so CI never runs them:

1. `git switch -c exp/<what>`
2. Run them with the CUDA build:
   `make <target> TORCH=cuda` (for example `make system-benchmark
   TORCH=cuda`, `make experiment SUITE=rerank TORCH=cuda`).
3. Commit the result files they write under `experiments/` (and update
   `docs/TASKS.md` / the reports if the conclusions change).
4. Open the pull request. The results stay in the repository as
   Markdown and JSON; nothing is published for them.

See `docs/DEPLOYMENT.md` for how the pages are deployed.
