# Results site: architecture and deployment

https://rag.peer4u.app shows everything the project has measured: the
retrieval lab, the external benchmarks (BEIR, CoIR), the soft-skills
lab and rubrics, the hardware runs (CPU against GPU) and every
document of the repository.

## Design

```
 Workstation (RTX 3070 Ti)          GitHub                       Peer4U server (ARM)
 ─────────────────────────          ──────                       ───────────────────
 make … TORCH=cuda                  pull request                 sanatorio-caddy-1 (TLS)
   └─ experiments/*.jsonl  ──git──►   CI: lint, tests,             │ rag.peer4u.app
      docs/*.md                          coverage, site build      ▼
                                    merge to main               ragputin-web (caddy, 128 MB)
                                      CD: build ─ rsync ───────►  /opt/ragputin/pruebas/
                                          (deploy key, rrsync)     (static files only)
```

Decisions, and why:

| Decision | Reason |
|---|---|
| Compute on the workstation, publish results | The GPU is there (6.4× faster than the best CPU plan, see the Hardware page). The server is Peer4U's production machine on the free tier: nothing heavy may run next to it. |
| Results are committed (`experiments/`) | They are the evidence behind every default. Versioning them makes each number traceable to a commit, and CI can rebuild the site from a clean checkout. |
| Static site, no server code | Nothing to crash, patch or scale; a 128 MB container is enough; Peer4U is unaffected. |
| `results_site` builder in Python, tested | Summaries (deltas against baselines, averages, GPU speed-up) are computed and unit-tested in Python. The browser only draws them. |
| No front-end build step | Plain HTML/CSS/JS; the only third-party code (Markdown rendering and sanitizing) is loaded from jsDelivr at fixed versions and allowed by the CSP. |
| CD writes through a restricted key | The deploy key is locked on the server to `rrsync /opt/ragputin/pruebas`. A leaked key can only replace the static files. |
| The Peer4U proxy is never touched by CI | Its block changes rarely and a broken block would stop Peer4U. It is installed by hand with `make infra-proxy`, which validates the whole Peer4U config first and rolls back on failure. |

A live search demo (the HTTP API on the workstation GPU, exposed through
a tunnel) is possible later. It is deliberately left out for now: it
would publish an endpoint served from the workstation.

## Pieces

| Path | What |
|---|---|
| `results_site/` | Builder: `python -m results_site build --out dist/site` |
| `web/` | Front end (single page, hash routes) |
| `infra/Caddyfile`, `infra/compose.yml` | The RAGPutin web container on the server |
| `infra/ragputin.caddy`, `infra/install-proxy.sh` | Block in the Peer4U Caddy and its safe installer |
| `.github/workflows/ci.yml` | CI (lint, tests, site) and CD (deploy on `main`) |

Server access details (address, DNS zone) are kept out of this public
repository.

## Operations

```sh
make site-serve        # preview locally
make deploy-site       # manual upload (normally CI does it)
make infra-up          # upload infra/ and restart the web container
make infra-proxy       # (re)install the rag.peer4u.app block, validated
```

`DEPLOY_TARGET` (default `ubuntu@rag.peer4u.app`) selects the server.

GitHub configuration used by the deploy job (environment `production`):

| Name | Kind | Value |
|---|---|---|
| `DEPLOY_SSH_KEY` | secret | private half of the deploy key |
| `DEPLOY_KNOWN_HOSTS` | secret | the server's SSH host keys (pinned) |
| `DEPLOY_HOST` | variable | server address |

On the server, `~/.ssh/authorized_keys` holds the public half as
`command="rrsync /opt/ragputin/pruebas",restrict ssh-ed25519 … ragputin-deploy`.
To revoke it, delete that line and the GitHub secret.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| Deploy job: `Permission denied (publickey)` | Secret or `authorized_keys` line missing or different |
| Deploy job: host key verification failed | Server host key changed: refresh `DEPLOY_KNOWN_HOSTS` |
| Smoke test keeps the old commit | Browser/proxy cache, or the upload went elsewhere: `ls /opt/ragputin/pruebas/data` |
| `502` on the site | `ragputin-web` down or off the `sanatorio_default` network: `docker compose ps` in `/opt/ragputin/infra` |
| Page shows "HTTP 404" for a data file | The section has no results yet (e.g. no hardware run committed) |
