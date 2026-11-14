# Deployment: the transcriber on rag.peer4u.app

https://rag.peer4u.app is the front door of the transcriber
(`docs/transcriber/ARCHITECTURE.md` §5): people open the pages there on
their phones to enrol their voice, record a meeting and play its
transcript. The server only serves static files and passes `/api/*`
along; the API, the models and every stored byte live on the
workstation.

The results site that used to be published at this address was retired
on 2026-11-14. The benchmark and experiment documents remain in the
repository (`docs/benchmarks/`, `experiments/`).

## What runs where

```
 phone ──HTTPS──► Peer4U Caddy (sanatorio-caddy-1, TLS, HSTS)
                    │  /opt/caddy-sitios/ragputin.caddy: reverse_proxy ragputin-web:80
                    ▼
                 ragputin-web (caddy:2.10-alpine, uid 1001, 128 MB, read-only)
                    ├─ /          static pages from /srv = /opt/ragputin/pruebas (CI writes them)
                    ├─ /healthz   answered by Caddy itself
                    └─ /api/*  ──► unix socket /run/tunnel/app.sock = /opt/ragputin/run/app.sock
                                     ▲  ssh -N -R (opened by the workstation)
 workstation: python -m transcriber serve
              API on 127.0.0.1:8770, GPU, data/transcriber/ (audio, voiceprints, transcripts)
```

| Piece | Where | Who changes it |
|---|---|---|
| Pages (`transcriber/adapters/inbound/web/`) + `version.json` | `/opt/ragputin/pruebas` | CI, on every merge to `main` |
| `infra/Caddyfile`, `infra/compose.yml` | `/opt/ragputin/infra` | `make infra-up`, by hand |
| `infra/tunnel-clean.sh` | `/opt/ragputin/infra` | `make infra-up`, by hand |
| Tunnel socket | `/opt/ragputin/run/app.sock` | sshd, while the workstation's tunnel is open |
| `infra/ragputin.caddy` | `/opt/caddy-sitios/ragputin.caddy` | `make infra-proxy`, by hand |
| API, models, data | workstation | `make transcriber-serve` |

What the inner Caddy does for `/api/*`: it sets `X-Client-IP` to the
caller's address (computed from the `X-Forwarded-For` the Peer4U Caddy
writes, trusting only private addresses, and overwriting whatever the
caller sent) and drops the forwarding headers; the API throttles wrong
access codes per `X-Client-IP` and never reads `X-Forwarded-For`. Bodies
are limited to 260 MiB (the API's largest is a 256 MiB upload), answers
are streamed (`flush_interval -1`), an enrolment may wait 15 minutes for
the GPU. With the tunnel down, `/api/*` answers
`502 {"code":"workstation_unavailable"}` and the pages say so; a body
over the limit gets `413 {"code":"payload_too_large"}`. Headers:
`Permissions-Policy: microphone=(self), camera=(), geolocation=()`, a
strict `Content-Security-Policy` (`'self'` only, no inline code),
`Referrer-Policy: no-referrer`, `X-Frame-Options: DENY`,
`X-Content-Type-Options: nosniff`; HSTS is the Peer4U Caddy's. The app's
files are served with `Cache-Control: no-cache` (they have no hashed
names, so a deploy never mixes old and new modules). No access log is
kept, and the access code never travels in a URL.

The container runs as uid/gid 1001 (`ubuntu`), because sshd creates the
socket with mode 0600 for the user that logged in
(`StreamLocalBindMask 0177`). The run directory is mounted read-only:
connecting to a socket needs write permission on the socket, which a
read-only mount does not take away. Port 80 needs no root: docker opens
the low ports inside the container's network namespace
(`net.ipv4.ip_unprivileged_port_start=0`, also set explicitly). Both
were checked with `infra/tests/docker_proof.py` (below).

## First-time setup

1. **Tunnel key** (workstation): `make tunnel-key` makes
   `~/.ssh/ragputin_tunnel` if it is missing and prints a line like

   ```
   restrict,port-forwarding,command="/opt/ragputin/infra/tunnel-clean.sh" ssh-ed25519 AAAA… ragputin-tunnel
   ```

   (see `infra/authorized_keys.tunnel.example`). Add it by hand to
   `~/.ssh/authorized_keys` of `ubuntu` on the server. The key can open
   no shell and forward nothing but what the workstation asks for.
   - The forced command runs for the clean-up connection the
     workstation makes first (`ssh <server> true`): it removes
     `/opt/ragputin/run/app.sock` if nothing listens on it. sshd has
     `StreamLocalBindUnlink no`, so a socket left by a dropped
     connection would otherwise block the next forward.
   - The forward itself (`ssh -N -R /opt/ragputin/run/app.sock:
     127.0.0.1:8770`) opens no session (`-N`), so the forced command does
     not run for it: the key only forwards.
2. **Container**: `make infra-up` creates `/opt/ragputin/{infra,pruebas}`
   and `/opt/ragputin/run` (mode 750, owner `ubuntu`), uploads
   `Caddyfile`, `compose.yml` and `tunnel-clean.sh` (mode 755), and runs
   `docker compose up -d`.
3. **Proxy block**: `make infra-proxy` (unchanged) installs
   `infra/ragputin.caddy` in the Peer4U Caddy, validates the whole
   Peer4U configuration and reloads, or puts the previous block back.
4. **Pages**: merge to `main` (or run the deploy job again).

`DEPLOY_TARGET` (default `ubuntu@rag.peer4u.app`) selects the server.
Server access details are kept out of this public repository.

## How a merge deploys the pages

1. CI (`.github/workflows/ci.yml`) runs `Lint`, `Tests` (with node 22
   for the pages' tests) and `Build site`. Branch protection requires
   these three names.
2. `Build site` runs `python -m transcriber.adapters.inbound.bundle --out
   dist/site --commit <short sha>`: the pages without their
   `package.json`, plus `version.json` (`{"commit", "built_at"}`).
3. On `main`, `Deploy site` rsyncs the bundle with `--delete` through the
   deploy key, which the server locks to `rrsync /opt/ragputin/pruebas`,
   and reads `https://rag.peer4u.app/version.json` until it shows the
   commit.

`make site` builds the same bundle locally. GitHub configuration of the
deploy job (environment `production`):

| Name | Kind | Value |
|---|---|---|
| `DEPLOY_SSH_KEY` | secret | private half of the deploy key |
| `DEPLOY_KNOWN_HOSTS` | secret | the server's SSH host keys (pinned) |
| `DEPLOY_HOST` | variable | server address |

On the server, `~/.ssh/authorized_keys` holds the deploy key as
`command="rrsync /opt/ragputin/pruebas",restrict ssh-ed25519 … ragputin-deploy`.

## Starting the workstation side

```sh
make transcriber-serve TORCH=cuda SERVE_ARGS='--keep-code'
```

It starts the API on 127.0.0.1, opens and supervises the tunnel
(clean-up, forward, reconnection with back-off) and prints the join
link `https://rag.peer4u.app/#k=<code>` with a QR code. `--keep-code`
keeps the code across restarts; `SERVE_ARGS='--no-tunnel --dev-pages'`
serves everything locally for development.

## Checking each hop

| Hop | Check | Expected |
|---|---|---|
| Pages | `curl -s https://rag.peer4u.app/version.json` | the deployed commit |
| Inner Caddy | `curl -s https://rag.peer4u.app/healthz` | `{"status":"ok"}` |
| Container | `docker compose ps` in `/opt/ragputin/infra` | `healthy` |
| Tunnel | the `tunnel` line printed by `transcriber serve`; on the server `ls -l /opt/ragputin/run` | `connected`; `srw------- ubuntu app.sock` |
| API through everything | `curl -s https://rag.peer4u.app/api/health` | `{"status":"ok"}` (502 `workstation_unavailable` when the workstation is off) |
| Access code | open the join link | the sessions list |

Locally, `python infra/tests/docker_proof.py` (also run by the tests
when docker is available) proves the Caddyfile in the server's image:
`caddy validate`, static page, `/healthz`, JSON 502 without the socket,
then through a 0600 socket: `/api/health`, `X-Client-IP` not spoofable,
a 5 MB upload, a ranged 206, a JSON 413 over the limit.

## Rolling back

- **Pages**: revert the commit on `main`; CI deploys the previous
  bundle. In a hurry, re-run the deploy job of an earlier successful
  run.
- **Container**: `git checkout <good commit> -- infra/` and
  `make infra-up`; the previous `Caddyfile`/`compose.yml` are back.
- **Proxy block**: `infra/install-proxy.sh` already restores the
  previous block when validation fails; to go back by choice, check out
  the earlier `infra/ragputin.caddy` and `make infra-proxy`.
- **Tunnel**: stop `transcriber serve`; the API is unreachable at once
  (`502`). To revoke the tunnel key, delete its line from
  `authorized_keys`.

## Rules on the shared server

The server is Peer4U's production machine.

- Touch only `/opt/ragputin` and `/opt/caddy-sitios/ragputin.caddy`.
  Never touch `/opt/sanatorio`, its containers or its Caddyfile.
- `caddy validate` before every reload of the Peer4U Caddy
  (`make infra-proxy` does it and rolls back on failure).
- Every container has memory, CPU and process caps, drops all
  capabilities, runs read-only and without new privileges.
- Images must exist for linux/arm64 (the server is ARM).
- CI never changes the proxy or the container; it only writes the
  pages, through a key locked to `/opt/ragputin/pruebas`.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `/api/*` gives 502 `workstation_unavailable` | `transcriber serve` not running, or the tunnel is `reconnecting` (see its line) |
| Tunnel keeps failing with `remote port forwarding failed` | A stale socket and the clean-up did not run: check the key's `command=` and that `tunnel-clean.sh` is executable |
| `/api/*` gives 502 although the tunnel is connected | The container cannot open the socket: it must run as the socket's owner (1001) |
| Deploy job: `Permission denied (publickey)` | Secret or `authorized_keys` line missing or different |
| Deploy job: host key verification failed | Server host key changed: refresh `DEPLOY_KNOWN_HOSTS` |
| Smoke test keeps the old commit | The upload went elsewhere: `cat /opt/ragputin/pruebas/version.json` |
| `502` on every page | `ragputin-web` down or off the `sanatorio_default` network: `docker compose ps` in `/opt/ragputin/infra` |
