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
                    ▼                                   (network sanatorio_default)
                 ragputin-web (caddy:2.10-alpine, uid 65534, 128 MB, read-only)
                    ├─ /          static pages from /srv = /opt/ragputin/pruebas (CI writes them)
                    ├─ /healthz   answered by Caddy itself
                    ├─ /_tunnel   WebSocket upgrades only ──► tunnel:8080 ─┐
                    └─ /api/*  ──────────────────────────────► tunnel:9000 │
                                                     (network ragputin_tunnel, internal)
                 ragputin-tunnel (jpillora/chisel:1.12.0, uid 65534, 64 MB, read-only)
                    chisel server: one user, allowed only R:0.0.0.0:9000
                                     ▲
                                     │ wss://rag.peer4u.app/_tunnel (opened by the workstation)
 workstation: python -m transcriber serve
              chisel client: R:0.0.0.0:9000 → 127.0.0.1:8770
              API on 127.0.0.1:8770, GPU, data/transcriber/ (audio, voiceprints, transcripts)
```

| Piece | Where | Who changes it |
|---|---|---|
| Pages (`transcriber/adapters/inbound/web/`) + `version.json` | `/opt/ragputin/pruebas` | CI, on every merge to `main` |
| `infra/Caddyfile`, `infra/compose.yml` | `/opt/ragputin/infra` | `make infra-up`, by hand |
| `TRUSTED_PROXIES` | `/opt/ragputin/infra/.env` | by hand, once |
| Tunnel users file (credential, allowed remote) | `/opt/ragputin/secrets/tunnel-users.json` (dir 700 `ubuntu`, file 600 uid 65534) | `make tunnel-secret` then `make infra-up` |
| `infra/ragputin.caddy` | `/opt/caddy-sitios/ragputin.caddy` | `make infra-proxy`, by hand |
| chisel client | `~/.local/share/ragputin/bin/chisel` (workstation) | `make tunnel-client` |
| Tunnel credential | `~/.config/ragputin/tunnel-credential` (workstation, mode 600) | `make tunnel-secret` |
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

`/_tunnel` reaches the chisel server only as a WebSocket upgrade on that
exact path; any other request there gets the static 404, so chisel's own
`/health` and `/version` are not reachable from outside. A body limit and
the API's timeouts do not apply to the tunnel's WebSocket, which lives
as long as the workstation is connected (chisel's keepalive every 25 s
keeps it busy through both proxies).

`trusted_proxies` comes from `TRUSTED_PROXIES` (default
`private_ranges`). On the server, put the subnet of `sanatorio_default`
in `/opt/ragputin/infra/.env`:

```sh
docker network inspect sanatorio_default -f '{{(index .IPAM.Config 0).Subnet}}'
echo 'TRUSTED_PROXIES=<that subnet>' > /opt/ragputin/infra/.env
```

This only narrows trust to that network: any container on
`sanatorio_default` (Peer4U's own, not ours) can still reach
`ragputin-web` directly and choose the `X-Client-IP` the API sees. The
impact is low: the header only selects which client wrong codes are
counted against, and a right code is never refused.

Both containers run as uid/gid 65534 with no capabilities (the web one
keeps `NET_BIND_SERVICE` for a root run of the image), read-only, without
new privileges, with memory, CPU and process caps and no published
port. Port 80 needs no root: docker opens the low ports inside the
container's network namespace (`net.ipv4.ip_unprivileged_port_start=0`,
also set explicitly). The pages are world-readable (CI uploads them with
modes 755/644). The chisel container is only on the internal network
`ragputin_tunnel`, which it shares with `ragputin-web` alone: no other
container of the host reaches chisel's port 8080 or the tunnel's port
9000. Its users file is mounted read-only; owner 65534 and mode 600 let
chisel, and nobody else in the container, read it. The chisel image is a
scratch image (the binary and CA roots), so it has no healthcheck of its
own; `docker compose exec web wget -qO- http://tunnel:8080/health`
answers `OK` when it runs. All of this, and the limits, were checked with
`infra/tests/docker_proof.py` (below).

## First-time setup

1. **Tunnel client and credential** (workstation):
   - `make tunnel-client` downloads the pinned chisel release for this
     machine from GitHub to `~/.local/share/ragputin/bin/chisel` and
     checks its SHA-256 against `infra/chisel-1.12.0.sha256` (copied
     from the release's checksums file); a mismatch installs nothing.
   - `make tunnel-secret` makes `~/.config/ragputin/tunnel-credential`
     (`workstation:<43 random characters>`, mode 600) if it is missing,
     and writes the matching server users file to
     `~/.config/ragputin/tunnel-users.json` (mode 600):

     ```json
     { "workstation:<password>": ["^R:0\\.0\\.0\\.0:9000$"] }
     ```

     chisel matches the regex against `R:<interface>:<port>` for a
     reverse remote, `<host>:<port>` for a forward one and `socks` for
     SOCKS, so this user can open the reverse port 9000 and nothing else.
     It uploads nothing.
2. **Containers**: `make infra-up` (`infra/install-infra.sh`) creates
   `/opt/ragputin/{infra,pruebas}` and `/opt/ragputin/secrets` (mode
   700), uploads `Caddyfile` and `compose.yml` to
   `/opt/ragputin/infra.next` and the users file next to its final place,
   validates (`caddy validate` in the pinned Caddy image, `docker compose
   config -q`), gives the users file to uid 65534 with mode 600 (a
   one-shot container of the Caddy image, the only way for `ubuntu` to
   `chown`), moves the new files in place, runs `docker compose up -d`
   and restarts `tunnel` (a replaced users file is a new inode). It also
   removes what the retired SSH tunnel left: `/opt/ragputin/run` and
   `infra/tunnel-clean.sh`. The tunnel key's line in `~/.ssh/authorized_keys`
   of `ubuntu` (`restrict,port-forwarding,command="…/tunnel-clean.sh" …
   ragputin-tunnel`) is removed by hand, and `~/.ssh/ragputin_tunnel*`
   on the workstation deleted.
3. **Proxy block**: `make infra-proxy` (unchanged) installs
   `infra/ragputin.caddy` in the Peer4U Caddy, validates the whole
   Peer4U configuration and reloads, or puts the previous block back.
   Its plain `reverse_proxy` already passes WebSocket upgrades.
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

It starts the API on 127.0.0.1, starts and supervises the chisel client
(restart with back-off; a client left by a crashed run, found through
`data/transcriber/tunnel.pid`, is ended first) and prints the join link
`https://rag.peer4u.app/#k=<code>` with a QR code. The `tunnel` line
says `connected` once chisel is connected and
`https://rag.peer4u.app/api/health` answers through it. `--keep-code`
keeps the code across restarts; `SERVE_ARGS='--no-tunnel --dev-pages'`
serves everything locally for development. Without the client or the
credential, `serve` stops at once and says which `make` target to run.

## Checking each hop

| Hop | Check | Expected |
|---|---|---|
| Pages | `curl -s https://rag.peer4u.app/version.json` | the deployed commit |
| Inner Caddy | `curl -s https://rag.peer4u.app/healthz` | `{"status":"ok"}` |
| Container | `docker compose ps` in `/opt/ragputin/infra` | `healthy` |
| Tunnel | the `tunnel` line printed by `transcriber serve`; on the server `docker compose logs tunnel` in `/opt/ragputin/infra` | `connected`; `session#N: Open (user=workstation … remotes=R:0.0.0.0:9000:127.0.0.1:8770)` |
| chisel server | `docker compose exec web wget -qO- http://tunnel:8080/health` | `OK` |
| API through everything | `curl -s https://rag.peer4u.app/api/health` | `{"status":"ok"}` (502 `workstation_unavailable` when the workstation is off) |
| Access code | open the join link | the sessions list |

Locally, `python infra/tests/docker_proof.py` (also run by the tests
when docker is available) builds the shape of `compose.yml` from local
containers of the pinned images, on two networks: an outer Caddy that
plays the Peer4U proxy, `web` and `tunnel`, and a chisel client on the
host network that connects through the outer Caddy and `web` to
`/_tunnel` and exposes a tiny local HTTP server. It checks `caddy
validate` (default and narrowed `TRUSTED_PROXIES`) and `docker compose
config`; the static page, its headers and `/healthz`; `/api/health` a
JSON 502 with no client; plain GETs of `/_tunnel`, `/_tunnel/health`
and `/_tunnel/version` a 404 that never reaches chisel; no credential,
a wrong password and another user refused (`/api` still 502); the user
refused for `R:0.0.0.0:9001`, `R:127.0.0.1:9000`, a forward to `web:80`,
a forward to chisel itself, `R:socks` and `socks`; with the client:
`/api/health` 200, `X-Client-IP` the address the outer proxy saw and not
spoofable, forwarding headers dropped, a 5 MB upload, a ranged 206, a
JSON 413 over the limit; a container on the outer network cannot reach
`tunnel:8080`, `tunnel:9000` or the container's address on either port;
during a 200 MB upload the tunnel container's memory peak (cgroup
`memory.peak`) stays under its 64 MiB limit; client killed → 502;
client started again → 200.

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
  (`502`). To revoke the credential, delete
  `~/.config/ragputin/tunnel-credential`, run `make tunnel-secret` (a new
  one) and `make infra-up`; chisel reloads its users file for new
  connections, and the restart ends the old one.
- **Back to the SSH tunnel**: not possible without reverting this change
  (`git revert`); the SSH key was retired on purpose (below).

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

## Residual risks

Moving the tunnel from SSH to HTTPS (2026-11-14, after a security
review) removed the worst case: the SSH tunnel key logged in as
`ubuntu`, who controls docker on the shared host, and OpenSSH cannot
restrict a key to one unix-socket forward, so a stolen key meant root on
Peer4U's production machine. What remains:

- **The tunnel credential** lets its holder serve `/api/*` for the
  domain while the workstation is not connected, that is impersonate the
  API to the phones (and so collect what they send: access codes,
  recordings). It gives nothing on the host. It lives only in
  `~/.config/ragputin/tunnel-credential` (mode 600) and, as the users
  file, in `/opt/ragputin/secrets`. The password is 43 random characters;
  chisel does not rate-limit logins, so its strength is the protection.
- **chisel is third-party software exposed to the internet** behind the
  two Caddies (WebSocket upgrades on `/_tunnel` only). A flaw before
  authentication would be reachable. It is pinned by version and digest,
  runs without capabilities, read-only, as uid 65534, with a 64 MiB cap,
  on a network only `ragputin-web` shares. Updating it is a deliberate
  change of the pinned tag, digest and checksums.
- **No server key pinning**: the client does not pass `--fingerprint`
  (the chisel server makes a new key at every start). The client trusts
  the TLS certificate of rag.peer4u.app; something able to intercept
  traffic inside the host's docker networks could pose as the chisel
  server, but it could equally read every `/api` request there.
- **Audio crosses the server in transit**: TLS ends at the Peer4U Caddy;
  between the containers it is plain HTTP; from the chisel server to the
  workstation it is inside chisel's SSH channel. Nothing is stored on the
  server.
- **`X-Client-IP` from a neighbour**: see `TRUSTED_PROXIES` above.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `/api/*` gives 502 `workstation_unavailable` | `transcriber serve` not running, or the tunnel is `reconnecting` (see its line) |
| `tunnel  reconnecting (… unable to authenticate)` | The credential and the server's users file differ: `make tunnel-secret` (keeps the credential, rewrites the users file) then `make infra-up` |
| `tunnel  reconnecting (… access to 'R:…' denied)` | `tunnel.remote_port` differs from the port in the users file |
| `tunnel  connecting (chisel connected; …/api/health answered 502)` | chisel is connected but `/api` does not reach it: `ragputin-web` and `ragputin-tunnel` must share the `ragputin_tunnel` network (`docker compose ps`, `docker network inspect ragputin_tunnel`) |
| `tunnel  reconnecting (… 404 …)` or `bad handshake` | `/_tunnel` does not reach chisel: the Caddyfile on the server is old (`make infra-up`) |
| The tunnel container restarts in a loop | Its users file is missing or unreadable: `ls -ln /opt/ragputin/secrets` should show `65534 65534 … tunnel-users.json` with mode 600 |
| Deploy job: `Permission denied (publickey)` | Secret or `authorized_keys` line missing or different |
| Deploy job: host key verification failed | Server host key changed: refresh `DEPLOY_KNOWN_HOSTS` |
| Smoke test keeps the old commit | The upload went elsewhere: `cat /opt/ragputin/pruebas/version.json` |
| `502` on every page | `ragputin-web` down or off the `sanatorio_default` network: `docker compose ps` in `/opt/ragputin/infra` |
