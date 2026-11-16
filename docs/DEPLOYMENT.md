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
              │ POST/DELETE /transcripts, GET/PUT /analysis/judge,
              │ POST/GET /analysis/sessions/{id}[/steps|/reports] (loopback, bearer token)
              ▼
            python -m src serve (make rag-serve): RAG API on 127.0.0.1:8000,
              data/transcripts/; not behind the tunnel, nothing of it on the server
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
| RAG API and its transcripts (`data/transcripts/`) | workstation, 127.0.0.1:8000 | `make rag-serve` |
| RAG ingest token | `~/.config/ragputin/rag-token` (workstation, mode 600) | `make rag-token` |

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
9000. From that network the container reaches `web:80` and itself, and
not docker's default bridge, the host's own address or the internet
("network unreachable"); it does reach services of the host listening
on 0.0.0.0 through the internal network's gateway address (measured
with the docker proof on the workstation, Docker 29.1.3; see residual
risks). Its users file is mounted read-only; owner 65534 and mode 600 let
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
     SOCKS. chisel 1.12.0 applies the list twice: to the remotes a client
     declares when it connects (`server/server_handler.go`) and again to
     the destination of every channel it later asks for
     (`share/tunnel/tunnel_out_ssh.go`), so a modified client cannot
     declare the allowed remote and then ask for other channels. The
     docker proof shows it with a hand-made client (below): with this
     credential every channel to another address is refused. What the
     credential gives is the reverse port 9000 in the `tunnel` container,
     that is everything `web` sends to `/api/*`. It uploads nothing.
2. **Containers**: `make infra-up` (`infra/install-infra.sh`, which
   runs `infra/install-remote.sh` on the server) creates
   `/opt/ragputin/{infra,pruebas}` and `/opt/ragputin/secrets` (mode
   700), uploads `Caddyfile` and `compose.yml` to
   `/opt/ragputin/infra.next` and the users file next to its final place,
   validates (`docker compose config -q` with the server's `infra/.env`,
   and `caddy validate` in the pinned Caddy image with the
   `TRUSTED_PROXIES` value compose resolves from that `.env`, so the
   value `web` gets is the one validated), gives the users file to uid
   65534 with mode 600 (a one-shot container of the Caddy image, the
   only way for `ubuntu` to `chown`), moves the new files in place, runs
   `docker compose up -d`, recreates `web` (the Caddyfile is a
   bind-mounted file and Caddy runs with `admin off`: a changed
   Caddyfile alone is never read by the running Caddy, and `up -d` does
   not recreate a container whose definition did not change) and
   restarts `tunnel` (a replaced users file is a new inode). It reports
   success only once `web` answers `/healthz` and reaches chisel's
   `/health` (60 s at most each; otherwise it prints the logs and
   fails). It also
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
`data/transcriber/tunnel.pid` with its arguments and start time, is
ended first) and prints the join link
`https://rag.peer4u.app/#k=<code>` with a QR code. The `tunnel` line
says `connected` once chisel is connected and
`https://rag.peer4u.app/api/health` answers through it; that check is
repeated every 5 s, and after 3 failures in a row the line says
`reconnecting` and the client is started again. `--keep-code`
keeps the code across restarts; `SERVE_ARGS='--no-tunnel --dev-pages'`
serves everything locally for development. Without the client or the
credential, `serve` stops at once and says which `make` target to run.

### The RAG on the workstation

The RAG runs on the workstation too, never on the server (owner's
decision of 2026-11-14): the server keeps only the pages and the
tunnel. The transcriber hands transcripts to it over loopback.

```sh
make rag-token                     # once: ~/.config/ragputin/rag-token, mode 600
make rag-serve TORCH=cuda          # RAG API on 127.0.0.1:8000
TRANSCRIBER_LOCAL_RAG=1 make transcriber-serve TORCH=cuda SERVE_ARGS='--keep-code'
```

- `make rag-token` makes the ingest token if it is missing (43 random
  URL-safe characters, file mode 600, directory 700) and checks it if it
  exists; it never prints it. Both services read it from that file, so
  it never appears on a command line, in a log or in a setting.
- `make rag-serve` runs `python -m src serve --host 127.0.0.1 --port
  8000 --token_file ~/.config/ragputin/rag-token`. With `--token_file`
  `serve` refuses any host that is not a loopback address, and refuses a
  token file others may read. It needs the index (`make index`) like any
  `serve`. It mounts `POST /transcripts`, `GET /transcripts/{id}` and
  `DELETE /transcripts/{id}`, all behind the bearer token.
- `TRANSCRIBER_LOCAL_RAG=1` makes the transcriber publish to
  `http://127.0.0.1:8000` with the token from the same file, when that
  file exists and the config file (`TRANSCRIBER_CONFIG`) sets no
  `publisher`; otherwise the directory publisher stays the default and
  `transcriber serve` says why on its first lines. Set it for the
  terminal commands too (`publish`, `sessions delete`), since each reads
  its own settings. Another RAG address or token file is set
  explicitly: `"publisher": "http"`, `"publish_http": {"url": ...,
  "token_file": "/abs/path"}`.
- Retention: a transcript stays in `data/transcripts/` until its
  session is deleted in the transcriber; a new publish of the same
  session replaces it. Deleting a session (web or `sessions delete`)
  with the HTTP publisher active then sends `DELETE
  /transcripts/{id}` (5 s at most per try, `publish_http.retract_timeout`)
  while still holding the session's lock, which a publish also holds
  until the RAG answers, so a publish in flight cannot leave a copy
  behind. The RAG's `404` with `"code": "transcript_not_found"` (no
  copy) is fine; any other `404` (a RAG without its transcript routes,
  another service on the port) and every other failure (RAG stopped,
  wrong token) is logged with the session id and the reason, never the
  transcript, and does not undo the deletion; delete the file by hand
  then. While an analysis of the session runs from the terminal
  (`skills analyze`), the RAG answers `409` and keeps the transcript
  and everything derived from it (the run would write its files
  again); only that answer is asked again, twice, after the
  publisher's back-off (1 s, then 2 s by default; #121), which covers a
  run that is just ending. Still `409` after that, the transcriber logs
  the failure like the others: stop the run, then
  `DELETE /transcripts/{id}` again. The directory
  publisher's files are not touched by a deletion.
- The tunnel forwards only `127.0.0.1:8770` (the transcriber); the RAG
  listens on another port and on loopback only, so nothing reaches it
  from outside the workstation (`tests/integration/test_rag_workstation.py`
  checks the ports, the Makefile recipe and the bind guard).

Check that a publish reached the RAG, on the workstation:

```sh
printf 'Authorization: Bearer %s\n' "$(cat ~/.config/ragputin/rag-token)" |
  curl -s -H @- http://127.0.0.1:8000/transcripts/<session id> | head -c 300
ls data/transcripts/            # <session id>.transcript.v1.json
```

(`printf` is a shell built-in and curl reads the header from its
input, so the token is in no process's arguments.)

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
| RAG on the workstation | `curl -s http://127.0.0.1:8000/health` | `{"status":"ok",...}` |

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

It also checks what the credential and the container can reach. A
hand-made chisel client (`infra/tests/chisel_probe`, Go, built with a
pinned Go image) authenticates with the credential, declares the
allowed `R:0.0.0.0:9000` and then asks for channels to `web:80`, chisel
itself, a listener on the workstation bound to 0.0.0.0 (through the
internal network's gateway, docker's default bridge and the machine's
own address), `1.1.1.1:443`, `socks`, `R:0.0.0.0:9000/udp` and
`R:0.0.0.0:9000` written as an address: every one is refused (`access
denied`, `SOCKS5 is not enabled`, or for the last one a dial error,
since it passes the list but is no address). A control user of the
proof's own users file, allowed everything, opens channels to `web:80`
and chisel with the same client, which shows the refusals come from
the list. `nc` in the container's network namespace (and the control
user) reach `web:80` and chisel and not the default bridge, the
machine's own address or `1.1.1.1:443`; the listener through the
internal network's gateway is printed as a `note` (reached here).
Last, it runs `infra/install-remote.sh` as on the server (docker CLI
image, its own compose project): a first install with
`TRUSTED_PROXIES` in `infra/.env` (validated and given to `web` with
that value), a Caddyfile-only change (served after the install; without
the recreation of `web` this check failed), and an invalid Caddyfile and
an invalid `TRUSTED_PROXIES` (both fail before anything is replaced).

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
  recordings). Beyond that reverse port it opens nothing: chisel 1.12.0
  checks its list on every channel, and the docker proof's hand-made
  client was refused every other destination (above). It lives only in
  `~/.config/ragputin/tunnel-credential` (mode 600) and, as the users
  file, in `/opt/ragputin/secrets`. The password is 43 random characters;
  chisel does not rate-limit logins, so its strength is the protection.
- **chisel is third-party software exposed to the internet** behind the
  two Caddies (WebSocket upgrades on `/_tunnel` only). A flaw before
  authentication would be reachable. It is pinned by version and digest,
  runs without capabilities, read-only, as uid 65534, with a 64 MiB cap,
  on a network only `ragputin-web` shares. Updating it is a deliberate
  change of the pinned tag, digest and checksums. Code running in that
  container could reach `web:80` and, through the internal network's
  gateway address, any service of the host that listens on 0.0.0.0
  (measured on the workstation with Docker 29.1.3; the server's Docker
  and firewall were not checked). The credential does not give this
  (above). Not changed: the network option
  `com.docker.network.bridge.gateway_mode_ipv4: isolated` removed the
  gateway address in a manual test on the workstation (the listener was
  then unreachable), but whether the server's Docker accepts it was not
  checked.
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
