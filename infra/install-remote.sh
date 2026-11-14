#!/bin/sh
# The server side of infra/install-infra.sh, streamed to the server over
# ssh (POSIX sh: the docker proof also runs it locally, in the docker
# CLI image, which has no bash).
# Usage: install-remote.sh <root> <caddy image>
#   <root> is /opt/ragputin on the server. Before it runs, <root>/infra.next
#   holds the new Caddyfile and compose.yml and
#   <root>/secrets/tunnel-users.json.new the users file.
#
# 1. Validate in <root>/infra.next, before anything running changes:
#    `docker compose config` (with the server's infra/.env, if there is
#    one), and `caddy validate` in the pinned image with the
#    TRUSTED_PROXIES value compose resolves from that .env (or its
#    default): the value web will really get is the one validated.
# 2. Give the users file to uid 65534 with mode 600; move the files in.
# 3. `docker compose up -d`, then recreate `web`: the Caddyfile is a
#    bind-mounted file and Caddy runs with `admin off`, so a changed
#    Caddyfile alone is never read by the running Caddy and `up -d` does
#    not recreate a container whose compose definition did not change.
#    Restart `tunnel` (a replaced users file is a new inode).
# 4. Wait until web answers /healthz and reaches chisel's /health (the
#    commands of web's healthcheck and docs/DEPLOYMENT.md); only then
#    print success. Otherwise print web's last log lines and fail.
set -eu
R="${1:?usage: install-remote.sh <root> <caddy image>}"
IMAGE="${2:?usage: install-remote.sh <root> <caddy image>}"
WAIT=60

cd "$R/infra.next"
if [ -f "$R/infra/.env" ]; then cp "$R/infra/.env" .env; fi
docker compose -f compose.yml config -q
proxies="$(docker compose -f compose.yml config --format json \
  | sed -n 's/^ *"TRUSTED_PROXIES": "\(.*\)",\{0,1\}$/\1/p' | head -n 1)"
if [ -z "$proxies" ]; then
  echo "Cannot read TRUSTED_PROXIES from docker compose config." >&2
  exit 1
fi
echo "caddy validate with TRUSTED_PROXIES=$proxies"
docker run --rm --network none -e "TRUSTED_PROXIES=$proxies" \
  -v "$R/infra.next/Caddyfile:/etc/caddy/Caddyfile:ro" \
  "$IMAGE" caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile
docker run --rm --network none --user 0 -v "$R/secrets:/s" "$IMAGE" \
  install -o 65534 -g 65534 -m 600 /s/tunnel-users.json.new /s/tunnel-users.json
rm -f "$R/secrets/tunnel-users.json.new"
cp Caddyfile compose.yml "$R/infra/"
cd "$R/infra"
docker compose up -d --remove-orphans
# A changed Caddyfile alone is only read by a new Caddy: recreate web.
docker compose up -d --no-deps --force-recreate web
# A replaced users file is a new inode: restart so the bind mount sees it.
docker compose restart tunnel

waited=0
until docker compose exec -T web wget -q -O /dev/null \
    http://127.0.0.1/healthz 2>/dev/null; do
  waited=$((waited + 1))
  if [ "$waited" -ge "$WAIT" ]; then
    echo "web does not answer /healthz after ${WAIT} s:" >&2
    docker compose ps >&2 || true
    docker compose logs --tail 40 web >&2 || true
    exit 1
  fi
  sleep 1
done
waited=0
until [ "$(docker compose exec -T web wget -q -O - http://tunnel:8080/health \
    2>/dev/null)" = "OK" ]; do
  waited=$((waited + 1))
  if [ "$waited" -ge "$WAIT" ]; then
    echo "web does not reach chisel (tunnel:8080/health) after ${WAIT} s:" >&2
    docker compose ps >&2 || true
    docker compose logs --tail 40 tunnel >&2 || true
    exit 1
  fi
  sleep 1
done
docker compose ps
rm -rf "$R/infra.next"
# The retired SSH tunnel.
rm -f "$R/infra/tunnel-clean.sh"
rm -rf "$R/run"
echo "RAGPutin containers up (web answers /healthz and reaches chisel)."
