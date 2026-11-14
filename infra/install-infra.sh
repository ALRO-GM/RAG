#!/usr/bin/env bash
# Upload infra/ and the tunnel's users file, validate, and (re)start the
# RAGPutin containers on the server.
# Usage: infra/install-infra.sh <ssh-target> <users-file>
#   (e.g. ubuntu@host ~/.config/ragputin/tunnel-users.json, written by
#   `make tunnel-secret`).
#
# Only /opt/ragputin is touched. The new Caddyfile and compose.yml are
# validated in /opt/ragputin/infra.next (caddy validate in the pinned
# image, docker compose config) before they replace the running ones.
# The users file goes to /opt/ragputin/secrets (directory 700, owner
# ubuntu); a one-shot container of the pinned Caddy image gives it to
# uid 65534 (the chisel container's user) with mode 600. The pieces of
# the retired SSH tunnel (/opt/ragputin/run, infra/tunnel-clean.sh) are
# removed. The tunnel key's line in ~/.ssh/authorized_keys is NOT
# touched: remove it by hand (docs/DEPLOYMENT.md).
set -euo pipefail
target="${1:?usage: $0 <ssh-target> <users-file>}"
users="${2:?usage: $0 <ssh-target> <users-file>}"
here="$(cd "$(dirname "$0")" && pwd)"
R=/opt/ragputin

if [ ! -f "$users" ]; then
  echo "No users file at $users: run \`make tunnel-secret\` first." >&2
  exit 1
fi
image="$(sed -n 's/^    image: \(caddy:[^ ]*\)$/\1/p' "$here/compose.yml" | head -n 1)"
if [ -z "$image" ]; then
  echo "No Caddy image in infra/compose.yml." >&2
  exit 1
fi

ssh "$target" "mkdir -p $R/infra $R/infra.next $R/pruebas && install -d -m 700 $R/secrets"
rsync -z --chmod=F644 "$here/Caddyfile" "$here/compose.yml" "$target:$R/infra.next/"
rsync -z --chmod=F600 "$users" "$target:$R/secrets/tunnel-users.json.new"
ssh "$target" "IMAGE='$image' bash -s" <<'REMOTE'
set -euo pipefail
R=/opt/ragputin
cd "$R/infra.next"
if [ -f "$R/infra/.env" ]; then cp "$R/infra/.env" .env; fi
docker run --rm --network none -v "$R/infra.next/Caddyfile:/etc/caddy/Caddyfile:ro" \
  "$IMAGE" caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile
docker compose -f compose.yml config -q
docker run --rm --network none --user 0 -v "$R/secrets:/s" "$IMAGE" \
  install -o 65534 -g 65534 -m 600 /s/tunnel-users.json.new /s/tunnel-users.json
rm -f "$R/secrets/tunnel-users.json.new"
cp Caddyfile compose.yml "$R/infra/"
cd "$R/infra"
docker compose up -d --remove-orphans
# A replaced users file is a new inode: restart so the bind mount sees it.
docker compose restart tunnel
docker compose ps
rm -rf "$R/infra.next"
# The retired SSH tunnel.
rm -f "$R/infra/tunnel-clean.sh"
rm -rf "$R/run"
echo "RAGPutin containers up."
REMOTE
