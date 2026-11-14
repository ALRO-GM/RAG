#!/usr/bin/env bash
# Upload infra/ and the tunnel's users file, validate, and (re)start the
# RAGPutin containers on the server.
# Usage: infra/install-infra.sh <ssh-target> <users-file>
#   (e.g. ubuntu@host ~/.config/ragputin/tunnel-users.json, written by
#   `make tunnel-secret`).
#
# Only /opt/ragputin is touched. The server side is
# infra/install-remote.sh: the new Caddyfile and compose.yml are
# validated in /opt/ragputin/infra.next (docker compose config, and
# caddy validate in the pinned image with the TRUSTED_PROXIES value
# compose resolves from the server's infra/.env) before they replace
# the running ones; then `web` is recreated (a changed Caddyfile alone
# is otherwise never read) and the script waits until web answers
# /healthz and reaches chisel before it reports success.
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
# The server side: validate, install, recreate web, wait until healthy.
ssh "$target" "sh -s -- $R '$image'" < "$here/install-remote.sh"
