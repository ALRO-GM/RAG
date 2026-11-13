#!/usr/bin/env bash
# Install infra/ragputin.caddy into the Peer4U Caddy, safely.
# Usage: infra/install-proxy.sh <ssh-target>   (e.g. ubuntu@host)
# Validates the complete Peer4U configuration before reloading and puts
# the previous block back if validation fails: a broken block would stop
# Caddy from starting and take Peer4U down.
set -euo pipefail
target="${1:?usage: $0 <ssh-target>}"
here="$(cd "$(dirname "$0")" && pwd)"
scp -q "$here/ragputin.caddy" "$target:/tmp/ragputin.caddy"
ssh "$target" 'bash -s' <<'REMOTE'
set -euo pipefail
T=/opt/caddy-sitios/ragputin.caddy
C="docker exec sanatorio-caddy-1 caddy"
ARGS="--config /etc/caddy/Caddyfile --adapter caddyfile"
if [ -f "$T" ]; then cp "$T" /tmp/ragputin.caddy.bak; else rm -f /tmp/ragputin.caddy.bak; fi
cp /tmp/ragputin.caddy "$T"
if $C validate $ARGS >/dev/null 2>&1; then
  $C reload $ARGS
  echo "Peer4U Caddy reloaded with the RAGPutin block."
else
  if [ -f /tmp/ragputin.caddy.bak ]; then cp /tmp/ragputin.caddy.bak "$T"; else rm -f "$T"; fi
  echo "Validation FAILED: previous state restored, nothing reloaded." >&2
  $C validate $ARGS 2>&1 | tail -5 >&2
  exit 1
fi
REMOTE
