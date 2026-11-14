#!/bin/sh
# Forced command of the workstation's tunnel key (infra/authorized_keys.tunnel.example),
# installed on the server as /opt/ragputin/infra/tunnel-clean.sh.
#
# `transcriber serve` first runs `ssh <server> true` with that key: sshd
# runs this script instead of `true`. It removes the socket of the
# reverse forward if a previous connection left it behind, because sshd
# (StreamLocalBindUnlink no) refuses to bind over an existing socket.
# A socket something still listens on (a live forward) is kept. The
# forward itself (`ssh -N -R ...`) opens no session, so this script does
# not run for it.
#
# Messages go to the error output: the workstation shows them as the
# reason of the clean-up. Override the path with RAGPUTIN_SOCKET (tests).
set -eu

SOCKET="${RAGPUTIN_SOCKET:-/opt/ragputin/run/app.sock}"

if [ ! -e "$SOCKET" ]; then
	echo "tunnel-clean: no socket to clean" >&2
	exit 0
fi
if [ ! -S "$SOCKET" ]; then
	echo "tunnel-clean: $SOCKET is not a socket; left alone" >&2
	exit 1
fi
# `ss -xlnH` lists the listening unix sockets, one per line, with the
# path as its own field.
if ss -xlnH 2>/dev/null | grep -qF " $SOCKET "; then
	echo "tunnel-clean: socket in use (a forward is open); kept" >&2
	exit 0
fi
rm -f -- "$SOCKET"
echo "tunnel-clean: stale socket removed" >&2
