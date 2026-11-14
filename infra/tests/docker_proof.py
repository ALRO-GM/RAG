"""Local proof of the server's web container, with docker (stdlib only).

Runs ``infra/Caddyfile`` in the same image as the server
(``caddy:2.10-alpine``), with the limits of ``infra/compose.yml``, as the
current user, a temporary ``/srv`` and a temporary run directory, behind
a second Caddy that plays the Peer4U proxy (it overwrites
``X-Forwarded-For`` as that one does). It checks, printing one line per
check:

1. ``caddy validate`` accepts the Caddyfile and ``docker compose config``
   accepts the compose file;
2. the static page and ``/healthz`` are served;
3. ``/api/health`` is a small JSON 502 while there is no socket;
4. with a tiny HTTP server on a unix socket (mode 0600) in the run
   directory: ``/api/health`` is proxied, ``X-Client-IP`` is the caller
   and cannot be spoofed, the forwarding headers are not passed on, a
   5 MB upload is accepted, a ranged GET comes back as 206, and a body
   over the limit is refused with 413.

Usage: ``python infra/tests/docker_proof.py`` (exit status 0 when every
check passes). Nothing is contacted but local containers.
"""

import http.client
import http.server
import json
import os
import shutil
import socket
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

IMAGE = "caddy:2.10-alpine"
INFRA = Path(__file__).resolve().parents[1]
UPLOAD = 5 * 1000 * 1000
LIMIT = 260 * 1024 * 1024
AUDIO = bytes(range(256)) * 400
WAIT = 30.0

PUBLIC_CLIENT = "198.51.100.23"
"""A public (documentation) address the outer proxy says it saw."""

OUTER = """{
\tauto_https off
\tadmin off
}
:80 {
\t@public header X-Proof-Public 1
\thandle @public {
\t\treverse_proxy ragputin-web:80 {
\t\t\theader_up X-Forwarded-For %s
\t\t}
\t}
\thandle {
\t\treverse_proxy ragputin-web:80
\t}
}
""" % PUBLIC_CLIENT
"""The Peer4U proxy as seen from here: it replaces X-Forwarded-For with
the address of its caller. On this workstation that caller is the docker
gateway, a private address; requests marked ``X-Proof-Public`` play a
caller with a public address instead."""


def docker(*args: str, check: bool = True) -> str:
    """Run a docker command; return its output."""
    done = subprocess.run(
        ["docker", *args], capture_output=True, text=True, check=False,
    )
    if check and done.returncode != 0:
        raise RuntimeError(
            f"docker {' '.join(args[:3])} failed: {done.stderr[-2000:]}"
        )
    return done.stdout + done.stderr


class UnixHttpServer(
    socketserver.ThreadingMixIn, socketserver.UnixStreamServer
):
    """An HTTP server on a unix socket (the workstation end of the tunnel)."""

    daemon_threads = True


class Backend(http.server.BaseHTTPRequestHandler):
    """Stands in for the transcriber API: echoes what reached it."""

    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:
        """Stay quiet."""

    def address_string(self) -> str:
        """A unix socket peer has no address."""
        return "unix"

    def _json(self, status: int, value: object) -> None:
        body = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        """Health, header echo and a ranged audio file."""
        if self.path == "/api/health":
            self._json(200, {"status": "ok"})
        elif self.path == "/api/echo":
            self._json(200, {k.lower(): v for k, v in self.headers.items()})
        elif self.path == "/api/audio":
            wanted = self.headers.get("Range", "")
            first, _, last = wanted.removeprefix("bytes=").partition("-")
            start, end = int(first), int(last)
            part = AUDIO[start:end + 1]
            self.send_response(206)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header(
                "Content-Range", f"bytes {start}-{end}/{len(AUDIO)}"
            )
            self.send_header("Content-Length", str(len(part)))
            self.end_headers()
            self.wfile.write(part)
        else:
            self._json(404, {"code": "not_found"})

    def do_POST(self) -> None:
        """Count the bytes of an upload."""
        length = int(self.headers.get("Content-Length", "0"))
        received = 0
        while received < length:
            data = self.rfile.read(min(1 << 20, length - received))
            if not data:
                break
            received += len(data)
        self._json(201, {"received": received})


def free_port() -> int:
    """Return a local port nothing listens on."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def request(
    port: int,
    method: str,
    path: str,
    headers: Optional[Dict[str, str]] = None,
    body: Optional[object] = None,
) -> Tuple[int, Dict[str, str], bytes]:
    """Make one request to the outer proxy; return status, headers, body."""
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
    try:
        connection.request(method, path, body=body,  # type: ignore[arg-type]
                           headers=headers or {})
        answer = connection.getresponse()
        data = answer.read()
        found = {k.lower(): v for k, v in answer.getheaders()}
        return answer.status, found, data
    except (ConnectionError, http.client.HTTPException) as exc:
        return 0, {"error": repr(exc)}, b""
    finally:
        connection.close()


def zeros(total: int, block: int = 1 << 20) -> Iterator[bytes]:
    """Yield ``total`` zero bytes in blocks."""
    sent = 0
    while sent < total:
        size = min(block, total - sent)
        sent += size
        yield bytes(size)


class Proof:
    """The containers, the temporary directories and the checks."""

    def __init__(self) -> None:
        """Name everything after a random tag."""
        tag = uuid.uuid4().hex[:8]
        self.network = f"ragputin-proof-{tag}"
        self.inner = f"ragputin-proof-web-{tag}"
        self.outer = f"ragputin-proof-outer-{tag}"
        self.work = Path(tempfile.mkdtemp(prefix="ragputin-proof-"))
        self.srv = self.work / "srv"
        self.run = self.work / "run"
        self.port = free_port()
        self.lines: List[str] = []
        self.failed = False
        self.server: Optional[UnixHttpServer] = None

    def check(self, what: str, ok: bool, detail: str = "") -> None:
        """Record and print one check."""
        line = f"{'ok  ' if ok else 'FAIL'} {what}" + (
            f": {detail}" if detail else ""
        )
        print(line, flush=True)
        self.lines.append(line)
        self.failed = self.failed or not ok

    def prepare(self) -> None:
        """Make /srv, the run directory and the outer proxy's config."""
        self.srv.mkdir()
        (self.srv / "index.html").write_text(
            "<!doctype html><title>proof</title><p>pages</p>\n"
        )
        (self.srv / "version.json").write_text('{"commit": "proof"}\n')
        self.run.mkdir(mode=0o750)
        os.chmod(self.run, 0o750)
        (self.work / "outer.caddy").write_text(OUTER)

    def validate(self) -> None:
        """caddy validate and docker compose config."""
        out = docker(
            "run", "--rm", "-v",
            f"{INFRA / 'Caddyfile'}:/etc/caddy/Caddyfile:ro", IMAGE,
            "caddy", "validate", "--config", "/etc/caddy/Caddyfile",
            "--adapter", "caddyfile", check=False,
        )
        self.check("caddy validate infra/Caddyfile",
                   "Valid configuration" in out,
                   out.strip().splitlines()[-1][-60:])
        if "Docker Compose" not in docker("compose", "version", check=False):
            print("skip docker compose config: the compose plugin is not"
                  " installed (the container below mirrors compose.yml)")
            return
        out = docker("compose", "-f", str(INFRA / "compose.yml"), "config",
                     "--quiet", check=False)
        self.check("docker compose config infra/compose.yml", out == "",
                   out.strip()[-300:])

    def start(self) -> None:
        """Start the web container as the current user, and the outer one."""
        user = f"{os.getuid()}:{os.getgid()}"
        docker("network", "create", self.network)
        docker(
            "run", "-d", "--name", self.inner, "--network", self.network,
            "--network-alias", "ragputin-web", "--user", user,
            "--read-only", "--tmpfs", f"/tmp:uid={os.getuid()}",
            "--tmpfs", f"/config:uid={os.getuid()}",
            "--tmpfs", f"/data:uid={os.getuid()}",
            "--cap-drop", "ALL", "--cap-add", "NET_BIND_SERVICE",
            "--security-opt", "no-new-privileges:true",
            "--sysctl", "net.ipv4.ip_unprivileged_port_start=0",
            "--memory", "128m", "--cpus", "0.5", "--pids-limit", "64",
            "-v", f"{INFRA / 'Caddyfile'}:/etc/caddy/Caddyfile:ro",
            "-v", f"{self.srv}:/srv:ro",
            "-v", f"{self.run}:/run/tunnel:ro",
            IMAGE,
        )
        docker(
            "run", "-d", "--name", self.outer, "--network", self.network,
            "-p", f"127.0.0.1:{self.port}:80",
            "-v", f"{self.work / 'outer.caddy'}:/etc/caddy/Caddyfile:ro",
            IMAGE,
        )
        deadline = time.monotonic() + WAIT
        while time.monotonic() < deadline:
            if request(self.port, "GET", "/healthz")[0] == 200:
                return
            time.sleep(0.2)
        raise RuntimeError("the containers did not answer: " + docker(
            "logs", self.inner, check=False)[-2000:])

    def serve_socket(self) -> None:
        """Start the stand-in API on run/app.sock, mode 0600."""
        path = self.run / "app.sock"
        self.server = UnixHttpServer(str(path), Backend)
        os.chmod(path, 0o600)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def checks_without_socket(self) -> None:
        """Pages, health and the JSON 502."""
        status, headers, body = request(self.port, "GET", "/")
        self.check("static page served", status == 200 and b"pages" in body,
                   f"{status} cache-control={headers.get('cache-control')}")
        csp = headers.get("content-security-policy", "")
        self.check("security headers on the page",
                   "default-src 'self'" in csp
                   and headers.get("referrer-policy") == "no-referrer"
                   and "microphone=(self)"
                   in headers.get("permissions-policy", ""),
                   headers.get("permissions-policy", ""))
        status, _, body = request(self.port, "GET", "/healthz")
        self.check("/healthz answered by Caddy", status == 200,
                   body.decode())
        status, headers, body = request(self.port, "GET", "/api/health")
        good = (
            status == 502
            and headers.get("content-type", "").startswith("application/json")
            and json.loads(body or b"{}").get("code")
            == "workstation_unavailable"
        )
        self.check("/api/health without the socket is a JSON 502", good,
                   f"{status} {body.decode(errors='replace')}")

    def checks_with_socket(self) -> None:
        """Proxying through the 0600 socket."""
        status, _, body = request(self.port, "GET", "/api/health")
        self.check("/api/health proxied through the unix socket",
                   status == 200 and json.loads(body) == {"status": "ok"},
                   f"{status} {body.decode(errors='replace')}")
        status, _, body = request(self.port, "GET", "/api/echo", {
            "X-Client-IP": "203.0.113.66",
            "X-Forwarded-For": "203.0.113.77",
            "Forwarded": "for=203.0.113.88",
        })
        seen = json.loads(body) if status == 200 else {}
        client = seen.get("x-client-ip", "")
        self.check(
            "X-Client-IP ignores what the caller sends",
            bool(client) and not client.startswith("203.0.113."),
            f"X-Client-IP={client!r}",
        )
        status, _, body = request(self.port, "GET", "/api/echo", {
            "X-Proof-Public": "1", "X-Client-IP": "203.0.113.66",
            "X-Forwarded-For": "203.0.113.77",
        })
        public = json.loads(body).get("x-client-ip") if status == 200 else None
        self.check(
            "X-Client-IP is the address the outer proxy saw",
            public == PUBLIC_CLIENT, f"X-Client-IP={public!r}",
        )
        leaked = [h for h in ("x-forwarded-for", "forwarded") if h in seen]
        self.check("forwarding headers are not passed to the API",
                   not leaked, f"passed on: {leaked}")
        status, _, body = request(
            self.port, "POST", "/api/upload",
            {"Content-Length": str(UPLOAD)}, zeros(UPLOAD),
        )
        self.check("a 5 MB upload is accepted",
                   status == 201 and json.loads(body)["received"] == UPLOAD,
                   f"{status} {body.decode(errors='replace')}")
        status, headers, body = request(
            self.port, "GET", "/api/audio", {"Range": "bytes=100-1099"}
        )
        self.check("a ranged GET passes through as 206",
                   status == 206 and body == AUDIO[100:1100]
                   and headers.get("content-range")
                   == f"bytes 100-1099/{len(AUDIO)}",
                   f"{status} {headers.get('content-range')}")
        over = LIMIT + 1024 * 1024
        status, _, body = request(
            self.port, "POST", "/api/upload",
            {"Content-Length": str(over)}, zeros(over),
        )
        self.check(f"a body over the limit ({over} bytes) is refused with"
                   " a JSON 413", status == 413
                   and b'"payload_too_large"' in body,
                   f"{status} {body[:120].decode(errors='replace')}")

    def clean(self) -> None:
        """Remove the containers, the network and the directories."""
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
        for name in (self.outer, self.inner):
            docker("rm", "-f", name, check=False)
        docker("network", "rm", self.network, check=False)
        shutil.rmtree(self.work, ignore_errors=True)


def main(argv: Optional[List[str]] = None,
         out: Callable[[str], None] = print) -> int:
    """Run every check; return 0 when they all pass."""
    proof = Proof()
    try:
        proof.prepare()
        proof.validate()
        proof.start()
        proof.checks_without_socket()
        proof.serve_socket()
        proof.checks_with_socket()
    except Exception as exc:  # report, clean up, fail
        proof.check("proof ran to the end", False, str(exc))
    finally:
        proof.clean()
    out("ALL CHECKS PASSED" if not proof.failed else "SOME CHECKS FAILED")
    return 1 if proof.failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
