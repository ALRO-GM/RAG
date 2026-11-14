"""Local proof of the server's two containers, with docker (stdlib only).

Builds the shape of ``infra/compose.yml`` out of local containers:

- an *outer* Caddy that plays the Peer4U proxy (``infra/ragputin.caddy``
  with its ``encode``; it overwrites ``X-Forwarded-For`` as that one
  does), on an ordinary network that stands for ``sanatorio_default``;
- ``web``: ``infra/Caddyfile`` in the pinned Caddy image with the limits
  of ``compose.yml``, on that network (alias ``ragputin-web``) and on an
  internal ``tunnel`` network;
- ``tunnel``: the pinned chisel server, with the flags of
  ``compose.yml`` and a users file owned by 65534 with mode 600, only on
  the internal network;
- a chisel client from the same image on the host network, which
  connects *through* the outer proxy and ``web`` to ``/_tunnel`` and
  opens ``R:0.0.0.0:9000`` back to a tiny HTTP server on this machine
  (the stand-in for the transcriber API).

It prints one line per check: ``caddy validate`` and ``docker compose
config``; static page, headers and ``/healthz``; ``/api/health`` a JSON
502 with no client; ``/_tunnel`` without a WebSocket upgrade a 404;
no credential, a wrong one, another reverse port, a forward and SOCKS
all refused (and ``/api`` still 502); with the client: proxied 200,
``X-Client-IP`` right and not spoofable, forwarding headers dropped, a
5 MB upload, a ranged GET 206, a JSON 413 over the limit; a container on
the outer network cannot reach chisel or port 9000; the tunnel
container's memory during a 200 MB upload stays under its limit; client
killed → 502; client started again → 200.

Usage: ``python infra/tests/docker_proof.py`` (exit status 0 when every
check passes). Nothing is contacted but local containers.
"""

import http.client
import http.server
import json
import os
import re
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

INFRA = Path(__file__).resolve().parents[1]
UPLOAD = 5 * 1000 * 1000
BIG_UPLOAD = 200 * 1000 * 1000
LIMIT = 260 * 1024 * 1024
AUDIO = bytes(range(256)) * 400
WAIT = 30.0
TUNNEL_MEMORY = 64 * 1024 * 1024
COMPOSE_CLI = "docker:29.1.3-cli"
"""Image with ``docker compose``, used when the plugin is not installed."""

USER = "workstation"
PASSWORD = "proof-" + uuid.uuid4().hex
REMOTE = "R:0.0.0.0:9000"
ALLOWED = r"^R:0\.0\.0\.0:9000$"

PUBLIC_CLIENT = "198.51.100.23"
"""A public (documentation) address the outer proxy says it saw."""

OUTER = """{
\tauto_https off
\tadmin off
}
:80 {
\tencode zstd gzip
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
"""The Peer4U proxy as seen from here (``infra/ragputin.caddy`` on port
80): it replaces X-Forwarded-For with the address of its caller. On this
workstation that caller is the docker gateway, a private address;
requests marked ``X-Proof-Public`` play a caller with a public address
instead."""


def image(service: str) -> str:
    """Return the pinned image of ``service`` in ``compose.yml``."""
    text = (INFRA / "compose.yml").read_text(encoding="utf-8")
    block = text.split(f"\n  {service}:\n", 1)[1]
    found = re.search(r"^    image: (\S+)$", block, re.MULTILINE)
    if found is None:
        raise RuntimeError(f"no image for {service} in compose.yml")
    return found.group(1)


def server_flags() -> List[str]:
    """Return the chisel server's command in ``compose.yml``."""
    text = (INFRA / "compose.yml").read_text(encoding="utf-8")
    block = text.split("\n  tunnel:\n", 1)[1].split("\n    command:\n", 1)[1]
    flags: List[str] = []
    for line in block.splitlines():
        if not line.startswith("      - "):
            break
        flags.append(line[len("      - "):])
    return flags


def run_docker(*args: str, timeout: float = 300) -> Tuple[int, str]:
    """Run a docker command; return its exit code and output."""
    done = subprocess.run(
        ["docker", *args], capture_output=True, text=True, check=False,
        timeout=timeout,
    )
    return done.returncode, done.stdout + done.stderr


def docker(*args: str, check: bool = True, timeout: float = 300) -> str:
    """Run a docker command; return its output (raise on failure)."""
    code, out = run_docker(*args, timeout=timeout)
    if check and code != 0:
        raise RuntimeError(
            f"docker {' '.join(args[:3])} failed: {out[-2000:]}"
        )
    return out


class Backend(http.server.BaseHTTPRequestHandler):
    """Stands in for the transcriber API: echoes what reached it."""

    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:
        """Stay quiet."""

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


class BackendServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    """The stand-in API on 127.0.0.1 (the workstation's end)."""

    daemon_threads = True


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
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
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
        self.outer_net = f"ragputin-proof-outer-{tag}"
        self.tunnel_net = f"ragputin-proof-tunnel-{tag}"
        self.web = f"ragputin-proof-web-{tag}"
        self.tunnel = f"ragputin-proof-tunnel-{tag}"
        self.outer = f"ragputin-proof-outer-{tag}"
        self.client = f"ragputin-proof-client-{tag}"
        self.work = Path(tempfile.mkdtemp(prefix="ragputin-proof-"))
        self.srv = self.work / "srv"
        self.secrets = self.work / "secrets"
        self.port = free_port()
        self.backend_port = free_port()
        self.web_image = image("web")
        self.chisel_image = image("tunnel")
        self.lines: List[str] = []
        self.failed = False
        self.server: Optional[BackendServer] = None

    def check(self, what: str, ok: bool, detail: str = "") -> None:
        """Record and print one check."""
        line = f"{'ok  ' if ok else 'FAIL'} {what}" + (
            f": {detail}" if detail else ""
        )
        print(line, flush=True)
        self.lines.append(line)
        self.failed = self.failed or not ok

    def prepare(self) -> None:
        """Make /srv, the users file (65534, 600) and the outer config."""
        self.srv.mkdir()
        os.chmod(self.srv, 0o755)
        (self.srv / "index.html").write_text(
            "<!doctype html><title>proof</title><p>pages</p>\n"
        )
        (self.srv / "version.json").write_text('{"commit": "proof"}\n')
        for name in ("index.html", "version.json"):
            os.chmod(self.srv / name, 0o644)
        self.secrets.mkdir(mode=0o700)
        draft = self.secrets / "users.json.new"
        draft.write_text(json.dumps({f"{USER}:{PASSWORD}": [ALLOWED]}))
        # As on the server (make infra-up): owner 65534, mode 600, in a
        # directory only its owner can enter.
        docker(
            "run", "--rm", "--network", "none", "--user", "0",
            "-v", f"{self.secrets}:/s", self.web_image,
            "sh", "-c", "install -o 65534 -g 65534 -m 600 /s/users.json.new"
            " /s/tunnel-users.json && rm /s/users.json.new",
        )
        (self.work / "outer.caddy").write_text(OUTER)
        os.chmod(self.work, 0o755)
        os.chmod(self.work / "outer.caddy", 0o644)

    def validate(self) -> None:
        """caddy validate (default and narrowed proxies), compose config."""
        for proxies in ("", "172.30.0.0/16 10.20.0.0/16"):
            env = ["-e", f"TRUSTED_PROXIES={proxies}"] if proxies else []
            out = docker(
                "run", "--rm", *env, "-v",
                f"{INFRA / 'Caddyfile'}:/etc/caddy/Caddyfile:ro",
                self.web_image, "caddy", "validate", "--config",
                "/etc/caddy/Caddyfile", "--adapter", "caddyfile",
                check=False,
            )
            self.check(
                "caddy validate infra/Caddyfile"
                + (f" (TRUSTED_PROXIES={proxies})" if proxies else ""),
                "Valid configuration" in out,
                "Valid configuration" if "Valid configuration" in out
                else out.strip()[-200:],
            )
        if "Docker Compose" in docker("compose", "version", check=False):
            command = ["compose", "-f", str(INFRA / "compose.yml"),
                       "config", "--quiet"]
        else:  # no compose plugin here: the docker CLI image has it
            docker("pull", "-q", COMPOSE_CLI)
            command = ["run", "--rm", "-v", f"{INFRA}:/opt/ragputin/infra:ro",
                       "-w", "/opt/ragputin/infra", COMPOSE_CLI,
                       "docker", "compose", "config", "--quiet"]
        code, out = run_docker(*command)
        self.check("docker compose config infra/compose.yml",
                   code == 0 and out.strip() == "", out.strip()[-300:])

    def start(self) -> None:
        """Start tunnel, web (as 65534) and the outer proxy."""
        docker("network", "create", self.outer_net)
        docker("network", "create", "--internal", self.tunnel_net)
        docker(
            "run", "-d", "--name", self.tunnel, "--network", self.tunnel_net,
            "--network-alias", "tunnel", "--user", "65534:65534",
            "--read-only", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges:true",
            "--memory", "64m", "--cpus", "0.25", "--pids-limit", "32",
            "-e", "GOMEMLIMIT=48MiB",
            "-v", f"{self.secrets / 'tunnel-users.json'}"
            ":/etc/chisel/users.json:ro",
            self.chisel_image, *server_flags(),
        )
        docker(
            "run", "-d", "--name", self.web, "--network", self.outer_net,
            "--network-alias", "ragputin-web", "--user", "65534:65534",
            "--read-only", "--tmpfs", "/tmp:uid=65534,gid=65534,mode=700",
            "--tmpfs", "/config:uid=65534,gid=65534,mode=700",
            "--tmpfs", "/data:uid=65534,gid=65534,mode=700",
            "--cap-drop", "ALL", "--cap-add", "NET_BIND_SERVICE",
            "--security-opt", "no-new-privileges:true",
            "--sysctl", "net.ipv4.ip_unprivileged_port_start=0",
            "--memory", "128m", "--cpus", "0.5", "--pids-limit", "64",
            "-v", f"{INFRA / 'Caddyfile'}:/etc/caddy/Caddyfile:ro",
            "-v", f"{self.srv}:/srv:ro",
            self.web_image,
        )
        docker("network", "connect", self.tunnel_net, self.web)
        docker(
            "run", "-d", "--name", self.outer, "--network", self.outer_net,
            "-p", f"127.0.0.1:{self.port}:80",
            "-v", f"{self.work / 'outer.caddy'}:/etc/caddy/Caddyfile:ro",
            self.web_image,
        )
        deadline = time.monotonic() + WAIT
        while time.monotonic() < deadline:
            if request(self.port, "GET", "/healthz")[0] == 200:
                return
            time.sleep(0.2)
        raise RuntimeError("the containers did not answer: " + docker(
            "logs", self.web, check=False)[-2000:])

    def serve_backend(self) -> None:
        """Start the stand-in API on 127.0.0.1."""
        self.server = BackendServer(("127.0.0.1", self.backend_port), Backend)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def run_client(
        self, *remotes: str, auth: Optional[str] = None,
        detach: bool = False, name: Optional[str] = None,
    ) -> str:
        """Run a chisel client on the host network through the proxies.

        Detached, it keeps running (``--max-retry-count 0``: it exits
        when the connection drops); otherwise its output is returned.
        """
        args = ["run", "--network", "host", "--user", "65534:65534",
                "--read-only", "--cap-drop", "ALL",
                "--security-opt", "no-new-privileges:true"]
        if auth is not None:
            args += ["-e", f"AUTH={auth}"]
        args += ["-d", "--name", name or self.client] if detach else ["--rm"]
        args += [self.chisel_image, "client", "--keepalive", "25s",
                 "--max-retry-count", "0",
                 f"http://127.0.0.1:{self.port}/_tunnel", *remotes]
        return docker(*args, check=False, timeout=60)

    def wait_api(self, status: int) -> Tuple[bool, str]:
        """Wait until ``/api/health`` answers ``status``."""
        deadline = time.monotonic() + WAIT
        found, body = 0, b""
        while time.monotonic() < deadline:
            found, _, body = request(self.port, "GET", "/api/health")
            if found == status:
                return True, f"{found} {body.decode(errors='replace')}"
            time.sleep(0.2)
        return False, f"{found} {body.decode(errors='replace')}"

    def checks_without_client(self) -> None:
        """Pages, health, the JSON 502 and /_tunnel for plain requests."""
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
        self.check("/api/health with no client connected is a JSON 502",
                   self._unavailable(status, headers, body),
                   f"{status} {body.decode(errors='replace')}")
        for path in ("/_tunnel", "/_tunnel/health", "/_tunnel/version"):
            status, _, body = request(self.port, "GET", path)
            self.check(f"plain GET {path} never reaches chisel (404)",
                       status == 404 and b"1.12" not in body,
                       f"{status} {body[:40]!r}")

    @staticmethod
    def _unavailable(
        status: int, headers: Dict[str, str], body: bytes
    ) -> bool:
        """Tell whether an answer is the JSON 502 of a missing tunnel."""
        return (
            status == 502
            and headers.get("content-type", "").startswith("application/json")
            and json.loads(body or b"{}").get("code")
            == "workstation_unavailable"
        )

    def checks_refused(self) -> None:
        """No credential, a wrong one and other remotes are refused."""
        target = f"{REMOTE}:127.0.0.1:{self.backend_port}"
        out = self.run_client(target)
        self.check("a client without a credential is refused",
                   "unable to authenticate" in out
                   and "Connected" not in out, self._last(out))
        out = self.run_client(target, auth=f"{USER}:wrong-password")
        self.check("a wrong password is refused",
                   "unable to authenticate" in out
                   and "Connected" not in out, self._last(out))
        out = self.run_client(target, auth=f"intruder:{PASSWORD}")
        self.check("another user with the right password is refused",
                   "unable to authenticate" in out, self._last(out))
        status, _, _ = request(self.port, "GET", "/api/health")
        self.check("/api still 502 after the refused clients",
                   status == 502, str(status))
        auth = f"{USER}:{PASSWORD}"
        for remote, what in (
            (f"R:0.0.0.0:9001:127.0.0.1:{self.backend_port}",
             "another reverse port (R:0.0.0.0:9001)"),
            (f"R:127.0.0.1:9000:127.0.0.1:{self.backend_port}",
             "the same port on another interface (R:127.0.0.1:9000)"),
            ("3000:ragputin-web:80", "a forward tunnel (3000:web:80)"),
            ("3001:127.0.0.1:8080", "a forward to chisel itself"),
            ("R:socks", "a reverse SOCKS proxy"),
            ("socks", "a SOCKS proxy"),
        ):
            out = self.run_client(remote, auth=auth)
            self.check(f"the user cannot open {what}",
                       "denied" in out and "Connected" not in out,
                       self._last(out))

    @staticmethod
    def _last(out: str) -> str:
        """Return the client's most telling line."""
        lines = [line for line in out.splitlines() if "error" in line
                 or "denied" in line or "Connected" in line]
        return (lines or out.splitlines() or [""])[-1][-110:]

    def connect(self) -> bool:
        """Start the right client; wait until /api answers through it."""
        self.run_client(
            f"{REMOTE}:127.0.0.1:{self.backend_port}",
            auth=f"{USER}:{PASSWORD}", detach=True,
        )
        ok, detail = self.wait_api(200)
        logs = docker("logs", self.client, check=False)
        self.check("the workstation's client connects through both proxies",
                   ok and "Connected" in logs, detail)
        return ok

    def checks_with_client(self) -> None:
        """Proxying through the tunnel."""
        status, _, body = request(self.port, "GET", "/api/health")
        self.check("/api/health proxied through the tunnel",
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

    def checks_isolation(self) -> None:
        """A neighbour on the outer network reaches neither chisel port."""
        address = docker(
            "inspect", "-f",
            "{{(index .NetworkSettings.Networks \"%s\").IPAddress}}"
            % self.tunnel_net, self.tunnel,
        ).strip()
        for target in ("tunnel:8080", "tunnel:9000",
                       f"{address}:8080", f"{address}:9000"):
            out = docker(
                "run", "--rm", "--network", self.outer_net, self.web_image,
                "sh", "-c", f"wget -q -T 3 -O - http://{target}/health"
                " >/dev/null 2>&1 && echo REACHED || echo UNREACHABLE",
                check=False, timeout=60,
            )
            self.check(f"a neighbour on the outer network cannot reach"
                       f" {target}", out.strip().endswith("UNREACHABLE"),
                       out.strip()[-40:])
        out = docker(
            "run", "--rm", "--network", self.tunnel_net, self.web_image,
            "wget", "-q", "-T", "3", "-O", "-", "http://tunnel:8080/health",
            check=False, timeout=60,
        )
        self.check("on the internal network chisel's /health answers",
                   out.strip() == "OK", out.strip()[-60:])

    def check_memory(self) -> None:
        """Upload 200 MB; the tunnel container stays under its limit."""
        identifier = docker("inspect", "-f", "{{.Id}}", self.tunnel).strip()
        peak_file = Path(
            f"/sys/fs/cgroup/system.slice/docker-{identifier}.scope"
            "/memory.peak"
        )
        samples: List[int] = []
        done = threading.Event()

        def sample() -> None:
            """Read the container's memory until the upload ends."""
            while not done.wait(0.5):
                out = docker("stats", "--no-stream", "--format",
                             "{{.MemUsage}}", self.tunnel, check=False)
                used = re.match(r"\s*([\d.]+)\s*([KMG]i?B)", out)
                if used:
                    scale = {"KiB": 1 << 10, "MiB": 1 << 20,
                             "GiB": 1 << 30, "KB": 1000, "MB": 1000 ** 2,
                             "GB": 1000 ** 3}[used.group(2)]
                    samples.append(int(float(used.group(1)) * scale))

        watcher = threading.Thread(target=sample, daemon=True)
        watcher.start()
        began = time.monotonic()
        status, _, body = request(
            self.port, "POST", "/api/upload",
            {"Content-Length": str(BIG_UPLOAD)}, zeros(BIG_UPLOAD),
        )
        seconds = time.monotonic() - began
        done.set()
        watcher.join(10)
        received = json.loads(body).get("received") if status == 201 else None
        self.check("a 200 MB upload goes through the tunnel",
                   received == BIG_UPLOAD,
                   f"{status} received={received} in {seconds:.1f} s")
        peak = int(peak_file.read_text()) if peak_file.exists() else max(
            samples or [0]
        )
        state = docker("inspect", "-f",
                       "{{.State.OOMKilled}} {{.RestartCount}}",
                       self.tunnel).strip()
        source = "cgroup memory.peak" if peak_file.exists() else "docker stats"
        self.check(
            "the tunnel container's memory stays under its 64 MiB limit",
            0 < peak < TUNNEL_MEMORY and state == "false 0",
            f"peak {peak / (1 << 20):.1f} MiB"
            f" ({source}"
            f"; {len(samples)} samples, max sampled"
            f" {max(samples or [0]) / (1 << 20):.1f} MiB); OOMKilled,"
            f" restarts: {state}",
        )

    def checks_reconnection(self) -> None:
        """Client killed → 502; started again → 200."""
        docker("rm", "-f", self.client, check=False)
        ok, detail = self.wait_api(502)
        self.check("client killed: /api/health is a 502 again", ok, detail)
        ok = self.connect()
        self.check("client started again: /api/health is 200 again", ok)

    def clean(self) -> None:
        """Remove the containers, the networks and the directories."""
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
        for name in (self.client, self.outer, self.web, self.tunnel):
            docker("rm", "-f", name, check=False)
        for network in (self.outer_net, self.tunnel_net):
            docker("network", "rm", network, check=False)
        shutil.rmtree(self.work, ignore_errors=True)


def main(argv: Optional[List[str]] = None,
         out: Callable[[str], None] = print) -> int:
    """Run every check; return 0 when they all pass."""
    proof = Proof()
    try:
        proof.prepare()
        proof.validate()
        proof.start()
        proof.serve_backend()
        proof.checks_without_client()
        proof.checks_refused()
        if proof.connect():
            proof.checks_with_client()
            proof.checks_isolation()
            proof.check_memory()
            proof.checks_reconnection()
    except Exception as exc:  # report, clean up, fail
        proof.check("proof ran to the end", False, str(exc))
    finally:
        proof.clean()
    out("ALL CHECKS PASSED" if not proof.failed else "SOME CHECKS FAILED")
    return 1 if proof.failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
