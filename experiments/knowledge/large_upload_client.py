"""Play the page: a large file in pieces through both services (#282).

Sends ``FILE`` through the transcriber's ``/api/knowledge/…`` in the
pieces the server announces, with a connection cut in the middle of the
5th piece (resumed from what the server holds), a piece at a gap on the
9th (``409``) and the 11th piece sent twice on the 12th (``200``, not
written again); completes it, polls the document until it is indexed
or failed, and samples the RSS of both service processes every 50 ms.
Prints one JSON document (the figures of docs/KNOWLEDGE.md §4.2).

How the run of §4.2 was made (CPU only, own ports; never the running
services or the GPU; generated files in a temp folder, deleted after):

1. ``large_upload_make_pdf.py big.pdf 107 2000`` and
   ``large_upload_make_video.py big.mp4 1000000000``.
2. A folder ``WORK`` with ``token`` and ``code`` (random, mode 600).
3. Start both services with ``systemd-run --user --unit=… --collect
   -p MemoryMax=10G --setenv=CUDA_VISIBLE_DEVICES=
   --setenv=HF_HUB_OFFLINE=1 --setenv=PYTHONPATH=<repo>``:
   ``large_upload_rag.py WORK 8790 8791 on`` (working directory
   ``WORK``) and ``large_upload_transcriber.py WORK 8791 8790``.
4. ``large_upload_client.py WORK 8791 RAG_PID TR_PID big.pdf big.pdf``
   (``systemctl --user show -p MainPID --value <unit>`` gives the pids),
   also under ``systemd-run``; then the same with ``big.mp4 clase.mp4``
   on fresh services and a fresh ``WORK``.
5. Stop the units; delete ``WORK`` and the generated files.

    python experiments/knowledge/large_upload_client.py \\
        WORK TR_PORT RAG_PID TR_PID FILE NAME
"""

import http.client
import json
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

MiB = 1048576


def rss(pid: int) -> int:
    """Return the resident memory of process ``pid`` in bytes."""
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return 0


class Sampler(threading.Thread):
    """The peak RSS of each process, per phase of the run."""

    def __init__(self, pids: Dict[str, int]) -> None:
        """Watch ``pids`` (name → pid)."""
        super().__init__(daemon=True)
        self.pids = pids
        self.phase = "idle"
        self.peaks: Dict[Tuple[str, str], int] = {}
        self.stop = False

    def run(self) -> None:
        """Sample every 50 ms until stopped."""
        while not self.stop:
            for who, pid in self.pids.items():
                key = (self.phase, who)
                self.peaks[key] = max(self.peaks.get(key, 0), rss(pid))
            time.sleep(0.05)


class Web:
    """The transcriber's ``/api`` with the access code."""

    def __init__(self, port: int, code: str) -> None:
        """Talk to ``port`` with ``code``."""
        self.port = port
        self.code = code

    def call(
        self, method: str, url: str, body: Optional[bytes] = None,
        kind: Optional[str] = None,
    ) -> Tuple[int, Any]:
        """Send one request; return the status and the JSON answer."""
        headers = {"X-Access-Code": self.code}
        if kind is not None:
            headers["Content-Type"] = kind
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=600)
        conn.request(method, "/api" + url, body=body, headers=headers)
        answer = conn.getresponse()
        raw = answer.read()
        conn.close()
        return answer.status, json.loads(raw) if raw.strip() else None

    def cut(self, url: str, data: bytes) -> None:
        """Send half a piece, then drop the connection."""
        sock = socket.create_connection(("127.0.0.1", self.port))
        sock.sendall((
            f"PUT /api{url} HTTP/1.1\r\nHost: x\r\n"
            f"X-Access-Code: {self.code}\r\n"
            f"Content-Type: application/octet-stream\r\n"
            f"Content-Length: {len(data)}\r\n\r\n"
        ).encode() + data[:len(data) // 2])
        sock.close()


def send(web: Web, path: Path, name: str, events: List[str]) -> Any:
    """Send the file in pieces (with the cut, gap and duplicate); return
    the document and the number of pieces."""
    size = path.stat().st_size
    status, upload = web.call(
        "POST", "/knowledge/domains/big/uploads",
        json.dumps({"name": name, "size": size,
                    "metadata": {"tags": ["282"]}}).encode(),
        "application/json",
    )
    assert status == 201, (status, upload)
    base = f"/knowledge/domains/big/uploads/{upload['id']}"
    piece, offset, n = upload["piece_bytes"], 0, 0
    octets = "application/octet-stream"
    with open(path, "rb") as handle:
        while offset < size:
            handle.seek(offset)
            data = handle.read(piece)
            n += 1
            if n == 5:
                web.cut(f"{base}/pieces/{offset}", data)
                time.sleep(0.5)
                _, view = web.call("GET", base)
                events.append(f"cut mid-piece at {offset}: server holds "
                              f"{view['received']}")
                offset = view["received"]
                continue
            if n == 9:
                status, _ = web.call(
                    "PUT", f"{base}/pieces/{offset + piece}", data, octets
                )
                events.append(f"gap: {status}")
            if n == 12:
                status, view = web.call(
                    "PUT", f"{base}/pieces/{offset - piece}", data, octets
                )
                events.append(f"duplicate: {status} received "
                              f"{view['received']}")
            status, view = web.call(
                "PUT", f"{base}/pieces/{offset}", data, octets
            )
            assert status == 200, (status, view)
            offset = view["received"]
    return base, n


def main() -> None:
    """Run it; print the figures."""
    work, port = Path(sys.argv[1]), int(sys.argv[2])
    pids = {"rag": int(sys.argv[3]), "transcriber": int(sys.argv[4])}
    path, name = Path(sys.argv[5]), sys.argv[6]
    web = Web(port, (work / "code").read_text().strip())
    sampler = Sampler(pids)
    sampler.start()
    time.sleep(1)
    web.call("POST", "/knowledge/domains", json.dumps({
        "id": "big", "title": "Big", "language": "es"}).encode(),
        "application/json")
    baseline = {who: rss(pid) for who, pid in pids.items()}
    events: List[str] = []
    sampler.phase = "receive"
    started = time.monotonic()
    base, pieces = send(web, path, name, events)
    sent = time.monotonic()
    status, doc = web.call("POST", f"{base}/complete")
    assert status == 202, (status, doc)
    completed = time.monotonic()
    sampler.phase = "process"
    progress: List[str] = []
    while True:
        _, docs = web.call("GET", "/knowledge/domains/big/documents")
        row = next(d for d in docs if d["id"] == doc["id"])
        seen = f"{row.get('pages_done')}/{row.get('pages')}"
        if row.get("pages_done") is not None and seen not in progress:
            progress.append(seen)
        if row["status"] in ("indexed", "failed"):
            break
        time.sleep(2)
    done = time.monotonic()
    sampler.stop = True
    time.sleep(0.1)
    peaks = {k: v for k, v in sampler.peaks.items() if k[0] != "idle"}
    print(json.dumps({
        "file": name, "bytes": path.stat().st_size, "pieces": pieces,
        "events": events,
        "upload_seconds": round(sent - started, 1),
        "complete_seconds": round(completed - sent, 1),
        "processing_seconds": round(done - completed, 1),
        "status": row["status"], "error": row.get("error"),
        "pages": row.get("pages"), "chunks": row.get("chunks"),
        "pages_done_seen": progress,
        "baseline_rss_mib": {
            k: round(v / MiB, 1) for k, v in baseline.items()
        },
        "peak_additional_mib": {
            f"{phase}/{who}": round((v - baseline[who]) / MiB, 1)
            for (phase, who), v in peaks.items()
        },
    }, indent=1), flush=True)


if __name__ == "__main__":
    main()
