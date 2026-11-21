# IMG worker: honouring the GPU lease (`gpu-lease.v1`)

Change the IMG project needs so it yields the GPU to the transcriber's
live mode (protocol: [ARCHITECTURE.md §7](ARCHITECTURE.md), file format:
[CONTRACTS.md §7](CONTRACTS.md)). It is **not applied** from this
repository: the owner (or an IMG session) applies it to
`IMG/worker/worker.py`. The transcriber side is in
`transcriber/adapters/outbound/fs_gpu_lease.py`.

Line numbers refer to `IMG/worker/worker.py` as read on 2026-11-21
(502 lines). The restart of the systemd unit is not needed to test the
logic, but the unit must be restarted to load the change.

## What the worker must do

1. Read `$XDG_RUNTIME_DIR/gpu-lease/lease.json`. The lease is **valid**
   when it parses, `schema == "gpu-lease.v1"`, `expires_at` is in the
   future and `/proc/<pid>` exists (never `os.kill`).
2. While a valid lease exists, do not claim jobs; sleep 5 s.
3. While generating, if a valid lease appears: interrupt ComfyUI,
   return the job to the server as retryable, stop its own ComfyUI
   child, write the ack. (The systemd unit keeps running.)
4. Resume claiming 30 s after the lease file has disappeared or stopped
   being valid.

## 1. Helpers (after `log`, around line 75)

```python
import datetime as dt

LEASE_DIR = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / "gpu-lease"
LEASE_RESUME_S = 30     # wait after the lease ends before claiming
LEASE_POLL_S = 5


def lease_valid():
    """The live lease if it counts (parses, not expired, pid alive), else None."""
    try:
        lease = json.loads((LEASE_DIR / "lease.json").read_text())
        expires = dt.datetime.fromisoformat(lease["expires_at"])
        if lease.get("schema") != "gpu-lease.v1" or expires <= dt.datetime.now(dt.timezone.utc):
            return None
        return lease if Path("/proc", str(int(lease["pid"]))).exists() else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def lease_ack():
    """Tell the live side the card is free (gpu-lease.ack.v1), atomically."""
    LEASE_DIR.mkdir(parents=True, exist_ok=True)
    body = {"schema": "gpu-lease.ack.v1", "holder": "img-worker", "freed": True,
            "at": dt.datetime.now(dt.timezone.utc).isoformat()}
    tmp = LEASE_DIR / ".ack.tmp"
    tmp.write_text(json.dumps(body))
    os.replace(tmp, LEASE_DIR / "lease.ack.json")
```

The ack must carry an `at` later than the lease's `acquired_at`: the
transcriber ignores older acks.

## 2. Before claiming (loop in `main`, line ~476)

```python
lease_ended = None                      # time the last lease stopped being valid
while True:
    lease = lease_valid()
    if lease:
        if comfy.running():             # idle ComfyUI still holds VRAM
            comfy.stop()
        lease_ack()                     # card is free: say so (cheap, idempotent)
        lease_ended = time.time()
        time.sleep(LEASE_POLL_S)
        continue
    if lease_ended and time.time() - lease_ended < LEASE_RESUME_S:
        time.sleep(LEASE_POLL_S)
        continue
    lease_ended = None
    try:
        job = server("/api/worker/claim", ...)   # unchanged
```

## 3. During generation (`Comfy.generate`, loop at line ~199)

It already polls every 3 s for cancellation; add the lease next to it:

```python
if time.time() - checked > 3:
    checked = time.time()
    if self.cancelled():
        self.interrupt()
        raise Cancelled()
    if lease_valid():
        self.interrupt()                # POST /interrupt + wait for the queue to empty
        raise LeaseTaken()
```

Define `class LeaseTaken(Exception)`. In `main`, handle it before the
generic `except Exception`:

```python
except LeaseTaken:
    log(f"Trabajo {job['id']} devuelto: la GPU la usa una sesión en vivo")
    try:
        server(f"/api/worker/jobs/{job['id']}/fail", {"error": "gpu lease taken", "retry": True})
    except OSError:
        pass
    comfy.stop()                        # frees the VRAM: stops only its own child
    lease_ack()
finally:
    ...                                 # unchanged
```

`long_video` / `multi_video` call `comfy.generate` per segment, so the
exception propagates from there with no extra change; the partial
segments are discarded and the job restarts from scratch when requeued.

## 4. Server side (IMG server)

`/api/worker/jobs/<id>/fail` must accept `{"retry": true}` and put the
job back in `queued` instead of `failed` (or add a dedicated
`/requeue` endpoint). Without it the interrupted job is lost; this is
the only change outside `worker.py`.

## Acceptance (from issue IMG-1)

- No claim while a valid lease exists; the idle ComfyUI is stopped.
- A lease appearing mid-job interrupts and requeues the job and writes
  `lease.ack.json` (`freed: true`).
- Claiming resumes 30 s after the lease file disappears.
- A lease whose holder was killed (pid gone, or `expires_at` past) does
  not block the worker.
