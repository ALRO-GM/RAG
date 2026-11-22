"""Which engine says the interviewer's opening line (cold first sentence).

The owner heard the voice bot's first sentence of each session sound
bad and the later ones fine. This script starts ``voicebot.interview``
sessions on a spare orchestrator (``experiments/voicebot/
natural_serve.py``, the saved «Ajustes» choices) with the running
transcriber (real live session, its GPU lease) and follows each
session's feed until the opening turn has spoken (or ``--wait``
seconds), then stops it. No audio is sent: the opening is said before
the candidate speaks. For every session it keeps, on its own clock from
the ``POST /sessions``:

* the feed's ``voice`` messages (the engine chosen, then each change of
  speaker with the reason the ones before it did not speak) and
  ``voice_state`` messages (the voice getting ready, if any);
* every ``speech`` message of the opening turn: when it arrived, its
  sentence number, sample rate (22050: Piper, 24000: Pocket or Qwen3)
  and length, and which engine spoke it (the last ``voice`` message
  before it).

One real Haiku call per session (the opening line). Usage::

    python experiments/voicebot/opening_voice.py --orchestrator URL \\
        --token-file T --plan PLAN_ID --runs 3 --out R.json
"""

import argparse
import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx


def follow(client: httpx.Client, url: str, headers: Dict[str, str],
           t0: float, items: List[Dict[str, Any]],
           done: threading.Event) -> None:
    """Append the feed's messages (with their arrival ms) until it ends."""
    try:
        with client.stream("GET", url, headers=headers,
                           timeout=None) as response:
            response.raise_for_status()
            name = "message"
            data: List[str] = []
            for line in response.iter_lines():
                if done.is_set():
                    return
                if line.startswith("event:"):
                    name = line[6:].strip()
                elif line.startswith("data:"):
                    data.append(line[5:].strip())
                elif not line and data:
                    if name == "end":
                        return
                    message = json.loads("\n".join(data))
                    message["_ms"] = round((time.monotonic() - t0) * 1000)
                    message.pop("audio", None)
                    items.append(message)
                    name, data = "message", []
                elif not line:
                    name = "message"
    except httpx.HTTPError:
        return


def opening(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarise the opening turn of one session's feed."""
    engine: Optional[str] = None
    voices: List[Dict[str, Any]] = []
    states: List[Dict[str, Any]] = []
    sentences: List[Dict[str, Any]] = []
    for message in items:
        kind = message.get("type")
        if kind == "voice":
            engine = str(message.get("engine"))
            voices.append({k: message.get(k) for k in
                           ("_ms", "engine", "clone", "reason")})
        elif kind == "voice_state":
            states.append({k: message.get(k) for k in ("_ms", "state")})
        elif kind == "speech" and message.get("turn") == 1:
            if message.get("text") or message.get("sample_rate") is None:
                sentences.append({
                    "ms": message["_ms"], "seq": message.get("seq"),
                    "sample_rate": message.get("sample_rate"),
                    "chars": len(str(message.get("text") or "")),
                    "engine": engine,
                })
    return {"voices": voices, "voice_states": states,
            "sentences": sentences,
            "first_engine": sentences[0]["engine"] if sentences else None}


def run_one(client: httpx.Client, base: str, headers: Dict[str, str],
            plan: str, session_id: str, wait: float) -> Dict[str, Any]:
    """Start one interview session, follow its opening, stop it."""
    items: List[Dict[str, Any]] = []
    done = threading.Event()
    t0 = time.monotonic()
    response = client.post(f"{base}/sessions", headers=headers, json={
        "product": "voicebot.interview", "audio": "mic", "plan": plan,
        "session_id": session_id, "candidate": "Prueba",
    }, timeout=60)
    response.raise_for_status()
    reader = threading.Thread(target=follow, args=(
        client, f"{base}/sessions/{session_id}/suggestions/stream",
        headers, t0, items, done), daemon=True)
    reader.start()
    deadline = t0 + wait
    while time.monotonic() < deadline:
        spoken = [m for m in items if m.get("type") == "bot"
                  and m.get("turn") == 1 and m.get("state") not in
                  (None, "thinking")]
        if spoken or any(m.get("type") == "turn" for m in items):
            break
        time.sleep(0.2)
    time.sleep(2.0)
    client.post(f"{base}/sessions/{session_id}/stop", headers=headers,
                timeout=30)
    done.set()
    return {"session": session_id, **opening(items)}


def main() -> int:
    """Run the sessions and write the results."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--orchestrator", default="http://127.0.0.1:8792")
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--wait", type=float, default=40.0)
    parser.add_argument("--pause", type=float, default=5.0)
    parser.add_argument("--label", default="")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    token = args.token_file.read_text(encoding="utf-8").strip()
    headers = {"Authorization": f"Bearer {token}"}
    base = args.orchestrator.rstrip("/")
    rows = []
    with httpx.Client() as client:
        for run in range(args.runs):
            session_id = time.strftime(f"opening-%H%M%S-{run}")
            row = run_one(client, base, headers, args.plan, session_id,
                          args.wait)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            rows.append(row)
            time.sleep(args.pause)
    args.out.write_text(json.dumps({
        "label": args.label, "date": time.strftime("%Y-%m-%d %H:%M"),
        "runs": rows,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
