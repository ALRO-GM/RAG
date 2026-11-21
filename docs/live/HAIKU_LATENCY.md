# Haiku latency through `claude -p` (live call assist)

Script: `experiments/live/haiku_latency.py`; raw data: `experiments/live/results.json`.
Claude Code 2.1.329, model `haiku`, adapter's isolation flags, 12 real calls per mode
(plus 4 smoke calls), synthetic Spanish customer-service / interview snippets.
Prompt: system ~700 tokens + 9 transcript lines + 3 knowledge chunks (~1,750 input tokens
in total); output requested at most 40 words.

| Mode | Metric | median | p90 | min | max |
|---|---|---|---|---|---|
| A cold `json` one-shot | wall (process start to exit) | 1.93 s | 5.56 s | 1.73 s | 5.87 s |
| A | `duration_ms` | 1.14 s | 4.69 s | 0.96 s | 5.02 s |
| A | `duration_api_ms` | 0.98 s | 4.58 s | 0.78 s | 4.88 s |
| B cold `stream-json` one-shot | first text token | 0.85 s | 3.48 s | 0.79 s | 4.35 s |
| B | completion | 1.36 s | 3.64 s | 1.14 s | 4.49 s |
| C persistent process | first token (per message) | 0.49 s | 0.60 s | 0.43 s | 0.85 s |
| C | completion | 0.83 s | 1.03 s | 0.70 s | 1.41 s |

Start-up overhead: process spawn to first `system` event is ~0.20 s (B, C). Mode A wall minus
`duration_ms` is ~0.8 s (process start, auth, exit); B reaches first token 0.85 s after spawn.

Tokens (`usage`): ~2 uncached input, 1,740 cache-creation on the first call (A: every call,
the cache is not read across cold calls here), ~1,380 cache-read plus ~360 creation in B;
output 44-90 tokens typically, with outliers of 384 and 797 (long replies despite the 40-word
rule; the p90 tails of A and B coincide with these, so cap output or stop at the first sentence).
In C the context grows with each turn (cache-read 1.4k to 3.4k+), so a long call needs a
periodic reset or a sliding window.

Verdict: usable. The persistent process (C) answers in under 1.5 s worst case, the cold modes
in about 2 s median but with 4-6 s tails. Use C with one warm process (and restart it to
trim context); fall back to B (stream, show first token at ~0.9 s) if a process cannot be kept.
Caveat: 12 samples per mode, one machine, subscription latency varies with load.
