# Voice bot (P3: VB-1 … VB-5)

Design: `docs/platform/ARCHITECTURE.md` §3 (P3), §6 (latency), §8
(licences). Product config: `config/products/voicebot.yaml`.

```
browser mic ──WS──▶ transcriber (live session, 1 channel, VAD hang 400 ms)
                          │ live.event.v1 (partials, finals)
                          ▼
                    orchestrator VoiceBot ──▶ POST /knowledge/query (support.demo)
                          │               ──▶ POST /reason/stream (conversational.voicebot@1, warm Haiku)
                          │ each finished sentence
                          ▼
                    Piper (own venv, own process) ── WAV per sentence
                          │ {"type": "speech", "audio": base64}
                          ▼
browser ◀──SSE── orchestrator feed (/api/assist/sessions/{id}/suggestions/stream)
```

## Barge-in and echo (VB-3)

The bot is *speaking* while its audio is still playing in the page. The
orchestrator estimates the end of playback from the durations of the
chunks it sent (`orchestrator/domain/bargein.py`, `Playback`).

- **The page's voice detector** (`EnergyVad` in
  `transcriber/adapters/inbound/web/js/lib/voicebot.js`): while the bot
  speaks, a microphone level above the threshold for 150 ms stops the
  playback at once in the browser and posts
  `POST /api/assist/sessions/{id}/barge-in`; the orchestrator cancels the
  turn in flight (no more reasoner text is read, no more sentences are
  synthesised) and the record says `interrupted_by: "vad"`.
- **The transcriber's words**: a user `partial` or `final` of two words
  or more while the bot speaks does the same (`interrupted_by:
  "speech"`), later (the first partial comes ~0.8 s after the speech
  starts) but without the page's detector.
- **A newer final** cancels a turn still in flight (`new_turn`), as in
  live assist.
- **Echo**: what the microphone hears is compared with the bot's last
  sentences; when at least 60 % of its words (and two or more) are words
  the bot just said, during playback or up to 1.5 s after it, it is the
  bot's own voice coming back: neither a barge-in nor a turn (skip
  reason `echo`).

**Headset assumption.** The voice bot is meant to be used with a
headset. With loudspeakers the microphone hears the bot: the echo check
catches the bot's words once they are transcribed, but the page's
level detector cannot tell the bot's voice from the user's, so the page
only listens for barge-in when "Llevo auriculares" is ticked (the
default). Without a headset: untick it and use push-to-talk; pressing
the button while the bot speaks is the barge-in. The browser's own echo
cancellation (`echoCancellation: true`) is asked for on this page only
(the live and record pages keep it off, as before).
