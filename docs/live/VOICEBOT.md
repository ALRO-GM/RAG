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

## Using it (VB-4)

1. The three services run as usual (`ragputin-servicios`): the RAG
   (knowledge `support.demo`, `POST /reason/stream`, the warm Haiku pool
   of `conversational.voicebot@1` started with the service), the
   orchestrator (`python -m orchestrator serve`, Piper on by default:
   `--tts piper`, `--piper-python ~/.local/share/ragputin/tts-venv/bin/python`,
   `--voices-dir ~/.local/share/ragputin/voices`; it warms the voice at
   start) and the transcriber (live sessions, GPU lease).
2. Open the transcriber's web, «Asistente de voz» (`#/voicebot`), choose
   *Mantener pulsado para hablar* (push-to-talk: hold the button or the
   space bar) or *Manos libres*, tick «Llevo auriculares» when wearing
   one, and press «Empezar a hablar».
3. The conversation shows what you said (partials dashed), the bot's
   sentences as they are spoken (marked «interrumpido» when cut), and
   under it the last turn's latency: end of your sentence → first voice.

Piper must be installed in its own venv (GPL-3.0, never imported; see
`experiments/skills/render_simulation.py` for the two commands). Without
it the orchestrator prints `voice bot off` and the page says the
assistant is not available; `--tts fake` answers with short tones (for
trying the page without Piper), `--tts none` turns it off.

### Transport

The bot's audio goes to the browser over the session's existing
Server-Sent Events stream (`/api/assist/sessions/{id}/suggestions/stream`,
through the transcriber's proxy): one message per sentence,
`{"type": "speech", "turn", "seq", "text", "format": "wav",
"sample_rate": 22050, "duration_ms", "audio": <base64 WAV>}`. It is the
simplest choice: no new socket, no new proxy route, the page already
follows that stream, and the browser decodes a WAV with
`decodeAudioData` and plays the sentences back to back (`SpeechQueue`).
Base64 adds a third to the size: a second of speech is 44 KB of WAV
(22 050 Hz, 16 bit) and about 59 KB in the stream; the time it adds is
inside the end-to-end measure of VB-5 below.
Other messages: `{"type": "bot", "turn", "state": "thinking"}` when a
turn starts, `{"type": "interrupt", "turn", "by": "new_turn" | "vad" |
"speech"}` and `{"type": "turn", "record": <TurnRecord>}` with the
per-stage latency (`event_lag_ms`, `knowledge_ms`, `first_text_ms`,
`first_sentence_ms`, `first_tts_ms`, `first_audio_ms`, `reasoner_ms`,
`total_ms`). A turn's audio is removed from the feed when the next turn
starts, so the session does not keep audio; the turn log
(`<log-dir>/<session>.voicebot.turns.jsonl`) has text and latency only.

The microphone path is the live page's (`WS /api/live/{id}/audio`,
16 kHz PCM). Push-to-talk keeps the socket sending *silence* while the
button is up, so the transcriber's endpointing (VAD hang 400 ms) still
closes the sentence when the button is released. The trigger is the
user's `final` (`config/products/voicebot.yaml`, `trigger.on: final`):
the live transcriber emits no `turn_end`, and in a one-person session a
final after the hang is the end of the user's turn. A newer final while
a turn is still being prepared replaces it.

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
