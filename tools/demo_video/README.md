# Demo video (DM-1, #233)

A two-minute recording of the product for showing it to people: the
home with the three services, the scripted interview of `#/live/demo`
playing in real time (suggestions, the age question with «Mejor evitar»
and «Reformulado», the job topics), and Lucía Marín's result (the
88,7 % ring, strengths, improvements, bars, «Más información»). Spanish
captions sit in a band under the page, so they never cover it. All the
data is the fictitious demo of `examples/demo/`.

## Re-record

```sh
make demo-video-tools   # once: a static ffmpeg in its own environment
make demo-video         # about 2 minutes
```

The video goes to `~/.local/share/ragputin/demo-video/demo-entrevista.mp4`
(outside the repository; never commit it), with eight frames spread over
it as PNG next to it for a look. The command prints a JSON report:
frames captured and written, duration, the capture rate, the frames held
longer than 0.2 s (they should all be still moments: cards, pauses
before a click), the scene marks and when each caption is shown.
`DEMO_ARGS='--keep-work'` keeps the temporary folder (logs, raw frames);
`DEMO_ARGS='--base-port 19041'` moves the four loopback ports (default
18841 to 18844) if they are taken.

## How it works

- `record.py` starts its own instances, never the services of the
  workstation: the transcriber's web (fake runtime, CPU, a temporary
  data directory), the orchestrator (fake knowledge and reasoner: the
  demo's suggestions are in its file) and `stand_in_rag.py`, which
  installs the demo result with the RAG's own code and serves the two
  analysis read routes the result page calls. `XDG_RUNTIME_DIR` points
  at the temporary folder, so their GPU lease is not the real one, and
  `CUDA_VISIBLE_DEVICES` is empty.
- Headless Google Chrome (`/usr/bin/google-chrome`) lays the page out at
  1536x792 CSS pixels with a scale factor of 1.25 (1920x990 pixels) in
  the light theme and Spanish. `overlay.js` adds the pointer, the click
  ring, gentle scrolling and the title cards; clicks are real DevTools
  mouse events. The screencast keeps every frame Chrome paints with its
  capture time.
- `timeline.py` (pure, tested in `tests/unit/tools/`) resamples the
  frames to a constant 30 fps on their capture times, so the video runs
  in real time with no speed-up, and holds the captions of the path,
  written relative to scene marks. ffmpeg adds the 90-pixel caption
  band and encodes H.264 High, yuv420p, BT.709, `+faststart`, no audio.

To change what the video says, edit `CAPTIONS` in `timeline.py`; to
change what it shows, the `play` function of `record.py`. The script
stops with a clear message when ffmpeg, Chrome, the font or a port is
missing, or when the page does not show what it expects.
