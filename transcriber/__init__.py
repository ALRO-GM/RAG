"""Transcriber: from a recorded session to attributed sentences.

A service separate from the RAG and the analyzer under ``src/``: it
owns the audio and the voiceprints and hands over only the text
contract ``transcript.v1`` (see ``docs/transcriber/ARCHITECTURE.md``).
It never imports ``src`` and ``src`` never imports it.
"""

__version__ = "0.1.0"
