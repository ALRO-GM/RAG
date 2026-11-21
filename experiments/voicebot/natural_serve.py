"""The orchestrator's ``serve`` with fixed voice-bot choices (#214).

For measuring each natural voice end to end
(``experiments/voicebot/first_audio.py``) without touching the choices
saved on «Ajustes»: the saved choices are read once from the RAG and
the ``--choice step=value`` pairs replace them for this process only.
Everything else is ``python -m orchestrator serve`` with the same
arguments (after ``--``)::

    python experiments/voicebot/natural_serve.py \\
        --choice voicebot.voice=pocket:lola -- --port 8792 \\
        --transcriber-code-file data/transcriber/access_code \\
        --token-file T
"""

import argparse
from typing import Dict, List, Optional, Sequence

from orchestrator.adapters.inbound import cli
from orchestrator.adapters.outbound.http_transcriber import read_code


class FixedSettings:
    """The saved choices with some replaced (a ``SettingsSource``)."""

    def __init__(self, choices: Dict[str, str]):
        """Keep the choices."""
        self.choices = choices

    def current(self) -> Optional[Dict[str, str]]:
        """Return the choices."""
        return dict(self.choices)


def parse_choices(pairs: Sequence[str]) -> Dict[str, str]:
    """Read ``step=value`` pairs."""
    choices: Dict[str, str] = {}
    for pair in pairs:
        step, sep, value = pair.partition("=")
        if not sep or not step.startswith("voicebot."):
            raise SystemExit(f"bad choice {pair!r}: voicebot.step=value")
        choices[step] = value
    return choices


def main(argv: Optional[List[str]] = None) -> int:
    """Build ``serve`` as the CLI does, replace the choices, run it."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--choice", action="append", default=[])
    parser.add_argument("serve", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    rest = [a for a in args.serve if a != "--"]
    serve = cli.build_parser().parse_args(["serve", *rest])
    sessions = cli.build_sessions(serve)
    saved = sessions.settings.current() if sessions.settings else None
    sessions.settings = FixedSettings({**(saved or {}),
                                       **parse_choices(args.choice)})
    token = read_code(serve.token_file) if serve.token_file else None
    import uvicorn

    from orchestrator.adapters.inbound.http_server import create_app

    print(f"orchestrator on http://{serve.host}:{serve.port} with "
          f"{sessions.settings.current()}", flush=True)
    uvicorn.run(create_app(sessions, token), host=serve.host,
                port=serve.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
