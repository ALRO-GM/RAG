"""Inbound adapter: the command line (``python -m transcriber ...``).

Thin wrappers over the use cases with human-readable output (``--json``
where a program may read it). Settings come from the environment
(:func:`~transcriber.config.load_settings`); ``TRANSCRIBER_RUNTIME=fake``
runs every command without torch.

``serve`` is the one process that runs the jobs: it builds the container
with recovery (:func:`~transcriber.bootstrap.start_container`). Every
other command uses :func:`~transcriber.bootstrap.build_container`,
which recovers nothing, so it may run while the service does. ``enrol``
and ``import --wait`` run the models in their own process and should
not be used while ``serve`` is working on a job (one GPU).
"""

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path, PurePath
from typing import (
    Any, Callable, Dict, List, Mapping, Optional, Sequence, TextIO,
)

from pydantic import ValidationError

from transcriber import __version__
from transcriber.application.ports import RecordedAudio
from transcriber.application.transcribe import TranscriptionOptions
from transcriber.bootstrap import (
    TranscriberContainer,
    access_code,
    build_container,
    build_tunnel,
    start_container,
)
from transcriber.config import TranscriberSettings, load_settings
from transcriber.domain.errors import (
    InvalidInputError,
    NotFoundError,
    TranscriberError,
)
from transcriber.domain.models import SessionState, Transcript

MODELS_WARNING = (
    "enrol and import --wait run the models in this process: do not use"
    " them while serve is working on a job (one GPU)."
)

WAIT_POLL_SECONDS = 0.2
"""How often ``import --wait`` looks at its job."""

RUNNER_CLOSE_TIMEOUT = 30.0
"""Seconds a command waits for the job runner to stop before exiting."""

FINISHED = (SessionState.DONE, SessionState.FAILED)


class UsageError(Exception):
    """The command was asked for something it cannot do (exit code 2)."""


class Output:
    """Where a command writes: the terminal's output and error streams."""

    def __init__(self, out: TextIO, err: TextIO) -> None:
        """Write results to ``out`` and messages to ``err``."""
        self.out = out
        self.err = err

    def line(self, text: str = "") -> None:
        """Write one line of result."""
        self.out.write(text + "\n")
        self.out.flush()

    def json(self, data: Any) -> None:
        """Write ``data`` as indented JSON."""
        self.line(json.dumps(data, indent=2, ensure_ascii=False))

    def note(self, text: str) -> None:
        """Write one line of information to the error stream."""
        self.err.write(text + "\n")
        self.err.flush()


def slug(text: str) -> str:
    """Return ``text`` as an id: ASCII, lowercase, words joined by ``-``.

    At most 48 characters; empty when nothing usable is left.
    """
    plain = unicodedata.normalize("NFKD", text)
    ascii_text = plain.encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", "-", ascii_text).strip("-")[:48].strip("-")


def clock_time(seconds: float) -> str:
    """Return ``seconds`` as ``hh:mm:ss`` (whole seconds, floored)."""
    whole = max(int(seconds), 0)
    return f"{whole // 3600:02d}:{whole % 3600 // 60:02d}:{whole % 60:02d}"


def transcript_lines(transcript: Transcript, lead: float = 0.0) -> List[str]:
    """Return one line per utterance: ``[hh:mm:ss] who (after X): text``.

    People are shown by name when they gave one, else by id. The time is
    ``start - lead`` (never below zero): where to seek in a player to
    hear the sentence with ``lead`` seconds of context.
    """
    names = {s.id: s.name or s.id for s in transcript.speakers}
    lines = []
    for utterance in transcript.utterances:
        who = names.get(utterance.speaker, utterance.speaker)
        after = ""
        if utterance.previous_speaker is not None:
            previous = names.get(
                utterance.previous_speaker, utterance.previous_speaker
            )
            after = f" (after {previous})"
        when = clock_time(max(utterance.start - lead, 0.0))
        lines.append(f"[{when}] {who}{after}: {utterance.text}")
    return lines


def _read_recording(path: Path) -> RecordedAudio:
    """Return the file at ``path`` as a recording of its extension."""
    kind = PurePath(path).suffix.lower().lstrip(".")
    if not kind:
        raise InvalidInputError(
            "The file name needs an extension (wav, m4a, mp3...)."
        )
    return RecordedAudio(path.read_bytes(), kind)


def _speaker_list(text: Optional[str]) -> Optional[List[str]]:
    """Return ``a,b`` as ``["a", "b"]`` (``None`` when not given)."""
    if text is None:
        return None
    found = [part.strip() for part in text.split(",") if part.strip()]
    if not found:
        raise UsageError("--expect needs at least one speaker id.")
    return found


class Commands:
    """The commands, over a container built on first use.

    ``build`` makes the container (``build_container`` for every
    command; ``serve`` uses its own). The container's job runner is
    closed by :meth:`close`.
    """

    def __init__(
        self,
        settings: TranscriberSettings,
        output: Output,
        build: Callable[[TranscriberSettings], TranscriberContainer] = (
            build_container
        ),
    ) -> None:
        """Keep the settings; nothing is built yet."""
        self.settings = settings
        self.output = output
        self._build = build
        self._container: Optional[TranscriberContainer] = None

    @property
    def container(self) -> TranscriberContainer:
        """Return the container, building it the first time."""
        if self._container is None:
            self._container = self._build(self.settings)
        return self._container

    def close(self) -> None:
        """Stop the job runner of the container, if one was built."""
        if self._container is not None:
            self._container.runner.close(RUNNER_CLOSE_TIMEOUT)

    def version(self, args: argparse.Namespace) -> int:
        """Print the transcriber version."""
        self.output.line(__version__)
        return 0

    def show_settings(self, args: argparse.Namespace) -> int:
        """Print the settings that matter to find things."""
        current = self.settings
        data = {
            "data_dir": str(current.data_dir),
            "runtime": current.runtime,
            "recogniser": current.recogniser,
            "voice_model": current.voice_model,
            "published": current.publish_target(),
            "tunnel_target": current.tunnel.target,
            "public_host": current.tunnel.public_host,
            "port": str(current.api.port),
        }
        if args.json:
            self.output.json(data)
        else:
            for key, value in data.items():
                self.output.line(f"{key}: {value}")
        return 0

    def serve(self, args: argparse.Namespace) -> int:
        """Run the API, the tunnel and the progress lines until Ctrl-C."""
        from transcriber.adapters.inbound.serve import (
            WEB_DIR,
            ServeOptions,
            dev_settings,
            serve,
        )

        settings = self.settings
        port = args.port if args.port is not None else settings.api.port
        if args.dev_pages:
            if not WEB_DIR.is_dir():
                raise UsageError(f"There are no pages in {WEB_DIR}.")
            settings = dev_settings(settings)
        code = access_code(settings, args.keep_code)
        container = start_container(settings)
        tunnel = None if args.no_tunnel else build_tunnel(settings, port)
        options = ServeOptions(
            port=port, tunnel=not args.no_tunnel, dev_pages=args.dev_pages
        )
        return serve(container, code, options, tunnel, self.output.out)

    def enrol(self, args: argparse.Namespace) -> int:
        """Enrol a person from a recording of the enrolment text."""
        if not args.consent:
            raise UsageError(
                "A voice is only enrolled with the person's explicit"
                " consent: add --consent once they agreed."
            )
        speaker_id = args.id or slug(args.name)
        if not speaker_id:
            raise UsageError("Choose an id for this person with --id.")
        recording = _read_recording(Path(args.file))
        voiceprint = self.container.enrol_speaker(
            speaker_id, args.name, args.language, recording, True,
            self.settings.api.consent_version,
        )
        data = {
            "id": voiceprint.speaker_id,
            "model": voiceprint.model,
            "enrolled_at": voiceprint.enrolled_at.isoformat(),
        }
        if args.json:
            self.output.json(data)
        else:
            self.output.line(
                f"Enrolled {args.name} as {voiceprint.speaker_id}"
                f" ({voiceprint.model})."
            )
        return 0

    def import_file(self, args: argparse.Namespace) -> int:
        """Import a recording; with ``--wait`` transcribe it here."""
        expected = _speaker_list(args.expect)
        if not args.wait and (args.speakers is not None or expected):
            raise UsageError(
                "--speakers and --expect apply to the transcription:"
                " use them with --wait."
            )
        path = Path(args.file)
        session_id = args.id or (
            f"{slug(path.stem) or 'import'}-"
            f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
        recording = _read_recording(path)
        c = self.container
        session = c.import_recording(
            session_id, args.language, recording, args.title
        )
        if not args.wait:
            if args.json:
                self.output.json({"id": session.id,
                                  "state": session.state.value})
            else:
                self.output.line(
                    f"Imported {session.id} ({session.state.value});"
                    " transcribe it from the pages, or import with --wait."
                )
            return 0
        c.queue_transcription(session.id, TranscriptionOptions(
            num_speakers=args.speakers, expected_speakers=expected,
        ))
        if not args.json:
            self.output.line(f"Imported {session.id}; transcribing.")
        return self._wait(session.id, args.json)

    def _wait(self, session_id: str, as_json: bool) -> int:
        """Show the job's progress until it ends; 1 if it failed."""
        c = self.container
        last = None
        while True:
            status = c.transcription_status(session_id)
            shown = (status.state, status.stage, int(status.fraction * 10))
            if shown != last and not as_json:
                stage = (
                    f" {status.stage}"
                    if status.stage and status.state not in FINISHED else ""
                )
                self.output.line(
                    f"{status.state.value}{stage}"
                    f" {int(status.fraction * 100)}%"
                )
            last = shown
            if status.state in FINISHED:
                break
            time.sleep(WAIT_POLL_SECONDS)
        session = c.show_session(session_id).session
        if as_json:
            self.output.json({"id": session_id, "state": session.state.value,
                              "failure": session.failure})
        elif session.state is SessionState.FAILED:
            self.output.line(f"Failed: {session.failure}")
        else:
            transcript = c.show_transcript(session_id)
            self.output.line(
                f"Done: {len(transcript.utterances)} sentences,"
                f" {len(transcript.speakers)} speakers."
                f" See them with: show {session_id}"
            )
        return 1 if session.state is SessionState.FAILED else 0

    def sessions(self, args: argparse.Namespace) -> int:
        """List the sessions, or delete one."""
        c = self.container
        if args.action == "delete":
            c.delete_session(args.session_id)
            self.output.line(f"Deleted {args.session_id}.")
            return 0
        listing = c.list_sessions()
        if args.json:
            self.output.json({
                "sessions": [
                    s.model_dump(mode="json", include={
                        "id", "title", "language", "started_at",
                        "duration", "state", "failure",
                    })
                    for s in listing.sessions
                ],
                "unreadable": listing.unreadable,
            })
            return 0
        if not listing.sessions:
            self.output.line("No sessions.")
        for s in listing.sessions:
            title = f"  {s.title}" if s.title else ""
            self.output.line(
                f"{s.id}  {s.state.value}  {clock_time(s.duration)}"
                f"  {s.language}{title}"
            )
        if listing.unreadable:
            self.output.line(f"({listing.unreadable} unreadable)")
        return 0

    def show(self, args: argparse.Namespace) -> int:
        """Show a session: its sentences when transcribed, else its state."""
        if args.lead < 0:
            raise UsageError("--lead cannot be negative.")
        c = self.container
        view = c.show_session(args.session_id)
        session = view.session
        try:
            transcript: Optional[Transcript] = c.show_transcript(session.id)
        except NotFoundError:
            transcript = None
        if args.json:
            data: Dict[str, Any] = {
                "session": session.model_dump(mode="json"),
                "status": view.status.model_dump(mode="json"),
            }
            if transcript is not None:
                data["transcript"] = transcript.model_dump(
                    mode="json",
                    exclude={"utterances": {"__all__": {"words"}}},
                )
            self.output.json(data)
            return 0
        title = f" - {session.title}" if session.title else ""
        self.output.line(
            f"{session.id}{title} ({session.state.value},"
            f" {clock_time(session.duration)})"
        )
        if session.failure:
            self.output.line(f"Failed: {session.failure}")
        if transcript is not None:
            for line in transcript_lines(transcript, args.lead):
                self.output.line(line)
        return 0

    def export(self, args: argparse.Namespace) -> int:
        """Write the ``transcript.v1`` document of a session."""
        document = self.container.show_transcript(
            args.session_id
        ).to_contract()
        text = json.dumps(document, indent=2, ensure_ascii=False) + "\n"
        if args.output is None:
            self.output.out.write(text)
            self.output.out.flush()
            return 0
        target = Path(args.output)
        target.write_text(text, encoding="utf-8")
        self.output.note(f"Wrote {target}")
        return 0

    def publish(self, args: argparse.Namespace) -> int:
        """Hand a finished transcript over to the RAG."""
        self.container.publish(args.session_id)
        self.output.line(
            f"Published {args.session_id} to "
            f"{self.settings.publish_target()}."
        )
        return 0

    def speakers(self, args: argparse.Namespace) -> int:
        """List the enrolled people, or forget one."""
        c = self.container
        if args.action == "forget":
            c.forget_speaker(args.speaker_id)
            self.output.line(f"Forgot {args.speaker_id}.")
            return 0
        found = c.list_speakers()
        if args.json:
            self.output.json({"speakers": [s._asdict() for s in found]})
            return 0
        if not found:
            self.output.line("Nobody is enrolled.")
        for speaker in found:
            self.output.line(
                f"{speaker.id}  {speaker.name or '-'}  {speaker.enrolled_at}"
            )
        return 0

    def enrolment_text(self, args: argparse.Namespace) -> int:
        """Print the text a person reads aloud to enrol."""
        guide = self.container.enrolment_text(
            args.language, args.name or None
        )
        chosen = guide.text
        if args.json:
            self.output.json({
                "language": chosen.language,
                "version": chosen.version,
                "template": chosen.template,
                "text": guide.reading,
                "consent_version": guide.consent_version,
            })
        else:
            self.output.line(guide.reading or chosen.template)
        return 0


def _json_flag(parser: argparse.ArgumentParser) -> None:
    """Add ``--json`` to ``parser``."""
    parser.add_argument(
        "--json", action="store_true", help="print JSON instead of text"
    )


def build_parser() -> argparse.ArgumentParser:
    """Return the parser of every command."""
    parser = argparse.ArgumentParser(
        prog="transcriber",
        description="From a recorded session to attributed sentences.",
        epilog=MODELS_WARNING,
    )
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")
    commands.required = True

    commands.add_parser("version", help="print the version")
    settings = commands.add_parser("settings", help="print the settings")
    _json_flag(settings)

    serve = commands.add_parser(
        "serve", help="run the API, the tunnel and print the join link",
        description=(
            "Run the API on 127.0.0.1, open the SSH tunnel to the server"
            " and print the join link with a QR code. This process runs"
            " the jobs. Ctrl-C stops it."
        ),
        epilog=MODELS_WARNING,
    )
    serve.add_argument("--port", type=int, help="local port of the API")
    serve.add_argument("--no-tunnel", action="store_true",
                       help="local only: no tunnel, an http://127.0.0.1"
                       " link")
    serve.add_argument("--dev-pages", action="store_true",
                       help="also serve the pages at / (development)")
    serve.add_argument("--keep-code", action="store_true",
                       help="reuse the access code kept in the data"
                       " directory (made on first use), so a printed QR"
                       " keeps working")

    enrol = commands.add_parser(
        "enrol", help="enrol a person's voice (runs the models here)",
        description=(
            "Enrol a person from a recording of the enrolment text, with"
            " their consent. " + MODELS_WARNING
        ),
    )
    enrol.add_argument("name", help="the person's name, as they give it")
    enrol.add_argument("file", help="recording of the enrolment text")
    enrol.add_argument("--consent", action="store_true",
                       help="the person agreed to the consent statement")
    enrol.add_argument("--id", help="speaker id (default: from the name)")
    enrol.add_argument("--language", default="es",
                       help="language of the text read (default: es)")
    _json_flag(enrol)

    imported = commands.add_parser(
        "import", help="import a recording made with another app",
        description=(
            "Import a recording as a new session. Without --wait it is"
            " left ready to transcribe (from the pages). With --wait it"
            " is transcribed in this process, showing progress. "
            + MODELS_WARNING
        ),
    )
    imported.add_argument("file", help="the recording (wav, m4a, mp3...)")
    imported.add_argument("--id", help="session id (default: from the file"
                          " name and the time)")
    imported.add_argument("--title", help="title of the session")
    imported.add_argument("--language", default="es",
                          help="language spoken (default: es)")
    imported.add_argument("--speakers", type=int,
                          help="how many people speak, when sure")
    imported.add_argument("--expect", help="enrolled ids present: ID,ID")
    imported.add_argument("--wait", action="store_true",
                          help="transcribe now and show progress")
    _json_flag(imported)

    sessions = commands.add_parser("sessions", help="list or delete sessions")
    _json_flag(sessions)
    session_actions = sessions.add_subparsers(dest="action", metavar="ACTION")
    deleted = session_actions.add_parser(
        "delete", help="delete a session with its audio and transcript"
    )
    deleted.add_argument("session_id", metavar="ID")

    show = commands.add_parser("show", help="show a session's sentences")
    show.add_argument("session_id", metavar="ID")
    show.add_argument("--lead", type=float, default=0.0, metavar="SECONDS",
                      help="show each time SECONDS earlier (where to seek"
                      " to hear the sentence with context)")
    _json_flag(show)

    export = commands.add_parser(
        "export", help="write the transcript.v1 document"
    )
    export.add_argument("session_id", metavar="ID")
    export.add_argument("--output", metavar="FILE",
                        help="write here instead of the terminal")

    publish = commands.add_parser(
        "publish", help="hand a finished transcript over to the RAG"
    )
    publish.add_argument("session_id", metavar="ID")

    speakers = commands.add_parser(
        "speakers", help="list or forget enrolled people"
    )
    _json_flag(speakers)
    speaker_actions = speakers.add_subparsers(dest="action", metavar="ACTION")
    forget = speaker_actions.add_parser(
        "forget", help="forget a person's voiceprint and consent"
    )
    forget.add_argument("speaker_id", metavar="ID")

    text = commands.add_parser(
        "enrolment-text", help="print the text to read aloud at enrolment"
    )
    text.add_argument("--language", default="es",
                      help="language of the text (default: es)")
    text.add_argument("--name", help="fill the text with this name")
    _json_flag(text)
    return parser


HANDLERS: Dict[str, Callable[[Commands, argparse.Namespace], int]] = {
    "version": Commands.version,
    "settings": Commands.show_settings,
    "serve": Commands.serve,
    "enrol": Commands.enrol,
    "import": Commands.import_file,
    "sessions": Commands.sessions,
    "show": Commands.show,
    "export": Commands.export,
    "publish": Commands.publish,
    "speakers": Commands.speakers,
    "enrolment-text": Commands.enrolment_text,
}


def main(
    argv: Optional[Sequence[str]] = None,
    environ: Optional[Mapping[str, str]] = None,
    out: Optional[TextIO] = None,
    err: Optional[TextIO] = None,
) -> int:
    """Run one command; return its exit code.

    0 on success, 1 when the transcriber refused or failed (the message
    goes to the error stream), 2 for a usage error.
    """
    output = Output(out or sys.stdout, err or sys.stderr)
    args = build_parser().parse_args(argv)
    try:
        settings = load_settings(os.environ if environ is None else environ)
    except (OSError, ValueError, ValidationError) as exc:
        output.note(f"transcriber: invalid settings: {exc}")
        return 2
    commands = Commands(settings, output)
    try:
        return HANDLERS[args.command](commands, args)
    except UsageError as exc:
        output.note(f"transcriber {args.command}: {exc}")
        return 2
    except (TranscriberError, OSError) as exc:
        output.note(f"transcriber {args.command}: error: {exc}")
        return 1
    finally:
        commands.close()
