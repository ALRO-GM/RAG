"""Inbound adapter: the command line (``python -m transcriber ...``)."""

from typing import Dict, Optional

from transcriber import __version__
from transcriber.bootstrap import TranscriberContainer, build_container


class TranscriberCLI:
    """Transcriber: from a recorded session to attributed sentences."""

    def __init__(
        self, container: Optional[TranscriberContainer] = None
    ) -> None:
        """Use ``container`` or build one with the default settings."""
        self._container = container or build_container()

    def version(self) -> str:
        """Print the transcriber version."""
        return __version__

    def settings(self) -> Dict[str, str]:
        """Print the settings in use."""
        current = self._container.settings
        return {
            "data_dir": str(current.data_dir),
            "recogniser": current.recogniser,
            "voice_model": current.voice_model,
        }
