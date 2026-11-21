"""The warm live reasoner runs Claude Code with ``--effort low``."""

from src.config import SkillsSettings
from src.skills_bootstrap import SkillsContainer


def test_live_effort_defaults_to_low_and_reaches_the_warm_backend() -> None:
    settings = SkillsSettings()
    assert settings.claude_live_effort == "low"
    backend = SkillsContainer(settings).warm_model("haiku")
    command = backend._judge.command("system")
    assert command[command.index("--effort") + 1] == "low"
