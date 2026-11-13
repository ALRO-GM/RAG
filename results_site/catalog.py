"""What the site publishes, relative to the repository root.

Adding a page for a new kind of result means adding an entry here and
a summary function in ``summaries``; the builder stays the same.
Only files committed to the repository are listed: the site must be
reproducible from a clean checkout (that is what CI builds).
"""

from typing import List, NamedTuple, Tuple


class DocGroup(NamedTuple):
    """A section of the documentation index and its glob patterns."""

    key: str
    title: str
    patterns: Tuple[str, ...]


DOC_GROUPS: List[DocGroup] = [
    DocGroup("project", "Proyecto", ("README.md", "CONTRIBUTING.md")),
    DocGroup("design", "Diseño y plan", (
        "docs/ARCHITECTURE.md", "docs/ROADMAP.md", "docs/TASKS.md",
        "docs/DEPLOYMENT.md",
    )),
    DocGroup("lab", "Laboratorio", (
        "experiments/REPORT.md", "experiments/general/*.md",
    )),
    DocGroup("benchmarks", "Benchmarks externos",
             ("docs/benchmarks/*.md",)),
    DocGroup("skills", "Soft skills", ("docs/skills/*.md",)),
]

#: Lab trials of the RAG pipeline (one JSON object per line).
LAB_TRIALS = "experiments/results/trials.jsonl"
#: Cross-validated tuning outcome, published as is.
LAB_TUNING = "experiments/results/tuning_tune_all.json"
#: Search profiles derived from the lab, published as is.
LAB_PROFILES = "experiments/profiles/*.json"
#: External benchmark rows (BEIR, CoIR).
BEIR_ROWS = "experiments/benchmarks/beir.jsonl"
#: Lab trials of the skills evidence retrieval.
SKILLS_TRIALS = "experiments/skills/results/trials.jsonl"
#: Runtime measurements (CPU plans against GPU), one run per line.
HARDWARE_RUNS = "experiments/hardware/runtime.jsonl"
#: Generalist study: benchmark rows of the suites (see GENERALIST.md).
GENERAL_ROWS = "experiments/general/benchmarks.jsonl"
#: Transcription and diarization on synthetic meetings.
AUDIO_MEETINGS = "experiments/audio/meetings.jsonl"
#: Competency rubrics.
RUBRICS = "knowledge/rubrics/*.json"
