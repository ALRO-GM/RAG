"""File-system side of the site: read the repository, write ``<out>/``.

Output layout (everything the browser fetches is relative)::

    <out>/index.html, app.js, style.css   copied from ``web/``
    <out>/data/manifest.json              build info + documentation index
    <out>/data/lab.json, beir.json, ...   summaries (see ``summaries``)
    <out>/raw/<repo path>                 the raw files, for download
    <out>/docs/<repo path>.md             Markdown rendered by the browser
"""

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, List, Optional

from results_site import SITE_FORMAT_VERSION, catalog
from results_site.summaries import (
    Record,
    SiteError,
    doc_title,
    parse_jsonl,
    summarize_beir,
    summarize_hardware,
    summarize_rubric,
    summarize_trials,
)


class SiteBuilder:
    """Build the static site from a repository checkout.

    Args:
        root: Repository root (sources are resolved against it).
        out: Output directory; it is emptied first, so it must not be
            the repository itself or one of its parents.
        web: Directory with the static front end (default ``root/web``).
    """

    def __init__(self, root: Path, out: Path, web: Optional[Path] = None):
        """Remember the directories; nothing is written yet."""
        self.root = root.resolve()
        self.out = out.resolve()
        self.web = (web or self.root / "web").resolve()
        if self.out == self.root or self.out in self.root.parents:
            raise SiteError(f"Refusing to build into {self.out}.")

    def build(
        self,
        commit: str = "",
        now: Optional[datetime] = None,
    ) -> Record:
        """Write the whole site and return its manifest."""
        if not (self.web / "index.html").is_file():
            raise SiteError(f"No front end at {self.web}/index.html.")
        if self.out.exists():
            shutil.rmtree(self.out)
        shutil.copytree(self.web, self.out)
        sections = {
            "lab": self._jsonl(catalog.LAB_TRIALS, summarize_trials),
            "beir": self._jsonl(catalog.BEIR_ROWS, summarize_beir),
            "skills": self._jsonl(catalog.SKILLS_TRIALS, summarize_trials),
            "hardware": self._jsonl(
                catalog.HARDWARE_RUNS, summarize_hardware
            ),
        }
        available = {k: v is not None for k, v in sections.items()}
        for key, value in sections.items():
            if value is not None:
                self._write_json(f"data/{key}.json", value)
        rubrics = self._rubrics()
        self._write_json("data/rubrics.json", rubrics)
        raw = self._publish_raw()
        manifest: Record = {
            "format": SITE_FORMAT_VERSION,
            "generated": (now or datetime.now(timezone.utc))
            .replace(microsecond=0).isoformat(),
            "commit": commit,
            "sections": available,
            "docs": self._docs(),
            "rubrics": len(rubrics),
            "raw": raw,
        }
        self._write_json("data/manifest.json", manifest)
        return manifest

    # -- sources ---------------------------------------------------------

    def _jsonl(
        self, rel: str, summarize: Callable[[List[Record]], Record]
    ) -> Optional[Record]:
        """Summarize a JSON Lines source, or ``None`` when it is absent."""
        path = self.root / rel
        if not path.is_file():
            return None
        rows = parse_jsonl(path.read_text(encoding="utf-8"), rel)
        return summarize(rows)

    def _rubrics(self) -> List[Record]:
        """Copy every rubric and return their index entries."""
        entries: List[Record] = []
        for path in sorted(self.root.glob(catalog.RUBRICS)):
            rel = path.relative_to(self.root).as_posix()
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise SiteError(f"{rel}: invalid JSON ({exc.msg}).")
            if not isinstance(data, dict):
                raise SiteError(f"{rel}: expected a JSON object.")
            entries.append(summarize_rubric(data, rel))
            self._copy(path, f"raw/{rel}")
        return entries

    def _docs(self) -> List[Record]:
        """Copy the Markdown documents and return the grouped index."""
        groups: List[Record] = []
        seen: set[str] = set()
        for group in catalog.DOC_GROUPS:
            items: List[Record] = []
            for pattern in group.patterns:
                for path in sorted(self.root.glob(pattern)):
                    rel = path.relative_to(self.root).as_posix()
                    if rel in seen or not path.is_file():
                        continue
                    seen.add(rel)
                    text = path.read_text(encoding="utf-8")
                    self._copy(path, f"docs/{rel}")
                    items.append({
                        "path": rel,
                        "title": doc_title(text, path.stem),
                        "bytes": len(text.encode("utf-8")),
                    })
            if items:
                groups.append({
                    "key": group.key, "title": group.title, "items": items,
                })
        return groups

    def _publish_raw(self) -> List[str]:
        """Copy the raw result files that exist; return their paths."""
        patterns = [
            catalog.LAB_TRIALS, catalog.LAB_TUNING, catalog.LAB_PROFILES,
            catalog.BEIR_ROWS, catalog.SKILLS_TRIALS, catalog.HARDWARE_RUNS,
            catalog.GENERAL_ROWS, catalog.AUDIO_MEETINGS,
        ]
        published: List[str] = []
        for pattern in patterns:
            for path in sorted(self.root.glob(pattern)):
                rel = path.relative_to(self.root).as_posix()
                self._copy(path, f"raw/{rel}")
                published.append(rel)
        return published

    # -- output ----------------------------------------------------------

    def _copy(self, source: Path, rel: str) -> None:
        """Copy ``source`` to ``<out>/<rel>``."""
        target = self.out / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)

    def _write_json(self, rel: str, value: Any) -> None:
        """Write compact UTF-8 JSON to ``<out>/<rel>``."""
        target = self.out / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(value, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
