"""How long each pass took: plan, lexicon, writing, every audit, every rewrite round, the table read.

Every pass appends one line to ``data/books/<book>/timings.jsonl`` when it ends, done, failed, stopped
or skipped, so a run that went on all night can be read back in the morning, also after the web app was
closed. A run is one job (or one CLI command); its id groups the passes it made.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

PASS_NAMES = {
    "ingest": "inlezen", "plan": "plan", "glossary": "lexicon", "script": "script schrijven", "audit": "audit",
    "revise": "herschrijven", "polish": "tafellezing", "continuity": "continuïteit", "render": "render",
    "chapter": "hele hoofdstuk",
}


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def new_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")


class PassTiming(BaseModel):
    run: str
    book_id: str
    chapter: str | None = None
    pass_name: str  # plan | glossary | script | audit | revise | polish | continuity | render | ingest | chapter
    label: str = ""  # "ronde 2", "na de tafellezing", ...
    status: str = "done"  # done | failed | stopped | skipped
    started_at: str
    finished_at: str
    seconds: float = 0.0
    llm_calls: int = 0

    @property
    def title(self) -> str:
        name = PASS_NAMES.get(self.pass_name, self.pass_name)
        return f"{name} ({self.label})" if self.label else name


def append(path: Path, entry: PassTiming) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(entry.model_dump_json() + "\n")


def load(path: Path) -> list[PassTiming]:
    if not path.is_file():
        return []
    out = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(PassTiming.model_validate(json.loads(raw)))
        except (ValueError, TypeError):
            continue  # a line cut off by a crash
    return out


def summary(entries: list[PassTiming]) -> dict:
    """Per chapter, the passes of the latest run that touched it; per book, when that run started and ended."""
    chapters: dict[str, dict] = {}
    for e in entries:
        if not e.chapter:
            continue
        current = chapters.get(e.chapter)
        if current is None or e.run > current["run"]:
            chapters[e.chapter] = current = {"run": e.run, "passes": [], "seconds": 0.0}
        if e.run == current["run"]:
            current["passes"].append(e.model_dump())
            if e.pass_name != "chapter" and e.status != "skipped":
                current["seconds"] += e.seconds
    latest = max((e.run for e in entries), default=None)
    in_latest = [e for e in entries if e.run == latest]
    book = None
    if in_latest:
        started = min(e.started_at for e in in_latest)
        finished = max(e.finished_at for e in in_latest)
        book = {"run": latest, "started_at": started, "finished_at": finished,
                "seconds": (datetime.fromisoformat(finished) - datetime.fromisoformat(started)).total_seconds()}
    return {"chapters": chapters, "latest_run": book}


def fmt_seconds(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours} u {minutes:02d} min"
    if minutes:
        return f"{minutes} min {secs:02d} s"
    return f"{secs} s"
