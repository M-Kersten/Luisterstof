"""Where every artifact of a book lives on disk.

Layout (per the build brief):

    data/books/<book_id>/
      source.pdf
      book.json
      glossary.json
      plans/ch03.plan.json
      scripts/ch03.script.json
      audits/ch03.audit.json
      render/ch03/blocks/*.mp3
      out/ch03.mp3
      out/ch03.transcript.json
"""

from __future__ import annotations

import re
from pathlib import Path

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def validate_id(value: str, what: str = "id") -> str:
    if not _SAFE_ID.match(value) or ".." in value:
        raise ValueError(f"invalid {what}: {value!r}")
    return value


class BookPaths:
    def __init__(self, data_dir: Path | str, book_id: str):
        self.data_dir = Path(data_dir)
        self.book_id = validate_id(book_id, "book_id")
        self.root = self.data_dir / "books" / self.book_id

    # Book level ---------------------------------------------------------
    @property
    def source_pdf(self) -> Path:
        return self.root / "source.pdf"

    @property
    def book_json(self) -> Path:
        return self.root / "book.json"

    @property
    def glossary_json(self) -> Path:
        return self.root / "glossary.json"

    @property
    def plans_dir(self) -> Path:
        return self.root / "plans"

    @property
    def scripts_dir(self) -> Path:
        return self.root / "scripts"

    @property
    def audits_dir(self) -> Path:
        return self.root / "audits"

    @property
    def render_dir(self) -> Path:
        return self.root / "render"

    @property
    def cache_dir(self) -> Path:
        return self.render_dir / "cache"

    @property
    def out_dir(self) -> Path:
        return self.root / "out"

    @property
    def figures_dir(self) -> Path:
        return self.root / "figures"

    # Chapter level ------------------------------------------------------
    def plan(self, chapter_id: str) -> Path:
        return self.plans_dir / f"{validate_id(chapter_id, 'chapter_id')}.plan.json"

    def script(self, chapter_id: str) -> Path:
        return self.scripts_dir / f"{validate_id(chapter_id, 'chapter_id')}.script.json"

    def audit(self, chapter_id: str) -> Path:
        return self.audits_dir / f"{validate_id(chapter_id, 'chapter_id')}.audit.json"

    def render_chapter(self, chapter_id: str) -> Path:
        return self.render_dir / validate_id(chapter_id, "chapter_id")

    def blocks_dir(self, chapter_id: str) -> Path:
        return self.render_chapter(chapter_id) / "blocks"

    def lines_dir(self, chapter_id: str) -> Path:
        return self.render_chapter(chapter_id) / "lines"

    def manifest(self, chapter_id: str, tier: str) -> Path:
        return self.render_chapter(chapter_id) / f"{tier}.manifest.json"

    def out_audio(self, chapter_id: str, tier: str = "final") -> Path:
        suffix = ".mp3" if tier == "final" else f".{tier}.mp3"
        return self.out_dir / f"{validate_id(chapter_id, 'chapter_id')}{suffix}"

    def transcript(self, chapter_id: str, tier: str = "final") -> Path:
        suffix = ".transcript.json" if tier == "final" else f".{tier}.transcript.json"
        return self.out_dir / f"{validate_id(chapter_id, 'chapter_id')}{suffix}"

    def approval(self, chapter_id: str) -> Path:
        return self.audits_dir / f"{validate_id(chapter_id, 'chapter_id')}.approved.json"

    # Helpers ------------------------------------------------------------
    def ensure(self) -> BookPaths:
        for d in (self.root, self.plans_dir, self.scripts_dir, self.audits_dir, self.render_dir,
                  self.cache_dir, self.out_dir, self.figures_dir):
            d.mkdir(parents=True, exist_ok=True)
        return self

    def resolve_artifact(self, relative: str) -> Path:
        """Resolve a user-supplied relative path inside the book folder, refusing traversal."""
        candidate = (self.root / relative).resolve()
        root = self.root.resolve()
        if root != candidate and root not in candidate.parents:
            raise ValueError("artifact path escapes the book folder")
        return candidate

    def list_artifacts(self) -> list[dict]:
        if not self.root.exists():
            return []
        items = []
        for p in sorted(self.root.rglob("*")):
            if p.is_file() and "cache" not in p.relative_to(self.root).parts:
                items.append({
                    "path": p.relative_to(self.root).as_posix(),
                    "bytes": p.stat().st_size,
                    "mtime": p.stat().st_mtime,
                })
        return items


def list_books(data_dir: Path | str) -> list[str]:
    base = Path(data_dir) / "books"
    if not base.exists():
        return []
    return sorted(p.name for p in base.iterdir() if p.is_dir() and _SAFE_ID.match(p.name))
