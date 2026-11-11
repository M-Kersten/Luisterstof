"""Entry point of stage 1: PDF -> Book."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from pipeline.ingest.pdf import detect_language, extract_pages, read_meta
from pipeline.ingest.structure import build_book, detect_structure, flatten, strip_numbering
from pipeline.models import Book

log = logging.getLogger(__name__)


def ingest_book(
    pdf_path: Path | str,
    book_id: str,
    *,
    llm=None,
    method: str = "auto",
    title: str | None = None,
    language: str | None = None,
) -> Book:
    meta = read_meta(pdf_path)
    pages = extract_pages(pdf_path)
    headings, used = detect_structure(pages, meta.toc, llm=llm, method=method)
    units = flatten(pages)
    book_title = title or meta.title or _title_from_pages(pages) or Path(pdf_path).stem
    lang = language or detect_language(pages)
    book = build_book(book_id, book_title, units, headings, language=lang, page_count=meta.page_count, method=used)
    log.info("ingested %s: %d chapters via %s", book_id, len(book.episode_chapters()), used)
    return book


def _title_from_pages(pages) -> str | None:
    """The largest text on the first pages, all its lines ("1." over "Wie was Willem van Oranje?"), without
    the numbering in front."""
    lines = [ln for p in pages[:3] for ln in p.lines()]
    if not lines:
        return None
    top = max(ln.size for ln in lines)
    run: list[str] = []
    for ln in lines:
        if abs(ln.size - top) < 0.3:
            run.append(ln.text.strip())
        elif run:
            break
    title = strip_numbering(" ".join(run)).strip()
    return title if re.search(r"[^\W\d_]{2}", title) else None
