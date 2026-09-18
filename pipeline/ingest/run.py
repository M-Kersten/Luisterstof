"""Entry point of stage 1: PDF -> Book."""

from __future__ import annotations

import logging
from pathlib import Path

from pipeline.ingest.figures import Captioner
from pipeline.ingest.pdf import detect_language, extract_pages, read_meta
from pipeline.ingest.structure import build_book, detect_structure, flatten
from pipeline.models import Book

log = logging.getLogger(__name__)


def ingest_book(
    pdf_path: Path | str,
    book_id: str,
    *,
    llm=None,
    captioner: Captioner | None = None,
    figures_dir: Path | None = None,
    method: str = "auto",
    title: str | None = None,
    language: str | None = None,
) -> Book:
    meta = read_meta(pdf_path)
    pages = extract_pages(pdf_path, captioner=captioner, figures_dir=figures_dir)
    headings, used = detect_structure(pages, meta.toc, llm=llm, method=method)
    units = flatten(pages)
    book_title = title or meta.title or _title_from_pages(pages) or Path(pdf_path).stem
    lang = language or detect_language(pages)
    book = build_book(book_id, book_title, units, headings, language=lang, page_count=meta.page_count, method=used)
    log.info("ingested %s: %d chapters via %s", book_id, len(book.episode_chapters()), used)
    return book


def _title_from_pages(pages) -> str | None:
    best = None
    for p in pages[:3]:
        for ln in p.lines():
            if best is None or ln.size > best.size:
                best = ln
    return best.text if best else None
