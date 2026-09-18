"""Chapter and section detection.

Order of strategies (per the brief): font-size heuristics first, TOC parse if
the PDF has one, LLM fallback on the first pages when both fail. Each strategy
produces a list of Heading objects positioned in the flattened unit stream;
``build_book`` turns headings plus units into a Book.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from statistics import median

from pydantic import BaseModel, Field
from rapidfuzz import fuzz

from pipeline.ingest.pdf import PageData, body_font_size, join_lines
from pipeline.models import Book, Chapter, Section

log = logging.getLogger(__name__)

NUM_H1_WORD = re.compile(r"^(hoofdstuk|chapter|deel|part|les|module)\s+(\d{1,3}|[ivxlc]{1,6})\b[:.]?\s*(.*)$", re.I)
NUM_H1_PLAIN = re.compile(r"^(\d{1,2})[.:]?\s+(\S.*)$")
NUM_H2 = re.compile(r"^(\d{1,2})\.(\d{1,2})[.:]?\s+(\S.*)$")
NUM_H3 = re.compile(r"^(\d{1,2})\.(\d{1,2})\.(\d{1,2})[.:]?\s+(\S.*)$")
NUMBERING_PREFIX = re.compile(r"^((hoofdstuk|chapter|deel|part|les|module)\s+)?(\d{1,3}(\.\d{1,3}){0,2}|[ivxlc]{1,6})[.:]?\s+", re.I)

MATH_CHARS = set("=+−*/^_∑∫√≈≤≥±∂∇∆∏αβγδεζηθικλμνξπρστυφχψωΓΔΘΛΞΠΣΦΨΩ∞|")


@dataclass
class Unit:
    index: int
    page: int
    kind: str  # line | figure | table
    text: str
    size: float
    bold: bool
    block_id: int


@dataclass
class Heading:
    text: str
    level: int  # 1 or 2
    page: int
    order: int  # index of the unit where the heading starts
    size: float = 0.0
    span: int = 1  # number of units the heading occupies

    @property
    def clean_title(self) -> str:
        return strip_numbering(self.text)


def strip_numbering(text: str) -> str:
    cleaned = NUMBERING_PREFIX.sub("", text.strip(), count=1).strip()
    return cleaned or text.strip()


def flatten(pages: list[PageData]) -> list[Unit]:
    units: list[Unit] = []
    block_id = 0
    for page in pages:
        for block in page.blocks:
            block_id += 1
            if block.kind == "text":
                for ln in block.lines:
                    units.append(Unit(len(units), page.number, "line", ln.text, ln.size, ln.bold, block_id))
            else:
                units.append(Unit(len(units), page.number, block.kind, block.text, 0.0, False, block_id))
    return units


def formula_density(text: str) -> float:
    if not text:
        return 0.0
    math = sum(1 for c in text if c in MATH_CHARS)
    digits = sum(1 for c in text if c.isdigit())
    return round((math + 0.25 * digits) / len(text), 4)


# ---------------------------------------------------------------------------
# Strategy 1: fonts
# ---------------------------------------------------------------------------

def _looks_like_heading_text(text: str) -> bool:
    words = text.split()
    if not words or len(text) > 120 or len(words) > 14:
        return False
    if re.fullmatch(r"[\d.\s]+", text):
        return False
    if text.endswith((",", ";", ":")):
        return False
    if text.endswith(".") and not NUMBERING_PREFIX.match(text):
        return False
    return True


def headings_from_fonts(units: list[Unit], body_size: float) -> list[Heading]:
    candidates: list[Heading] = []
    block_sizes = Counter(u.block_id for u in units if u.kind == "line")
    for u in units:
        if u.kind != "line" or not _looks_like_heading_text(u.text):
            continue
        big = u.size >= body_size * 1.15
        level: int | None = None
        if NUM_H3.match(u.text):
            continue
        if NUM_H1_WORD.match(u.text) and (big or u.bold):
            level = 1
        elif NUM_H2.match(u.text) and (big or u.bold):
            level = 2
        elif NUM_H1_PLAIN.match(u.text) and (big or u.bold):
            level = 1
        elif big:
            level = 0  # size decides later
        elif u.bold and u.size >= body_size and len(u.text.split()) <= 8 and block_sizes[u.block_id] == 1:
            level = 2
        if level is not None:
            candidates.append(Heading(u.text, level, u.page, u.index, u.size))

    if not candidates:
        return []

    # Anchor unnumbered sizes on numbered headings when available.
    l1_sizes = sorted({c.size for c in candidates if c.level == 1}, reverse=True)
    l2_sizes = sorted({c.size for c in candidates if c.level == 2}, reverse=True)
    undecided_sizes = sorted({c.size for c in candidates if c.level == 0}, reverse=True)
    for c in candidates:
        if c.level != 0:
            continue
        if l1_sizes and c.size >= min(l1_sizes) - 0.25:
            c.level = 1
        elif l2_sizes and c.size <= max(l2_sizes) + 0.25:
            c.level = 2
        elif undecided_sizes and c.size >= undecided_sizes[0] - 0.25 and c.size >= body_size * 1.3:
            c.level = 1
        else:
            c.level = 2

    # Merge multi-line headings (same level, same size, consecutive units, same page).
    merged: list[Heading] = []
    for c in candidates:
        prev = merged[-1] if merged else None
        if prev and prev.level == c.level and prev.page == c.page and abs(prev.size - c.size) < 0.3 \
                and c.order == prev.order + prev.span and not NUMBERING_PREFIX.match(c.text):
            prev.text = f"{prev.text} {c.text}"
            prev.span += 1
        else:
            merged.append(c)
    return merged


# ---------------------------------------------------------------------------
# Strategy 2: TOC bookmarks
# ---------------------------------------------------------------------------

def _find_title_unit(units: list[Unit], title: str, page: int, window: int = 1, min_score: float = 80.0) -> Unit | None:
    target = strip_numbering(title).casefold()
    best, best_score = None, min_score
    for u in units:
        if u.kind != "line" or abs(u.page - page) > window:
            continue
        cand = strip_numbering(u.text).casefold()
        score = fuzz.ratio(cand, target)
        if score > best_score or (score == best_score and best is not None and u.page == page and best.page != page):
            best, best_score = u, score
    return best


def headings_from_toc(toc: list[list], units: list[Unit]) -> list[Heading]:
    headings: list[Heading] = []
    for entry in toc:
        if len(entry) < 3:
            continue
        level, title, page = int(entry[0]), str(entry[1]).strip(), int(entry[2])
        if level > 2 or not title or page < 1:
            continue
        unit = _find_title_unit(units, title, page)
        if unit is None:
            first = next((u for u in units if u.page >= page), None)
            if first is None:
                continue
            headings.append(Heading(title, level, page, first.order if False else first.index, 0.0, span=0))
        else:
            headings.append(Heading(unit.text, level, unit.page, unit.index, unit.size))
    headings.sort(key=lambda h: h.order)
    return headings


# ---------------------------------------------------------------------------
# Strategy 3: LLM on the first pages
# ---------------------------------------------------------------------------

class ChapterGuess(BaseModel):
    title: str = Field(description="Titel van het hoofdstuk zoals gedrukt, zonder nummer.")
    number: str | None = Field(description="Hoofdstuknummer zoals gedrukt, of null.")
    printed_page: int | None = Field(description="Paginanummer uit de inhoudsopgave, of null als onbekend.")


class StructureGuess(BaseModel):
    has_printed_toc: bool
    chapters: list[ChapterGuess]


STRUCTURE_SYSTEM = (
    "Je krijgt de eerste pagina's van een Nederlands studieboek, met paginamarkeringen. "
    "Bepaal de hoofdstukindeling. Gebruik de gedrukte inhoudsopgave als die er is; anders leid je "
    "hoofdstukken af uit de koppen. Geef alleen hoofdstukken op het hoogste niveau, geen paragrafen. "
    "Voorwerk (voorwoord, inhoudsopgave, inleiding tot het boek) hoort er niet bij."
)


def headings_from_llm(units: list[Unit], llm, max_pages: int = 30, max_chars: int = 60000) -> list[Heading]:
    from pipeline.llm import LLMRequest

    parts: list[str] = []
    total = 0
    current_page = None
    for u in units:
        if u.page > max_pages or total > max_chars:
            break
        if u.page != current_page:
            current_page = u.page
            marker = f"\n=== pagina {u.page} ===\n"
            parts.append(marker)
            total += len(marker)
        parts.append(u.text + "\n")
        total += len(u.text) + 1
    request = LLMRequest(
        task="structure",
        system=STRUCTURE_SYSTEM,
        user="".join(parts),
        schema=StructureGuess,
        max_tokens=4096,
        effort="medium",
        cache_system=False,
    )
    guess = llm.generate(request)
    assert isinstance(guess, StructureGuess)

    headings: list[Heading] = []
    offset: int | None = None
    last_index = -1
    for ch in guess.chapters:
        title = ch.title.strip()
        if not title:
            continue
        expected_page = (ch.printed_page + (offset or 0)) if ch.printed_page else None
        best, best_score = None, 82.0
        target = title.casefold()
        for u in units:
            if u.kind != "line" or u.index <= last_index:
                continue
            if expected_page is not None and abs(u.page - expected_page) > 3 and offset is not None:
                continue
            cand = strip_numbering(u.text).casefold()
            if abs(len(cand) - len(target)) > max(12, len(target)):
                continue
            score = fuzz.ratio(cand, target)
            if score > best_score:
                best, best_score = u, score
        if best is None:
            log.warning("LLM chapter %r not located in text", title)
            continue
        if ch.printed_page and offset is None:
            offset = best.page - ch.printed_page
        headings.append(Heading(best.text, 1, best.page, best.index, best.size))
        last_index = best.index
    return headings


# ---------------------------------------------------------------------------
# Validation and book assembly
# ---------------------------------------------------------------------------

def _chapter_char_counts(units: list[Unit], headings: list[Heading]) -> list[int]:
    l1 = [h for h in headings if h.level == 1]
    counts = []
    for i, h in enumerate(l1):
        end = l1[i + 1].order if i + 1 < len(l1) else len(units)
        counts.append(sum(len(u.text) for u in units[h.order + h.span : end]))
    return counts


def validate_headings(units: list[Unit], headings: list[Heading], min_chapters: int = 2,
                      max_chapters: int = 120, min_median_chars: int = 600) -> bool:
    counts = _chapter_char_counts(units, headings)
    if not (min_chapters <= len(counts) <= max_chapters):
        return False
    return median(counts) >= min_median_chars


def build_book(
    book_id: str,
    title: str,
    units: list[Unit],
    headings: list[Heading],
    *,
    language: str = "nl",
    page_count: int | None = None,
    method: str = "unknown",
    min_chapter_chars: int = 300,
    front_matter_min_chars: int = 200,
) -> Book:
    headings = sorted(headings, key=lambda h: h.order)
    heading_at: dict[int, Heading] = {h.order: h for h in headings}

    # Walk the unit stream into chapters -> sections -> paragraphs.
    chapters: list[dict] = []
    current: dict | None = None
    section: dict | None = None

    def new_chapter(h: Heading | None) -> dict:
        ch = {"title": h.clean_title if h else title, "level": 1 if h else 0, "sections": [], "heading": h}
        chapters.append(ch)
        return ch

    def new_section(ch: dict, h: Heading | None) -> dict:
        sec = {"title": h.clean_title if h else None, "paragraphs": [], "pages": [], "figures": 0, "tables": 0,
               "current_block": None, "buffer": []}
        ch["sections"].append(sec)
        return sec

    def flush(sec: dict | None) -> None:
        if sec and sec["buffer"]:
            sec["paragraphs"].append(join_lines(sec["buffer"]))
            sec["buffer"] = []

    skip_until = -1
    for u in units:
        if u.index < skip_until:
            continue
        h = heading_at.get(u.index)
        if h is not None:
            skip_until = u.index + h.span
            flush(section)
            if h.level == 1 or current is None:
                current = new_chapter(h if h.level == 1 else None)
                section = new_section(current, h if h.level == 2 else None)
            else:
                section = new_section(current, h)
            if h.level == 1:
                continue
            if h.span == 0:
                # TOC entry without a located title line: keep the unit as text.
                pass
            else:
                continue
        if current is None:
            current = new_chapter(None)
            section = new_section(current, None)
        assert section is not None
        if u.kind == "line":
            if section["current_block"] != u.block_id:
                flush(section)
                section["current_block"] = u.block_id
            section["buffer"].append(u.text)
        else:
            flush(section)
            section["paragraphs"].append(u.text)
            section["figures" if u.kind == "figure" else "tables"] += 1
        section["pages"].append(u.page)
    flush(section)

    # Fold tiny chapters: the first becomes front matter, later ones merge backwards.
    folded: list[dict] = []
    for ch in chapters:
        text_len = sum(len(p) for s in ch["sections"] for p in s["paragraphs"])
        if ch["level"] == 1 and text_len < min_chapter_chars:
            if folded and folded[-1]["level"] == 1:
                for s in ch["sections"]:
                    if s["title"] is None:
                        s["title"] = ch["title"]
                folded[-1]["sections"].extend(ch["sections"])
                continue
            ch["level"] = 0
        folded.append(ch)
    chapters = folded

    warnings: list[str] = []
    out_chapters: list[Chapter] = []
    n_real = 0
    for ch in chapters:
        text_len = sum(len(p) for s in ch["sections"] for p in s["paragraphs"])
        if ch["level"] == 0:
            if text_len < front_matter_min_chars:
                continue
            ch_id = "ch00"
            ch_title = ch["title"] if ch["heading"] else "Voorwerk"
        else:
            n_real += 1
            ch_id = f"ch{n_real:02d}"
            ch_title = ch["title"]
        sections: list[Section] = []
        k = 0
        for s in ch["sections"]:
            text = "\n\n".join(p for p in s["paragraphs"] if p.strip())
            if not text.strip():
                continue
            k += 1
            sec_title = s["title"] or (ch_title if k == 1 else f"{ch_title} ({k})")
            pages = (min(s["pages"]), max(s["pages"])) if s["pages"] else None
            sections.append(Section(id=f"{ch_id}.{k}", title=sec_title, text=text, pages=pages,
                                    formula_density=formula_density(text), figure_count=s["figures"],
                                    table_count=s["tables"]))
        if not sections:
            continue
        p0 = min(s.pages[0] for s in sections if s.pages) if any(s.pages for s in sections) else 1
        p1 = max(s.pages[1] for s in sections if s.pages) if any(s.pages for s in sections) else p0
        out_chapters.append(Chapter(id=ch_id, title=ch_title, level=ch["level"], pages=(p0, p1), sections=sections))

    if n_real == 0:
        warnings.append("no chapters detected; whole book treated as one episode")
        if out_chapters:
            out_chapters[0].level = 1
            out_chapters[0].id = "ch01"
            for i, s in enumerate(out_chapters[0].sections, start=1):
                s.id = f"ch01.{i}"
    return Book(book_id=book_id, title=title, language=language, chapters=out_chapters, page_count=page_count,
                structure_method=method, warnings=warnings)


def detect_structure(pages: list[PageData], toc: list[list] | None, llm=None, method: str = "auto") -> tuple[list[Heading], str]:
    units = flatten(pages)
    body = body_font_size(pages)
    if method in ("auto", "fonts"):
        h = headings_from_fonts(units, body)
        if method == "fonts" or validate_headings(units, h):
            return h, "fonts"
        log.info("font heuristics rejected (%d level-1 headings)", sum(1 for x in h if x.level == 1))
    if method in ("auto", "toc") and toc:
        h = headings_from_toc(toc, units)
        if method == "toc" or validate_headings(units, h):
            return h, "toc"
        log.info("toc rejected")
    if method in ("auto", "llm") and llm is not None:
        h = headings_from_llm(units, llm)
        if method == "llm" or validate_headings(units, h):
            return h, "llm"
        log.info("llm structure rejected")
    return [], "single"
