"""PDF extraction with pymupdf.

Produces a list of PageData objects: text lines with font information (for
heading detection), plus figures and tables already converted to text at
their position in the reading order. Structure detection lives in
``structure.py``; this module knows nothing about chapters.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

try:  # silence the layout-package advertisement
    pymupdf.TOOLS.unset_quad_corrections(True)
except Exception:  # pragma: no cover
    pass

from pipeline.ingest.figures import Captioner, caption_kind, figure_to_text, is_caption, table_to_text

log = logging.getLogger(__name__)

BlockKind = str  # "text" | "figure" | "table"


@dataclass
class TextLine:
    text: str
    size: float
    bold: bool
    bbox: tuple[float, float, float, float]
    font: str = ""

    @property
    def y(self) -> float:
        return self.bbox[1]


@dataclass
class Block:
    kind: BlockKind
    page: int  # 1-based
    bbox: tuple[float, float, float, float]
    lines: list[TextLine] = field(default_factory=list)
    text: str = ""  # for figure/table blocks: the spoken text
    image_path: str | None = None

    @property
    def y(self) -> float:
        return self.bbox[1]


@dataclass
class PageData:
    number: int  # 1-based
    width: float
    height: float
    blocks: list[Block] = field(default_factory=list)

    def lines(self) -> list[TextLine]:
        return [ln for b in self.blocks if b.kind == "text" for ln in b.lines]


@dataclass
class PdfMeta:
    title: str | None
    page_count: int
    toc: list[list]  # [[level, title, page], ...]


_HYPHEN_END = re.compile(r"(\w)[-‐]$")
_PAGE_NUMBER = re.compile(r"^\s*(pagina\s+)?\d{1,4}\s*$", re.IGNORECASE)


def read_meta(pdf_path: Path | str) -> PdfMeta:
    with pymupdf.open(pdf_path) as doc:
        title = (doc.metadata or {}).get("title") or None
        return PdfMeta(title=title.strip() if title else None, page_count=doc.page_count, toc=doc.get_toc())


def _span_bold(span: dict) -> bool:
    flags = span.get("flags", 0)
    font = (span.get("font") or "").lower()
    return bool(flags & 16) or "bold" in font or "black" in font or "heavy" in font


def _line_from_dict(line: dict) -> TextLine | None:
    spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
    if not spans:
        return None
    text = "".join(s["text"] for s in line["spans"]).strip()
    text = re.sub(r"[ \t]+", " ", text)
    weights = Counter()
    for s in spans:
        weights[round(float(s.get("size", 0)), 1)] += len(s["text"].strip())
    size = weights.most_common(1)[0][0]
    total = sum(len(s["text"].strip()) for s in spans)
    bold_chars = sum(len(s["text"].strip()) for s in spans if _span_bold(s))
    x0, y0, x1, y1 = line["bbox"]
    return TextLine(text=text, size=size, bold=bold_chars >= 0.6 * total, bbox=(x0, y0, x1, y1),
                    font=spans[0].get("font", ""))


def _inside(inner: tuple, outer: tuple, tol: float = 2.0) -> bool:
    return inner[0] >= outer[0] - tol and inner[1] >= outer[1] - tol and inner[2] <= outer[2] + tol and inner[3] <= outer[3] + tol


def _extract_tables(page: pymupdf.Page) -> list[tuple[tuple, list[list]]]:
    try:
        found = page.find_tables()
    except Exception as exc:  # pragma: no cover - depends on pymupdf build
        log.debug("find_tables failed on page %s: %s", page.number, exc)
        return []
    tables = []
    for t in found.tables:
        try:
            rows = t.extract()
        except Exception:
            continue
        if rows and len(rows) >= 2 and max(len(r) for r in rows) >= 2:
            tables.append((tuple(t.bbox), rows))
    return tables


def _figure_regions(page: pymupdf.Page, min_area_share: float = 0.04) -> list[tuple]:
    """Bounding boxes of images and vector drawings large enough to matter."""
    page_area = max(page.rect.width * page.rect.height, 1.0)
    regions: list[tuple] = []
    for info in page.get_image_info():
        bbox = tuple(info["bbox"])
        area = max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])
        if area / page_area >= min_area_share:
            regions.append(bbox)
    try:
        for rect in page.cluster_drawings():
            bbox = tuple(rect)
            area = rect.width * rect.height
            if area / page_area >= min_area_share and not any(_inside(bbox, r) for r in regions):
                regions.append(bbox)
    except Exception:  # pragma: no cover
        pass
    return regions


def _nearest_caption(lines: list[TextLine], bbox: tuple, kind: str, max_dist: float = 60.0) -> TextLine | None:
    best, best_d = None, max_dist
    for ln in lines:
        if not is_caption(ln.text) or caption_kind(ln.text) != kind:
            continue
        d = min(abs(ln.bbox[1] - bbox[3]), abs(bbox[1] - ln.bbox[3]))
        if d < best_d:
            best, best_d = ln, d
    return best


def extract_pages(
    pdf_path: Path | str,
    *,
    captioner: Captioner | None = None,
    figures_dir: Path | None = None,
    max_pages: int | None = None,
) -> list[PageData]:
    pages: list[PageData] = []
    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            if max_pages is not None and page.number >= max_pages:
                break
            pages.append(_extract_page(page, captioner, figures_dir))
    _strip_running_headers(pages)
    return pages


def _extract_page(page: pymupdf.Page, captioner: Captioner | None, figures_dir: Path | None) -> PageData:
    number = page.number + 1
    data = PageData(number=number, width=page.rect.width, height=page.rect.height)
    raw = page.get_text("dict", sort=True)
    tables = _extract_tables(page)
    table_boxes = [bbox for bbox, _ in tables]
    figure_boxes = _figure_regions(page)

    text_blocks: list[Block] = []
    for b in raw.get("blocks", []):
        if b.get("type") != 0:
            continue
        bbox = tuple(b["bbox"])
        if any(_inside(bbox, tb) for tb in table_boxes):
            continue  # table text is rendered by table_to_text instead
        lines = [ln for ln in (_line_from_dict(l) for l in b.get("lines", [])) if ln]
        if lines:
            text_blocks.append(Block(kind="text", page=number, bbox=bbox, lines=lines))

    all_lines = [ln for blk in text_blocks for ln in blk.lines]
    used_captions: set[int] = set()
    extra_blocks: list[Block] = []

    for bbox, rows in tables:
        cap = _nearest_caption(all_lines, bbox, "table")
        cap_text = cap.text if cap else None
        if cap:
            used_captions.add(id(cap))
        text = table_to_text(rows, cap_text)
        if text:
            extra_blocks.append(Block(kind="table", page=number, bbox=bbox, text=text))

    for i, bbox in enumerate(figure_boxes):
        cap = _nearest_caption(all_lines, bbox, "figure")
        cap_text = cap.text if cap else None
        if cap:
            used_captions.add(id(cap))
        description = None
        image_path = None
        if captioner is not None or figures_dir is not None:
            try:
                pix = page.get_pixmap(clip=pymupdf.Rect(bbox), dpi=110)
                png = pix.tobytes("png")
            except Exception as exc:  # pragma: no cover
                log.warning("could not rasterise figure on page %s: %s", number, exc)
                png = None
            if png and figures_dir is not None:
                figures_dir.mkdir(parents=True, exist_ok=True)
                out = figures_dir / f"p{number:04d}_f{i + 1}.png"
                out.write_bytes(png)
                image_path = str(out)
            if png and captioner is not None:
                context = " ".join(ln.text for ln in all_lines if abs(ln.bbox[1] - bbox[1]) < 250)
                try:
                    description = captioner.describe(png, cap_text, context)
                except Exception as exc:
                    log.warning("captioner failed on page %s: %s", number, exc)
        text = figure_to_text(cap_text, description)
        if text and (cap_text or description):
            extra_blocks.append(Block(kind="figure", page=number, bbox=bbox, text=text, image_path=image_path))

    # Drop caption lines that were absorbed into a figure/table block.
    for blk in text_blocks:
        blk.lines = [ln for ln in blk.lines if id(ln) not in used_captions]
    text_blocks = [b for b in text_blocks if b.lines]

    data.blocks = sorted(text_blocks + extra_blocks, key=lambda b: (round(b.bbox[1] / 4), b.bbox[0]))
    return data


def _strip_running_headers(pages: list[PageData], band: float = 0.08, min_share: float = 0.3) -> None:
    """Remove page numbers and running heads that repeat across pages."""
    if len(pages) < 4:
        for p in pages:
            _drop_lines(p, lambda ln: _PAGE_NUMBER.match(ln.text) is not None)
        return
    counter: Counter[str] = Counter()
    for p in pages:
        seen: set[str] = set()
        for ln in p.lines():
            if ln.y < p.height * band or ln.bbox[3] > p.height * (1 - band):
                key = re.sub(r"\d+", "#", ln.text.strip().lower())
                seen.add(key)
        counter.update(seen)
    threshold = max(2, int(len(pages) * min_share))
    repeated = {k for k, n in counter.items() if n >= threshold}

    def is_running(ln: TextLine, p: PageData) -> bool:
        in_band = ln.y < p.height * band or ln.bbox[3] > p.height * (1 - band)
        if not in_band:
            return False
        key = re.sub(r"\d+", "#", ln.text.strip().lower())
        return key in repeated or _PAGE_NUMBER.match(ln.text) is not None

    for p in pages:
        _drop_lines(p, lambda ln, p=p: is_running(ln, p))


def _drop_lines(page: PageData, predicate) -> None:
    for blk in page.blocks:
        if blk.kind == "text":
            blk.lines = [ln for ln in blk.lines if not predicate(ln)]
    page.blocks = [b for b in page.blocks if b.kind != "text" or b.lines]


def body_font_size(pages: list[PageData]) -> float:
    weights: Counter[float] = Counter()
    for p in pages:
        for ln in p.lines():
            weights[round(ln.size * 2) / 2] += len(ln.text)
    if not weights:
        return 10.0
    return weights.most_common(1)[0][0]


def join_lines(lines: list[str]) -> str:
    """Join physical lines into flowing text, repairing hyphenation."""
    out = ""
    for raw in lines:
        piece = raw.strip()
        if not piece:
            continue
        if out and _HYPHEN_END.search(out):
            out = _HYPHEN_END.sub(r"\1", out) + piece
        elif out:
            out += " " + piece
        else:
            out = piece
    return out


def block_text(block: Block) -> str:
    if block.kind != "text":
        return block.text
    return join_lines([ln.text for ln in block.lines])


def detect_language(pages: list[PageData]) -> str:
    nl = {"de", "het", "een", "en", "van", "is", "dat", "niet", "zijn", "op", "voor", "met", "ook", "wordt"}
    en = {"the", "and", "of", "is", "that", "not", "are", "on", "for", "with", "also", "this"}
    counts = Counter()
    for p in pages[:40]:
        for ln in p.lines():
            for w in re.findall(r"[a-zA-Z]+", ln.text.lower()):
                if w in nl:
                    counts["nl"] += 1
                if w in en:
                    counts["en"] += 1
    return "en" if counts["en"] > counts["nl"] * 1.5 else "nl"
