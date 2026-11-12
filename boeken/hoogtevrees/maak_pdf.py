"""Zet hoogtevrees.md om in hoogtevrees.pdf (reportlab, DejaVu-fonts).

Opmaak afgestemd op de ingest van Luisterstof (pipeline/ingest/structure.py):
- titelpagina met de boektitel als grootste tekst;
- elk hoofdstuk op een nieuwe pagina, kop 'Hoofdstuk N Titel' in een duidelijk groter vet lettertype;
- paragrafen 'N.M Titel' in een middelgroot vet lettertype;
- lopende tekst 10,5 pt in één kolom, geen kopregels, alleen een paginanummer onderaan;
- PDF-bladwijzers voor hoofdstukken en paragrafen;
- een alinea wordt alleen aan het eind van een zin over twee pagina's verdeeld.

Gebruik: python maak_pdf.py [invoer.md] [uitvoer.pdf]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import BaseDocTemplate, Frame, PageBreak, PageTemplate, Paragraph, Spacer

HERE = Path(__file__).resolve().parent
FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")

pdfmetrics.registerFont(TTFont("DejaVuSerif", str(FONT_DIR / "DejaVuSerif.ttf")))
pdfmetrics.registerFont(TTFont("DejaVuSerif-Bold", str(FONT_DIR / "DejaVuSerif-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DejaVuSans", str(FONT_DIR / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", str(FONT_DIR / "DejaVuSans-Bold.ttf")))

BODY = ParagraphStyle("body", fontName="DejaVuSerif", fontSize=10.5, leading=15, alignment=TA_JUSTIFY,
                      spaceAfter=7, allowWidows=0, allowOrphans=0)
H1 = ParagraphStyle("h1", fontName="DejaVuSans-Bold", fontSize=18, leading=24, alignment=TA_LEFT,
                    spaceBefore=36, spaceAfter=22, keepWithNext=1)
H2 = ParagraphStyle("h2", fontName="DejaVuSans-Bold", fontSize=13, leading=17, alignment=TA_LEFT,
                    spaceBefore=16, spaceAfter=8, keepWithNext=1)
SOURCE = ParagraphStyle("source", fontName="DejaVuSerif", fontSize=8.5, leading=11.5, alignment=TA_LEFT,
                        leftIndent=0, firstLineIndent=0, spaceAfter=5)
TITLE = ParagraphStyle("title", fontName="DejaVuSans-Bold", fontSize=36, leading=44, alignment=TA_CENTER)
SUBTITLE = ParagraphStyle("subtitle", fontName="DejaVuSans", fontSize=15, leading=21, alignment=TA_CENTER)

SENTENCE_END = re.compile(r"[.!?][’”')]*$")
SENTENCE_START = re.compile(r"^[‘“'(]*[A-ZÀ-ÖØ-Þ0-9]")


def _line_text(line) -> str:
    if isinstance(line, tuple):  # plain paragraph: (extra space, [words])
        return " ".join(line[1])
    words = getattr(line, "words", [])
    return "".join(getattr(w, "text", "") for w in words)


class SentenceParagraph(Paragraph):
    """A paragraph that stays on one page, or else breaks only where a line ends a sentence.

    The ingest reads every text block as a paragraph and joins the two halves of a split paragraph only when
    the second half starts in lower case. A break after 'Gibson' | 'en Walk' would leave a sentence in two
    pieces, and even a break between sentences turns one paragraph into two. So a paragraph that does not fit
    moves to the next page whole. Only a paragraph taller than a whole page (none in this book) is split, and
    then only after a line that ends a sentence.
    """

    FULL_PAGE = A4[1] - 50 * mm  # frame height: top and bottom margins are 25 mm each

    def split(self, availWidth, availHeight):
        self.wrap(availWidth, availHeight)
        lines = self.blPara.lines
        leading = self.style.leading
        fits = int(availHeight / leading)
        if fits >= len(lines):
            return [self]
        if len(lines) * leading < self.FULL_PAGE * 0.9:
            return []  # fits on a fresh page: move it there whole
        best = 0
        for k in range(2, min(fits, len(lines) - 2) + 1):
            here = _line_text(lines[k - 1]).rstrip()
            nxt = _line_text(lines[k]).lstrip()
            if SENTENCE_END.search(here) and SENTENCE_START.match(nxt) and not re.search(r"\b[A-Z]\.$", here):
                best = k
        if best == 0:
            return []
        return Paragraph.split(self, availWidth, best * leading + 0.01)


class Heading(Paragraph):
    def __init__(self, text, style, level, key):
        super().__init__(escape(text), style)
        self.outline = (text, level, key)


class BookTemplate(BaseDocTemplate):
    def __init__(self, filename, **kw):
        super().__init__(filename, pagesize=A4, leftMargin=25 * mm, rightMargin=25 * mm, topMargin=25 * mm,
                         bottomMargin=25 * mm, **kw)
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="main",
                      leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        self.addPageTemplates([PageTemplate(id="page", frames=[frame], onPage=self._footer)])

    @staticmethod
    def _footer(canvas, doc):
        if doc.page == 1:
            return
        canvas.saveState()
        canvas.setFont("DejaVuSans", 9)
        canvas.drawCentredString(A4[0] / 2, 14 * mm, str(doc.page))
        canvas.restoreState()

    def afterFlowable(self, flowable):
        outline = getattr(flowable, "outline", None)
        if outline:
            text, level, key = outline
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text, key, level=level, closed=level == 0)


def parse(md: str):
    """Yield (kind, text) items: title, subtitle, chapter, section, sources, source, para."""
    para: list[str] = []

    def flush():
        if para:
            yield "para", " ".join(para)
            para.clear()

    for raw in md.splitlines():
        line = raw.strip()
        if not line:
            yield from flush()
            continue
        if line.startswith("# Hoofdstuk"):
            yield from flush()
            yield "chapter", line[2:].strip()
        elif line.startswith("# "):
            yield from flush()
            yield "title", line[2:].strip()
        elif line.startswith("*") and line.endswith("*") and len(line) > 2:
            yield from flush()
            yield "subtitle", line.strip("*").strip()
        elif line == "## Bronnen":
            yield from flush()
            yield "sources", "Bronnen"
        elif line.startswith("## "):
            yield from flush()
            yield "section", line[3:].strip()
        elif line.startswith("- "):
            yield from flush()
            yield "source", line[2:].strip()
        else:
            para.append(line)
    yield from flush()


def build(md_path: Path, pdf_path: Path) -> None:
    items = list(parse(md_path.read_text(encoding="utf-8")))
    title = next(t for k, t in items if k == "title")
    subtitle = next((t for k, t in items if k == "subtitle"), "")
    story = [Spacer(1, 70 * mm), Paragraph(escape(title), TITLE), Spacer(1, 10 * mm),
             Paragraph(escape(subtitle), SUBTITLE)]
    n_ch = 0
    n_sec = 0
    for kind, text in items:
        if kind in ("title", "subtitle"):
            continue
        if kind == "chapter":
            n_ch += 1
            story += [PageBreak(), Heading(text, H1, 0, f"h{n_ch}")]
        elif kind in ("section", "sources"):
            n_sec += 1
            story.append(Heading(text, H2, 1, f"s{n_sec}"))
        elif kind == "source":
            story.append(SentenceParagraph(escape(text), SOURCE))  # never split a source over two pages
        else:
            story.append(SentenceParagraph(escape(text), BODY))
    doc = BookTemplate(str(pdf_path), title=title, subject=subtitle, author="Luisterstof studieboek",
                       creator="maak_pdf.py (reportlab)", lang="nl-NL")
    doc.build(story)


if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "hoogtevrees.md"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "hoogtevrees.pdf"
    build(src, out)
    print(f"geschreven: {out}")
