from pipeline.fake_handlers import default_fake_llm
from pipeline.ingest.figures import figure_to_text, is_caption, table_to_text
from pipeline.ingest.pdf import body_font_size, extract_pages, join_lines, read_meta
from pipeline.ingest.run import ingest_book
from pipeline.ingest.structure import (
    detect_structure,
    flatten,
    formula_density,
    headings_from_fonts,
    headings_from_llm,
    strip_numbering,
)


def test_extraction_strips_running_headers_and_converts_tables(sample_pdf):
    pages = extract_pages(sample_pdf)
    assert len(pages) == 4
    assert body_font_size(pages) == 10.0
    texts = [ln.text for p in pages for ln in p.lines()]
    assert "Statistiek voor beginners" not in texts[2:]  # running head removed (title page keeps it)
    assert not any(t.strip() == "3" for t in texts)  # page numbers removed
    tables = [b for p in pages for b in p.blocks if b.kind == "table"]
    assert len(tables) == 1 and "Kolommen: Uitkomst, Kans" in tables[0].text and "Rij 1" in tables[0].text


def test_structure_by_fonts_and_toc(sample_pdf, sample_pdf_notoc):
    for pdf, expect in ((sample_pdf, "fonts"), (sample_pdf_notoc, "fonts")):
        book = ingest_book(pdf, "demo")
        assert book.structure_method == expect
        assert [c.title for c in book.episode_chapters()] == ["Kansrekening", "Verdelingen", "Schatten en toetsen"]
        assert [s.id for s in book.chapters[0].sections] == ["ch01.1", "ch01.2"]
        assert book.chapters[0].pages == (2, 2)
        assert book.language == "nl"
    toc_book = ingest_book(sample_pdf, "demo", method="toc")
    assert toc_book.structure_method == "toc"
    assert [c.title for c in toc_book.episode_chapters()] == ["Kansrekening", "Verdelingen", "Schatten en toetsen"]
    assert toc_book.chapters[0].sections[1].table_count == 1


def test_llm_fallback_locates_chapters(sample_pdf_notoc):
    pages = extract_pages(sample_pdf_notoc)
    units = flatten(pages)
    heads = headings_from_llm(units, default_fake_llm())
    assert [h.clean_title for h in heads] == ["Kansrekening", "Verdelingen", "Schatten en toetsen"]
    llm_book = ingest_book(sample_pdf_notoc, "demo", llm=default_fake_llm(), method="llm")
    assert llm_book.structure_method == "llm" and len(llm_book.episode_chapters()) == 3


def test_single_fallback_when_nothing_found(sample_pdf):
    pages = extract_pages(sample_pdf)
    # Pretend every line is body text: no fonts, no toc, no llm -> single chapter
    for p in pages:
        for ln in p.lines():
            ln.size, ln.bold = 10.0, False
    heads, method = detect_structure(pages, toc=None, llm=None)
    assert heads == [] and method == "single"


def test_helpers():
    assert strip_numbering("3.2 De p-waarde") == "De p-waarde"
    assert strip_numbering("Hoofdstuk 4: Toetsen") == "Toetsen"
    assert join_lines(["een ver-", "schil", "in tekst"]) == "een verschil in tekst"
    assert formula_density("z = (x - mu) / sigma") > formula_density("Een gewone zin zonder symbolen.")
    assert is_caption("Figuur 3.2: De verdeling") and is_caption("Tabel 1.1 Kansen") and not is_caption("De figuur toont")
    assert figure_to_text("Figuur 1", "Een stijgende lijn.") == "[Figuur 1. Een stijgende lijn.]"
    assert figure_to_text(None, None) == ""
    text = table_to_text([["a", "b"], ["1", "2"], [None, ""]], "Tabel 2.1: X")
    assert text.startswith("[Tabel 2.1: X.") and "Rij 1: a 1, b 2." in text


def test_meta(sample_pdf):
    meta = read_meta(sample_pdf)
    assert meta.title == "Statistiek voor beginners" and meta.page_count == 4 and meta.toc[0][1] == "Kansrekening"


def test_font_headings_levels(sample_pdf):
    pages = extract_pages(sample_pdf)
    heads = headings_from_fonts(flatten(pages), body_font_size(pages))
    levels = [(h.level, h.clean_title) for h in heads]
    assert (1, "Kansrekening") in levels and (2, "Wat is een kans") in levels
