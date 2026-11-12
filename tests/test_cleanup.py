"""What a study book prints for a reader with a pen stays out of the podcast's source text."""

from pipeline.ingest.cleanup import CleanupReport, clean_paragraphs, is_exercise_heading, rejoin_broken
from pipeline.ingest.pdf import Block, PageData, TextLine, _absorb_unlabelled_captions, _line_from_dict
from pipeline.ingest.run import _title_from_pages
from pipeline.ingest.structure import Unit, build_book
from pipeline.models import Chapter, Section
from pipeline.plan.content_plan import ClaimOut, PlanOut, build_plan


def _span(text, sup=False, size=12.0):
    return {"text": text, "size": 8.0 if sup else size, "flags": 1 if sup else 0, "font": "Arial"}


def _line(*spans):
    return {"spans": list(spans), "bbox": (70, 100, 500, 112)}


def test_superscript_footnote_numbers_go_and_exponents_stay():
    footnoted = _line_from_dict(_line(_span("Hij begon een opstand"), _span("1", sup=True), _span(" tegen de koning.")))
    assert footnoted.text == "Hij begon een opstand tegen de koning."
    quoted = _line_from_dict(_line(_span("noemt ze ‘ketters’"), _span("7", sup=True), _span(" en straft ze")))
    assert quoted.text == "noemt ze ‘ketters’ en straft ze"
    assert _line_from_dict(_line(_span("de oppervlakte is x"), _span("2", sup=True))).text == "de oppervlakte is x2"
    assert _line_from_dict(_line(_span("dus (a+b)"), _span("2", sup=True))).text == "dus (a+b)2"


def test_fill_ins_footnotes_repeats_and_page_breaks():
    report = CleanupReport()
    paragraphs = [
        "Willem van Oranje wordt in 1533 in Duitsland geboren. Zijn ouders zijn protestant.",
        "Het volkslied heet daarom ‘Het Wilhelmus’*.",
        "* Wil je het Wilhelmus horen? Kijk dan dit filmpje.",
        "Willem van Oranje wordt in ______ in Duitsland geboren. Zijn ouders zijn ______.",
        "Willem van Oranje wordt in 1533 in Duitsland geboren. Zijn ouders zijn protestants.",
        "Op 10",
        "juli 1584 hebben ze een afspraak.",
    ]
    kept = clean_paragraphs(paragraphs, [], report)
    assert kept == [
        "Willem van Oranje wordt in 1533 in Duitsland geboren. Zijn ouders zijn protestant.",
        "Het volkslied heet daarom ‘Het Wilhelmus’.",
        "Op 10 juli 1584 hebben ze een afspraak.",
    ]
    assert (report.fill_ins, report.footnotes, report.repeats) == (1, 1, 1)
    assert rejoin_broken(["Een kop zonder punt", "Hoofdletter, dus nieuw."]) == ["Een kop zonder punt", "Hoofdletter, dus nieuw."]


def test_exercise_headings():
    for heading in ("Oefening 1", "Oefeningen", "Opdracht 3.2", "Vragen bij hoofdstuk 4", "Test jezelf", "Exercises"):
        assert is_exercise_heading(heading), heading
    for text in ("Vragen die de koning stelde over zijn geloof waren altijd lastig.", "De opdrachtgever", "Oefenen"):
        assert not is_exercise_heading(text), text


def _units(rows):
    return [Unit(i, page, "line", text, size, bold, block) for i, (page, text, size, bold, block) in enumerate(rows)]


def test_an_exercise_block_is_left_out_up_to_the_next_heading():
    from pipeline.ingest.structure import Heading

    units = _units([
        (1, "Een lange alinea over de beeldenstorm en wat de koning daarvan vond, met genoeg tekst om te tellen. " * 2, 12, False, 1),
        (2, "Oefening 1", 14, True, 2),
        (2, "1. Waarom wordt Willem prins? A Omdat zijn vader koning is.", 12, False, 3),
        (2, "De moord op Willem", 12, True, 4),
        (2, "In 1584 wordt Willem in Delft vermoord door Balthasar Gerards, die beneden op hem wacht. " * 2, 12, False, 5),
    ])
    headings = [Heading("De moord op Willem", 2, 2, 3, 12.0)]
    book = build_book("b", "Wie was Willem van Oranje?", units, headings)
    text = " ".join(s.text for ch in book.chapters for s in ch.sections)
    assert "Waarom wordt Willem prins" not in text and "Oefening" not in text
    assert "beeldenstorm" in text and "Balthasar Gerards" in text
    chapter = book.chapters[0]
    assert chapter.title == "Wie was Willem van Oranje?" and chapter.sections[0].title == "Wie was Willem van Oranje?"
    assert any("characters of exercises" in w for w in book.warnings)



def test_a_source_list_is_left_out_up_to_the_next_heading():
    from pipeline.ingest.cleanup import is_reference_heading
    from pipeline.ingest.structure import Heading

    for heading in ("Bronnen", "Literatuur", "Referenties", "Noten:", "Further reading"):
        assert is_reference_heading(heading), heading
    assert not is_reference_heading("Bronnen van stress in het dagelijks leven")  # a real section title
    units = _units([
        (1, "1 Wat is hoogtevrees?", 18, True, 0),
        (1, "Op een glazen vloer gaat het hart binnen tien seconden sneller slaan, lang voordat je het zelf merkt. " * 4, 12, False, 1),
        (2, "1.6 Bronnen", 14, True, 2),
        (2, "Gibson, E. J., & Walk, R. D. (1960). The visual cliff. Scientific American, 202(4), 64-71.", 10, False, 3),
        (3, "2 Het evenwicht en de ogen", 18, True, 4),
        (3, "Rechtop staan is een kunststuk dat je evenwichtsorgaan, je ogen en je spieren samen leveren. " * 4, 12, False, 5),
    ])
    headings = [Heading("1 Wat is hoogtevrees?", 1, 1, 0, 18.0), Heading("1.6 Bronnen", 2, 2, 2, 14.0),
                Heading("2 Het evenwicht en de ogen", 1, 3, 4, 18.0)]
    book = build_book("b", "Hoogtevrees", units, headings)
    text = " ".join(s.text for ch in book.chapters for s in ch.sections)
    assert "Scientific American" not in text and "Bronnen" not in [s.title for ch in book.chapters for s in ch.sections]
    assert "glazen vloer" in text and "kunststuk" in text
    assert [ch.title for ch in book.chapters] == ["Wat is hoogtevrees?", "Het evenwicht en de ogen"]
    assert any("characters of source lists" in w for w in book.warnings)


def _page(blocks, pictures):
    return PageData(number=1, width=595, height=842, blocks=blocks, picture_boxes=pictures)


def _text_block(text, bbox, size=10.0):
    return Block(kind="text", page=1, bbox=bbox, lines=[TextLine(text=text, size=size, bold=False, bbox=bbox)])


def test_captions_under_a_picture_become_its_text_and_name_rows_go():
    caption = _text_block("Het prinsdom Oranje (het roze stuk op de kaart) ligt in Zuid-Frankrijk", (71, 490, 376, 512))
    names = _text_block("Willem van Oranje Karel de Vijfde Het hof van Karel de Vijfde in Brussel", (71, 714, 491, 733))
    body = _text_block("De koning van Spanje vindt dat goed, maar dan moet Willem naar Brussel.", (71, 526, 526, 581), size=12.0)
    far = _text_block("10 het leger : de groep militairen die voor een land vechten", (71, 200, 200, 212))
    page = _page([caption, body, names, far], [(71, 327, 294, 500), (71, 595, 162, 721), (166, 595, 272, 720)])
    _absorb_unlabelled_captions([page], body_size=12.0)
    kinds = [(b.kind, b.text or b.lines[0].text) for b in page.blocks]
    assert ("figure", "[Het prinsdom Oranje (het roze stuk op de kaart) ligt in Zuid-Frankrijk.]") in kinds
    assert not any("Karel de Vijfde" in t for _, t in kinds)  # a row of names under portraits
    assert ("text", "De koning van Spanje vindt dat goed, maar dan moet Willem naar Brussel.") in kinds
    assert ("text", "10 het leger : de groep militairen die voor een land vechten") in kinds  # not next to a picture


def test_a_title_split_over_two_lines_is_read_whole():
    lines = [TextLine("1.", 18.0, True, (71, 71, 80, 91)), TextLine("Wie was Willem van Oranje?", 18.0, True, (71, 92, 354, 110)),
             TextLine("Willem van Oranje is een belangrijk persoon.", 12.0, False, (71, 134, 514, 150))]
    page = _page([Block(kind="text", page=1, bbox=(71, 71, 514, 150), lines=lines)], [])
    assert _title_from_pages([page]) == "Wie was Willem van Oranje?"


def test_plain_prose_is_never_formula_dense():
    prose = "In 1568 is het eerste gevecht. In 1581 laat Nederland weten dat het onafhankelijk is. " * 4
    chapter = Chapter(id="ch01", title="T", pages=(1, 2), sections=[Section(id="ch01.1", title="T", text=prose,
                                                                              formula_density=0.002)])
    out = PlanOut(summary="s", learning_objectives=["l"], definitions=[], misconceptions=[], worked_example=None,
                  expert_domain=None, expert_reason=None, formula_dense_sections=["ch01.1"],
                  key_claims=[ClaimOut(claim="In 1568 is het eerste gevecht.", section="ch01.1",
                                       source_quote="In 1568 is het eerste gevecht.", difficulty=2, exam_relevance=4)])
    assert build_plan(chapter, out).formula_dense_sections == []
