"""Generate a small synthetic Dutch study book as PDF for tests and demos.

Usage: python -m pipeline.sample_book out.pdf [--no-toc]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pymupdf

BODY = 10
H1 = 20
H2 = 14

CHAPTERS = [
    {
        "title": "Kansrekening",
        "sections": [
            ("Wat is een kans", [
                "Een kans is een getal tussen 0 en 1 dat uitdrukt hoe waarschijnlijk een gebeurtenis is. "
                "Een kans van 0 betekent dat de gebeurtenis nooit optreedt, een kans van 1 dat ze zeker optreedt. "
                "De som van de kansen van alle mogelijke uitkomsten van een experiment is altijd precies 1.",
                "Bij een eerlijke dobbelsteen is de kans op elk van de zes ogen gelijk aan 1/6, ca. 16,7 %. "
                "De kans op een even aantal ogen is 3/6, dus 0,5. Dit volgt uit de regel dat kansen van elkaar "
                "uitsluitende gebeurtenissen mogen worden opgeteld.",
            ]),
            ("Voorwaardelijke kans", [
                "De voorwaardelijke kans P(A|B) is de kans op A gegeven dat B is opgetreden. Ze wordt berekend "
                "als P(A en B) gedeeld door P(B), mits P(B) groter is dan 0. Twee gebeurtenissen heten "
                "onafhankelijk als P(A|B) gelijk is aan P(A).",
                "Een veelgemaakte fout is het verwarren van P(A|B) met P(B|A). De kans dat iemand die ziek is "
                "positief test, is iets anders dan de kans dat iemand die positief test ziek is. Het verschil "
                "wordt bepaald door hoe vaak de ziekte voorkomt, de zgn. prevalentie.",
            ]),
        ],
        "table": [["Uitkomst", "Kans"], ["1 oog", "1/6"], ["even", "1/2"], ["hoger dan 4", "1/3"]],
    },
    {
        "title": "Verdelingen",
        "sections": [
            ("De binomiale verdeling", [
                "De binomiale verdeling beschrijft het aantal successen in n onafhankelijke pogingen met elk "
                "dezelfde succeskans p. De verwachtingswaarde is n keer p en de variantie is n keer p keer (1 - p). "
                "Voor n = 10 en p = 0,3 is de verwachting dus 3 en de variantie 2,1.",
                "De verdeling is symmetrisch als p precies 0,5 is en scheef in alle andere gevallen. Bij grote n "
                "wordt de binomiale verdeling goed benaderd door een normale verdeling met dezelfde verwachting "
                "en variantie, de zgn. normale benadering.",
            ]),
            ("De normale verdeling", [
                "De normale verdeling is een continue, klokvormige verdeling die volledig wordt vastgelegd door "
                "twee parameters: het gemiddelde mu en de standaardafwijking sigma. Ongeveer 68 % van de waarden "
                "ligt binnen één standaardafwijking van het gemiddelde, ongeveer 95 % binnen twee.",
                "Een z-score geeft aan hoeveel standaardafwijkingen een waarde van het gemiddelde af ligt: "
                "z = (x - mu) / sigma. Een z-score van 2 is dus twee standaardafwijkingen boven het gemiddelde, "
                "wat bij een normale verdeling in ca. 2,3 % van de gevallen voorkomt.",
            ]),
        ],
        "table": None,
    },
    {
        "title": "Schatten en toetsen",
        "sections": [
            ("Betrouwbaarheidsintervallen", [
                "Een betrouwbaarheidsinterval van 95 % is een interval dat bij herhaald steekproeftrekken in 95 % "
                "van de gevallen de werkelijke parameter bevat. Het zegt niets over de kans dat de parameter in "
                "dit ene interval ligt, een misverstand dat o.a. in de media veel voorkomt.",
                "De breedte van het interval hangt af van de steekproefomvang: vier keer zo veel waarnemingen "
                "halveert de breedte, omdat de standaardfout evenredig is met 1 gedeeld door de wortel van n.",
            ]),
            ("De p-waarde", [
                "De p-waarde is de kans om, als de nulhypothese waar is, een resultaat te vinden dat minstens zo "
                "extreem is als het waargenomen resultaat. Een kleine p-waarde (bijv. onder 0,05) geldt als bewijs "
                "tegen de nulhypothese. De p-waarde is niet de kans dat de nulhypothese waar is.",
                "Een statistisch significant resultaat hoeft niet praktisch relevant te zijn. Bij een zeer grote "
                "steekproef wordt vrijwel elk verschil significant, hoe klein ook. Effectgrootte en p-waarde "
                "moeten daarom altijd samen worden gerapporteerd.",
            ]),
        ],
        "table": None,
    },
]


def build(path: Path | str, *, with_toc: bool = True, with_numbers: bool = True) -> Path:
    path = Path(path)
    doc = pymupdf.open()
    toc: list[list] = []

    # Title page
    page = doc.new_page()
    page.insert_text((72, 200), "Statistiek voor beginners", fontsize=28, fontname="helv")
    page.insert_text((72, 240), "Een studieboek", fontsize=14, fontname="helv")
    page.insert_text((72, 780), "1", fontsize=9, fontname="helv")

    for ci, ch in enumerate(CHAPTERS, start=1):
        page = doc.new_page()
        y = 90
        page.insert_text((72, 40), "Statistiek voor beginners", fontsize=8, fontname="helv")
        heading = f"{ci} {ch['title']}" if with_numbers else ch["title"]
        page.insert_text((72, y), heading, fontsize=H1, fontname="hebo")
        toc.append([1, ch["title"], page.number + 1])
        y += 40
        for si, (stitle, paragraphs) in enumerate(ch["sections"], start=1):
            sheading = f"{ci}.{si} {stitle}" if with_numbers else stitle
            if y > 700:
                page.insert_text((72, 780), str(page.number + 1), fontsize=9, fontname="helv")
                page = doc.new_page()
                page.insert_text((72, 40), "Statistiek voor beginners", fontsize=8, fontname="helv")
                y = 90
            page.insert_text((72, y), sheading, fontsize=H2, fontname="hebo")
            toc.append([2, stitle, page.number + 1])
            y += 24
            for para in paragraphs:
                rect = pymupdf.Rect(72, y, 523, y + 400)
                page.insert_textbox(rect, para, fontsize=BODY, fontname="helv", lineheight=1.3)
                # estimate height: chars per line ~ 95
                lines = max(1, len(para) // 90 + 1)
                y += int(lines * BODY * 1.3) + 14
            y += 6
        if ch.get("table"):
            page.insert_text((72, y), f"Tabel {ci}.1: Kansen bij een dobbelsteen", fontsize=BODY, fontname="helv")
            y += 16
            rows = ch["table"]
            col_w = 120
            for r, row in enumerate(rows):
                for c, cell in enumerate(row):
                    x0, y0 = 72 + c * col_w, y + r * 18
                    rect = pymupdf.Rect(x0, y0, x0 + col_w, y0 + 18)
                    page.draw_rect(rect, color=(0, 0, 0), width=0.5)
                    page.insert_text((x0 + 4, y0 + 13), cell, fontsize=BODY, fontname="helv")
            y += len(rows) * 18 + 20
        page.insert_text((72, 780), str(page.number + 1), fontsize=9, fontname="helv")

    if with_toc:
        doc.set_toc(toc)
    doc.set_metadata({"title": "Statistiek voor beginners", "author": "Testauteur"})
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path)
    doc.close()
    return path


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "data/sample_book.pdf"
    build(out, with_toc="--no-toc" not in sys.argv)
    print(out)
