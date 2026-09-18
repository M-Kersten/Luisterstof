"""Figures and tables become sentences.

A listener cannot see a diagram, so anything that carries meaning must be
converted to text at its position in the reading order. Tables are converted
deterministically. Figures get their caption, plus a vision description when
an LLM is available.
"""

from __future__ import annotations

import base64
import re
from typing import Protocol

from pydantic import BaseModel, Field

CAPTION_RE = re.compile(
    r"^\s*(?P<kind>Figuur|Fig\.|Figure|Afbeelding|Grafiek|Diagram|Schema|Tabel|Table|Kader|Box)"
    r"\s*(?P<num>\d+([.,]\d+)*)?\s*[:.\-–]?\s*(?P<rest>.*)$",
    re.IGNORECASE,
)

TABLE_KINDS = {"tabel", "table"}


def is_caption(text: str) -> bool:
    m = CAPTION_RE.match(text.strip())
    return bool(m and (m.group("num") or m.group("rest")))


def caption_kind(text: str) -> str:
    m = CAPTION_RE.match(text.strip())
    if not m:
        return "figure"
    return "table" if m.group("kind").lower().rstrip(".") in TABLE_KINDS else "figure"


def _clean_cell(cell: object) -> str:
    if cell is None:
        return ""
    return re.sub(r"\s+", " ", str(cell)).strip()


def table_to_text(rows: list[list[object]], caption: str | None = None, max_rows: int = 30) -> str:
    """Spell a table out as sentences a listener can follow."""
    cleaned = [[_clean_cell(c) for c in row] for row in rows if row and any(_clean_cell(c) for c in row)]
    if not cleaned:
        return ""
    header, body = cleaned[0], cleaned[1:]
    label = caption.strip().rstrip(".") if caption else "Tabel"
    parts = [f"[{label}."]
    if any(header):
        parts.append("Kolommen: " + ", ".join(h or f"kolom {i + 1}" for i, h in enumerate(header)) + ".")
    for i, row in enumerate(body[:max_rows], start=1):
        cells = []
        for j, value in enumerate(row):
            if not value:
                continue
            name = header[j] if j < len(header) and header[j] else f"kolom {j + 1}"
            cells.append(f"{name} {value}")
        if cells:
            parts.append(f"Rij {i}: " + ", ".join(cells) + ".")
    if len(body) > max_rows:
        parts.append(f"Nog {len(body) - max_rows} rijen weggelaten.")
    parts.append("]")
    return " ".join(parts)


def figure_to_text(caption: str | None, description: str | None) -> str:
    caption = (caption or "").strip().rstrip(".")
    description = (description or "").strip()
    if not caption and not description:
        return ""
    if caption and description:
        return f"[{caption}. {description}]"
    return f"[{caption or 'Figuur'}. {description}]".replace(". ]", ".]")


class FigureDescription(BaseModel):
    carries_meaning: bool = Field(description="False voor decoratie, logo's of foto's zonder leerinhoud.")
    description: str = Field(
        description="Twee tot vier Nederlandse zinnen die de inhoud voor een luisteraar beschrijven. Leeg als carries_meaning false is."
    )


class Captioner(Protocol):
    def describe(self, png: bytes, caption: str | None, context: str) -> str: ...


class LLMCaptioner:
    """Describe a figure with a vision-capable LLM. Returns '' for decorative images."""

    SYSTEM = (
        "Je beschrijft figuren uit een Nederlands studieboek voor een podcast. De luisteraar ziet "
        "niets. Beschrijf wat de figuur laat zien en wat de leerinhoud ervan is, in twee tot vier "
        "zinnen. Noem assen, richtingen en relaties expliciet. Beschrijf geen kleuren of stijl. "
        "Als de figuur decoratief is (logo, sfeerfoto, versiering), zet carries_meaning op false."
    )

    def __init__(self, llm):
        self.llm = llm

    def describe(self, png: bytes, caption: str | None, context: str) -> str:
        from pipeline.llm import LLMRequest, image_block, text_block

        prompt = "Bijschrift: " + (caption or "(geen)") + "\n\nOmringende tekst:\n" + context[:1500]
        request = LLMRequest(
            task="figure_caption",
            system=self.SYSTEM,
            user=[image_block(base64.standard_b64encode(png).decode("ascii")), text_block(prompt)],
            schema=FigureDescription,
            max_tokens=1024,
            effort="low",
            cache_system=False,
        )
        result = self.llm.generate(request)
        assert isinstance(result, FigureDescription)
        return result.description.strip() if result.carries_meaning else ""
