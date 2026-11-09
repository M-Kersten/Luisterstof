"""Figures and tables become sentences.

A listener cannot see a diagram, so anything that carries meaning must be
converted to text at its position in the reading order. Tables are converted
deterministically; a figure contributes its caption. The writing model runs
through the OpenCode text endpoint, so there is no vision path.
"""

from __future__ import annotations

import re

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


def figure_to_text(caption: str | None) -> str:
    """A figure without a caption carries nothing a listener can hear."""
    caption = (caption or "").strip().rstrip(".")
    return f"[{caption}.]" if caption else ""
