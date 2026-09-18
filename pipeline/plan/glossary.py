"""Stage 2b: the lexicon (glossary.json).

With a Dutch source there is no translation layer. What remains is
pronunciation: English loanwords, notation and abbreviations. The lexicon is
per book, so a term sounds the same in chapter 9 as it did in chapter 2.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from pipeline.llm import LLM, LLMRequest, text_block
from pipeline.models import Chapter, Glossary, LexiconEntry

log = logging.getLogger(__name__)

DEFAULTS_FILE = "lexicon_defaults.yaml"


def load_defaults(cast_dir: Path | str) -> list[LexiconEntry]:
    path = Path(cast_dir) / DEFAULTS_FILE
    if not path.is_file():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return [LexiconEntry.model_validate(e) for e in data.get("entries", [])]


def load_or_create_glossary(glossary_path: Path, book_id: str, cast_dir: Path | str | None = None) -> Glossary:
    glossary = Glossary.load_or_none(glossary_path)
    if glossary is None:
        glossary = Glossary(book_id=book_id)
        if cast_dir is not None:
            glossary.merge(load_defaults(cast_dir))
    return glossary


# ---------------------------------------------------------------------------
# Candidate scan (deterministic) and LLM proposal
# ---------------------------------------------------------------------------

_DOTTED_ABBR = re.compile(r"\b(?:[A-Za-z]{1,3}\.){2,}")
_SHORT_ABBR = re.compile(r"\b(bijv|bv|ca|enz|resp|zgn|nl|vs|etc|afk|ong|max|min|incl|excl)\.", re.IGNORECASE)
_ACRONYM = re.compile(r"\b[A-Z][A-Z0-9]{1,6}\b")
_NOTATION = [
    re.compile(r"\d+[.,]\d+(?:\s?[×x]\s?10\^?-?\d+)?"),
    re.compile(r"\d+\s?%"),
    re.compile(r"\b[a-zA-Z]\^-?\d+"),
    re.compile(r"\b\d+/\d+\b"),
    re.compile(r"\b[A-Za-z]\((?:[A-Za-z|,\s]+)\)"),
    re.compile(r"[=<>≤≥≈±√∑∫]"),
    re.compile(r"\b\d+(?:[.,]\d+)?\s?(?:km|m|cm|mm|kg|g|mg|s|ms|Hz|kHz|MHz|J|kJ|W|kW|V|A|Pa|kPa|mol|°C|K)\b"),
]


def scan_candidates(text: str, limit: int = 60) -> dict[str, list[str]]:
    """Cheap regex pass that lists things the LLM must not overlook."""
    abbreviations: list[str] = []
    for m in _DOTTED_ABBR.finditer(text):
        abbreviations.append(m.group(0))
    for m in _SHORT_ABBR.finditer(text):
        abbreviations.append(m.group(0))
    acronyms = [m.group(0) for m in _ACRONYM.finditer(text) if len(m.group(0)) >= 2]
    notation: list[str] = []
    for rx in _NOTATION:
        notation.extend(m.group(0) for m in rx.finditer(text))

    def uniq(items: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for it in items:
            key = it.strip()
            if key and key.casefold() not in seen:
                seen.add(key.casefold())
                out.append(key)
        return out[:limit]

    return {"abbreviations": uniq(abbreviations), "acronyms": uniq(acronyms), "notation": uniq(notation)}


class LexiconEntryOut(BaseModel):
    surface: str = Field(description="Exacte schrijfwijze zoals in de tekst.")
    kind: Literal["loanword_en", "notation", "abbreviation"]
    spoken: str = Field(description="Hoe het uitgesproken moet worden, als Nederlandse fonetische spelling of voluit geschreven.")
    note: str | None = Field(description="Korte toelichting, of null.")


class LexiconOut(BaseModel):
    entries: list[LexiconEntryOut]


LEXICON_SYSTEM = """Je stelt een uitspraaklexicon samen voor een Nederlandse studiepodcast die met tekst-naar-spraak wordt ingesproken.

Drie categorieën, alle drie moeten voor de opname zijn opgelost:
1. loanword_en: Engelse leenwoorden in Nederlandse zinnen (bijvoorbeeld "gradient descent", "feedback loop", "overfitting"). Geef een Nederlandse fonetische spelling die een Nederlands TTS-model consequent goed uitspreekt (bijvoorbeeld "greedient discent"). Eén spelling per term, die wordt overal hergebruikt.
2. notation: formules, eenheden, exponenten, breuken, bereiken, symbolen. Schrijf voluit zoals een docent het hardop zegt ("ongeveer anderhalf keer tien tot de derde", "P van A gegeven B", "een zesde").
3. abbreviation: afkortingen, voluit geschreven ("onder andere", "met betrekking tot"). Acroniemen die als woord worden gezegd blijven een woord; acroniemen die per letter gaan, spel je met streepjes of spaties ("C B S").

Regels:
- Neem alleen op wat in de tekst voorkomt. Sla over wat al in het bestaande lexicon staat.
- surface moet letterlijk overeenkomen met de tekst, zodat het automatisch kan worden vervangen.
- Geen gewone Nederlandse woorden, geen eigennamen die al goed klinken.
- Alles in het Nederlands.
"""


def propose_lexicon(chapter: Chapter, glossary: Glossary, llm: LLM) -> list[LexiconEntry]:
    text = chapter.full_text()
    candidates = scan_candidates(text)
    existing = "\n".join(f"- {e.surface} -> {e.spoken}" for e in glossary.entries[:300]) or "(leeg)"
    cand_text = "\n".join(
        f"{k}: " + ", ".join(v) for k, v in candidates.items() if v
    ) or "(geen kandidaten gevonden door de scanner)"
    request = LLMRequest(
        task="lexicon",
        system=[
            text_block(LEXICON_SYSTEM),
            text_block(f"# Hoofdstuk {chapter.id}: {chapter.title}\n\n{text}"),
        ],
        user=(
            f"Bestaand lexicon (niet herhalen):\n{existing}\n\n"
            f"Kandidaten uit een regex-scan (beoordeel ze, en zoek zelf verder naar leenwoorden):\n{cand_text}\n\n"
            "Geef het aanvullende lexicon voor dit hoofdstuk."
        ),
        schema=LexiconOut,
        effort="medium",
        max_tokens=8000,
    )
    out = llm.generate(request)
    assert isinstance(out, LexiconOut)
    entries: list[LexiconEntry] = []
    for e in out.entries:
        surface = e.surface.strip()
        if not surface or glossary.find(surface) or surface.casefold() not in text.casefold():
            continue
        entries.append(LexiconEntry(surface=surface, kind=e.kind, spoken=e.spoken.strip(), note=e.note,
                                    source_chapter=chapter.id, lock=e.kind == "loanword_en"))
    return entries


# ---------------------------------------------------------------------------
# Application before render
# ---------------------------------------------------------------------------

def _boundary_pattern(surface: str) -> str:
    escaped = re.escape(surface)
    prefix = r"(?<![\w])" if surface[0].isalnum() else ""
    suffix = r"(?![\w])" if surface[-1].isalnum() else ""
    return prefix + escaped + suffix


def _compile(glossary: Glossary) -> tuple[re.Pattern | None, dict[str, LexiconEntry]]:
    entries = sorted(glossary.entries, key=lambda e: -len(e.surface))
    if not entries:
        return None, {}
    table = {e.surface.casefold(): e for e in entries}
    pattern = re.compile("|".join(_boundary_pattern(e.surface) for e in entries), re.IGNORECASE)
    return pattern, table


def apply_lexicon(text: str, glossary: Glossary) -> str:
    """Replace every surface form with its spoken form in a single pass."""
    pattern, table = _compile(glossary)
    if pattern is None:
        return text

    def repl(m: re.Match) -> str:
        entry = table.get(m.group(0).casefold())
        if entry is None:
            return m.group(0)
        spoken = entry.spoken
        if m.group(0)[:1].isupper() and spoken[:1].islower():
            spoken = spoken[0].upper() + spoken[1:]
        return spoken

    out = pattern.sub(repl, text)
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)
    return out.strip()


def canonicalise(text: str, glossary: Glossary) -> str:
    """Map both surface and spoken forms to one token, for tolerant transcript comparison."""
    pairs: list[tuple[str, str]] = []
    for e in glossary.entries:
        canon = re.sub(r"[^\w]+", "", e.surface.casefold()) or "sym"
        pairs.append((e.spoken.strip(), canon))
        pairs.append((e.surface, canon))
    pairs.sort(key=lambda p: -len(p[0]))
    out = text
    for form, canon in pairs:
        if not form.strip():
            continue
        out = re.sub(_boundary_pattern(form.strip()), f" {canon} ", out, flags=re.IGNORECASE)
    return out
