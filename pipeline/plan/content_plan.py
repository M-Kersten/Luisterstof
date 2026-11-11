"""Stage 2: the content plan for one chapter.

This is the accuracy backbone. The model proposes claims with verbatim
source quotes; this module resolves every quote into a character span in
the section text so the audit can check the script against the source.
"""

from __future__ import annotations

import logging
from statistics import mean

from pydantic import BaseModel, Field
from rapidfuzz import fuzz

from pipeline.models import (
    Book,
    Cast,
    Chapter,
    ContentPlan,
    Definition,
    KeyClaim,
    Misconception,
    SourceSpan,
    WorkedExample,
)

log = logging.getLogger(__name__)

FORMULA_DENSITY_THRESHOLD = 0.02
EXPERT_DIFFICULTY_THRESHOLD = 4.0


# LLM output schema (kept free of dicts and numeric constraints; see llm.py).
class ClaimOut(BaseModel):
    claim: str = Field(description="Eén feitelijke bewering, volledig en zelfstandig leesbaar, in het Nederlands.")
    section: str = Field(description="Sectie-id waar de bewering vandaan komt, bijvoorbeeld ch03.2.")
    source_quote: str = Field(
        description="Letterlijk citaat uit die sectie van 20 tot 300 tekens dat de bewering onderbouwt. Exact overnemen, niets herformuleren."
    )
    difficulty: int = Field(description="1 (triviaal) tot 5 (zeer moeilijk voor een eerstejaars).")
    exam_relevance: int = Field(description="1 (bijzaak) tot 5 (zeker tentamenstof).")


class DefinitionOut(BaseModel):
    term: str
    definition: str = Field(description="Definitie in eigen woorden, trouw aan de bron.")
    section: str
    source_quote: str = Field(description="Letterlijk citaat waarin de bron de term definieert.")


class MisconceptionOut(BaseModel):
    wrong: str = Field(description="De verleidelijke, foute opvatting.")
    right: str = Field(description="Wat er wel klopt, volgens de bron.")
    why_tempting: str = Field(description="Waarom studenten hier intrappen.")


class WorkedExampleOut(BaseModel):
    setup: str
    steps: list[str]
    answer: str


class PlanOut(BaseModel):
    summary: str = Field(description="Eén alinea die het hoofdstuk zakelijk samenvat. Geen grappen.")
    learning_objectives: list[str] = Field(description="Drie tot zeven leerdoelen, elk één zin.")
    key_claims: list[ClaimOut]
    definitions: list[DefinitionOut]
    misconceptions: list[MisconceptionOut] = Field(description="Twee tot vijf misvattingen die het hoofdstuk uitlokt.")
    worked_example: WorkedExampleOut | None = Field(description="Een uitgewerkt voorbeeld uit de bron, of null als het hoofdstuk er geen leent.")
    expert_domain: str | None = Field(
        description="Vakgebied waarvoor een gastexpert nodig is omdat geen van de hosts het geloofwaardig kan dragen, anders null."
    )
    expert_reason: str | None = Field(description="Korte reden voor de expert, of null.")
    formula_dense_sections: list[str] = Field(
        description="Sectie-ids die zo notatiezwaar zijn dat de podcast de vorm en betekenis moet beschrijven in plaats van symbolen voor te lezen."
    )


def _hosts_summary(cast: Cast | None) -> str:
    if cast is None or not cast.hosts:
        return "Geen castinformatie beschikbaar."
    parts = [f"- {h.name} ({h.role}): achtergrond: {h.background.strip()}" for h in cast.hosts]
    return "De hosts en hun achtergrond, om te bepalen of een gastexpert nodig is:\n" + "\n".join(parts)


def resolve_span(section_text: str, quote: str, min_score: float = 80.0) -> tuple[int, int, float] | None:
    """Locate a (possibly imperfect) verbatim quote in the section text."""
    quote = " ".join(quote.split())
    if not quote:
        return None
    idx = section_text.find(quote)
    if idx >= 0:
        return idx, idx + len(quote), 100.0
    lowered = section_text.casefold()
    idx = lowered.find(quote.casefold())
    if idx >= 0:
        return idx, idx + len(quote), 99.0
    try:
        alignment = fuzz.partial_ratio_alignment(quote, section_text)
    except Exception:  # pragma: no cover
        return None
    if alignment is None or alignment.score < min_score:
        return None
    return alignment.dest_start, alignment.dest_end, float(alignment.score)


def _locate(chapter: Chapter, section_id: str, quote: str, warnings: list[str], what: str) -> SourceSpan:
    section = chapter.section(section_id)
    candidates = [section] if section else []
    candidates += [s for s in chapter.sections if s is not section]
    for sec in candidates:
        hit = resolve_span(sec.text, quote)
        if hit:
            start, end, score = hit
            if sec is not section:
                warnings.append(f"{what}: citaat gevonden in {sec.id} in plaats van {section_id}")
            return SourceSpan(section=sec.id, start=start, end=end, quote=quote, match_score=round(score, 1))
    fallback = section or chapter.sections[0]
    warnings.append(f"{what}: citaat niet gevonden in de bron, hele sectie {fallback.id} als bron genomen")
    return SourceSpan(section=fallback.id, start=0, end=len(fallback.text), quote=quote, match_score=0.0)


def _clamp(value: int, lo: int = 1, hi: int = 5) -> int:
    return max(lo, min(hi, int(value)))


def build_plan(chapter: Chapter, out: PlanOut) -> ContentPlan:
    warnings: list[str] = []
    claims: list[KeyClaim] = []
    for i, c in enumerate(out.key_claims, start=1):
        cid = f"c{i}"
        span = _locate(chapter, c.section, c.source_quote, warnings, cid)
        claims.append(KeyClaim(id=cid, claim=c.claim.strip(), source_span=span,
                               difficulty=_clamp(c.difficulty), exam_relevance=_clamp(c.exam_relevance)))

    definitions = [
        Definition(term=d.term.strip(), definition=d.definition.strip(),
                   source_span=_locate(chapter, d.section, d.source_quote, warnings, f"definitie {d.term}"))
        for d in out.definitions
    ]
    misconceptions = [Misconception(wrong=m.wrong, right=m.right, why_tempting=m.why_tempting) for m in out.misconceptions]
    worked = None
    if out.worked_example and out.worked_example.steps:
        worked = WorkedExample(setup=out.worked_example.setup, steps=out.worked_example.steps, answer=out.worked_example.answer)

    # Difficulty per section and the deterministic expert rule.
    per_section: dict[str, list[int]] = {}
    for c in claims:
        per_section.setdefault(c.source_span.section, []).append(c.difficulty)
    section_difficulty = {sid: round(mean(v), 2) for sid, v in per_section.items()}
    hard_sections = [sid for sid, d in section_difficulty.items() if d > EXPERT_DIFFICULTY_THRESHOLD]

    expert_domain = (out.expert_domain or "").strip() or None
    expert_reason = (out.expert_reason or "").strip() or None
    needs_expert = bool(expert_domain) or bool(hard_sections)
    if hard_sections and not expert_domain:
        expert_domain = chapter.title
        expert_reason = "gemiddelde moeilijkheid boven 4 in sectie(s) " + ", ".join(hard_sections)

    dense = set(s.id for s in chapter.sections if s.formula_density >= FORMULA_DENSITY_THRESHOLD)
    # The model's own call counts only where the text has some notation at all: plain prose with a few
    # years in it is never "formula-dense", whatever a fill-in exercise made it look like.
    dense.update(sid for sid in out.formula_dense_sections
                 if chapter.section(sid) and chapter.section(sid).formula_density >= FORMULA_DENSITY_THRESHOLD / 4)

    return ContentPlan(
        chapter_id=chapter.id,
        chapter_title=chapter.title,
        summary=out.summary.strip(),
        learning_objectives=[o.strip() for o in out.learning_objectives if o.strip()],
        key_claims=claims,
        definitions=definitions,
        misconceptions=misconceptions,
        worked_example=worked,
        needs_expert=needs_expert,
        expert_domain=expert_domain,
        expert_reason=expert_reason,
        formula_dense_sections=sorted(dense),
        section_difficulty=section_difficulty,
        warnings=warnings,
    )


def span_text(book: Book, span: SourceSpan, context: int = 0) -> str:
    section = book.section(span.section)
    start = max(0, span.start - context)
    end = min(len(section.text), span.end + context)
    return section.text[start:end]
