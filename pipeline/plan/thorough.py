"""Thorough content plan and lexicon: many small passes instead of one big one.

One call over a whole chapter suits a frontier model. A local model loses
sections, picks the wrong claims and paraphrases its "verbatim" quotes. So:

1. per section (long sections in parts): candidate claims, definitions,
   misconceptions, worked examples, with the section's text as the only source;
2. every quote is checked against that text; failed ones go back once to be
   fixed, and a candidate whose quote still isn't in the text is dropped;
   a substantial section that yields nothing gets a second pass;
3. a chapter pass picks and merges the claims by candidate id, so quotes are
   never rewritten there, and writes summary and learning objectives;
4. review rounds compare the selection with the section list and the candidate
   pool and add or remove candidates.

The result goes through the same ``build_plan`` as the single-call plan.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from pydantic import BaseModel, Field

from pipeline.llm import LLM, LLMRequest, text_block
from pipeline.models import Book, Cast, Chapter, ContentPlan, Glossary, LexiconEntry, Section
from pipeline.plan.content_plan import (
    ClaimOut,
    DefinitionOut,
    MisconceptionOut,
    PlanOut,
    WorkedExampleOut,
    _hosts_summary,
    build_plan,
    resolve_span,
)

log = logging.getLogger(__name__)

MAX_PART_CHARS = 12000  # a section longer than this is read in parts
SUBSTANTIAL_SECTION_CHARS = 400  # below this an empty harvest is plausible
QUOTE_MIN_SCORE = 95.0  # near-literal only; the single-call plan accepts 80, which lets paraphrases through
MAX_CLAIMS = 25

Progress = Callable[[str, dict], None]


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class SectionClaimOut(BaseModel):
    claim: str = Field(description="Eén feitelijke bewering uit deze tekst, zelfstandig leesbaar, in het Nederlands.")
    source_quote: str = Field(description="Letterlijk citaat uit deze tekst van 20 tot 300 tekens dat de bewering onderbouwt. "
                                          "Exact overnemen, inclusief getallen en leestekens.")
    difficulty: int = Field(description="1 (triviaal) tot 5 (zeer moeilijk voor een eerstejaars).")
    exam_relevance: int = Field(description="1 (bijzaak) tot 5 (zeker tentamenstof).")


class SectionDefinitionOut(BaseModel):
    term: str
    definition: str = Field(description="Definitie in eigen woorden, trouw aan de bron.")
    source_quote: str = Field(description="Letterlijk citaat waarin de tekst de term definieert.")


class SectionPlanOut(BaseModel):
    claims: list[SectionClaimOut] = Field(description="Alle beweringen die ertoe doen, meestal twee tot acht.")
    definitions: list[SectionDefinitionOut]
    misconceptions: list[MisconceptionOut] = Field(description="Misvattingen die deze tekst weerlegt; leeg als er geen zijn.")
    worked_example: WorkedExampleOut | None = Field(description="Een uitgewerkt voorbeeld uit deze tekst, of null.")
    formula_dense: bool = Field(description="True als de tekst zo notatiezwaar is dat voorlezen van symbolen niet werkt.")


class QuoteFixOut(BaseModel):
    class Fix(BaseModel):
        id: str = Field(description="Het id van het citaat, bijvoorbeeld q2.")
        source_quote: str = Field(description="De exacte passage uit de tekst, letterlijk gekopieerd, of leeg als die er niet is.")

    fixes: list[Fix]


class PickedClaimOut(BaseModel):
    candidate: str = Field(description="Id van de kandidaat (bijvoorbeeld k7) waar deze bewering op rust.")
    claim: str = Field(description="De bewering, eventueel samengevoegd met een dubbele kandidaat of scherper geformuleerd, "
                                   "zonder iets toe te voegen wat niet in de kandidaat staat.")
    difficulty: int = Field(description="1 tot 5.")
    exam_relevance: int = Field(description="1 tot 5.")


class PlanMergeOut(BaseModel):
    summary: str = Field(description="Eén alinea die het hoofdstuk zakelijk samenvat. Geen grappen.")
    learning_objectives: list[str] = Field(description="Drie tot zeven leerdoelen, elk één zin.")
    key_claims: list[PickedClaimOut] = Field(description="De kernbeweringen van het hoofdstuk, 6 tot 25, uit elke sectie.")
    definitions: list[str] = Field(description="Ids van de definities die in het plan horen (bijvoorbeeld d2).")
    misconceptions: list[MisconceptionOut] = Field(description="Twee tot vijf misvattingen, samengevoegd uit de kandidaten.")
    worked_example: str | None = Field(description="Id van het beste uitgewerkte voorbeeld (bijvoorbeeld w1), of null.")
    expert_domain: str | None = Field(description="Vakgebied waarvoor een gastexpert nodig is, anders null.")
    expert_reason: str | None = Field(description="Korte reden voor de expert, of null.")


class PlanReviewOut(BaseModel):
    problems: list[str] = Field(description="Wat ontbreekt, te mager of dubbel is. Leeg als het plan klopt.")
    add: list[str] = Field(description="Kandidaat-ids die in het plan moeten.")
    remove: list[str] = Field(description="Kandidaat-ids die eruit moeten (dubbel of bijzaak).")


SECTION_SYSTEM = """Je leest één sectie van een Nederlands studieboek en haalt eruit wat een student moet weten, als bouwsteen voor het inhoudsplan van een studiepodcast.

Werk uitsluitend vanuit de tekst hieronder. Voeg geen kennis van buiten toe. Wat hier fout gaat, wordt later met overtuiging fout verteld.

Regels:
- Een bewering is één zelfstandig leesbare zin die letterlijk uit de tekst te herleiden is. Neem alle beweringen op die ertoe doen, ook de lastige.
- Geef bij elke bewering en definitie een letterlijk citaat van 20 tot 300 tekens. Kopieer het teken voor teken uit de tekst. Verzin of herformuleer geen citaten.
- difficulty en exam_relevance: 1 tot 5.
- Misvattingen alleen als de tekst ze expliciet of impliciet weerlegt.
- Alles in het Nederlands.
"""

MERGE_SYSTEM = """Je stelt het inhoudsplan samen voor één hoofdstuk van een Nederlands studieboek, als basis voor een studiepodcast. De kandidaten hieronder zijn per sectie uit de brontekst gehaald en hun citaten zijn gecontroleerd.

Regels:
- Kies de kernbeweringen: minimaal 6, maximaal 25, en elke sectie met inhoud komt aan bod. Tentamenstof en het moeilijkste begrip gaan voor bijzaken.
- Verwijs naar kandidaten met hun id. Twee kandidaten die hetzelfde zeggen, voeg je samen onder het id van de beste.
- Formuleer een bewering alleen scherper als er daarbij niets bij komt wat niet in de kandidaat staat.
- De samenvatting is zakelijk, geen grappen, alleen wat in de kandidaten staat.
- Een gastexpert is nodig als het hoofdstuk een vakgebied raakt dat geen van de twee hosts geloofwaardig kan dragen. Anders expert_domain null.
- Alles in het Nederlands.
"""

REVIEW_SYSTEM = """Je controleert een concept-inhoudsplan voor een studiepodcast tegen de secties van het hoofdstuk en de volledige lijst kandidaten.

Zoek wat er mis is: een sectie met inhoud die ontbreekt of maar één bijzaak heeft, het moeilijkste begrip dat ontbreekt, tentamenstof die is weggelaten, twee beweringen die hetzelfde zeggen, bijzaken die plaats innemen. Stel voor welke kandidaten erbij moeten en welke eruit, met hun id. Is het plan in orde, geef dan lege lijsten. Alles in het Nederlands.
"""


# ---------------------------------------------------------------------------
# Section pass
# ---------------------------------------------------------------------------

def section_parts(section: Section, max_chars: int = MAX_PART_CHARS) -> list[str]:
    """The section's text in parts of at most ``max_chars``, split on paragraph boundaries."""
    text = section.text
    if len(text) <= max_chars:
        return [text]
    parts, current = [], ""
    for para in text.split("\n\n"):
        while len(para) > max_chars:  # one enormous paragraph: hard split
            head, para = para[:max_chars], para[max_chars:]
            if current:
                parts.append(current)
                current = ""
            parts.append(head)
        if current and len(current) + 2 + len(para) > max_chars:
            parts.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        parts.append(current)
    return parts


class Candidates:
    def __init__(self) -> None:
        self.claims: list[tuple[str, str, SectionClaimOut]] = []  # (id, section id, claim)
        self.definitions: list[tuple[str, str, SectionDefinitionOut]] = []
        self.misconceptions: list[MisconceptionOut] = []
        self.examples: list[tuple[str, WorkedExampleOut]] = []
        self.formula_dense: set[str] = set()
        self.warnings: list[str] = []

    def claim(self, cid: str) -> tuple[str, str, SectionClaimOut] | None:
        return next((c for c in self.claims if c[0] == cid), None)

    def per_section(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for _, sid, _ in self.claims:
            counts[sid] = counts.get(sid, 0) + 1
        return counts


def _section_request(book: Book, chapter: Chapter, section: Section, text: str, label: str,
                     fix: str | None = None) -> LLMRequest:
    outline = "\n".join(f"- {s.id}: {s.title}" for s in chapter.sections)
    user = (f"Haal de bouwstenen uit {label}.\n\nDe secties van dit hoofdstuk, ter oriëntatie:\n{outline}")
    if fix:
        user += "\n\n" + fix
    return LLMRequest(
        task="plan_section",
        system=[text_block(SECTION_SYSTEM),
                text_block(f"# Boek: {book.title}\n# Hoofdstuk {chapter.id}: {chapter.title}\n\n"
                           f"### Sectie {section.id}: {section.title}\n{text}")],
        user=user,
        schema=SectionPlanOut,
        effort="high",
    )


def _fix_quotes(llm: LLM, section: Section, text: str, failed: list[tuple[str, str, str]]) -> dict[str, str]:
    """failed: (quote id, what it supports, quote). Returns quote id -> corrected quote (verified)."""
    listing = "\n".join(f"[{qid}] bij: {what}\n     citaat: \"{quote}\"" for qid, what, quote in failed)
    request = LLMRequest(
        task="plan_quotes",
        system=[text_block("Je corrigeert citaten. Een citaat moet teken voor teken in de tekst staan. Zoek per citaat de passage "
                           "in de tekst die hetzelfde zegt en kopieer die letterlijk (20 tot 300 tekens). Staat het er niet, geef dan "
                           "een lege source_quote."),
                text_block(f"### Sectie {section.id}: {section.title}\n{text}")],
        user=f"Deze citaten staan niet letterlijk in de tekst:\n{listing}",
        schema=QuoteFixOut,
        effort="medium",
    )
    out = llm.generate(request)
    assert isinstance(out, QuoteFixOut)
    return {f.id: f.source_quote for f in out.fixes
            if f.source_quote.strip() and resolve_span(text, f.source_quote, QUOTE_MIN_SCORE)}


def harvest_section(book: Book, chapter: Chapter, section: Section, llm: LLM, pool: Candidates,
                    progress: Progress | None = None) -> None:
    parts = section_parts(section)
    found = 0
    for k, text in enumerate(parts, start=1):
        label = f"sectie {section.id}" + (f" (deel {k} van {len(parts)})" if len(parts) > 1 else "")
        if progress:
            progress("section", {"section": section.id, "part": k, "parts": len(parts)})
        out = llm.generate(_section_request(book, chapter, section, text, label))
        assert isinstance(out, SectionPlanOut)
        if not out.claims and len(text) >= SUBSTANTIAL_SECTION_CHARS:
            out = llm.generate(_section_request(
                book, chapter, section, text, label,
                fix=f"Een eerdere poging vond in deze tekst geen enkele bewering, terwijl de tekst {len(text)} tekens telt. "
                    "Lees opnieuw en haal de beweringen eruit die een student moet kennen."))
            assert isinstance(out, SectionPlanOut)
        found += _collect(llm, section, text, out, pool)
    if not found and len(section.text) >= SUBSTANTIAL_SECTION_CHARS:
        pool.warnings.append(f"sectie {section.id}: geen enkele bewering met een controleerbaar citaat gevonden")


def _collect(llm: LLM, section: Section, text: str, out: SectionPlanOut, pool: Candidates) -> int:
    items: list[tuple[str, str, str]] = []  # (qid, what, quote)
    for i, c in enumerate(out.claims):
        items.append((f"q{i + 1}", c.claim, c.source_quote))
    offset = len(out.claims)
    for j, d in enumerate(out.definitions):
        items.append((f"q{offset + j + 1}", f"definitie {d.term}", d.source_quote))
    failed = [it for it in items if not resolve_span(text, it[2], QUOTE_MIN_SCORE)]
    fixed = _fix_quotes(llm, section, text, failed) if failed else {}
    quotes = {qid: fixed.get(qid, quote) for qid, _, quote in items}
    ok = {qid for qid, _, _ in items if qid in fixed or resolve_span(text, quotes[qid], QUOTE_MIN_SCORE)}

    kept = 0
    for i, c in enumerate(out.claims):
        qid = f"q{i + 1}"
        if qid not in ok:
            pool.warnings.append(f"sectie {section.id}: bewering geschrapt, citaat staat niet in de bron: {c.claim[:80]}")
            continue
        pool.claims.append((f"k{len(pool.claims) + 1}", section.id, c.model_copy(update={"source_quote": quotes[qid]})))
        kept += 1
    for j, d in enumerate(out.definitions):
        qid = f"q{offset + j + 1}"
        if qid not in ok:
            pool.warnings.append(f"sectie {section.id}: definitie {d.term} geschrapt, citaat staat niet in de bron")
            continue
        pool.definitions.append((f"d{len(pool.definitions) + 1}", section.id, d.model_copy(update={"source_quote": quotes[qid]})))
    pool.misconceptions.extend(out.misconceptions)
    if out.worked_example and out.worked_example.steps:
        pool.examples.append((f"w{len(pool.examples) + 1}", out.worked_example))
    if out.formula_dense:
        pool.formula_dense.add(section.id)
    return kept


# ---------------------------------------------------------------------------
# Chapter pass and review
# ---------------------------------------------------------------------------

def _pool_text(pool: Candidates, chapter: Chapter) -> str:
    rows = ["## Kandidaat-beweringen (id, sectie, moeilijkheid, tentamen)"]
    rows += [f"[{cid}] {sid} (m{c.difficulty}, t{c.exam_relevance}) {c.claim}" for cid, sid, c in pool.claims]
    rows.append("\n## Kandidaat-definities")
    rows += [f"[{did}] {sid} {d.term}: {d.definition}" for did, sid, d in pool.definitions] or ["(geen)"]
    rows.append("\n## Kandidaat-misvattingen")
    rows += [f"- fout: {m.wrong} | juist: {m.right} | verleidelijk: {m.why_tempting}" for m in pool.misconceptions] or ["(geen)"]
    rows.append("\n## Uitgewerkte voorbeelden")
    rows += [f"[{wid}] {w.setup}" for wid, w in pool.examples] or ["(geen)"]
    rows.append("\n## Secties")
    counts = pool.per_section()
    rows += [f"- {s.id}: {s.title} ({len(s.text)} tekens, {counts.get(s.id, 0)} kandidaten)" for s in chapter.sections]
    return "\n".join(rows)


def _selection_text(selected: list[PickedClaimOut], pool: Candidates, chapter: Chapter) -> str:
    per: dict[str, int] = {}
    rows = ["## Concept: gekozen beweringen"]
    for p in selected:
        found = pool.claim(p.candidate)
        sid = found[1] if found else "?"
        per[sid] = per.get(sid, 0) + 1
        rows.append(f"[{p.candidate}] {sid} (m{p.difficulty}, t{p.exam_relevance}) {p.claim}")
    rows.append("\n## Gekozen per sectie")
    rows += [f"- {s.id}: {per.get(s.id, 0)}" + ("  <- geen enkele" if not per.get(s.id) and pool.per_section().get(s.id) else "")
             for s in chapter.sections]
    hardest = max(pool.claims, key=lambda c: (c[2].difficulty, c[2].exam_relevance), default=None)
    if hardest:
        rows.append(f"\nMoeilijkste kandidaat: [{hardest[0]}] {hardest[2].claim}")
    return "\n".join(rows)


def _clean_selection(selected: list[PickedClaimOut], pool: Candidates) -> list[PickedClaimOut]:
    seen: set[str] = set()
    out = []
    for p in selected:
        cid = p.candidate.strip().strip("[]")
        if cid in seen or pool.claim(cid) is None:
            if pool.claim(cid) is None:
                pool.warnings.append(f"samenstelling verwees naar onbekende kandidaat {p.candidate}")
            continue
        seen.add(cid)
        out.append(p.model_copy(update={"candidate": cid}))
    return out[:MAX_CLAIMS]


def plan_chapter_thorough(book: Book, chapter_id: str, llm: LLM, cast: Cast | None = None, *,
                          review_rounds: int = 1, progress: Progress | None = None) -> ContentPlan:
    chapter = book.chapter(chapter_id)
    pool = Candidates()
    for section in chapter.sections:
        if section.text.strip():
            harvest_section(book, chapter, section, llm, pool, progress)
    if not pool.claims:
        raise ValueError(f"no claim with a verifiable quote in any section of {chapter_id}; check the ingest of this chapter")

    if progress:
        progress("merge", {"candidates": len(pool.claims)})
    pool_text = _pool_text(pool, chapter)
    merged = llm.generate(LLMRequest(
        task="plan_merge",
        system=[text_block(MERGE_SYSTEM + "\n" + _hosts_summary(cast)),
                text_block(f"# Boek: {book.title}\n# Hoofdstuk {chapter.id}: {chapter.title}\n\n{pool_text}")],
        user=f"Stel het inhoudsplan samen voor hoofdstuk {chapter.id} ({chapter.title}).",
        schema=PlanMergeOut,
        effort="high",
    ))
    assert isinstance(merged, PlanMergeOut)
    selected = _clean_selection(merged.key_claims, pool)

    for round_no in range(1, max(0, review_rounds) + 1):
        if progress:
            progress("review", {"round": round_no, "claims": len(selected)})
        review = llm.generate(LLMRequest(
            task="plan_review",
            system=[text_block(REVIEW_SYSTEM), text_block(pool_text)],
            user=_selection_text(selected, pool, chapter),
            schema=PlanReviewOut,
            effort="high",
        ))
        assert isinstance(review, PlanReviewOut)
        remove = {r.strip().strip("[]") for r in review.remove}
        chosen = {p.candidate for p in selected}
        additions = [a.strip().strip("[]") for a in review.add if a.strip().strip("[]") not in chosen]
        additions = [a for a in additions if pool.claim(a)]
        if not remove & chosen and not additions:
            break
        selected = [p for p in selected if p.candidate not in remove]
        for cid in additions:
            _, _, c = pool.claim(cid)  # type: ignore[misc]
            selected.append(PickedClaimOut(candidate=cid, claim=c.claim, difficulty=c.difficulty, exam_relevance=c.exam_relevance))
        selected = selected[:MAX_CLAIMS]
        pool.warnings.extend(f"review {round_no}: {p}" for p in review.problems[:5])

    wanted = {x.strip().strip("[]") for x in merged.definitions}
    definitions = [(sid, d) for did, sid, d in pool.definitions if did in wanted]
    example = next((w for wid, w in pool.examples if wid == (merged.worked_example or "").strip().strip("[]")), None)
    out = PlanOut(
        summary=merged.summary,
        learning_objectives=merged.learning_objectives,
        key_claims=[ClaimOut(claim=p.claim, section=pool.claim(p.candidate)[1],  # type: ignore[index]
                             source_quote=pool.claim(p.candidate)[2].source_quote,  # type: ignore[index]
                             difficulty=p.difficulty, exam_relevance=p.exam_relevance) for p in selected],
        definitions=[DefinitionOut(term=d.term, definition=d.definition, section=sid, source_quote=d.source_quote)
                     for sid, d in definitions],
        misconceptions=merged.misconceptions or pool.misconceptions[:5],
        worked_example=example,
        expert_domain=merged.expert_domain,
        expert_reason=merged.expert_reason,
        formula_dense_sections=sorted(pool.formula_dense),
    )
    plan = build_plan(chapter, out)
    plan.warnings = pool.warnings + plan.warnings
    return plan


# ---------------------------------------------------------------------------
# Lexicon, per section
# ---------------------------------------------------------------------------

def propose_lexicon_thorough(chapter: Chapter, glossary: Glossary, llm: LLM,
                             progress: Progress | None = None) -> list[LexiconEntry]:
    """One lexicon call per section part. Each sees the entries found so far, so a term gets one spelling."""
    from pipeline.plan.glossary import propose_lexicon

    working = glossary.model_copy(deep=True)
    found: list[LexiconEntry] = []
    for section in chapter.sections:
        for k, text in enumerate(section_parts(section), start=1):
            if not text.strip():
                continue
            if progress:
                progress("lexicon_section", {"section": section.id, "part": k})
            part = Chapter(id=chapter.id, title=chapter.title, pages=chapter.pages,
                           sections=[section.model_copy(update={"text": text})])
            entries = propose_lexicon(part, working, llm)
            working.merge(entries)
            found.extend(entries)
    return found
