"""Thorough content plan and lexicon: many small passes instead of one big one.

One call over a whole chapter suits a frontier model. A local model loses
sections, picks the wrong claims and paraphrases its "verbatim" quotes. So:

1. per part (short sections together up to ``PART_CHARS``, long sections in
   pieces): candidate claims, definitions, misconceptions, worked examples,
   with that text as the only source;
2. every quote is checked against that text. A near miss is replaced by the
   source passage itself; a quote that isn't there at all goes back once to
   be fixed, and a candidate whose quote still isn't in the text is dropped;
   a substantial part that yields nothing gets a second pass;
3. a chapter pass picks and merges the claims by candidate id, so quotes are
   never rewritten there, and writes summary and learning objectives;
4. review rounds compare the selection with the section list and the candidate
   pool and add or remove candidates.

The result goes through the same ``build_plan`` as the single-call plan.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

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

PART_CHARS = 8000  # one call reads up to this much: short sections together, a longer one in parts
SUBSTANTIAL_SECTION_CHARS = 400  # below this an empty harvest is plausible
QUOTE_MIN_SCORE = 95.0  # near-literal only; the single-call plan accepts 80, which lets paraphrases through
REPAIR_MIN_SCORE = 85.0  # between this and QUOTE_MIN_SCORE the quote is replaced by the source passage itself
MAX_CLAIMS = 25

Progress = Callable[[str, dict], None]


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class SectionClaimOut(BaseModel):
    section: str = Field(description="Id van de sectie waar het citaat staat.")
    claim: str = Field(description="Eén feitelijke bewering uit deze tekst, zelfstandig leesbaar, in het Nederlands.")
    source_quote: str = Field(description="Letterlijk citaat uit deze tekst van 20 tot 300 tekens dat de bewering onderbouwt. "
                                          "Exact overnemen, inclusief getallen en leestekens.")
    difficulty: int = Field(description="1 (triviaal) tot 5 (zeer moeilijk voor een eerstejaars).")
    exam_relevance: int = Field(description="1 (bijzaak) tot 5 (zeker tentamenstof).")


class SectionDefinitionOut(BaseModel):
    section: str = Field(description="Id van de sectie waar het citaat staat.")
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

def section_parts(section: Section, max_chars: int = PART_CHARS) -> list[str]:
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


@dataclass
class Part:
    """What one call reads: several short sections together, or one piece of a long section."""

    pieces: list[tuple[Section, str]]  # (section, its text in this part)
    label: str

    def text(self) -> str:
        return "\n\n".join(f"### Sectie {s.id}: {s.title}\n{t}" for s, t in self.pieces)

    def chars(self) -> int:
        return sum(len(t) for _, t in self.pieces)


def chapter_parts(chapter: Chapter, max_chars: int = PART_CHARS) -> list[Part]:
    """Consecutive short sections share a call up to ``max_chars``; a longer section is read in parts.
    Every call has a fixed cost (instructions, schema, a JSON answer), so a heading with two paragraphs
    under it isn't worth a call of its own."""
    parts: list[Part] = []
    group: list[tuple[Section, str]] = []

    def flush() -> None:
        if group:
            ids = [s.id for s, _ in group]
            label = f"sectie {ids[0]}" if len(ids) == 1 else "de secties " + ", ".join(ids[:-1]) + f" en {ids[-1]}"
            parts.append(Part(list(group), label))
            group.clear()

    for section in chapter.sections:
        if not section.text.strip():
            continue
        pieces = section_parts(section, max_chars)
        if len(pieces) > 1:
            flush()
            parts += [Part([(section, t)], f"sectie {section.id} (deel {k} van {len(pieces)})")
                      for k, t in enumerate(pieces, start=1)]
            continue
        if group and sum(len(t) for _, t in group) + len(section.text) > max_chars:
            flush()
        group.append((section, section.text))
    flush()
    return parts


class Candidates:
    def __init__(self) -> None:
        self.claims: list[tuple[str, str, SectionClaimOut]] = []  # (id, section id, claim)
        self.definitions: list[tuple[str, str, SectionDefinitionOut]] = []
        self.misconceptions: list[MisconceptionOut] = []
        self.examples: list[tuple[str, WorkedExampleOut]] = []
        self.formula_dense: set[str] = set()
        self.warnings: list[str] = []
        self.repaired = 0  # quotes put right from the source text, without a model call

    def claim(self, cid: str) -> tuple[str, str, SectionClaimOut] | None:
        return next((c for c in self.claims if c[0] == cid), None)

    def per_section(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for _, sid, _ in self.claims:
            counts[sid] = counts.get(sid, 0) + 1
        return counts


def _section_request(book: Book, chapter: Chapter, part: Part, fix: str | None = None) -> LLMRequest:
    outline = "\n".join(f"- {s.id}: {s.title}" for s in chapter.sections)
    user = (f"Haal de bouwstenen uit {part.label}. Geef bij elke bewering en definitie het id van de sectie "
            f"waar het citaat staat.\n\nDe secties van dit hoofdstuk, ter oriëntatie:\n{outline}")
    if fix:
        user += "\n\n" + fix
    return LLMRequest(
        task="plan_section",
        system=[text_block(SECTION_SYSTEM),
                text_block(f"# Boek: {book.title}\n# Hoofdstuk {chapter.id}: {chapter.title}\n\n{part.text()}")],
        user=user,
        schema=SectionPlanOut,
        effort="high",
    )


def _snap(text: str, start: int, end: int) -> str:
    """The source passage at [start, end), widened to whole words."""
    while start > 0 and not text[start - 1].isspace():
        start -= 1
    while end < len(text) and not text[end].isspace():
        end += 1
    return " ".join(text[start:end].split())


def _place(part: Part, quote: str, hint: str) -> tuple[str, str, bool] | None:
    """(section id, verified quote, repaired) for a quote in this part, or None when it isn't there.

    A quote that is near-literal but not quite (a changed word, a dropped comma) is replaced by the
    passage it was taken from. That is what the quote-fix call would do, without the call."""
    best: tuple[float, bool, Section, str, int, int] | None = None
    for section, text in part.pieces:
        hit = resolve_span(text, quote, REPAIR_MIN_SCORE)
        if hit:
            key = (hit[2], section.id == hint)
            if best is None or key > best[:2]:
                best = (hit[2], section.id == hint, section, text, hit[0], hit[1])
    if best is None:
        return None
    score, _, section, text, start, end = best
    if score >= QUOTE_MIN_SCORE:
        return section.id, quote, False
    return section.id, _snap(text, start, end), True


def _fix_quotes(llm: LLM, part: Part, failed: list[tuple[str, str, str]]) -> dict[str, str]:
    """failed: (quote id, what it supports, quote). Returns quote id -> the model's corrected quote (unverified)."""
    listing = "\n".join(f"[{qid}] bij: {what}\n     citaat: \"{quote}\"" for qid, what, quote in failed)
    request = LLMRequest(
        task="plan_quotes",
        system=[text_block("Je corrigeert citaten. Een citaat moet teken voor teken in de tekst staan. Zoek per citaat de passage "
                           "in de tekst die hetzelfde zegt en kopieer die letterlijk (20 tot 300 tekens). Staat het er niet, geef dan "
                           "een lege source_quote."),
                text_block(part.text())],
        user=f"Deze citaten staan niet letterlijk in de tekst:\n{listing}",
        schema=QuoteFixOut,
        effort="medium",
    )
    out = llm.generate(request)
    assert isinstance(out, QuoteFixOut)
    return {f.id: f.source_quote for f in out.fixes if f.source_quote.strip()}


def harvest_part(book: Book, chapter: Chapter, part: Part, llm: LLM, pool: Candidates,
                 progress: Progress | None = None) -> None:
    if progress:
        progress("section", {"sections": [s.id for s, _ in part.pieces], "chars": part.chars()})
    out = llm.generate(_section_request(book, chapter, part))
    assert isinstance(out, SectionPlanOut)
    if not out.claims and part.chars() >= SUBSTANTIAL_SECTION_CHARS:
        out = llm.generate(_section_request(
            book, chapter, part,
            fix=f"Een eerdere poging vond in deze tekst geen enkele bewering, terwijl de tekst {part.chars()} tekens telt. "
                "Lees opnieuw en haal de beweringen eruit die een student moet kennen."))
        assert isinstance(out, SectionPlanOut)
    _collect(llm, part, out, pool)


def _collect(llm: LLM, part: Part, out: SectionPlanOut, pool: Candidates) -> None:
    first = part.pieces[0][0].id
    items: list[tuple[str, str, str, str]] = []  # (qid, what, quote, section hint)
    for i, c in enumerate(out.claims):
        items.append((f"q{i + 1}", c.claim, c.source_quote, c.section.strip()))
    offset = len(out.claims)
    for j, d in enumerate(out.definitions):
        items.append((f"q{offset + j + 1}", f"definitie {d.term}", d.source_quote, d.section.strip()))

    placed = {qid: _place(part, quote, hint) for qid, _, quote, hint in items}
    failed = [(qid, what, quote) for qid, what, quote, _ in items if placed[qid] is None]
    if failed:
        hints = {qid: hint for qid, _, _, hint in items}
        for qid, quote in _fix_quotes(llm, part, failed).items():
            if qid in placed and placed[qid] is None:
                placed[qid] = _place(part, quote, hints.get(qid, first))
    pool.repaired += sum(1 for v in placed.values() if v and v[2])

    for i, c in enumerate(out.claims):
        where = placed[f"q{i + 1}"]
        if where is None:
            pool.warnings.append(f"sectie {c.section or first}: bewering geschrapt, citaat staat niet in de bron: {c.claim[:80]}")
            continue
        pool.claims.append((f"k{len(pool.claims) + 1}", where[0], c.model_copy(update={"source_quote": where[1], "section": where[0]})))
    for j, d in enumerate(out.definitions):
        where = placed[f"q{offset + j + 1}"]
        if where is None:
            pool.warnings.append(f"sectie {d.section or first}: definitie {d.term} geschrapt, citaat staat niet in de bron")
            continue
        pool.definitions.append((f"d{len(pool.definitions) + 1}", where[0],
                                 d.model_copy(update={"source_quote": where[1], "section": where[0]})))
    pool.misconceptions.extend(out.misconceptions)
    if out.worked_example and out.worked_example.steps:
        pool.examples.append((f"w{len(pool.examples) + 1}", out.worked_example))
    if out.formula_dense:
        pool.formula_dense.update(s.id for s, _ in part.pieces)


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
                          review_rounds: int = 1, part_chars: int = PART_CHARS,
                          progress: Progress | None = None) -> ContentPlan:
    chapter = book.chapter(chapter_id)
    pool = Candidates()
    for part in chapter_parts(chapter, part_chars):
        harvest_part(book, chapter, part, llm, pool, progress)
    counts = pool.per_section()
    pool.warnings += [f"sectie {s.id}: geen enkele bewering met een controleerbaar citaat gevonden"
                      for s in chapter.sections if len(s.text.strip()) >= SUBSTANTIAL_SECTION_CHARS and not counts.get(s.id)]
    if pool.repaired:
        log.info("%s: %d quotes repaired from the source text", chapter_id, pool.repaired)
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

def propose_lexicon_thorough(chapter: Chapter, glossary: Glossary, llm: LLM, *, part_chars: int = PART_CHARS,
                             progress: Progress | None = None) -> list[LexiconEntry]:
    """One lexicon call per part (the same parts as the plan). Each sees the entries found so far,
    so a term gets one spelling."""
    from pipeline.plan.glossary import propose_lexicon

    working = glossary.model_copy(deep=True)
    found: list[LexiconEntry] = []
    for part in chapter_parts(chapter, part_chars):
        if progress:
            progress("lexicon_section", {"sections": [s.id for s, _ in part.pieces], "chars": part.chars()})
        sub = Chapter(id=chapter.id, title=chapter.title, pages=chapter.pages,
                      sections=[s.model_copy(update={"text": t}) for s, t in part.pieces])
        entries = propose_lexicon(sub, working, llm)
        working.merge(entries)
        found.extend(entries)
    return found
