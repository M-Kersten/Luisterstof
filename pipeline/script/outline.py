"""Stage 3, before a line is written: the thread of the episode.

A content plan is a list of claims in the order the book has them. Cut into equal slices it gives blocks
that each say five things and lead nowhere. The outline decides what the episode is about (one question a
listener would ask), which step answers which part of it, in what order the claims have to be told for a
listener to follow, and where a block hands over to the next.

One model call. Whatever comes back is put right here: every claim ends up in exactly one block, unknown
ids are dropped, and when the model gives nothing usable the claims go into blocks in book order, which
is what the writer did before there was an outline.
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, Field

from pipeline.llm import LLM, LLMError, LLMRequest, LLMUnavailable, text_block
from pipeline.models import Cast, Chapter, ContentPlan, EpisodeOutline, KeyClaim, OutlineBlock

log = logging.getLogger(__name__)

MAX_MOMENTS = 2  # a story or an argument in every block is a format, not a conversation
OPENING_CHARS = 1500  # how much of the chapter's own opening the model sees
MOMENTS = ("anekdote", "meningsverschil", "plagen", "kwartje")


class OutlineBlockOut(BaseModel):
    question: str = Field(description="De deelvraag die dit blok beantwoordt, zoals een luisteraar hem zou stellen. "
                                      "Gewone woorden, geen vakterm.")
    claims: list[str] = Field(description="Id's van de beweringen in dit blok, in de volgorde waarin je ze vertelt.")
    bridge: str = Field(description="De vraag die het antwoord van dit blok oproept en waar het volgende blok mee "
                                    "begint. Leeg bij het laatste blok.")
    moment: Literal["", "anekdote", "meningsverschil", "plagen", "kwartje"] = Field(
        description="Een menselijk moment, alleen als het hier echt past. Meestal leeg.")
    moment_note: str = Field(description="Bij anekdote: welk verhaal van welke host, en bij welke bewering het hoort. "
                                         "Bij meningsverschil: waarover. Anders leeg.")


class OutlineOut(BaseModel):
    question: str = Field(description="De ene vraag waar deze aflevering antwoord op geeft, zoals een nieuwsgierige "
                                      "vriend hem zou stellen. Gewone woorden, geen vakterm.")
    opening: str = Field(description="Een herkenbare situatie om in te beginnen, in twee zinnen. Bij voorkeur de "
                                     "situatie waarmee het hoofdstuk zelf opent.")
    blocks: list[OutlineBlockOut]
    takeaway: str = Field(description="Het antwoord op de hoofdvraag in één of twee zinnen die je hardop tegen een "
                                      "vriend zou zeggen. Alleen wat in de beweringen staat.")


OUTLINE_SYSTEM = """Je maakt het draaiboek voor één aflevering van een Nederlandse studiepodcast: twee mensen die samen een hoofdstuk van een studieboek doorpraten. Je schrijft geen dialoog. Jij bepaalt de rode draad, zodat het gesprek ergens heen gaat in plaats van losse punten af te werken.

Regels:
- De hoofdvraag is één vraag in gewone woorden, zonder vakterm. Alles in de aflevering helpt die vraag te beantwoorden.
- Verdeel de beweringen over het gevraagde aantal blokken. Elk blok beantwoordt één deelvraag. De deelvragen volgen uit elkaar: het antwoord op het ene blok roept de vraag van het volgende op. Die overgang zet je in bridge.
- Elke bewering komt precies één keer voor, met zijn id. Laat er geen weg en verzin er geen bij.
- Zet de beweringen in de volgorde waarin je ze moet vertellen. Wat je nodig hebt om iets te snappen komt eerst: de opstelling van een proef vóór de uitkomst, een begrip vóór het cijfer erover, de gewone verklaring vóór het onderzoek dat hem onderuit haalt. Dat mag afwijken van de volgorde in het boek.
- Beweringen die bij elkaar horen (dezelfde proef, hetzelfde begrip, dezelfde verklaring) staan in hetzelfde blok.
- Een menselijk moment krijgt een blok alleen als het daar echt past, in hooguit twee blokken. anekdote: een host vertelt iets uit eigen leven dat de bewering duidelijker maakt; kies dan in moment_note één verhaal uit de lijst van die host dat er inhoudelijk mee te maken heeft, en past er geen, dan geen anekdote. meningsverschil: de hosts zijn het oneens over wat de stof betekent. kwartje: de vragensteller snapt het opeens en zegt het in eigen woorden terug. plagen: een korte plaagstoot over iets wat eerder in deze aflevering is gezegd.
- Alles in het Nederlands.
"""


def _stories(cast: Cast) -> str:
    rows = ["## Verhalen die de hosts kunnen vertellen (alleen voor een anekdote, en alleen als er een past)"]
    for h in cast.hosts:
        rows.append(f"### {h.name}")
        rows += [f"- {s}" for s in h.stories] or ["- (geen)"]
    return "\n".join(rows)


def _plan_rows(plan: ContentPlan, claims: list[KeyClaim]) -> str:
    rows = [f"# Hoofdstuk {plan.chapter_id}: {plan.chapter_title}", plan.summary, "",
            "## Beweringen (id, moeilijkheid, tekst, in de volgorde van het boek)"]
    for c in claims:
        rows.append(f"[{c.id}] (m{c.difficulty}) {c.claim}" + (f" (vooraf nodig: {c.context})" if c.context else ""))
    rows.append("\n## Begrippen")
    rows += [f"- {d.term}: {d.definition}" for d in plan.definitions] or ["- (geen)"]
    rows.append("\n## Misvattingen")
    rows += [f"- fout: {m.wrong} | juist: {m.right}" for m in plan.misconceptions] or ["- (geen)"]
    return "\n".join(rows)


def fallback_outline(plan: ContentPlan, claims: list[KeyClaim], n_blocks: int) -> EpisodeOutline:
    """Claims in book order, in equal blocks. No questions, no bridges, no moments."""
    n_blocks = max(1, min(n_blocks, len(claims))) if claims else 0
    blocks = []
    for i in range(n_blocks):
        size = len(claims) / n_blocks
        chunk = claims[round(i * size):round((i + 1) * size)]
        if chunk:
            blocks.append(OutlineBlock(claims=[c.id for c in chunk]))
    return EpisodeOutline(question=_as_question(plan.chapter_title), blocks=blocks, planned=False)


def _as_question(title: str) -> str:
    title = " ".join((title or "").split())
    return title if title.endswith("?") else ""


def repair_outline(out: OutlineOut, plan: ContentPlan, claims: list[KeyClaim], n_blocks: int) -> EpisodeOutline:
    """The model's outline with every claim in exactly one block. A claim the model forgot goes into the
    block that holds its neighbour in the book, right after it."""
    known = [c.id for c in claims]
    seen: set[str] = set()
    blocks: list[OutlineBlock] = []
    for b in out.blocks:
        ids = []
        for raw in b.claims:
            cid = raw.strip().strip("[]")
            if cid in known and cid not in seen:
                seen.add(cid)
                ids.append(cid)
        if ids:
            blocks.append(OutlineBlock(question=" ".join(b.question.split()), claims=ids, bridge=" ".join(b.bridge.split()),
                                       moment=b.moment if b.moment in MOMENTS else "",
                                       moment_note=" ".join(b.moment_note.split())))
    if not blocks:
        return fallback_outline(plan, claims, n_blocks)
    for i, cid in enumerate(known):
        if cid in seen:
            continue
        before = next((p for p in reversed(known[:i]) if p in seen), None)
        home = next((b for b in blocks if before in b.claims), blocks[0]) if before else blocks[0]
        home.claims.insert(home.claims.index(before) + 1 if before in home.claims else 0, cid)
        seen.add(cid)
    kept = 0
    for b in blocks:
        if b.moment:
            kept += 1
            if kept > MAX_MOMENTS:
                b.moment, b.moment_note = "", ""
    blocks[-1].bridge = ""
    return EpisodeOutline(question=" ".join(out.question.split()) or _as_question(plan.chapter_title),
                          opening=" ".join(out.opening.split()), blocks=blocks,
                          takeaway=" ".join(out.takeaway.split()), planned=True)


def plan_outline(plan: ContentPlan, claims: list[KeyClaim], cast: Cast, llm: LLM, *, n_blocks: int,
                 chapter: Chapter | None = None) -> EpisodeOutline:
    """The outline for the claims the hosts tell themselves (a guest's claims are not in ``claims``)."""
    if not claims:
        return fallback_outline(plan, claims, n_blocks)
    opening = ""
    if chapter is not None:
        opening = ("\n\n## Zo begint het hoofdstuk zelf\n" + " ".join(chapter.full_text().split())[:OPENING_CHARS])
    request = LLMRequest(
        task="script_outline",
        system=[text_block(OUTLINE_SYSTEM), text_block(_plan_rows(plan, claims) + opening + "\n\n" + _stories(cast))],
        user=f"Maak het draaiboek voor deze aflevering in {n_blocks} blokken. Elke bewering precies één keer: "
             + ", ".join(c.id for c in claims) + ".",
        schema=OutlineOut,
    )
    try:
        out = llm.generate(request)
    except LLMUnavailable:
        raise  # the model is gone: every next call fails the same way
    except LLMError as exc:  # no handler in a fake, an unparseable answer: the episode is still written
        log.warning("no outline for %s (%s); claims go into blocks in book order", plan.chapter_id, exc)
        return fallback_outline(plan, claims, n_blocks)
    assert isinstance(out, OutlineOut)
    return repair_outline(out, plan, claims, n_blocks)
