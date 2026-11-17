"""Stage 3: dialogue generation.

The episode format is fixed by a deterministic blueprint (which segments,
which claims each one must cover, how long it may run). The model writes one
segment at a time with the previous lines as context, so the stable prefix
(rules, cast, continuity, plan, glossary) is cached across calls.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from rapidfuzz import fuzz

from pipeline.config import Settings
from pipeline.cues import cue_reaction, pure_reaction, speakable, strip_cues
from pipeline.llm import LLM, LLMRequest, text_block
from pipeline.models import (
    AuditIssue,
    AuditResult,
    Book,
    Cast,
    ContentPlan,
    ContinuityEntry,
    EpisodeOutline,
    Glossary,
    Guest,
    KeyClaim,
    Line,
    Overlap,
    Script,
    Segment,
    SegmentType,
)
from pipeline.script.cast import cast_bible_text, continuity_text, pick_guest, story_seeds, voice_samples_text
from pipeline.script.outline import MAX_MOMENTS, fallback_outline, plan_outline
from pipeline.tags import known_tags

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Blueprint
# ---------------------------------------------------------------------------

@dataclass
class SegmentBrief:
    type: SegmentType
    title: str
    covers: list[str]
    target_chars: int
    instructions: str
    speakers: list[str]
    extra: dict = field(default_factory=dict)


SEGMENT_SECONDS = {
    "cold_open": 55,
    "recap": 90,
    "outro": 40,
}


# How long a claim takes to land in conversation, by its difficulty (1-5). The hosts tell the material as a
# story rather than state it, which takes longer than a list of facts. With a short frame around the body
# this sizes the episode to its material: eight easy pages make a short episode, a dense chapter a long
# one, and STUDIEPODCAST_TARGET_MINUTES is the ceiling.
SECONDS_PER_CLAIM = {1: 15, 2: 25, 3: 40, 4: 60, 5: 85}
SECONDS_PER_MISCONCEPTION = 10
MIN_EPISODE_MINUTES = 6
GUEST_MIN_MINUTES = 15  # a guest in a ten-minute episode is padding
GUEST_MAX_CLAIMS = 4
GUEST_INTRO_SECONDS = 40  # bringing someone in, and the hosts saying it back in plain words
GUEST_TALK_FACTOR = 1.3  # three people take longer over the same point than two
REEXPLAIN_MIN_DIFFICULTY = 4  # a second route only for a claim that is actually hard
REEXPLAIN_SECONDS = 120
QUIZ_SECONDS_PER_QUESTION = 50
HUMAN_MOMENT_SECONDS = 45  # a story, an argument or an aha, told at length
BODY_SHORTFALL = 0.9  # below this share of what the claims need, the extras go before the claims are squeezed
PASSAGE_CONTEXT = 400  # characters of the book on either side of a claim's quote that the writer gets to read


def _block_count(claims: list[KeyClaim]) -> int:
    return max(3, min(5, round(len(claims) / 4))) if claims else 3


def claim_seconds(claims: list[KeyClaim]) -> float:
    return float(sum(SECONDS_PER_CLAIM.get(max(1, min(5, c.difficulty)), 35) for c in claims))


def human_moment(kind: str, explainer, skeptic, *, note: str = "", teller=None) -> str:
    """The instruction for one human moment. Which block gets which moment is the outline's call: a story in
    every block is a format, and a format is what makes a conversation sound written."""
    teller = teller or explainer
    listener = skeptic if teller is explainer else explainer
    text = {
        "anekdote": (f"{teller.name} vertelt iets uit eigen leven dat dit punt duidelijker maakt. Als een echt verhaal, "
                     "in een paar lange beurten: waar en wanneer, wie erbij was (zeg erbij wie dat is), wat er misging "
                     f"of verraste, hoe het afliep. {listener.name} reageert op het verhaal zelf. Daarna zegt "
                     f"{teller.name} hardop wat het verhaal met de stof te maken heeft. Een verhaal dat niet over dit "
                     "punt gaat, vertel je niet."),
        "meningsverschil": (f"{explainer.name} en {skeptic.name} zijn het hier echt oneens over wat de stof betekent, "
                            "allebei met een punt. Het wordt niet in één regel opgelost: ze duwen een paar keer heen en "
                            "weer en komen er samen uit. Wie een vraag stelt of een bezwaar maakt, krijgt daar antwoord op."),
        "plagen": (f"{skeptic.name} plaagt {explainer.name} (of andersom) met iets wat eerder in deze aflevering is "
                   "gezegd, en de ander pareert. Kort en warm, daarna verder met de stof."),
        "kwartje": (f"Bij {skeptic.name} valt hoorbaar het kwartje: eerst verwarring, dan het moment dat het klikt, en dan "
                    f"vertelt {skeptic.name} het in een paar eigen zinnen terug. {explainer.name} zegt of het klopt en "
                    "praat er niet meteen overheen."),
    }.get(kind, "")
    if text and note:
        text += f" Uit het draaiboek: {note}"
    return text


def _moment_teller(note: str, explainer, skeptic):
    """Whose story an anecdote is: the host the outline's note names, else the explainer."""
    low = note.casefold()
    named = [h for h in (explainer, skeptic) if h.name.casefold() in low]
    return named[0] if len(named) == 1 else explainer


def quiz_questions(target_minutes: int) -> int:
    return 2 if target_minutes < 10 else 3 if target_minutes < 18 else 4


def guest_claims(plan: ContentPlan, guest: Guest | None, target_minutes: int) -> list[KeyClaim]:
    """The claims a guest tells: the hardest few, in the order of the book. They are the guest's alone. Told
    by the hosts first and by the guest again, a claim is heard twice in the same words."""
    if guest is None or not plan.needs_expert or target_minutes < GUEST_MIN_MINUTES:
        return []
    claims = list(plan.key_claims)
    share = max(1, min(GUEST_MAX_CLAIMS, len(claims) // 3))
    hard = {c.id for c in sorted(claims, key=lambda c: -c.difficulty)[:share]}
    return [c for c in claims if c.id in hard]


@dataclass
class Shape:
    """What an episode holds and how its seconds are divided. The claims come first: a guest and a second
    route are extras, and they go when the body would otherwise get less time than its claims need."""
    body_claims: list[KeyClaim]
    guest_claims: list[KeyClaim]
    reexplain: KeyClaim | None
    n_quiz: int
    seconds: dict[str, float]


def _body_need(claims: list[KeyClaim], plan: ContentPlan, moments: int) -> float:
    return claim_seconds(claims) + SECONDS_PER_MISCONCEPTION * len(plan.misconceptions) + HUMAN_MOMENT_SECONDS * moments


def plan_shape(plan: ContentPlan, *, guest: Guest | None, target_minutes: int, has_recap: bool = False,
               moments: int = 0) -> Shape:
    total = target_minutes * 60
    claims = list(plan.key_claims)
    n_quiz = quiz_questions(target_minutes)
    seconds = {"cold_open": float(SEGMENT_SECONDS["cold_open"]), "outro": float(SEGMENT_SECONDS["outro"]),
               "quiz": float(QUIZ_SECONDS_PER_QUESTION * n_quiz),
               "recap": min(float(SEGMENT_SECONDS["recap"]), 0.08 * total) if has_recap else 0.0,
               "guest": 0.0, "reexplain": 0.0}
    frame = seconds["cold_open"] + seconds["outro"] + seconds["quiz"] + seconds["recap"]
    hardest = plan.hardest_claim()
    second_route = hardest if hardest is not None and hardest.difficulty >= REEXPLAIN_MIN_DIFFICULTY else None
    for_guest = guest_claims(plan, guest, target_minutes)

    def room(guest_part: list[KeyClaim], again: KeyClaim | None) -> tuple[float, float]:
        """(seconds the body gets, seconds its claims need) with these extras in."""
        ids = {c.id for c in guest_part}
        guest_s = claim_seconds(guest_part) * GUEST_TALK_FACTOR + GUEST_INTRO_SECONDS if guest_part else 0.0
        need = _body_need([c for c in claims if c.id not in ids], plan, moments)
        return total - frame - guest_s - (REEXPLAIN_SECONDS if again else 0.0), need

    got, need = room(for_guest, second_route)
    if got < BODY_SHORTFALL * need and second_route is not None:
        second_route = None
        got, need = room(for_guest, None)
    if got < BODY_SHORTFALL * need and for_guest:
        for_guest = []
        got, need = room([], None)
    if for_guest:
        seconds["guest"] = claim_seconds(for_guest) * GUEST_TALK_FACTOR + GUEST_INTRO_SECONDS
    if second_route is not None:
        seconds["reexplain"] = float(REEXPLAIN_SECONDS)
    seconds["body"] = max(3 * 60.0, got)
    ids = {c.id for c in for_guest}
    return Shape(body_claims=[c for c in claims if c.id not in ids], guest_claims=for_guest, reexplain=second_route,
                 n_quiz=n_quiz, seconds=seconds)


def episode_minutes(plan: ContentPlan, max_minutes: int, *, with_guest: bool = False) -> int:
    """The episode length this plan carries, between MIN_EPISODE_MINUTES and ``max_minutes``: what the claims
    need, plus the frame around them, plus the second route and the guest where the plan asks for them."""
    claims = list(plan.key_claims)
    need = _body_need(claims, plan, min(MAX_MOMENTS, _block_count(claims)))
    hardest = plan.hardest_claim()
    if hardest is not None and hardest.difficulty >= REEXPLAIN_MIN_DIFFICULTY:
        need += REEXPLAIN_SECONDS
    if with_guest and plan.needs_expert:
        need += GUEST_INTRO_SECONDS
    minutes = MIN_EPISODE_MINUTES
    for n_quiz in (2, 3, 4):
        frame = SEGMENT_SECONDS["cold_open"] + SEGMENT_SECONDS["outro"] + QUIZ_SECONDS_PER_QUESTION * n_quiz
        minutes = math.ceil((need + frame) / 60)
        if quiz_questions(minutes) <= n_quiz:
            break
    return max(MIN_EPISODE_MINUTES, min(max_minutes, minutes))


def _about(text: str, claims: list[KeyClaim]) -> float:
    return max((fuzz.token_set_ratio(text, c.claim) for c in claims), default=0.0)


def _assign_misconceptions(misconceptions: list, clusters: list[list[KeyClaim]], open_claim: KeyClaim | None = None,
                           min_score: float = 50.0) -> tuple[object | None, dict[int, object]]:
    """Each misconception goes to the block whose claims it is about (at most one per block), never by its
    position in the list: a misconception about 1584 does not belong in the block about 1566. With
    ``open_claim`` the opening may take one that is about that claim."""
    def about(m, claims: list[KeyClaim]) -> float:
        return _about(f"{m.wrong} {m.right}", claims)

    pool = list(misconceptions)
    opener = None
    if open_claim is not None and pool:
        best = max(pool, key=lambda m: about(m, [open_claim]))
        if about(best, [open_claim]) >= min_score + 10:
            opener = best
            pool.remove(best)
    pairs = sorted(((about(m, cl), i, k) for k, m in enumerate(pool) for i, cl in enumerate(clusters)), reverse=True)
    per_block: dict[int, object] = {}
    used: set[int] = set()
    for score, i, k in pairs:
        if score < min_score or i in per_block or k in used:
            continue
        per_block[i] = pool[k]
        used.add(k)
    return opener, per_block


def _chars(seconds: float, cps: float) -> int:
    return int(seconds * cps * 0.9)


def _claim_text(c: KeyClaim) -> str:
    return f"[{c.id}] (moeilijkheid {c.difficulty}, tentamen {c.exam_relevance}) {c.claim}"


def _claim_rows(claims: list[KeyClaim], *, numbered: bool = False) -> list[str]:
    rows = []
    for n, c in enumerate(claims, start=1):
        rows.append((f"{n}. " if numbered else "") + _claim_text(c))
        if c.context:
            rows.append(f"   eerst vertellen, anders kan de luisteraar dit niet plaatsen: {c.context}")
    return rows


def _sentence_window(text: str, start: int, end: int) -> tuple[int, int]:
    """[start, end) widened to whole sentences."""
    before = max(text.rfind(mark, 0, start) for mark in (". ", "? ", "! ", "\n"))
    after = min((i for i in (text.find(mark, end) for mark in (". ", "? ", "! ", "\n")) if i >= 0), default=len(text) - 1)
    return (before + 1 if before >= 0 else 0), min(len(text), after + 1)


def source_passages(book: Book | None, claims: list[KeyClaim], context: int = PASSAGE_CONTEXT) -> list[str]:
    """What the book says around these claims, as rows for a brief. A claim's own quote is one sentence, and
    a writer who gets one sentence reads it out. The paragraph around it has who did the study, what the
    setup was and why the outcome is what it is. Passages that overlap are joined, so a block about one
    experiment gets that experiment once. Without a book: the bare quotes."""
    spans: list[tuple[str, int, int, list[str]]] = []
    loose: list[str] = []
    for c in claims:
        span = c.source_span
        text = None
        if book is not None:
            try:
                text = book.section(span.section).text
            except KeyError:
                text = None
        if text is None or span.end <= span.start or span.end > len(text):
            if span.quote:
                loose.append(f"  bij {c.id}: \"{span.quote[:300]}\"")
            continue
        start, end = _sentence_window(text, max(0, span.start - context), min(len(text), span.end + context))
        spans.append((span.section, start, end, [c.id]))
    spans.sort(key=lambda s: (s[0], s[1]))
    merged: list[tuple[str, int, int, list[str]]] = []
    for sec, start, end, ids in spans:
        if merged and merged[-1][0] == sec and start <= merged[-1][2]:
            last = merged[-1]
            merged[-1] = (sec, last[1], max(last[2], end), last[3] + ids)
        else:
            merged.append((sec, start, end, ids))
    rows = []
    for sec, start, end, ids in merged:
        passage = " ".join(book.section(sec).text[start:end].split())  # type: ignore[union-attr]
        rows.append(f"  bij {', '.join(ids)}: \"{passage}\"")
    return rows + loose


BACKGROUND_HEADER = ("Wat er in het boek omheen staat. Lees het om te snappen wie, wat en waarom; vertel het na in je "
                     "eigen woorden en neem er geen zin letterlijk uit over:")


def _fit_outline(outline: EpisodeOutline | None, plan: ContentPlan, claims: list[KeyClaim]) -> EpisodeOutline:
    """The outline over exactly these claims: an id that is not among them goes, a claim no block has is added
    to the last one. Without an outline the claims go into blocks in book order."""
    if outline is None or not outline.blocks:
        return fallback_outline(plan, claims, _block_count(claims))
    known = {c.id for c in claims}
    seen: set[str] = set()
    blocks = []
    for b in outline.blocks:
        ids = [cid for cid in b.claims if cid in known and cid not in seen]
        seen.update(ids)
        if ids:
            blocks.append(b.model_copy(update={"claims": ids}))
    if not blocks:
        return fallback_outline(plan, claims, _block_count(claims))
    blocks[-1].claims += [c.id for c in claims if c.id not in seen]
    return outline.model_copy(update={"blocks": blocks})


def _quiz_claims(blocks: list[list[KeyClaim]], n: int) -> list[KeyClaim]:
    """One question per block where that works, the hardest of the claims that matter for the exam: a quiz on
    the four easiest facts tests whether someone was awake, not whether they understood."""
    def rank(c: KeyClaim) -> tuple:
        return (-c.exam_relevance, -c.difficulty)

    everything = [c for block in blocks for c in block]
    if any(c.exam_relevance >= 4 for c in everything):
        blocks = [[c for c in block if c.exam_relevance >= 4] for block in blocks]
        everything = [c for block in blocks for c in block]
    picked = sorted((min(block, key=rank) for block in blocks if block), key=rank)[:n]
    rest = sorted((c for c in everything if c not in picked), key=rank)
    picked += rest[:max(0, n - len(picked))]
    order = {c.id: i for i, c in enumerate(everything)}
    return sorted(picked, key=lambda c: order[c.id])


def build_blueprint(
    plan: ContentPlan,
    cast: Cast,
    *,
    guest: Guest | None,
    previous_plan: ContentPlan | None,
    next_chapter_title: str | None,
    target_minutes: int,
    chars_per_second: float,
    outline: EpisodeOutline | None = None,
    book: Book | None = None,
) -> list[SegmentBrief]:
    explainer, skeptic = cast.by_role("explainer"), cast.by_role("skeptic")
    hosts = [explainer.id, skeptic.id]
    briefs: list[SegmentBrief] = []

    moments = sum(1 for b in (outline.blocks if outline else []) if b.moment)
    shape = plan_shape(plan, guest=guest, target_minutes=target_minutes, has_recap=previous_plan is not None,
                       moments=moments)
    seconds = shape.seconds
    outline = _fit_outline(outline, plan, shape.body_claims)
    by_id = {c.id: c for c in plan.key_claims}
    clusters = [[by_id[cid] for cid in b.claims] for b in outline.blocks]
    question = outline.question or (plan.chapter_title if (plan.chapter_title or "").endswith("?") else "")

    # Opening: a place to stand and the question the episode answers.
    _, block_misconceptions = _assign_misconceptions(plan.misconceptions, clusters)
    briefs.append(SegmentBrief(
        type="cold_open", title="Koude opening",
        covers=[],
        target_chars=_chars(seconds["cold_open"], chars_per_second),
        speakers=hosts,
        instructions=(
            "Dit is het begin van de aflevering. De luisteraar weet nog niets: niet waar het over gaat en niet wat "
            "jullie eerder bespraken. Begin daarom niet midden in een discussie.\n"
            + (f"Begin in deze situatie: {outline.opening}\n" if outline.opening else
               "Begin in een herkenbare situatie uit het gewone leven die bij dit hoofdstuk hoort.\n")
            + f"{explainer.name} schetst die situatie in gewone woorden, {skeptic.name} reageert als iemand die het "
            "herkent of juist niet. Na drie regels weet de luisteraar waar deze aflevering over gaat.\n"
            "Wat hier niet in mag: een vakterm, een getal, de naam van een onderzoeker, een vergelijking of "
            "beeldspraak, iets uit het leven van de hosts, een verwijzing naar iets wat nog niet is uitgelegd.\n"
            + (f"Het eindigt met de vraag waar de aflevering om draait, hardop gesteld door {skeptic.name}: "
               f"\"{question}\" Die vraag wordt hier nog niet beantwoord.\n" if question else
               f"Het eindigt met de vraag waar de aflevering om draait, hardop gesteld door {skeptic.name} in "
               "gewone woorden. Die vraag wordt hier nog niet beantwoord.\n")
            + "Geen welkom en geen namen als begroeting. Dit segment dekt geen beweringen: 'covers' blijft leeg."
        ),
    ))

    # Recap of the previous episode.
    if previous_plan is not None:
        prev_top = sorted(previous_plan.key_claims, key=lambda c: (-c.exam_relevance, -c.difficulty))
        prev_claim = prev_top[0] if prev_top else None
        briefs.append(SegmentBrief(
            type="recap", title="Terugblik",
            covers=[f"{previous_plan.chapter_id}:{prev_claim.id}"] if prev_claim else [],
            target_chars=_chars(seconds["recap"], chars_per_second),
            speakers=hosts,
            instructions=(
                f"Eén kernpunt uit de vorige aflevering ({previous_plan.chapter_title or previous_plan.chapter_id}). "
                f"{skeptic.name} stelt het eerst als vraag aan de luisteraar en laat een korte stilte vallen "
                "(pause_after_ms 2500 op die regel) voordat het antwoord komt. Spaced repetition, gratis.\n"
                + (f"Het kernpunt: {prev_claim.claim}\nBron van dat punt: {prev_claim.source_span.quote or ''}\n"
                   f"Zet op de regel met het antwoord covers: ['{previous_plan.chapter_id}:{prev_claim.id}']" if prev_claim else "")
            ),
            extra={"previous_chapter": previous_plan.chapter_id},
        ))

    # Body blocks: one question each, in the order of the outline.
    needs = [claim_seconds(cl) + (HUMAN_MOMENT_SECONDS if b.moment else 0) + (SECONDS_PER_MISCONCEPTION if i in block_misconceptions else 0)
             for i, (b, cl) in enumerate(zip(outline.blocks, clusters, strict=True))]
    example_block = None
    if plan.worked_example and plan.worked_example.steps and clusters:
        example_text = f"{plan.worked_example.setup} {plan.worked_example.answer}"
        example_block = max(range(len(clusters)), key=lambda i: _about(example_text, clusters[i]))
    explained: set[str] = set()  # each definition is handed to one block only, so it is explained once
    for i, (block, cluster) in enumerate(zip(outline.blocks, clusters, strict=True), start=1):
        misc = block_misconceptions.get(i - 1)
        relevant_defs = [d for d in plan.definitions if d.term.casefold() not in explained
                         and any(d.term.casefold() in c.claim.casefold() for c in cluster)]
        explained.update(d.term.casefold() for d in relevant_defs)
        previous = outline.blocks[i - 2] if i > 1 else None
        lines = []
        if block.question:
            lines.append(f"Dit blok beantwoordt één vraag: \"{block.question}\"")
        if previous is not None and previous.bridge:
            lines.append(f"Het vorige blok eindigde op deze vraag, begin daar: \"{previous.bridge}\"")
        elif previous is None:
            lines.append("De opening eindigde op de vraag van de aflevering. Begin daar, zonder opnieuw te beginnen.")
        lines.append(
            f"Zo loopt het gesprek: {skeptic.name} wil het antwoord weten en vraagt wat de luisteraar zou vragen. "
            f"{explainer.name} heeft het hoofdstuk gelezen en vertelt het in eigen woorden, en begint elke beurt met "
            f"een antwoord op wat {skeptic.name} net vroeg of zei. {skeptic.name} kent de stof niet: hij raadt, "
            "twijfelt en zegt terug wat hij net hoorde, maar brengt zelf geen feiten, namen of jaartallen uit het "
            "hoofdstuk in.")
        lines.append("Vertel deze punten in deze volgorde. Elk punt volgt uit het vorige, en je zegt hardop hoe "
                     "('en daarom', 'maar dan', 'dus als dat zo is'). Elk punt komt in minstens één regel terug, "
                     "feitelijk correct, met de id in 'covers':")
        lines += _claim_rows(cluster, numbered=True)
        passages = source_passages(book, cluster)
        if passages:
            lines.append(BACKGROUND_HEADER)
            lines += passages
        if relevant_defs:
            lines.append("Begrippen die hier voor het eerst vallen. Zeg het woord één keer, zeg er meteen in gewone "
                         "woorden bij wat het betekent, en nooit als woordenboekregel ('term: omschrijving'):")
            lines += [f"- {d.term}: {d.definition}" for d in relevant_defs]
        if misc:
            lines.append(
                f"Misvatting in dit blok: {skeptic.name} denkt hardop '{misc.wrong}', op een moment dat dat logisch "
                f"volgt uit wat net gezegd is (verleidelijk omdat: {misc.why_tempting}). {explainer.name} zegt wat er "
                f"wel klopt: '{misc.right}'. {explainer.name} trapt er zelf niet in."
            )
        if block.moment:
            teller = _moment_teller(block.moment_note, explainer, skeptic)
            moment = human_moment(block.moment, explainer, skeptic, note=block.moment_note, teller=teller)
            lines.append("Menselijk moment in dit blok: " + moment)
            if block.moment == "anekdote":
                seeds = story_seeds(teller).strip()
                if seeds:
                    lines.append("Waar het verhaal uit kan komen (kies er hooguit één, en alleen als het past):\n" + seeds)
        if example_block == i - 1:
            lines.append(
                "Werk het uitgewerkte voorbeeld uit de bron door in dialoog, nadat de opstelling is uitgelegd: "
                f"{plan.worked_example.setup} Stappen: " + " | ".join(plan.worked_example.steps)
                + f" Antwoord: {plan.worked_example.answer}"
            )
        if block.bridge:
            lines.append(f"Het blok eindigt als de vraag beantwoord is, en met de vraag die dat antwoord oproept: "
                         f"\"{block.bridge}\"")
        elif i == len(clusters):
            lines.append("Dit is het laatste blok van het verhaal: het eindigt als de vraag beantwoord is.")
        briefs.append(SegmentBrief(
            type="body", title=f"Blok {i}",
            covers=[c.id for c in cluster],
            target_chars=_chars(seconds["body"] * needs[i - 1] / max(1.0, sum(needs)), chars_per_second),
            speakers=hosts,
            instructions="\n".join(lines),
        ))

    # Guest block: the hardest claims, told here for the first time.
    if shape.guest_claims:
        hard = shape.guest_claims
        passages = source_passages(book, hard)
        briefs.append(SegmentBrief(
            type="guest", title=f"Gast: {guest.name}",
            covers=[c.id for c in hard],
            target_chars=_chars(seconds["guest"], chars_per_second),
            speakers=hosts + [guest.id],
            instructions=(
                f"Driespreker-blok met gast {guest.name} (spreker-id {guest.id}), vakgebied: {plan.expert_domain}. "
                f"Reden voor de gast: {plan.expert_reason or 'de stof is te moeilijk voor de hosts alleen'}. "
                f"{skeptic.name} zegt in één zin wie {guest.name} is en waarom die erbij is; geen welkomstritueel.\n"
                "De punten hieronder zijn het moeilijkste deel van het hoofdstuk en zijn nog niet verteld. "
                f"{guest.name} vertelt ze, {skeptic.name} vraagt door tot hij het kan navertellen, {explainer.name} zegt "
                f"het na in gewone woorden. {guest.name} praat als iemand die het vak kent en het aan vrienden uitlegt: "
                "geen spreuken, en geen adviezen of feiten die niet in het boek staan.\n"
                + "\n".join(_claim_rows(hard, numbered=True))
                + ("\n" + BACKGROUND_HEADER + "\n" + "\n".join(passages) if passages else "")
            ),
        ))

    # Wacht, opnieuw: the why behind the hardest claim.
    if shape.reexplain is not None:
        hardest = shape.reexplain
        order = [c.id for c in plan.key_claims]
        at = order.index(hardest.id)
        nearby = [c for c in plan.key_claims[max(0, at - 2):at + 3] if c.id != hardest.id
                  and c.source_span.section == hardest.source_span.section]
        passages = source_passages(book, [hardest], context=2 * PASSAGE_CONTEXT)
        briefs.append(SegmentBrief(
            type="reexplain", title="Wacht, opnieuw",
            covers=[hardest.id],
            target_chars=_chars(seconds["reexplain"], chars_per_second),
            speakers=hosts,
            instructions=(
                f"Het moeilijkste punt van de aflevering krijgt een tweede uitleg. {skeptic.name} zegt eerlijk dat hij "
                f"het nog niet kan navertellen en wat hem precies dwarszit. {explainer.name} legt het opnieuw uit langs "
                "een andere weg: begin bij één concreet geval en loop van daar naar de regel.\n"
                "Deze keer gaat het om het waarom. Het punt zelf is al genoemd, dus herhaal die zin niet: leg uit hoe "
                "het kan dat het zo is, met wat er in het boek omheen staat. Geeft het boek geen verklaring, zeg dat "
                "dan in één eerlijke zin en zeg wat er wel vaststaat. Praat niet over de uitleg zelf ('ik pak een ander "
                "beeld'), geef de uitleg.\n"
                f"Aan het eind vertelt {skeptic.name} het in eigen woorden na, en {explainer.name} zegt of dat klopt.\n"
                "Het punt:\n" + "\n".join(_claim_rows([hardest]))
                + ("\nWat er al over verteld is en waar het waarom in zit:\n" + "\n".join(f"- {c.claim}" for c in nearby)
                   if nearby else "")
                + ("\n" + BACKGROUND_HEADER + "\n" + "\n".join(passages) if passages else "")
            ),
        ))

    # Quiz.
    quiz_claims = _quiz_claims(clusters + ([shape.guest_claims] if shape.guest_claims else []), shape.n_quiz)
    quiz_misc = max(plan.misconceptions, key=lambda m: _about(f"{m.wrong} {m.right}", quiz_claims), default=None)
    quiz_sources = [f"  bron {c.id}: \"{(c.source_span.quote or '')[:300]}\"" for c in quiz_claims]
    briefs.append(SegmentBrief(
        type="quiz", title="Quiz",
        covers=[c.id for c in quiz_claims],
        target_chars=_chars(seconds["quiz"], chars_per_second),
        speakers=hosts,
        instructions=(
            f"{skeptic.name} overhoort {explainer.name} met precies {len(quiz_claims)} vragen, elk over één van deze "
            f"punten. {skeptic.name} heeft de antwoorden voor zich liggen. Vraag naar begrip: waarom, hoe, wat betekent "
            "het. Geen vraag naar een los getal of een jaartal.\n"
            "Na elke vraag valt een stilte voor de luisteraar: zet pause_after_ms op 3000 op de regel met de vraag.\n"
            f"{explainer.name} antwoordt zoals je het een vriend uitlegt: het klopt met het boek, maar het is geen zin "
            f"uit het boek. Na elk antwoord zegt {skeptic.name} in een paar woorden of het klopt, zodat de luisteraar "
            "het eigen antwoord kan nakijken.\n"
            f"Precies één antwoord is fout: {explainer.name} geeft daar het antwoord dat veel mensen zouden geven"
            + (f" (bijvoorbeeld deze misvatting: '{quiz_misc.wrong}')" if quiz_misc else "")
            + f", en {skeptic.name} verbetert het met wat er wel klopt en waarom.\n"
            "Tussen de vragen geen zijpaden, geen verhalen en geen grappen over iets anders.\n"
            + "\n".join(_claim_text(c) for c in quiz_claims)
            + "\n" + "\n".join(quiz_sources)
        ),
    ))

    # Outro: the answer, and what comes next.
    briefs.append(SegmentBrief(
        type="outro", title="Uitsmijter",
        covers=[],
        target_chars=_chars(seconds["outro"], chars_per_second),
        speakers=hosts,
        instructions=(
            "Kort. "
            + (f"De aflevering begon met de vraag \"{question}\" " if question else "De aflevering begon met een vraag. ")
            + f"{skeptic.name} zegt in één of twee gewone zinnen wat daar nu het antwoord op is, zoals je het na afloop "
            f"tegen een vriend zou zeggen, en {explainer.name} zegt of dat klopt. Geen opsomming van wat er besproken "
            "is en geen bedankje.\n"
            + (f"Waar het antwoord op neerkomt: {outline.takeaway}\n" if outline.takeaway else "")
            + (f"Daarna één haakje naar het volgende hoofdstuk: '{next_chapter_title}', als een vraag die nu open ligt. "
               if next_chapter_title else "Daarna één vraag die nog open ligt. ")
            + "Geen verwijzing naar iets wat niet in deze aflevering is verteld."
        ),
    ))
    return briefs


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------

class LineOut(BaseModel):
    speaker: str = Field(description="Spreker-id uit de cast (bijvoorbeeld tessa, joris of de gast-id).")
    text: str = Field(description="De gesproken tekst. Bij een onderbroken regel eindigt de tekst op een afgebroken zinsdeel met een kastlijntje (—).")
    tags: list[str] = Field(description="Nul tot twee emotie-tags uit de toegestane lijst.")
    covers: list[str] = Field(description="Id's van beweringen die deze regel feitelijk overbrengt (bijvoorbeeld c3). Leeg voor grappen, vragen en reacties.")
    overlap: Literal["none", "interrupt", "backchannel"] = Field(
        description="Of deze regel over de vorige regel heen valt. interrupt: de vorige regel wordt afgekapt. backchannel: korte reactie eronder (ja, precies, lach)."
    )
    pause_after_ms: int = Field(description="Bewuste stilte na deze regel in milliseconden, meestal 0.")


BEATS_DESCRIPTION = ("Eerst, voor je de regels schrijft: in twee of drie zinnen het verloop van dit segment. Waar begint het "
                     "verhaal (een moment, een persoon, een raadsel), welk menselijk moment, wie wil wat, waar de lach of "
                     "het 'o!' valt. Wordt niet uitgesproken.")


def _ask_for_beats(schema: dict) -> None:
    """The model is asked for the beats first; an answer without them is still accepted, not sent back."""
    required = schema.setdefault("required", [])
    if "beats" not in required:
        required.insert(0, "beats")


class SegmentOut(BaseModel):
    model_config = ConfigDict(json_schema_extra=_ask_for_beats)
    beats: str = Field(default="", description=BEATS_DESCRIPTION)
    lines: list[LineOut]


class PhraseOut(BaseModel):
    text: str = Field(description="Een stuk van de regel, letterlijk. Alle stukken samen vormen precies de tekst van de regel.")
    delivery: str = Field(description="Voordracht van dit stuk, zelfde labels als bij de regel, of leeg.")
    pause_after: str = Field(description="Stilte na dit stuk: none, short of beat.")


class PerformanceLineOut(LineOut):
    delivery: str = Field(description="Hoe de regel gebracht wordt: explain, think, excite, react, interrupt, realize, disagree, setup, punchline.")
    mood: str = Field(description="Toestand van de spreker nu: confident, challenged, surprised, amused, thoughtful, calm.")
    timing: str = Field(description="Hoe snel deze regel volgt op de vorige: immediate, hesitate, search, deliberate. Leeg bij overlap.")
    phrases: list[PhraseOut] = Field(description="Alleen bij langere regels met een omslag erin: de regel opgeknipt in stukken. Anders leeg.")
    reaction: str = Field(description="Alleen voor een kale reactie: laugh, chuckle, sigh, hm, ja, oh, wacht, precies. Anders leeg.")


class PerformanceSegmentOut(BaseModel):
    model_config = ConfigDict(json_schema_extra=_ask_for_beats)
    beats: str = Field(default="", description=BEATS_DESCRIPTION)
    lines: list[PerformanceLineOut]


PERFORMANCE_RULES = """
## Voordracht (performance-labels)
Elke regel krijgt labels die bepalen hoe hij klinkt. Kies ze op basis van wat er in het gesprek gebeurt, nooit om variatie te maken.
- delivery: explain (uitleg, rustiger), think (hardop denken, trager), excite (enthousiast, sneller), react (reactie op de ander), interrupt (kort en snel ertussen), realize (het kwartje valt), disagree (tegenspreken, snel), setup (aanloop naar een grap), punchline (de grap zelf, snel na de aanloop).
- mood: de toestand van de spreker (confident, challenged, surprised, amused, thoughtful, calm). Een toestand blijft staan tot er iets gebeurt dat hem verandert. Een uitdaging maakt iemand challenged, een onverwacht punt surprised, een grap amused.
- timing: hoe snel deze regel volgt. immediate bij tegenspreken en snelle wisselingen, hesitate als iemand even moet nadenken, search als iemand naar woorden zoekt, deliberate vóór een realisatie of een belangrijk punt. Leeg bij interrupt en backchannel.
- phrases: knip een regel alleen op als er binnen de regel een omslag zit, bijvoorbeeld "Ik snap wat je bedoelt..." (react) + "maar wacht even—" (interrupt, pause_after none) + "dat kan toch helemaal niet?" (disagree). Na een realisatie een beat, dan trager verder. De stukken samen zijn letterlijk de regeltekst.
- reaction: kleine reacties (hm, ja, oh, wacht, precies, een lach, een zucht) zijn eigen korte regels met reaction gezet, meestal als backchannel onder de ander door. Iemand reageert zonder het woord over te nemen.
- De tekst wordt letterlijk uitgesproken. Schrijf daarom het geluid zelf ("Haha.", "Hehe.", "Pff.", "Hm."), nooit een beschrijving als "(lacht)", "[chuckle]", "*zucht*" of "grinnikt". Ook niet midden in een zin.
"""

WRITER_RULES = """Je schrijft het script van een Nederlandse studiepodcast met een vaste cast. Eén aflevering per hoofdstuk van een studieboek. Alles in het Nederlands.

## Wat dit is
Twee mensen die samen een hoofdstuk doorpraten, voor iemand die luistert om het te leren. Het is een gesprek, geen les en geen hoorspel. Na afloop kan de luisteraar de hoofdvraag van de aflevering beantwoorden en uitleggen waarom het antwoord klopt.

## Wie weet wat
- De explainer heeft het hoofdstuk gelezen en legt het uit. Alles wat uit de stof komt (feiten, namen, jaartallen, getallen, vaktermen) komt van de explainer, of van de gast als die er is.
- De skeptic heeft het hoofdstuk niet gelezen. Hij vraagt wat de luisteraar zou vragen, gokt, twijfelt, en zegt in eigen woorden terug wat hij net hoorde. Hij brengt zelf niets uit de stof in. Een vakterm of een naam hoort hij eerst van de explainer. Alleen in de quiz heeft hij de antwoorden voor zich.
- Een misvatting komt van de skeptic, op het moment dat ze logisch volgt uit wat net gezegd is. De explainer trapt er niet in en doet ook niet alsof.

## Gewone woorden
- Schrijf zoals je tegen een vriend praat aan de keukentafel. Kies het woord dat je hardop zou gebruiken: "nuttig" en niet "functioneel", "hoe vaak het voorkomt" en niet "prevalentie", "ooit in je leven" en niet "levenslang", "voorzichtig zijn" en niet "behoedzaamheid".
- Een vakterm die de luisteraar moet leren, zeg je één keer, met er meteen bij wat hij betekent in gewone woorden. Daarna mag je hem gewoon gebruiken. Vaktermen die de luisteraar niet hoeft te kennen, laat je weg.
- Gebruik alleen woorden die bestaan en waarvan je zeker bent. Geen zelfbedachte samenstellingen, geen plechtige of dichterlijke zinnen, geen spreuken.
- Zinnen die je in één adem zegt. Geen puntkomma. Geen dubbele punt met een rijtje erachter. Geen zinsbouw die alleen op papier werkt ("Voorzichtig concludeerden zij dat...").

## Niets uit de lucht
- De luisteraar kent alleen wat in deze aflevering hardop is gezegd. Elke persoon, elk onderzoek, elke proef, elk begrip en elk voorwerp wordt eerst voorgesteld: wie of wat is het, en waarom gaat het er nu over. "Die enquête", "de onderzoekers" en "dat model" kunnen pas als ze genoemd zijn.
- Een onderzoek krijgt de eerste keer een gezicht: wie het deed of waar, wat ze deden, en dan pas wat eruit kwam. Een proef leg je uit voordat je de uitkomst vertelt.
- Dat geldt ook voor het leven van de hosts. Een broer, een zus, een hobby of een oude grap bestaat voor de luisteraar pas als het in deze aflevering is verteld. Vertel er alleen iets over waar de opdracht van het segment erom vraagt.
- Een vergelijking gebruik je alleen als ze iets uitlegt. Je zegt erbij wat met wat overeenkomt, en je houdt het bij één beeld per blok. Een beeld waar je niet verder mee komt, laat je los; stapel er geen tweede op.

## De draad
- De aflevering beantwoordt één hoofdvraag (zie de opbouw onderaan). Elk blok beantwoordt een deelvraag en eindigt bij de vraag die dat antwoord oproept.
- Binnen een blok volgt elk punt uit het vorige, en dat zeg je hardop: "en daarom", "maar dan klopt er iets niet", "dus als dat zo is". Spring niet van onderwerp zonder te zeggen waarom je daar nu heen gaat.
- Een vraag krijgt antwoord in de eerstvolgende beurt van de ander. Wie iets vraagt en een ander onderwerp terugkrijgt, zit niet in een gesprek.
- Vertel bij een uitkomst ook hoe het komt en wat het betekent, als de stof dat zegt. Een getal of een onderzoek zonder betekenis blijft niet hangen.
- Wat al verteld is, herhaal je niet in dezelfde woorden. Terugverwijzen mag: "dat is dus dat alarm van net".

## Nauwkeurigheid (niet onderhandelbaar)
- Dit is studiemateriaal. Elke feitelijke bewering komt uit het inhoudsplan en de bronfragmenten. Voeg geen feiten toe die niet in de bron staan, ook geen adviezen.
- Getallen, jaartallen en namen uit de bron zijn heilig, en de richting van een verband ook (meer of minder, sneller of trager). De formulering is van jou: de achtergrond uit het boek in een opdracht is er om het te begrijpen, neem er geen zin uit over.
- Eén getal per zin is genoeg voor een luisteraar. Staan er in de stof vier percentages bij elkaar, verdeel ze dan over het gesprek en zeg bij elk wat het betekent.
- Praat over het onderwerp, niet over de bron. Geen "de tekst zegt", "er staat", "volgens de bron": vertel wat er gebeurde of hoe het zit.
- Markeer regels die een bewering overbrengen met de claim-id in 'covers'. Grappen, vragen en reacties krijgen geen covers.
- Als een host expres iets fout zegt (misvatting, quizfout), corrigeer het in het gesprek binnen twee regels, en zet de covers alleen op de regel met de correcte informatie. Wie verbetert, verbetert wat er echt fout was: zeg nooit "nee" tegen iets wat klopt.

## Humor
- Humor is bijvangst van een echt gesprek: een verkeerde gok, een droge reactie, iemand die zichzelf betrapt. Het gaat over wat er net gezegd is. Liever geen grap dan een gezochte.
- Verboden: actuele grappen, grappen die in elke aflevering zouden werken, grappen waarvoor de luisteraar iets moet zien, en grappen over iets uit het leven van de hosts dat in deze aflevering niet is verteld.
- Een verbale tic is een kruiderij: hooguit één keer per segment, nooit als vaste opening van regels. De tics van de ander zijn verboden terrein.
- Geen vaste formules. Laat regels niet steeds met dezelfde woorden beginnen en gebruik geen sjabloon dat terugkomt (zoals steeds "een X is geen Y").

## Vertellen
- Een blok begint ergens waar je kunt staan: een moment, een persoon, een plek, een vraag waar toen nog niemand het antwoord op had. Van daaruit bouwt het op: wat stond er op het spel, wat wist nog niemand, wat liep anders dan gedacht. De beweringen vallen in het verhaal op de plek waar ze thuishoren, met hun oorzaak en hun gevolg erbij.
- Bij verhalende stof (geschiedenis, een verloop van gebeurtenissen) vertel je in de volgorde waarin het gebeurde, met wie, wanneer en waarom. Bij stof zonder verloop in de tijd (een begrip, een rekenregel, een mechanisme) is de gedachtegang het verhaal: begin bij een raadsel of een situatie en loop samen naar het antwoord.
- Een feit staat nooit kaal. Een jaartal krijgt een persoon of een plek, een besluit krijgt een reden, een getal krijgt iets om het mee te vergelijken. Rijtjes losse zinnetjes zijn verboden, zoals "Nee. 1533, Duitsland. Zijn ouders waren protestant." of "Een kind ging. De koning wilde het.": zo klinkt een spiekbriefje dat wordt voorgelezen. Vertel hetzelfde in een lopende zin, met wie, waarom en wat het betekende.
- De skeptic twijfelt aan de stof: waarom gebeurde dit, hoe hangt het samen, wat betekent het. Niet aan de woordkeus van de ander, en niet aan wat er wel of niet in de bron staat. Hij praat in hele zinnen: gokt hardop en zit er soms naast, maakt de zin van de ander af.

## Verboden taal (de linter blokkeert het script bij één treffer)
- AI-podcastvulling: "dat is echt fascinerend", "goede vraag", "laten we eens duiken in", "aan het eind van de dag", "welkom bij", "in deze aflevering", "kortom", "samengevat", "in een notendop".
- Elke zin die samenvat wat er net is besproken. Geen recap aan het einde van een blok. Alleen de uitsmijter beantwoordt in één of twee zinnen de hoofdvraag.
- Driedelige opsommingen als retorisch sierstukje ("snel, simpel en slim").
- Constructies van het type "niet X maar Y" als stijlfiguur.
- Zinnen die het gesprek aankondigen in plaats van voeren.

## Overlap
- Nooit over een definitie, een getal, een formule of een vakterm heen. Wat de luisteraar nodig heeft, moet schoon zijn.
- Alleen bij reacties, overgangen, een host die enthousiast wordt, een grap die landt.
- Maximaal 4 onderbrekingen per 10 minuten; daarboven wordt energie chaos.
- interrupt: de regel ervóór moet eindigen op een afgebroken zinsdeel dat bedoeld is om te verdwijnen, met een kastlijntje: "...en dan verschuift dus de hele—". Jij schrijft dat fragment.
- backchannel: kort ("ja", "precies", een lach), verandert de tijdlijn niet.
- De eerste regel van een segment heeft nooit overlap. Overlap is altijd met een andere spreker dan de vorige regel.

## Vorm
- Spreektaal zoals mensen echt praten als ze iets vertellen: zinnen die kronkelen, met 'en toen', 'want', 'terwijl', 'alleen', 'dus eigenlijk'; een zijpad dat terugkomt ("en dat is trouwens, nee, dat komt zo, eerst dit"); een spoor dat doodloopt ("je zou denken dat het andersom zit, en zo dacht ik er ook lang over, tot..."). Daartussen korte reacties, een vraag, een stilte. Lange en korte zinnen wisselen elkaar af; een rij korte zinnen achter elkaar klinkt als een opsomming.
- Een vertelbeurt is één regel, ook als hij drie tot vijf zinnen lang is (tot ongeveer 400 tekens). Knip een verhaal niet op in regels van één feit per regel; de ander reageert eronder met een backchannel of onderbreekt waar het spannend wordt. Een kaal "Ja." of "Nee." is een reactie, nooit het begin van een rijtje feiten.
- Elke regel reageert op wat er net gezegd is: op de woorden, de toon, de grap van de ander. Iemand die alleen het volgende punt afwerkt, klinkt als een robot.
- Mensen praten niet foutloos: een zin opnieuw beginnen, zichzelf verbeteren ("nee, anders:"), lachen om iets specifieks, iets niet weten en dat zeggen. Met mate, en nooit in een regel die een getal of definitie draagt.
- Schrijf getallen, formules en afkortingen zoals de bron ze schrijft; de uitspraak wordt later automatisch opgelost. Notatiezware stof beschrijf je in vorm en betekenis in plaats van symbolen voor te lezen.
- Geen regieaanwijzingen in de tekst, ook niet tussen haakjes: de tekst wordt letterlijk uitgesproken. Emotie gaat via tags. Toegestane tags: {tags}.
- Chatterbox leest hoofdletters als nadruk (harder, trager op dat woord) en gebruikt komma's en punten om adempauzes te plaatsen. Zet een enkel woord in KAPITALEN wanneer een host het écht benadrukt, niet elke zin, en varieer leestekens: een kort zinnetje met een punt klinkt anders dan een lange komma-zin.
- Houd de streefduur aan: ongeveer {cps} tekens per seconde spreektijd.

## Emotionele continuïteit
- Een tag op de vorige regel is niet gebonden aan één spreker. Als de vorige regel excited, surprised of serious draagt, negeer dat niet zomaar in de eerstvolgende regel van de andere host, dat leest als twee monologen naast elkaar.
- Twee geldige reacties, kies wat bij het onderwerp past: meebewegen (de ander wordt ook enthousiaster of juist serieuzer, met een eigen tag) of tegenwicht bieden (de ander blijft kalm, remt af, brengt het terug naar de feiten, ook met een eigen tag die dat laat horen, bijvoorbeeld deadpan of skeptical).
- Dit hoeft niet op elke regel, maar een opbouw naar een hoogtepunt (een analogie die steeds enthousiaster wordt tot hij breekt) of een afkoeling na een moeilijk punt moet voelbaar zijn over een paar regels, niet alleen op de ene regel waar het toevallig gebeurt.
- Tessa jaagt van nature vaker op, Joris remt vaker af, maar dat mag omdraaien als de stof erom vraagt: ook Joris mag meegesleept raken, ook Tessa mag ergens serieus van worden.
"""


def _plan_text(plan: ContentPlan) -> str:
    parts = [f"## Inhoudsplan hoofdstuk {plan.chapter_id}: {plan.chapter_title}", plan.summary,
             "### Leerdoelen"] + [f"- {o}" for o in plan.learning_objectives]
    parts.append("### Beweringen (id, moeilijkheid, tentamenrelevantie)")
    parts += [_claim_text(c) for c in plan.key_claims]
    parts.append("### Definities")
    parts += [f"- {d.term}: {d.definition}" for d in plan.definitions] or ["- (geen)"]
    parts.append("### Misvattingen")
    parts += [f"- fout: {m.wrong} | juist: {m.right} | verleidelijk omdat: {m.why_tempting}" for m in plan.misconceptions] or ["- (geen)"]
    if plan.formula_dense_sections:
        parts.append("### Notatiezware secties (vorm en betekenis beschrijven, geen symbolen voorlezen): "
                     + ", ".join(plan.formula_dense_sections))
    return "\n".join(parts)


def _glossary_text(glossary: Glossary | None) -> str:
    if glossary is None or not glossary.entries:
        return "## Lexicon\n(leeg)"
    loan = [e for e in glossary.entries if e.kind == "loanword_en"]
    if not loan:
        return "## Lexicon\nGeen vastgelegde leenwoorden. Schrijf termen zoals de bron."
    return "## Lexicon: vaste schrijfwijze van leenwoorden (gebruik de linkerkant in het script)\n" + "\n".join(
        f"- {e.surface}" for e in loan[:150]
    )


def _lines_text(script: Script, limit_chars: int = 30000) -> str:
    rows = []
    for seg in script.segments:
        for line in seg.lines:
            tag = f" ({', '.join(line.tags)})" if line.tags else ""
            ov = f" [{line.overlap.mode}]" if line.overlap.mode != "none" else ""
            perf = [f"{k}={v}" for k, v in (("delivery", line.delivery), ("mood", line.mood), ("timing", line.timing),
                                            ("reaction", line.reaction)) if v]
            perf_text = f" {{{', '.join(perf)}}}" if perf else ""
            rows.append(f"[{line.id}] {line.speaker}{tag}{ov}{perf_text}: {line.text}")
    text = "\n".join(rows)
    return text[-limit_chars:] if len(text) > limit_chars else text


# Warnings about how the dialogue *speaks* rather than what it says. A script can pass the audit and
# still read as two monologues next to each other, and the render cannot fix that: an interruption
# that is not in the script is not in the audio. Quality mode hands these back to the writer as well,
# so a flat stretch costs a rewrite round instead of a line in the audit json.
NATURALNESS_RULES = frozenset({
    "too_few_overlaps",          # not enough interruptions and backchannels for the length
    "flat_emotional_reaction",   # nobody reacts when something surprising lands
    "cut_word_missing",          # an interruption that breaks off mid-word cannot be staged
    "quiz_pause",                # a quiz question nobody waits for
    "ignores_previous",          # a line that walks past what the other one just said
    "stiff",                     # written language nobody would say out loud
    "formal",                    # a word too solemn or bookish for a conversation
    "forced_image",              # a comparison that explains nothing
    "telegraphic",               # facts in strings of three-word sentences instead of a story
    "source_copy",               # a sentence lifted from the book
    "claim_repeated",            # the same claim told again in another segment
    "near_duplicate",            # nearly the same line twice
    "skeptic_fact",              # the host who has not read the chapter brings in a fact from it
    "term_before_intro",         # a term or a name that host uses before it has been introduced
    "written_syntax",            # semicolons and colon lists in spoken text
    "number_overload",           # more numbers in one line than a listener can hold
    "unknown_word",              # a word that is in no dictionary, in the book or in the cast
    "no_antecedent",             # a reference the listener has never been given
    "unanswered",                # a question the next turn does not answer
    "nonsense",                  # a sentence a listener cannot make sense of
    "contradicts",               # a correction of something that was right
    "not_landed",                # a hard claim that is stated but never explained
})

# What a listener cannot follow, or is told without being able to explain it afterwards. Unlike a flat
# stretch this is not a matter of polish: a segment with one of these goes back to the writer in every mode,
# with its brief, because the fix can need something the table read may not add (the why behind a claim).
COMPREHENSION_RULES = frozenset({"no_antecedent", "unanswered", "nonsense", "contradicts", "not_landed"})

# How many of the flattest body segments a script-wide note is handed to. Rewriting everything to
# please one warning would put the whole episode at the model's mercy.
FLAT_SEGMENT_TARGETS = 3


def naturalness_issues(audit: AuditResult) -> list[AuditIssue]:
    """The warnings worth a rewrite in quality mode: overlap, reactions, cut words, quiz beats."""
    return [i for i in audit.warnings() if i.rule in NATURALNESS_RULES]


def comprehension_issues(audit: AuditResult) -> list[AuditIssue]:
    """The warnings worth a rewrite in any mode: what a first-time listener cannot follow."""
    return [i for i in audit.warnings() if i.rule in COMPREHENSION_RULES]


def _thread_text(outline: EpisodeOutline | None) -> str:
    """The thread of the episode as every segment call sees it: the question, and what each block answers."""
    if outline is None or not (outline.question or any(b.question for b in outline.blocks)):
        return ""
    rows = []
    if outline.question:
        rows.append(f"Hoofdvraag van de aflevering: {outline.question}")
    for i, b in enumerate(outline.blocks, start=1):
        if b.question:
            rows.append(f"- Blok {i} beantwoordt: {b.question}" + (f" En leidt naar: {b.bridge}" if b.bridge else ""))
    if outline.takeaway:
        rows.append(f"Waar het op uitkomt (pas in de uitsmijter hardop): {outline.takeaway}")
    return "\n".join(rows) + "\n\n"


class Writer:
    def __init__(self, llm: LLM, cast: Cast, settings: Settings, continuity: list[ContinuityEntry] | None = None,
                 *, performance: bool = False):
        self.llm = llm
        self.cast = cast
        self.settings = settings
        self.continuity = continuity or []
        self.performance = performance
        self.progress: Callable[[int, int, str], None] | None = None  # (segment number, segments, title)
        # (segment number, segments, the segment just written, whether it was the table read): for the watch page
        self.on_segment: Callable[[int, int, Segment, bool], None] | None = None

    # System prefix: identical for every segment call of one episode.
    def _system(self, plan: ContentPlan, glossary: Glossary | None, guest: Guest | None, briefs: list[SegmentBrief],
                episode_outline: EpisodeOutline | None = None) -> list[dict]:
        rules = WRITER_RULES.format(tags=", ".join(known_tags()), cps=int(self.settings.chars_per_second))
        if self.performance:
            rules += PERFORMANCE_RULES
        outline = "## Opbouw van deze aflevering\n" + _thread_text(episode_outline) + "\n".join(
            f"{i + 1}. {b.title} ({b.type}, ~{b.target_chars} tekens, dekt {', '.join(b.covers) or 'niets'})"
            for i, b in enumerate(briefs)
        )
        samples = voice_samples_text(self.cast)
        return [
            text_block(rules),
            text_block(cast_bible_text(self.cast, guest) + ("\n\n" + samples if samples else "")
                       + "\n\n" + continuity_text(self.continuity)),
            text_block(_plan_text(plan) + "\n\n" + _glossary_text(glossary) + "\n\n" + outline),
        ]

    def _user(self, brief: SegmentBrief, script: Script, interrupts_left: int, fix: str | None = None,
              edit: str | None = None) -> str:
        prior = _lines_text(script)
        parts = [
            f"# Schrijf nu segment: {brief.title} (type {brief.type})",
            f"Streeflengte: ongeveer {brief.target_chars} tekens gesproken tekst (± 15%).",
            f"Toegestane sprekers: {', '.join(brief.speakers)}.",
            f"Onderbrekingen (interrupt) die nog mogen in de rest van de aflevering: {interrupts_left}.",
            "Opdracht:\n" + brief.instructions,
        ]
        if fix:
            parts.append("## Herschrijf: los deze problemen op en behoud wat werkte\n"
                         "Los een probleem op in de inhoud zelf: verwijs niet naar 'de tekst' of 'de bron' om het te "
                         "ontwijken, en laat het segment niet langer worden dan de streeflengte.\n" + fix)
        if edit:
            parts.append(edit)
        parts.append("## Regels tot nu toe (ga hier naadloos op verder, herhaal niets)\n" + (prior or "(nog niets)"))
        return "\n\n".join(parts)

    def polish(self, script: Script, audit: AuditResult | None, plan: ContentPlan, glossary: Glossary | None) -> Script:
        """The table read: once the facts are right, every segment is rewritten for how it sounds. The facts are
        frozen (a line with covers keeps its content and its covers); everything around them may change: lines
        that answer each other, a host who restarts a sentence, a reaction under the other one's line. The
        audit's conversation notes for a segment (lines that ignore the previous one, stiff lines) go along."""
        guest = self.cast.guest(script.guest_id) if script.guest_id else None
        notes: dict[str, list[str]] = {}
        for issue in (audit.issues if audit else []):
            if issue.rule in NATURALNESS_RULES and issue.line_id:
                notes.setdefault(issue.line_id, []).append(issue.message)
        briefs = [SegmentBrief(type=s.type, title=s.title or s.type, covers=list(s.covers), target_chars=max(200, s.char_count),
                               instructions=s.brief or "", speakers=self._segment_speakers(s)) for s in script.segments]
        system = self._system(plan, glossary, guest, briefs, script.outline)
        budget = self._interrupt_budget(script.target_minutes)
        new_script = Script(episode_id=script.episode_id, target_minutes=script.target_minutes, title=script.title,
                            guest_id=script.guest_id, outline=script.outline, revision=script.revision + 1)
        todo = [s for s in script.segments if s.lines]
        for n, seg in enumerate(todo, start=1):
            if self.progress:
                self.progress(n, len(todo), f"tafellezing: {seg.title or seg.type}")
            i = script.segments.index(seg)
            used = sum(1 for line in new_script.lines() if line.overlap.mode == "interrupt")
            seg_notes = [f"- regel {line.id}: {m}" for line in seg.lines for m in notes.get(line.id, [])]
            edit = (POLISH_INSTRUCTIONS
                    + ("\nOpmerkingen bij deze versie:\n" + "\n".join(seg_notes) if seg_notes else "")
                    + "\n\nHuidige versie van dit segment:\n" + "\n".join(_edit_row(line) for line in seg.lines))
            request = LLMRequest(
                task="script_polish",
                system=system,
                user=self._user(briefs[i], new_script, max(0, budget - used), edit=edit),
                schema=SegmentOut,
                metadata={"segment": seg.type, "polish": True},
            )
            out = self.llm.generate(request)
            assert isinstance(out, SegmentOut)
            new_seg = self._materialise(out, briefs[i], _with_max_id(new_script, script))
            new_seg.beats = seg.beats or new_seg.beats
            new_script.segments.append(new_seg if new_seg.lines else seg)
            if self.on_segment:
                self.on_segment(n, len(todo), new_script.segments[-1], True)
        return new_script

    def _speaker_id(self, raw: str, allowed: list[str]) -> str | None:
        key = raw.strip().casefold()
        for sid in allowed:
            if key == sid.casefold():
                return sid
        for sid in allowed:
            try:
                name = self.cast.speaker(sid).name.casefold()
            except KeyError:
                continue
            if key == name or fuzz.ratio(key, name) > 85 or fuzz.ratio(key, sid) > 85:
                return sid
        return None

    def _materialise(self, out: SegmentOut, brief: SegmentBrief, script: Script) -> Segment:
        seg = Segment(type=brief.type, covers=list(brief.covers), title=brief.title, brief=brief.instructions,
                      beats=(getattr(out, "beats", "") or "").strip() or None)
        prev: Line | None = None
        for raw in out.lines:
            text = raw.text.strip()
            if not text:
                continue
            cue = cue_reaction(text)
            text = speakable(text) if cue else (strip_cues(text) or text)  # a TTS voice reads stage cues aloud
            speaker = self._speaker_id(raw.speaker, brief.speakers)
            if speaker is None:
                speaker = brief.speakers[0]
                text = text  # keep the text; the audit will flag the speaker mismatch through tics if any
                log.warning("unknown speaker %r in %s, mapped to %s", raw.speaker, brief.title, speaker)
            line = Line(
                id=script.next_line_id() if not seg.lines else _bump(seg.lines[-1].id),
                speaker=speaker,
                text=text,
                tags=[t for t in raw.tags if t in known_tags()][:2],
                covers=[c.strip() for c in raw.covers if c.strip()],
                pause_after_ms=max(0, int(raw.pause_after_ms or 0)),
                **(_performance_fields(raw, text) if isinstance(raw, PerformanceLineOut) else {}),
            )
            if cue and self.performance and line.reaction is None:
                line.reaction = cue
            if line.reaction and pure_reaction(line) is None:
                line.reaction = None  # a labelled full sentence keeps its words; only bare reactions are reactions
            mode = raw.overlap if raw.overlap in ("interrupt", "backchannel") else "none"
            if mode != "none" and prev is not None and prev.speaker != speaker:
                cut = None
                if mode == "interrupt":
                    words = prev.text.rstrip("—…-. ").split()
                    cut = words[-1] if words else None
                line.overlap = Overlap(mode=mode, target=prev.id, cut_word=cut)
            seg.lines.append(line)
            prev = line
        return seg

    def write(
        self,
        plan: ContentPlan,
        book: Book | None,
        glossary: Glossary | None,
        *,
        previous_plan: ContentPlan | None = None,
        next_chapter_title: str | None = None,
        target_minutes: int | None = None,
    ) -> Script:
        # A guest only when the cast has someone in this field: a stand-in expert has nothing of their own to say.
        guest = pick_guest(self.cast, plan.expert_domain, generic=False) if plan.needs_expert else None
        target_minutes = target_minutes or episode_minutes(plan, self.settings.target_minutes, with_guest=guest is not None)
        shape = plan_shape(plan, guest=guest, target_minutes=target_minutes, has_recap=previous_plan is not None,
                           moments=MAX_MOMENTS)
        outline = self.outline(plan, book, shape.body_claims)
        # With the moments the outline really has, the extras may fit after all: the blocks follow the final shape.
        shape = plan_shape(plan, guest=guest, target_minutes=target_minutes, has_recap=previous_plan is not None,
                           moments=sum(1 for b in outline.blocks if b.moment))
        outline = _fit_outline(outline, plan, shape.body_claims)
        briefs = build_blueprint(
            plan, self.cast, guest=guest, previous_plan=previous_plan, next_chapter_title=next_chapter_title,
            target_minutes=target_minutes, chars_per_second=self.settings.chars_per_second, outline=outline, book=book,
        )
        with_guest = any(b.type == "guest" for b in briefs)
        script = Script(episode_id=plan.chapter_id, target_minutes=target_minutes, title=plan.chapter_title,
                        guest_id=guest.id if guest and with_guest else None, outline=outline)
        guest = guest if with_guest else None
        system = self._system(plan, glossary, guest, briefs, outline)
        budget = self._interrupt_budget(target_minutes)
        for n, brief in enumerate(briefs, start=1):
            if self.progress:
                self.progress(n, len(briefs), brief.title)
            used = sum(1 for line in script.lines() if line.overlap.mode == "interrupt")
            request = LLMRequest(
                task="script_segment",
                system=system,
                user=self._user(brief, script, max(0, budget - used)),
                schema=SegmentOut,
                metadata={"segment": brief.type},
            )
            out = self.llm.generate(request)
            assert isinstance(out, SegmentOut)
            script.segments.append(self._materialise(out, brief, script))
            log.info("segment %s: %d lines", brief.title, len(script.segments[-1].lines))
            if self.on_segment:
                self.on_segment(n, len(briefs), script.segments[-1], False)
        return script

    def outline(self, plan: ContentPlan, book: Book | None, claims: list[KeyClaim]) -> EpisodeOutline:
        """The thread of the episode: one question, the steps that answer it, the order the claims are told in."""
        chapter = None
        if book is not None:
            try:
                chapter = book.chapter(plan.chapter_id)
            except KeyError:
                chapter = None
        return plan_outline(plan, claims, self.cast, self.llm, n_blocks=_block_count(claims), chapter=chapter)

    def write_scene(self, plan: ContentPlan, glossary: Glossary | None, brief: SegmentBrief, *,
                    fix: str | None = None, episode_id: str | None = None) -> Script:
        """One stand-alone segment with performance labels (the prototype scene)."""
        script = Script(episode_id=episode_id or plan.chapter_id, target_minutes=2, title=brief.title)
        request = LLMRequest(
            task="script_scene",
            system=self._system(plan, glossary, None, [brief]),
            user=self._user(brief, script, 2, fix=fix),
            schema=PerformanceSegmentOut if self.performance else SegmentOut,
            metadata={"segment": brief.type},
        )
        out = self.llm.generate(request)
        script.segments.append(self._materialise(out, brief, script))  # type: ignore[arg-type]
        return script

    def _interrupt_budget(self, target_minutes: int) -> int:
        return int(math.ceil(target_minutes / 10 * self.settings.max_interrupts_per_10min))

    def revise(self, script: Script, audit: AuditResult, plan: ContentPlan, glossary: Glossary | None,
               *, also_naturalness: bool = False) -> Script:
        """Rewrite only the segments with problems, feeding the problems back to the model.

        Blocking issues always go back to the writer, and so does what a listener cannot follow
        (``COMPREHENSION_RULES``). With ``also_naturalness`` the warnings about the shape of the dialogue go along,
        which is the only place interruptions and reactions can still come from: the mixer cannot overlap two lines
        the script never marked. Those notes ride along with the segments that are rewritten anyway; a segment with
        nothing wrong in it is only rewritten for its shape when nothing else in the script is wrong. A rewritten
        segment is new text, and new text brings sentences of its own for the next check to find fault with.
        """
        must, must_episode = self._route(script, audit.blocking() + comprehension_issues(audit))
        style, style_episode = self._route(script, [i for i in naturalness_issues(audit) if i.rule not in COMPREHENSION_RULES]
                                           if also_naturalness else [])
        if not must and not must_episode and not style and not style_episode:
            return script
        guest = self.cast.guest(script.guest_id) if script.guest_id else None
        per_segment: dict[int, list[str]] = {}
        if must_episode:  # about the episode as a whole, so no single segment owns it
            for idx in self._flat_segments(script):
                notes = must.setdefault(idx, [])
                notes += [n for n in must_episode if n not in notes]
        if must:
            for idx, notes in must.items():
                per_segment[idx] = notes + style.get(idx, []) + [n for n in style_episode if n not in notes]
        else:
            per_segment = style
            if style_episode:
                for idx in self._flat_segments(script):
                    notes = per_segment.setdefault(idx, [])
                    notes += [n for n in style_episode if n not in notes]

        briefs = [SegmentBrief(type=s.type, title=s.title or s.type, covers=list(s.covers), target_chars=max(200, s.char_count),
                               instructions=s.brief or "", speakers=self._segment_speakers(s)) for s in script.segments]
        system = self._system(plan, glossary, guest, briefs, script.outline)
        budget = self._interrupt_budget(script.target_minutes)
        new_script = Script(episode_id=script.episode_id, target_minutes=script.target_minutes, title=script.title,
                            guest_id=script.guest_id, outline=script.outline, revision=script.revision + 1)
        todo = sorted(per_segment)
        for i, seg in enumerate(script.segments):
            if i not in per_segment:
                new_script.segments.append(seg)
                continue
            if self.progress:
                self.progress(todo.index(i) + 1, len(todo), f"herschrijven: {seg.title or seg.type}")
            used = sum(1 for line in new_script.lines() if line.overlap.mode == "interrupt")
            fix = "\n".join(per_segment[i]) + "\n\nHuidige versie van dit segment:\n" + "\n".join(
                f"[{line.id}] {line.speaker}: {line.text}" for line in seg.lines
            )
            request = LLMRequest(
                task="script_segment",
                system=system,
                user=self._user(briefs[i], new_script, max(0, budget - used), fix=fix),
                schema=SegmentOut,
                metadata={"segment": seg.type, "revision": True},
            )
            out = self.llm.generate(request)
            assert isinstance(out, SegmentOut)
            # Fresh ids for the rewritten segment so overlap targets stay unambiguous.
            new_seg = self._materialise(out, briefs[i], _with_max_id(new_script, script))
            new_script.segments.append(new_seg)
        return new_script

    def _route(self, script: Script, issues: list[AuditIssue]) -> tuple[dict[int, list[str]], list[str]]:
        """The issues as notes for the writer, per segment index, and the ones about the episode as a whole."""
        per_segment: dict[int, list[str]] = {}
        episode_notes: list[str] = []
        for issue in issues:
            idx = None
            if issue.line_id:
                for i, seg in enumerate(script.segments):
                    if any(line.id == issue.line_id for line in seg.lines):
                        idx = i
                        break
            elif issue.claim_id:
                for i, seg in enumerate(script.segments):
                    if issue.claim_id in seg.covers:
                        idx = i
                        break
                if idx is None:
                    idx = next((i for i, s in enumerate(script.segments) if s.type == "body"), 0)
            where = f"regel {issue.line_id}: " if issue.line_id else ""
            note = f"- {where}{issue.message}" + (f" (suggestie: {issue.suggestion})" if issue.suggestion else "")
            if idx is None:
                if note not in episode_notes:
                    episode_notes.append(note)
                continue
            per_segment.setdefault(idx, []).append(note)
        return per_segment, episode_notes

    def _segment_speakers(self, seg: Segment) -> list[str]:
        ids = [h.id for h in self.cast.hosts]
        if seg.type == "guest":
            ids += [g.id for g in self.cast.guests]
        return ids

    def _flat_segments(self, script: Script) -> list[int]:
        """The body segments with the least overlapping speech, emptiest first.

        Where an episode-wide note about the shape of the dialogue gets delivered. Longest segments
        among the flat ones go first: they have room for a reaction between the sentences.
        """
        scored = []
        for i, seg in enumerate(script.segments):
            if seg.type != "body" or not seg.lines:
                continue
            overlaps = sum(1 for line in seg.lines if line.overlap.mode != "none")
            scored.append((overlaps, -len(seg.lines), i))
        scored.sort()
        return [i for _, _, i in scored[:FLAT_SEGMENT_TARGETS]]


POLISH_INSTRUCTIONS = """## Tafellezing: maak hier een echt gesprek van
Dit segment klopt inhoudelijk. Lees het in je hoofd hardop en herschrijf het zo dat het klinkt als deze twee mensen, die elkaar kennen en het ergens over hebben.
- Elke regel reageert op de vorige: op de woorden, de toon of de grap. Een vraag krijgt antwoord in de eerstvolgende beurt, een verhaal krijgt een reactie.
- Gewone woorden. Vervang elk woord dat je niet hardop tegen een vriend zou zeggen, elk woord dat niet bestaat en elke zin die naar een boek klinkt. Haal puntkomma's en dubbele punten uit gesproken zinnen.
- Niets uit de lucht. Een naam, een onderzoek, een begrip of iets uit het leven van de hosts dat hier valt zonder dat het in de aflevering is voorgesteld, stel je voor of haal je weg. Een beeld dat niets uitlegt, haal je weg.
- De draad. Elk punt volgt hoorbaar uit het vorige; zet er de zin tussen die zegt hoe ("en daarom", "maar dan"). Een punt dat alleen genoemd wordt, krijgt zijn waarom erbij als dat in de opdracht staat.
- Maak er een verhaal van. Waar een host feiten opsomt in korte losse zinnen ("Nee. 1533, Duitsland."), smeed je ze tot een lopende zin met een persoon, een reden of een gevolg erbij. Een vertelbeurt mag lang zijn; korte zinnen zijn voor reacties en vragen.
- De feiten liggen vast, de formulering niet. Een regel met covers zegt na afloop hetzelfde feit, met dezelfde getallen, namen en richting, en houdt zijn covers. Zeg hem wel in spreektaal: een zin die uit het boek is overgenomen, herschrijf je in eigen woorden, en een regel met veel getallen knip je in twee (de covers gaan mee naar de regel met het feit). Voeg geen feit toe en laat er geen weg.
- Wat uit de stof komt, zegt de explainer of de gast. Staat een feit bij de skeptic, geef het dan aan de explainer en laat de skeptic ernaar vragen of erop reageren.
- Ongeveer even lang (± 20%). Geef ook de beats: in twee zinnen wat je aan het gesprek veranderde."""


def _edit_row(line: Line) -> str:
    """A line as the dialogue edit sees it: who, how, which claims it carries, and how it meets the line before."""
    meta = []
    if line.tags:
        meta.append("tags: " + ", ".join(line.tags))
    if line.covers:
        meta.append("covers: " + ", ".join(line.covers))
    if line.overlap.mode != "none":
        meta.append("overlap: " + line.overlap.mode)
    if line.pause_after_ms:
        meta.append(f"pauze: {line.pause_after_ms}")
    return f"[{line.id}] {line.speaker}" + (f" ({'; '.join(meta)})" if meta else "") + f": {line.text}"


def _label(value: str | None, allowed: tuple[str, ...]) -> str | None:
    value = (value or "").strip().casefold()
    return value if value in allowed else None


def _performance_fields(raw: PerformanceLineOut, text: str) -> dict:
    """Keep only labels from the known vocabularies; phrases that don't rebuild the text are dropped by Line itself."""
    from pipeline.performance import DELIVERIES, MOODS, PHRASE_PAUSES, REACTIONS, TIMINGS

    phrases = [{"text": strip_cues(p.text), "delivery": _label(p.delivery, DELIVERIES),
                "pause_after": _label(p.pause_after, PHRASE_PAUSES) or "none"}
               for p in raw.phrases if strip_cues(p.text)]
    return {
        "delivery": _label(raw.delivery, DELIVERIES),
        "mood": _label(raw.mood, MOODS),
        "timing": _label(raw.timing, TIMINGS),
        "reaction": _label(raw.reaction, REACTIONS),
        "phrases": phrases if len(phrases) > 1 else [],
    }


def _bump(line_id: str) -> str:
    return f"l{int(line_id[1:]) + 1:03d}"


def _with_max_id(new_script: Script, old_script: Script) -> Script:
    """A view whose next_line_id continues after both scripts' highest id."""
    highest = 0
    for s in (new_script, old_script):
        for line in s.lines():
            highest = max(highest, int(line.id[1:]))

    class _View:
        def next_line_id(self) -> str:
            return f"l{highest + 1:03d}"

    return _View()  # type: ignore[return-value]


def write_script(plan: ContentPlan, book: Book | None, glossary: Glossary | None, cast: Cast, llm: LLM,
                 settings: Settings, *, continuity: list[ContinuityEntry] | None = None,
                 previous_plan: ContentPlan | None = None, next_chapter_title: str | None = None) -> Script:
    return Writer(llm, cast, settings, continuity).write(plan, book, glossary, previous_plan=previous_plan,
                                                        next_chapter_title=next_chapter_title)
