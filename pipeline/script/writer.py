"""Stage 3: dialogue generation.

The episode format is fixed by a deterministic blueprint (which segments,
which claims each one must cover, how long it may run). The model writes one
segment at a time with the previous lines as context, so the stable prefix
(rules, cast, continuity, plan, glossary) is cached across calls.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field
from rapidfuzz import fuzz

from pipeline.config import Settings
from pipeline.cues import cue_reaction, speakable, strip_cues
from pipeline.llm import LLM, LLMRequest, text_block
from pipeline.models import (
    AuditResult,
    Book,
    Cast,
    ContentPlan,
    ContinuityEntry,
    Glossary,
    Guest,
    KeyClaim,
    Line,
    Overlap,
    Script,
    Segment,
    SegmentType,
)
from pipeline.script.cast import cast_bible_text, continuity_text, pick_guest
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
    "guest": 6 * 60,
    "reexplain": 3 * 60,
    "quiz": 3.5 * 60,
    "outro": 30,
}


def _chars(seconds: float, cps: float) -> int:
    return int(seconds * cps * 0.9)


def _claim_text(c: KeyClaim) -> str:
    return f"[{c.id}] (moeilijkheid {c.difficulty}, tentamen {c.exam_relevance}) {c.claim}"


def _cluster(claims: list[KeyClaim], n_blocks: int) -> list[list[KeyClaim]]:
    if not claims:
        return []
    n_blocks = max(1, min(n_blocks, len(claims)))
    size = len(claims) / n_blocks
    clusters: list[list[KeyClaim]] = []
    for i in range(n_blocks):
        start, end = round(i * size), round((i + 1) * size)
        chunk = claims[start:end]
        if chunk:
            clusters.append(chunk)
    return clusters


def build_blueprint(
    plan: ContentPlan,
    cast: Cast,
    *,
    guest: Guest | None,
    previous_plan: ContentPlan | None,
    next_chapter_title: str | None,
    target_minutes: int,
    chars_per_second: float,
) -> list[SegmentBrief]:
    explainer, skeptic = cast.by_role("explainer"), cast.by_role("skeptic")
    hosts = [explainer.id, skeptic.id]
    briefs: list[SegmentBrief] = []

    total_seconds = target_minutes * 60
    fixed = SEGMENT_SECONDS["cold_open"] + SEGMENT_SECONDS["reexplain"] + SEGMENT_SECONDS["quiz"] + SEGMENT_SECONDS["outro"]
    if previous_plan is not None:
        fixed += SEGMENT_SECONDS["recap"]
    if guest is not None and plan.needs_expert:
        fixed += SEGMENT_SECONDS["guest"]
    body_seconds = max(6 * 60, total_seconds - fixed)

    claims = list(plan.key_claims)
    top = sorted(claims, key=lambda c: (-c.exam_relevance, c.difficulty))
    hardest = plan.hardest_claim()

    # Cold open: an argument about something from the chapter.
    open_claim = top[0] if top else None
    misconception = plan.misconceptions[0] if plan.misconceptions else None
    briefs.append(SegmentBrief(
        type="cold_open", title="Koude opening",
        covers=[open_claim.id] if open_claim else [],
        target_chars=_chars(SEGMENT_SECONDS["cold_open"], chars_per_second),
        speakers=hosts,
        instructions=(
            "De hosts zitten al midden in een meningsverschil over iets uit dit hoofdstuk. Geen intro, geen welkom, "
            "geen namen noemen als begroeting. Begin midden in een zin of midden in een bezwaar. "
            + (f"Onderwerp van het meningsverschil: {open_claim.claim}" if open_claim else "")
            + (f"\nEventueel als inzet: de misvatting '{misconception.wrong}'." if misconception else "")
        ),
    ))

    # Recap of the previous episode.
    if previous_plan is not None:
        prev_top = sorted(previous_plan.key_claims, key=lambda c: (-c.exam_relevance, -c.difficulty))
        prev_claim = prev_top[0] if prev_top else None
        briefs.append(SegmentBrief(
            type="recap", title="Terugblik",
            covers=[f"{previous_plan.chapter_id}:{prev_claim.id}"] if prev_claim else [],
            target_chars=_chars(SEGMENT_SECONDS["recap"], chars_per_second),
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

    # Body blocks.
    n_blocks = max(3, min(5, round(len(claims) / 4))) if claims else 3
    clusters = _cluster(claims, n_blocks)
    per_block_chars = _chars(body_seconds / max(1, len(clusters)), chars_per_second)
    definitions = {d.term.casefold(): d for d in plan.definitions}
    for i, cluster in enumerate(clusters, start=1):
        misc = plan.misconceptions[i] if i < len(plan.misconceptions) else None
        relevant_defs = [d for d in plan.definitions if any(d.term.casefold() in c.claim.casefold() for c in cluster)]
        lines = ["Beweringen die dit blok moet dekken (gebruik de id's in 'covers'):"]
        lines += [_claim_text(c) for c in cluster]
        lines += [f"  bron {c.id}: \"{(c.source_span.quote or '')[:300]}\"" for c in cluster]
        if relevant_defs:
            lines.append("Definities die hier thuishoren (letterlijk correct houden, nooit onder overlap):")
            lines += [f"- {d.term}: {d.definition}" for d in relevant_defs]
        if misc:
            lines.append(
                f"Loop bewust in deze misvatting en corrigeer hem daarna: '{misc.wrong}' -> '{misc.right}' "
                f"(verleidelijk omdat: {misc.why_tempting})."
            )
        if i == 1 and plan.worked_example and plan.worked_example.steps:
            lines.append(
                "Werk het uitgewerkte voorbeeld uit de bron door in dialoog: "
                f"{plan.worked_example.setup} Stappen: " + " | ".join(plan.worked_example.steps)
                + f" Antwoord: {plan.worked_example.answer}"
            )
        briefs.append(SegmentBrief(
            type="body", title=f"Blok {i}",
            covers=[c.id for c in cluster],
            target_chars=per_block_chars,
            speakers=hosts,
            instructions=(
                f"{skeptic.name} duwt, {explainer.name} legt uit. Elke bewering hierboven moet in minstens één regel "
                "terugkomen, feitelijk correct, met de bijbehorende id in 'covers'.\n" + "\n".join(lines)
            ),
        ))
    _ = definitions

    # Guest block.
    if guest is not None and plan.needs_expert:
        hard = sorted(claims, key=lambda c: -c.difficulty)[:4]
        briefs.append(SegmentBrief(
            type="guest", title=f"Gast: {guest.name}",
            covers=[c.id for c in hard],
            target_chars=_chars(SEGMENT_SECONDS["guest"], chars_per_second),
            speakers=hosts + [guest.id],
            instructions=(
                f"Driespreker-blok met gast {guest.name} (spreker-id {guest.id}), vakgebied: {plan.expert_domain}. "
                f"Reden voor de gast: {plan.expert_reason or 'de stof is te moeilijk voor de hosts alleen'}. "
                f"De gast wordt kort en zonder ceremonie binnengehaald (geen welkomstritueel). {skeptic.name} test de gast, "
                f"{explainer.name} vertaalt naar een beeld. De gast dekt deze beweringen:\n"
                + "\n".join(_claim_text(c) for c in hard)
                + "\n" + "\n".join(f"  bron {c.id}: \"{(c.source_span.quote or '')[:300]}\"" for c in hard)
            ),
        ))

    # Wacht, opnieuw.
    if hardest is not None:
        briefs.append(SegmentBrief(
            type="reexplain", title="Wacht, opnieuw",
            covers=[hardest.id],
            target_chars=_chars(SEGMENT_SECONDS["reexplain"], chars_per_second),
            speakers=hosts,
            instructions=(
                f"{skeptic.name} weigert de eerdere uitleg van het moeilijkste begrip en eist een tweede route. "
                f"{explainer.name} legt het opnieuw uit langs een compleet andere weg (ander beeld, andere volgorde, "
                "van het voorbeeld naar de regel in plaats van andersom). Het begrip:\n"
                + _claim_text(hardest) + f"\n  bron: \"{(hardest.source_span.quote or '')[:400]}\""
            ),
        ))

    # Quiz.
    quiz_pool = [c for c in top if c.exam_relevance >= 4] or top
    quiz_claims = quiz_pool[:4]
    briefs.append(SegmentBrief(
        type="quiz", title="Quiz",
        covers=[c.id for c in quiz_claims],
        target_chars=_chars(SEGMENT_SECONDS["quiz"], chars_per_second),
        speakers=hosts,
        instructions=(
            f"{skeptic.name} overhoort {explainer.name} met precies vier vragen, elk gebaseerd op één van deze beweringen. "
            f"Na elke vraag valt een stilte voor de luisteraar: zet pause_after_ms op 3000 op de regel met de vraag. "
            f"{explainer.name} heeft er precies één fout, {skeptic.name} corrigeert met de juiste informatie uit de bron. "
            "Vragen en antwoorden moeten exact kloppen met de bron.\n"
            + "\n".join(_claim_text(c) for c in quiz_claims)
            + "\n" + "\n".join(f"  bron {c.id}: \"{(c.source_span.quote or '')[:300]}\"" for c in quiz_claims)
        ),
    ))

    # Outro.
    briefs.append(SegmentBrief(
        type="outro", title="Uitsmijter",
        covers=[],
        target_chars=_chars(SEGMENT_SECONDS["outro"], chars_per_second),
        speakers=hosts,
        instructions=(
            "Dertig seconden. Geen samenvatting van wat er is besproken, geen bedankje aan de luisteraar. "
            + (f"Een haakje naar het volgende hoofdstuk: '{next_chapter_title}'. " if next_chapter_title else "Een haakje naar wat er nog open ligt. ")
            + "Mag eindigen op een open vraag of een plaagstoot."
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


class SegmentOut(BaseModel):
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

## De twee motoren
De rollen zijn tegelijk de leermotor en de comedymotor. De explainer legt uit met beelden en duwt die soms te ver; de skeptic accepteert geen hand-wave en stelt de vraag die de luisteraar heeft. Het meningsverschil is echt en gaat over de stof.

## Nauwkeurigheid (niet onderhandelbaar)
- Dit is studiemateriaal. Elke feitelijke bewering komt uit het inhoudsplan en de bronfragmenten. Voeg geen feiten toe die niet in de bron staan.
- Getallen, definities en formuleringen uit de bron zijn heilig. Een analogie mag simplificeren, nooit tegenspreken.
- Markeer regels die een bewering overbrengen met de claim-id in 'covers'. Grappen, vragen en reacties krijgen geen covers.
- Als een host expres iets fout zegt (misvatting, quizfout), corrigeer het in het gesprek binnen twee regels, en zet de covers alleen op de regel met de correcte informatie.

## Humor
- Humor hangt aan de stof. Toegestane bronnen: een analogie die tot hij breekt wordt doorgevoerd, een host die het begrip verkeerd toepast, een callback uit het continuïteitslogboek, de hosts die het oneens zijn over hoe moeilijk iets is.
- Verboden: actuele grappen, grappen die in elke aflevering zouden werken, grappen waarvoor de luisteraar iets moet zien.
- Elke host houdt zich aan de eigen verbale tics. De tics van de ander zijn verboden terrein.

## Verboden taal (de linter blokkeert het script bij één treffer)
- AI-podcastvulling: "dat is echt fascinerend", "goede vraag", "laten we eens duiken in", "aan het eind van de dag", "welkom bij", "in deze aflevering", "kortom", "samengevat", "in een notendop".
- Elke zin die samenvat wat er net is besproken. Geen recap aan het einde van een blok.
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
- Spreektaal: korte zinnen, contracties, halve zinnen mogen. Wissel zinslengte sterk af.
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


class Writer:
    def __init__(self, llm: LLM, cast: Cast, settings: Settings, continuity: list[ContinuityEntry] | None = None,
                 *, performance: bool = False):
        self.llm = llm
        self.cast = cast
        self.settings = settings
        self.continuity = continuity or []
        self.performance = performance

    # System prefix: identical for every segment call of one episode.
    def _system(self, plan: ContentPlan, glossary: Glossary | None, guest: Guest | None, briefs: list[SegmentBrief]) -> list[dict]:
        rules = WRITER_RULES.format(tags=", ".join(known_tags()), cps=int(self.settings.chars_per_second))
        if self.performance:
            rules += PERFORMANCE_RULES
        outline = "## Opbouw van deze aflevering\n" + "\n".join(
            f"{i + 1}. {b.title} ({b.type}, ~{b.target_chars} tekens, dekt {', '.join(b.covers) or 'niets'})"
            for i, b in enumerate(briefs)
        )
        return [
            text_block(rules),
            text_block(cast_bible_text(self.cast, guest) + "\n\n" + continuity_text(self.continuity)),
            text_block(_plan_text(plan) + "\n\n" + _glossary_text(glossary) + "\n\n" + outline),
        ]

    def _user(self, brief: SegmentBrief, script: Script, interrupts_left: int, fix: str | None = None) -> str:
        prior = _lines_text(script)
        parts = [
            f"# Schrijf nu segment: {brief.title} (type {brief.type})",
            f"Streeflengte: ongeveer {brief.target_chars} tekens gesproken tekst (± 15%).",
            f"Toegestane sprekers: {', '.join(brief.speakers)}.",
            f"Onderbrekingen (interrupt) die nog mogen in de rest van de aflevering: {interrupts_left}.",
            "Opdracht:\n" + brief.instructions,
        ]
        if fix:
            parts.append("## Herschrijf: los deze problemen op en behoud wat werkte\n" + fix)
        parts.append("## Regels tot nu toe (ga hier naadloos op verder, herhaal niets)\n" + (prior or "(nog niets)"))
        return "\n\n".join(parts)

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
        seg = Segment(type=brief.type, covers=list(brief.covers), title=brief.title, brief=brief.instructions)
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
        target_minutes = target_minutes or self.settings.target_minutes
        guest = pick_guest(self.cast, plan.expert_domain) if plan.needs_expert else None
        briefs = build_blueprint(
            plan, self.cast, guest=guest, previous_plan=previous_plan, next_chapter_title=next_chapter_title,
            target_minutes=target_minutes, chars_per_second=self.settings.chars_per_second,
        )
        script = Script(episode_id=plan.chapter_id, target_minutes=target_minutes, title=plan.chapter_title,
                        guest_id=guest.id if guest and plan.needs_expert else None)
        system = self._system(plan, glossary, guest, briefs)
        budget = self._interrupt_budget(target_minutes)
        for brief in briefs:
            used = sum(1 for line in script.lines() if line.overlap.mode == "interrupt")
            request = LLMRequest(
                task="script_segment",
                system=system,
                user=self._user(brief, script, max(0, budget - used)),
                schema=SegmentOut,
                effort="high",
                metadata={"segment": brief.type},
            )
            out = self.llm.generate(request)
            assert isinstance(out, SegmentOut)
            script.segments.append(self._materialise(out, brief, script))
            log.info("segment %s: %d lines", brief.title, len(script.segments[-1].lines))
        return script

    def write_scene(self, plan: ContentPlan, glossary: Glossary | None, brief: SegmentBrief, *,
                    fix: str | None = None, episode_id: str | None = None) -> Script:
        """One stand-alone segment with performance labels (the prototype scene)."""
        script = Script(episode_id=episode_id or plan.chapter_id, target_minutes=2, title=brief.title)
        request = LLMRequest(
            task="script_scene",
            system=self._system(plan, glossary, None, [brief]),
            user=self._user(brief, script, 2, fix=fix),
            schema=PerformanceSegmentOut if self.performance else SegmentOut,
            effort="high",
            metadata={"segment": brief.type},
        )
        out = self.llm.generate(request)
        script.segments.append(self._materialise(out, brief, script))  # type: ignore[arg-type]
        return script

    def _interrupt_budget(self, target_minutes: int) -> int:
        return int(math.ceil(target_minutes / 10 * self.settings.max_interrupts_per_10min))

    def revise(self, script: Script, audit: AuditResult, plan: ContentPlan, glossary: Glossary | None) -> Script:
        """Rewrite only the segments with blocking issues, feeding the issues back to the model."""
        blocking = audit.blocking()
        if not blocking:
            return script
        guest = self.cast.guest(script.guest_id) if script.guest_id else None
        per_segment: dict[int, list[str]] = {}
        for issue in blocking:
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
            if idx is None:
                continue
            where = f"regel {issue.line_id}: " if issue.line_id else ""
            per_segment.setdefault(idx, []).append(f"- {where}{issue.message}" + (f" (suggestie: {issue.suggestion})" if issue.suggestion else ""))

        briefs = [SegmentBrief(type=s.type, title=s.title or s.type, covers=list(s.covers), target_chars=max(200, s.char_count),
                               instructions=s.brief or "", speakers=self._segment_speakers(s)) for s in script.segments]
        system = self._system(plan, glossary, guest, briefs)
        budget = self._interrupt_budget(script.target_minutes)
        new_script = Script(episode_id=script.episode_id, target_minutes=script.target_minutes, title=script.title,
                            guest_id=script.guest_id, revision=script.revision + 1)
        for i, seg in enumerate(script.segments):
            if i not in per_segment:
                new_script.segments.append(seg)
                continue
            used = sum(1 for line in new_script.lines() if line.overlap.mode == "interrupt")
            fix = "\n".join(per_segment[i]) + "\n\nHuidige versie van dit segment:\n" + "\n".join(
                f"[{line.id}] {line.speaker}: {line.text}" for line in seg.lines
            )
            request = LLMRequest(
                task="script_segment",
                system=system,
                user=self._user(briefs[i], new_script, max(0, budget - used), fix=fix),
                schema=SegmentOut,
                effort="high",
                metadata={"segment": seg.type, "revision": True},
            )
            out = self.llm.generate(request)
            assert isinstance(out, SegmentOut)
            # Fresh ids for the rewritten segment so overlap targets stay unambiguous.
            new_seg = self._materialise(out, briefs[i], _with_max_id(new_script, script))
            new_script.segments.append(new_seg)
        return new_script

    def _segment_speakers(self, seg: Segment) -> list[str]:
        ids = [h.id for h in self.cast.hosts]
        if seg.type == "guest":
            ids += [g.id for g in self.cast.guests]
        return ids


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
