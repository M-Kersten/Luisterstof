"""Stage 3b: audit. Three blocking checks: support, coverage, lint.

Support and the semantic half of lint need a model; coverage, structure and
the phrase/pattern half of lint are deterministic. A script cannot go to the
premium render with open blocking issues.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Callable
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field
from rapidfuzz import fuzz

from pipeline.config import Settings
from pipeline.llm import LLM, LLMRequest, text_block
from pipeline.models import (
    AuditIssue,
    AuditResult,
    Book,
    Cast,
    ContentPlan,
    CoverageReport,
    Glossary,
    KeyClaim,
    Script,
    SupportReport,
)
from pipeline.plan.content_plan import span_text
from pipeline.script.cast import BannedRules, voice_sample_lines
from pipeline.script.writer import quiz_questions

log = logging.getLogger(__name__)

PlanLookup = Callable[[str], ContentPlan | None]

DEFINITIONAL = re.compile(
    r"\b(is gedefinieerd als|definiëren we als|noemen we|heet|betekent|is per definitie|wordt gedefinieerd|is gelijk aan|formule)\b",
    re.IGNORECASE,
)
NUMBERISH = re.compile(r"\d")
QUESTION = re.compile(r"\?\s*$")


# ---------------------------------------------------------------------------
# Deterministic checks
# ---------------------------------------------------------------------------

def check_structure(script: Script, cast: Cast) -> list[AuditIssue]:
    issues: list[AuditIssue] = []
    known = {h.id for h in cast.hosts} | {g.id for g in cast.guests}
    if not script.segments:
        return [AuditIssue(check="structure", rule="empty_script", message="Het script heeft geen segmenten.")]
    if script.segments[0].type != "cold_open":
        issues.append(AuditIssue(check="structure", severity="warning", rule="no_cold_open", message="Het script begint niet met een koude opening."))
    if script.segments[-1].type != "outro":
        issues.append(AuditIssue(check="structure", severity="warning", rule="no_outro", message="Het script eindigt niet met een outro."))
    prev = None
    for seg in script.segments:
        if not seg.lines:
            issues.append(AuditIssue(check="structure", rule="empty_segment", message=f"Segment {seg.title or seg.type} heeft geen regels."))
        for i, line in enumerate(seg.lines):
            if line.speaker not in known:
                issues.append(AuditIssue(check="structure", rule="unknown_speaker", line_id=line.id,
                                         message=f"Onbekende spreker '{line.speaker}'."))
            if seg.type != "guest" and line.speaker in {g.id for g in cast.guests}:
                issues.append(AuditIssue(check="structure", rule="guest_outside_guest_segment", line_id=line.id,
                                         message="Een gast spreekt buiten het gastsegment."))
            ov = line.overlap
            if ov.mode != "none":
                if i == 0:
                    issues.append(AuditIssue(check="structure", rule="overlap_at_segment_start", line_id=line.id,
                                             message="De eerste regel van een segment mag geen overlap hebben."))
                elif ov.target != prev.id:
                    issues.append(AuditIssue(check="structure", rule="overlap_target", line_id=line.id,
                                             message="Overlap moet de direct voorafgaande regel als doel hebben."))
                elif prev.speaker == line.speaker:
                    issues.append(AuditIssue(check="structure", rule="overlap_same_speaker", line_id=line.id,
                                             message="Een spreker kan zichzelf niet onderbreken."))
            prev = line
    return issues


def check_coverage(script: Script, plan: ContentPlan, minimum_relevance: int = 3) -> tuple[list[AuditIssue], CoverageReport]:
    required = [c.id for c in plan.claims_with_relevance(minimum_relevance)]
    covered_all: set[str] = set()
    for line in script.lines():
        covered_all.update(line.covers)
    covered = [c for c in required if c in covered_all]
    missing = [c for c in required if c not in covered_all]
    issues = [
        AuditIssue(check="coverage", rule="missing_claim", claim_id=cid,
                   message=f"Bewering {cid} (tentamenrelevantie {plan.claim(cid).exam_relevance}) komt in geen enkele regel terug: {plan.claim(cid).claim}")
        for cid in missing
    ]
    known_ids = {c.id for c in plan.key_claims}
    for line in script.lines():
        for cid in line.covers:
            if ":" in cid:
                continue
            if cid not in known_ids:
                issues.append(AuditIssue(check="coverage", severity="warning", rule="unknown_claim", line_id=line.id, claim_id=cid,
                                         message=f"Regel verwijst naar onbekende bewering {cid}."))
    return issues, CoverageReport(required=required, covered=covered, missing=missing)


def _glossary_terms(glossary: Glossary | None) -> list[str]:
    if glossary is None:
        return []
    return [e.surface for e in glossary.entries if e.kind in ("loanword_en", "notation") and len(e.surface) > 1]


def lint_rules(script: Script, cast: Cast, banned: BannedRules, glossary: Glossary | None, settings: Settings) -> list[AuditIssue]:
    issues: list[AuditIssue] = []
    markers = {}
    for speaker in list(cast.hosts) + list(cast.guests):
        markers[speaker.id] = [m.casefold() for m in speaker.tic_markers if m.strip()]
    terms = [t.casefold() for t in _glossary_terms(glossary)]

    def norm(text: str) -> str:
        return " ".join(text.split()).casefold()

    prev = None
    interrupts = 0
    backchannels = 0
    reaction_opportunities = 0
    reactions = 0
    for seg in script.segments:
        for line in seg.lines:
            if prev is not None and prev.speaker != line.speaker and prev.tags:
                reaction_opportunities += 1
                if line.tags:
                    reactions += 1
            text = norm(line.text)
            for phrase in banned.phrases:
                if re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text):
                    issues.append(AuditIssue(check="lint", rule="banned_phrase", line_id=line.id,
                                             message=f"Verboden zinsdeel '{phrase}'.",
                                             suggestion="Schrap het of zeg wat je bedoelt zonder vulling."))
            for name, severity, rx in banned.patterns:
                if rx.search(line.text):
                    issues.append(AuditIssue(check="lint", severity=severity, rule=name, line_id=line.id,  # type: ignore[arg-type]
                                             message=f"Patroon '{name}' gevonden: {rx.search(line.text).group(0)!r}."))
            for other, phrases in markers.items():
                if other == line.speaker:
                    continue
                for m in phrases:
                    if re.search(r"(?<!\w)" + re.escape(m) + r"(?!\w)", text):
                        issues.append(AuditIssue(check="lint", rule="tic_leak", line_id=line.id,
                                                 message=f"'{m}' is een tic van {other}, niet van {line.speaker}."))
            ov = line.overlap
            if ov.mode != "none" and prev is not None:
                target_text = prev.text
                if NUMBERISH.search(target_text) or DEFINITIONAL.search(target_text) or any(
                        re.search(r"(?<!\w)" + re.escape(t) + r"(?!\w)", target_text.casefold()) for t in terms):
                    issues.append(AuditIssue(check="lint", rule="overlap_over_content", line_id=line.id,
                                             message="Overlap valt over een regel met een getal, definitie of vakterm.",
                                             suggestion="Verplaats de overlap naar een reactie of overgang."))
                if ov.mode == "interrupt":
                    interrupts += 1
                    if not prev.ends_with_fragment:
                        issues.append(AuditIssue(check="lint", rule="interrupt_without_fragment", line_id=prev.id,
                                                 message="Een onderbroken regel moet eindigen op een afgebroken zinsdeel met een kastlijntje (—).",
                                                 suggestion="Laat de regel eindigen als '...en dan verschuift dus de hele—'."))
                    if ov.cut_word and ov.cut_word.casefold() not in target_text.casefold():
                        issues.append(AuditIssue(check="lint", severity="warning", rule="cut_word_missing", line_id=line.id,
                                                 message=f"Afkapwoord '{ov.cut_word}' staat niet in de onderbroken regel."))
                if ov.mode == "backchannel":
                    backchannels += 1
                    if len(line.text.split()) > 8:
                        issues.append(AuditIssue(check="lint", rule="backchannel_too_long", line_id=line.id,
                                                 message="Een backchannel is kort: hooguit een paar woorden."))
            if not line.covers and seg.type in ("body", "guest", "reexplain", "quiz", "recap") and (
                    NUMBERISH.search(line.text) and len(line.text) > 60):
                issues.append(AuditIssue(check="lint", severity="warning", rule="uncovered_fact", line_id=line.id,
                                         message="Regel bevat getallen maar dekt geen bewering; controleer of dit feitelijk klopt."))
            prev = line

    minutes = script.estimated_seconds(settings.chars_per_second) / 60
    budget = int(math.ceil(max(minutes, 1) / 10 * settings.max_interrupts_per_10min))
    if interrupts > budget:
        issues.append(AuditIssue(check="lint", rule="too_many_interrupts",
                                 message=f"{interrupts} onderbrekingen bij een budget van {budget} (max {settings.max_interrupts_per_10min} per 10 minuten)."))
    connective = interrupts + backchannels
    min_connective = max(1, math.ceil(minutes / 10 * settings.min_connective_per_10min))
    if connective < min_connective:
        issues.append(AuditIssue(check="lint", severity="warning", rule="too_few_overlaps",
                                 message=f"Maar {connective} onderbrekingen en backchannels in ~{minutes:.1f} min "
                                         f"(richtlijn: minstens {min_connective}). Een script dat zo overlap-arm is "
                                         "klinkt als twee monologen naast elkaar, hoe goed de audio ook wordt gemixt.",
                                 suggestion="Voeg meer korte reacties ('ja precies', een korte onderbreking) toe tussen de hosts."))
    if reaction_opportunities >= 4:
        reaction_rate = reactions / reaction_opportunities
        if reaction_rate < settings.min_emotional_reaction_rate:
            issues.append(AuditIssue(check="lint", severity="warning", rule="flat_emotional_reaction",
                                     message=f"De ene host draagt {reaction_opportunities}x een emotie-tag terwijl de "
                                             f"ander aan het woord komt; die reageert daar zelf maar {reactions}x "
                                             f"({reaction_rate:.0%}) met een eigen tag op (richtlijn: minstens "
                                             f"{settings.min_emotional_reaction_rate:.0%}). Emotie die genegeerd wordt "
                                             "door de ander klinkt als twee monologen, ook als de tekst inhoudelijk klopt.",
                                     suggestion="Laat de andere host meebewegen (ook enthousiaster of kalmer worden) of "
                                                "bewust tegenwicht bieden, met een eigen tag, in plaats van neutraal door te praten."))
    target = script.target_minutes
    if minutes < target * 0.8 or minutes > target * 1.25:
        issues.append(AuditIssue(check="lint", severity="warning", rule="duration",
                                 message=f"Geschatte duur {minutes:.1f} min bij een streefduur van {target} min."))

    quiz = next((s for s in script.segments if s.type == "quiz"), None)
    if quiz is not None:
        questions = [line for line in quiz.lines if QUESTION.search(line.text) and line.speaker == cast.by_role("skeptic").id]
        wanted = quiz_questions(script.target_minutes)
        if len(questions) < wanted:
            issues.append(AuditIssue(check="lint", severity="warning", rule="quiz_questions",
                                     message=f"De quiz heeft {len(questions)} vragen van de skeptic; deze aflevering vraagt er {wanted}."))
        if not any(line.pause_after_ms > 0 for line in quiz.lines):
            issues.append(AuditIssue(check="lint", severity="warning", rule="quiz_pause",
                                     message="Geen enkele quizvraag heeft een stilte (pause_after_ms) voor de luisteraar."))
    return issues


# ---------------------------------------------------------------------------
# Repetition: what a writing model falls back on when it runs out of ideas
# ---------------------------------------------------------------------------

STOCK_DENIAL = re.compile(r"\b(is|zijn|blijft|blijven)\s+(nog\s+)?geen\b", re.IGNORECASE)  # "een deur is geen brood"
SOURCE_TALK = re.compile(r"\b(in de tekst|de tekst|de bron|volgens de bron|het boek|er staat|staat er)\b", re.IGNORECASE)
MAX_STOCK_DENIALS = 3  # per episode
MAX_SOURCE_TALK = 2  # per episode
MIN_DUPLICATE_CHARS = 40
SAMPLE_COPY_SCORE = 85.0  # a line this close to a voice-sample line was copied, not written


def _opening(text: str, words: int = 2) -> str:
    return " ".join(re.findall(r"[^\W\d_]+", text.casefold())[:words])


def lint_repetition(script: Script, cast: Cast, plan: ContentPlan | None = None) -> list[AuditIssue]:
    """Tics on every line, one joke template thirty times, definitions read out like a dictionary, the hosts
    talking about "de tekst" instead of the subject. Each of these is what a model does on autopilot, and each
    is blocking: a listener hears it within a minute."""
    issues: list[AuditIssue] = []
    markers = {sp.id: [m.casefold() for m in sp.tic_markers if m.strip()] for sp in list(cast.hosts) + list(cast.guests)}
    names = {sp.id: sp.name for sp in list(cast.hosts) + list(cast.guests)}
    terms = {d.term.casefold().strip() for d in (plan.definitions if plan else [])}
    stock_hits: dict[int, list] = {}
    source_hits: dict[int, list] = {}
    seen: dict[str, str] = {}
    samples = [" ".join(t.split()).casefold() for t in voice_sample_lines(cast) if len(t) >= 20]
    for i, seg in enumerate(script.segments):
        by_speaker: dict[str, list] = {}
        for line in seg.lines:
            by_speaker.setdefault(line.speaker, []).append(line)
            text = " ".join(line.text.split())
            folded = text.casefold()
            if any(STOCK_DENIAL.search(sent) and len(sent.split()) <= 9 for sent in re.split(r"(?<=[.!?])\s+", text)):
                stock_hits.setdefault(i, []).append(line)
            if SOURCE_TALK.search(text):
                source_hits.setdefault(i, []).append(line)
            bare = re.sub(r"^(oké dus|oke dus|nou|dus|ja)[,.]?\s*", "", folded)
            head, colon, _ = bare.partition(":")
            if colon and head.strip() in terms:
                issues.append(AuditIssue(check="lint", rule="definition_recital", line_id=line.id,
                                         message=f"De definitie van '{head.strip()}' wordt voorgelezen als woordenboekregel.",
                                         suggestion="Leg het begrip uit in eigen woorden, als deel van het gesprek."))
            if len(folded) >= 20 and any(fuzz.ratio(folded, sample) >= SAMPLE_COPY_SCORE for sample in samples):
                issues.append(AuditIssue(check="lint", rule="sample_copy", line_id=line.id,
                                         message="Deze regel komt uit de stemvoorbeelden van de cast.",
                                         suggestion="De voorbeelden laten de toon horen; schrijf een eigen regel over de stof."))
            key = re.sub(r"[^\w ]", "", folded)
            if len(key) >= MIN_DUPLICATE_CHARS and seg.type not in ("quiz", "recap"):  # repeating is their job
                if key in seen:
                    issues.append(AuditIssue(check="lint", rule="duplicate_line", line_id=line.id,
                                             message=f"Letterlijk dezelfde regel als {seen[key]}."))
                else:
                    seen[key] = line.id
        for speaker, lines in by_speaker.items():
            name = names.get(speaker, speaker)
            allowed = 1 + len(lines) // 15
            for marker in markers.get(speaker, []):
                hits = [ln for ln in lines if re.search(r"(?<!\w)" + re.escape(marker) + r"(?!\w)", ln.text.casefold())]
                if len(hits) > allowed:
                    issues.append(AuditIssue(check="lint", rule="tic_overuse", line_id=hits[allowed].id,
                                             message=f"{name} zegt '{marker}' {len(hits)} keer in dit segment; een tic "
                                                     f"werkt hooguit {allowed} keer.",
                                             suggestion="Laat de tic weg en begin de regel met de inhoud."))
            if len(lines) >= 5:
                openings: dict[str, list] = {}
                for ln in lines:
                    openings.setdefault(_opening(ln.text), []).append(ln)
                for opening, same in openings.items():
                    if opening and opening not in markers.get(speaker, []) and len(same) >= 3 and len(same) >= 0.4 * len(lines):
                        issues.append(AuditIssue(check="lint", rule="repeated_opening", line_id=same[2].id,
                                                 message=f"{name} begint {len(same)} van {len(lines)} regels in dit "
                                                         f"segment met '{opening}'.",
                                                 suggestion="Varieer hoe de regels beginnen."))
    if sum(len(v) for v in stock_hits.values()) > MAX_STOCK_DENIALS:
        total = sum(len(v) for v in stock_hits.values())
        for lines in stock_hits.values():
            issues.append(AuditIssue(check="lint", rule="stock_phrase", line_id=lines[0].id,
                                     message=f"Vaste formule 'een X is geen Y' ({total} keer in de aflevering).",
                                     suggestion="Laat de twijfel over de stof gaan: vraag waarom, hoe of wat het betekent."))
    if sum(len(v) for v in source_hits.values()) > MAX_SOURCE_TALK:
        total = sum(len(v) for v in source_hits.values())
        for lines in source_hits.values():
            issues.append(AuditIssue(check="lint", rule="source_talk", line_id=lines[0].id,
                                     message=f"De hosts praten {total} keer over 'de tekst' of 'de bron' in plaats van over "
                                             "het onderwerp.",
                                     suggestion="Vertel wat er gebeurde of hoe het zit; laat de verwijzing naar de bron weg."))
    return issues


# ---------------------------------------------------------------------------
# Told once, in their own words, by the host who knows
# ---------------------------------------------------------------------------

SOURCE_COPY_SCORE = 88.0  # a line this close to the book's sentence was read out, not told
SOURCE_COPY_MIN_CHARS = 40
NEAR_DUPLICATE_SCORE = 85.0
NEAR_DUPLICATE_MIN_CHARS = 60
RETELLING_SEGMENTS = ("quiz", "recap", "reexplain")  # going over a claim again is what these are for
NAME_IN_CLAIM = re.compile(r"(?<=[a-zà-ÿ,;:] )[A-ZÀ-Ý][a-zà-ÿ]{3,}(?:-[A-Za-zà-ÿ]+)?")


def _flat(text: str) -> str:
    return " ".join(re.sub(r"[^\w ]", " ", text.casefold()).split())


def lint_sources(script: Script, plan: ContentPlan) -> list[AuditIssue]:
    """A line that is the book's sentence, or the plan's sentence, with a speaker label in front. It is
    correct and it sounds like a lecture: nobody says "die basisalarmering is functioneel" to a friend."""
    issues: list[AuditIssue] = []
    for line in script.lines():
        text = _flat(line.text)
        for cid in line.covers:
            claim = plan.claim(cid)
            if claim is None:
                continue
            quote = _flat(claim.source_span.quote or "")
            stated = _flat(claim.claim)
            copied = len(quote) >= SOURCE_COPY_MIN_CHARS and len(text) >= SOURCE_COPY_MIN_CHARS and (
                fuzz.partial_ratio(quote, text) >= SOURCE_COPY_SCORE if len(quote) <= len(text)
                else fuzz.ratio(quote, text) >= SOURCE_COPY_SCORE)
            recited = len(stated) >= SOURCE_COPY_MIN_CHARS and fuzz.ratio(stated, text) >= SOURCE_COPY_SCORE
            if copied or recited:
                issues.append(AuditIssue(
                    check="lint", severity="warning", rule="source_copy", line_id=line.id, claim_id=cid,
                    message="Deze regel is de zin uit het boek, niet iets wat iemand hardop tegen een ander zegt.",
                    suggestion="Zeg hetzelfde feit in eigen woorden, zoals je het een vriend vertelt; de getallen "
                               "en namen blijven gelijk."))
                break
    return issues


def lint_homes(script: Script, plan: ContentPlan, cast: Cast) -> list[AuditIssue]:
    """Every claim has one place where it is told, and it is told by someone who can know it.

    - claim_repeated: the same claim carried by lines in more than one telling segment.
    - near_duplicate: nearly the same sentence twice.
    - skeptic_fact: the host who has not read the chapter is the first to state a claim.
    - term_before_intro: that host is the first to use a term the chapter defines, or a name from it."""
    issues: list[AuditIssue] = []
    skeptic = cast.by_role("skeptic")
    told_in: dict[str, int] = {}  # claim id -> index of the segment that told it first
    covered: set[str] = set()
    earlier: list[tuple[str, str]] = []  # (line id, flattened text) of the long lines so far
    for i, seg in enumerate(script.segments):
        for line in seg.lines:
            for cid in line.covers:
                if ":" in cid:
                    continue
                if seg.type not in RETELLING_SEGMENTS:
                    if cid in told_in and told_in[cid] != i:
                        issues.append(AuditIssue(
                            check="lint", severity="warning", rule="claim_repeated", line_id=line.id, claim_id=cid,
                            message=f"Bewering {cid} is al verteld in een eerder segment en wordt hier opnieuw gebracht.",
                            suggestion="Verwijs ernaar in een paar woorden of laat de regel weg; vertel hier wat nieuw is."))
                    told_in.setdefault(cid, i)
                if line.speaker == skeptic.id and cid not in covered and seg.type not in RETELLING_SEGMENTS:
                    issues.append(AuditIssue(
                        check="lint", severity="warning", rule="skeptic_fact", line_id=line.id, claim_id=cid,
                        message=f"{skeptic.name} heeft het hoofdstuk niet gelezen en brengt hier als eerste bewering {cid}.",
                        suggestion=f"Laat {skeptic.name} ernaar vragen of ernaar gokken, en geef het feit aan wie het weet."))
            covered.update(line.covers)
            text = _flat(line.text)
            if len(text) >= NEAR_DUPLICATE_MIN_CHARS and seg.type not in ("quiz", "recap"):
                twin = next((lid for lid, other in earlier if other != text and fuzz.ratio(other, text) >= NEAR_DUPLICATE_SCORE), None)
                if twin:
                    issues.append(AuditIssue(check="lint", severity="warning", rule="near_duplicate", line_id=line.id,
                                             message=f"Bijna dezelfde zin als {twin}.",
                                             suggestion="Wat al gezegd is, hoeft niet nog eens in dezelfde woorden."))
                earlier.append((line.id, text))

    cast_names = {sp.name.casefold() for sp in list(cast.hosts) + list(cast.guests)}
    terms = {d.term.strip() for d in plan.definitions if len(d.term.strip()) >= 5}
    names = {m.group(0) for c in plan.key_claims for m in NAME_IN_CLAIM.finditer(c.claim)}
    for term in sorted(terms | {n for n in names if n.casefold() not in cast_names}):
        rx = re.compile(r"(?<!\w)" + re.escape(term.casefold()) + r"(?!\w)")
        for seg in script.segments:
            first = next((line for line in seg.lines if rx.search(line.text.casefold())), None)
            if first is None:
                continue
            if first.speaker == skeptic.id and seg.type != "quiz":
                issues.append(AuditIssue(
                    check="lint", severity="warning", rule="term_before_intro", line_id=first.id,
                    message=f"{skeptic.name} gebruikt '{term}' voordat iemand het heeft uitgelegd of genoemd.",
                    suggestion=f"Laat de uitlegger '{term}' eerst voorstellen: wie of wat is het, en waarom gaat het er nu over."))
            break
    return issues


# ---------------------------------------------------------------------------
# Can it be said, and can it be heard
# ---------------------------------------------------------------------------

NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
COLON_LIST = re.compile(r":\s+\S")
MAX_NUMBERS_PER_LINE = 3


def lint_speakable(script: Script) -> list[AuditIssue]:
    """Punctuation that only exists on paper, and more numbers than an ear can hold in one go."""
    issues: list[AuditIssue] = []
    for line in script.lines():
        if ";" in line.text or (line.covers and COLON_LIST.search(line.text)):
            issues.append(AuditIssue(
                check="lint", severity="warning", rule="written_syntax", line_id=line.id,
                message="Een puntkomma of een dubbele punt met een rijtje erachter: dat is een zin voor op papier.",
                suggestion="Maak er twee gesproken zinnen van, of één lopende zin met 'en', 'want' of 'maar'."))
        numbers = NUMBER.findall(line.text)
        if len(numbers) > MAX_NUMBERS_PER_LINE:
            issues.append(AuditIssue(
                check="lint", severity="warning", rule="number_overload", line_id=line.id,
                message=f"{len(numbers)} getallen in één regel; een luisteraar onthoudt er één, hooguit twee.",
                suggestion="Verdeel de getallen over het gesprek en zeg bij elk wat het betekent."))
    return issues


SPOKEN_WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*")
SOUNDS = frozenset({"haha", "hahaha", "hehe", "hihi", "pff", "pfff", "hmm", "hmmm", "eh", "ehm", "uh", "uhm", "oeh",
                    "jeetje", "tja", "nou", "oké", "oke", "joh", "ach", "goh", "poeh", "aha", "oei", "wauw"})
LINKS = ("", "s", "e", "en")  # what Dutch puts between the parts of a compound
MIN_WORD = 4


@lru_cache(maxsize=1)
def _dictionary():
    """The Dutch word list, or None when the optional package is not installed (the check is then skipped)."""
    try:
        from spellchecker import SpellChecker
    except ImportError:
        return None
    try:
        return SpellChecker(language="nl")
    except Exception:  # pragma: no cover - a broken install must not stop an audit
        log.warning("the Dutch word list could not be loaded; unknown words are not checked")
        return None


def known_words(*texts: str) -> frozenset[str]:
    """Every word in these texts: the book, the cast, the plan. A word the chapter uses is a word the hosts may use."""
    return frozenset(w.casefold() for text in texts for w in SPOKEN_WORD.findall(text or ""))


def lint_spelling(script: Script, known: frozenset[str] = frozenset()) -> list[AuditIssue]:
    """Words that exist nowhere: not in a Dutch word list, not in the book, not in the cast, and not as a
    compound of words that do. "vertrust" for "vertraagt" turns a finding into noise, and a voice reads it
    out as written. Names (capitalised mid-sentence) are left alone."""
    dictionary = _dictionary()
    if dictionary is None:
        return []

    def exists(word: str) -> bool:
        return word in known or word in SOUNDS or word in dictionary

    def compound(word: str, depth: int = 0) -> bool:
        if exists(word):
            return True
        if depth >= 2:
            return False
        for cut in range(MIN_WORD, len(word) - MIN_WORD + 1):
            if not exists(word[:cut]):
                continue
            rest = word[cut:]
            if any(rest.startswith(link) and len(rest) - len(link) >= MIN_WORD and compound(rest[len(link):], depth + 1)
                   for link in LINKS):
                return True
        return False

    issues: list[AuditIssue] = []
    flagged: set[str] = set()
    for line in script.lines():
        for m in SPOKEN_WORD.finditer(line.text):
            word = m.group(0)
            at_start = not line.text[:m.start()].strip() or line.text[:m.start()].rstrip()[-1:] in ".!?:…—\"'“‘"
            if len(word) < MIN_WORD or word.isupper() or (word[0].isupper() and not at_start):
                continue  # short words, emphasis in capitals and names
            low = word.casefold()
            if low in flagged or compound(low):
                continue
            flagged.add(low)
            issues.append(AuditIssue(
                check="lint", severity="warning", rule="unknown_word", line_id=line.id,
                message=f"'{word}' staat in geen woordenlijst, niet in het boek en niet in de cast.",
                suggestion="Is het een verschrijving of een verzonnen woord, kies dan het gewone woord dat bedoeld is."))
    return issues


# ---------------------------------------------------------------------------
# Telegraphic lines: facts read off a crib sheet instead of told
# ---------------------------------------------------------------------------

SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")
WORD = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*")  # "1533" and "Oranje-Nassau" are one word each
TELEGRAPHIC_MEAN_WORDS = 5.0  # two or more sentences averaging fewer words than this read as a list of facts
TELEGRAPHIC_MIN_WORDS = 6  # "Ja. Precies." is a reaction, not a list
TELEGRAPHIC_SHARE = 0.2  # of a segment's substantive lines before the segment counts as told in fragments
TELEGRAPHIC_MAX_HITS = 4  # per segment: enough for the rewrite to see the pattern
NARRATIVE_SEGMENTS = ("cold_open", "recap", "body", "guest", "reexplain", "outro")  # the quiz asks short questions


def sentence_lengths(text: str) -> list[int]:
    """Words per sentence, empty sentences left out."""
    return [n for n in (len(WORD.findall(part)) for part in SENTENCE_END.split(text.strip())) if n]


def is_telegraphic(text: str) -> bool:
    """"Nee. 1533, Duitsland. Zijn ouders waren protestant." Several sentences, each a bare fact."""
    lengths = sentence_lengths(text)
    return (len(lengths) >= 2 and sum(lengths) >= TELEGRAPHIC_MIN_WORDS
            and sum(lengths) / len(lengths) < TELEGRAPHIC_MEAN_WORDS)


def lint_telegraphic(script: Script) -> list[AuditIssue]:
    """A segment in which the hosts state facts in strings of three-word sentences. One such line is a
    deadpan; a pattern of them sounds like a lesson read aloud, and a listener tunes out. A warning per line
    (up to a few per segment), so the rewrite and the table read know where to tell instead of list."""
    issues: list[AuditIssue] = []
    for seg in script.segments:
        if seg.type not in NARRATIVE_SEGMENTS:
            continue
        substantive = [line for line in seg.lines if line.overlap.mode != "backchannel"
                       and sum(sentence_lengths(line.text)) >= TELEGRAPHIC_MIN_WORDS]
        hits = [line for line in substantive if is_telegraphic(line.text)]
        if len(hits) < 2 or len(hits) < TELEGRAPHIC_SHARE * len(substantive):
            continue
        for line in hits[:TELEGRAPHIC_MAX_HITS]:
            issues.append(AuditIssue(
                check="lint", severity="warning", rule="telegraphic", line_id=line.id,
                message=f"Losse korte zinnetjes, als een spiekbriefje dat wordt voorgelezen ({len(hits)} van "
                        f"{len(substantive)} regels in dit segment).",
                suggestion="Vertel het: smeed de feiten tot een lopende zin met een persoon, een reden of een "
                           "gevolg erbij, en laat de uitlegger een verhaal opbouwen."))
    return issues


# ---------------------------------------------------------------------------
# LLM checks
# ---------------------------------------------------------------------------

class SupportVerdict(BaseModel):
    line_id: str
    supported: bool = Field(description="True als elke feitelijke bewering in de regel door de bronfragmenten wordt gedekt.")
    problem: str | None = Field(description="Wat er niet klopt of niet in de bron staat, anders null.")


class SupportOut(BaseModel):
    verdicts: list[SupportVerdict]


UNCOVERED_GUEST_MIN_CHARS = 40  # shorter than this a guest's line is a reaction, not a statement

SUPPORT_SYSTEM = """Je controleert een podcastscript op feitelijke ondersteuning door de bron. Dit is studiemateriaal; een grappige aflevering die iets verkeerd vertelt is erger dan geen aflevering.

Per regel krijg je de tekst en de bronfragmenten die de regel zou moeten dekken. Beoordeel:
- Wordt elke feitelijke bewering in de regel gedekt door de fragmenten? Getallen, definities, richtingen van verbanden en voorwaarden moeten kloppen.
- Analogieën en beelden mogen simplificeren zolang ze de bron niet tegenspreken.
- Een parafrase in eigen woorden is ondersteund als de inhoud klopt; letterlijk citeren is niet nodig. Een regel als "hij wilde Alva het land uit hebben" bij de bron "Hij wil Alva weg krijgen uit Nederland" is ondersteund.
- Oordeel "niet ondersteund" alleen als de regel iets beweert dat niet in de fragmenten staat of ze tegenspreekt. Een detail weglaten is geen fout.
- Vragen, grappen, reacties en emotie zijn geen beweringen; beoordeel alleen de feitelijke inhoud.
- Een regel die bewust iets fout zegt als onderdeel van een misvatting of quizfout is alleen in orde als dat uit de regel zelf blijkt (de spreker twijfelt, of het is een quizantwoord dat daarna wordt gecorrigeerd).
Geef voor elke regel een oordeel. Wees streng op inhoud, soepel op stijl."""


class LintHit(BaseModel):
    line_id: str
    rule: Literal["summary_sentence", "rhetorical_triplet", "topical_joke", "generic_joke", "visual_joke", "announcing", "filler",
                  "ignores_previous", "stiff", "formal", "forced_image"]
    explanation: str


class LintOut(BaseModel):
    hits: list[LintHit]


LINT_SYSTEM = """Je bent de linter van een Nederlandse studiepodcast. Je zoekt in het script naar zinnen die de regels overtreden. Wees precies: alleen echte overtredingen, met regel-id.

Regels:
- summary_sentence: een zin die samenvat of verpakt wat er net is besproken ("dus wat we nu weten is...", "kortom", een afsluitende recap van een blok). Uitzondering: in het segment outro mag in één of twee zinnen het antwoord op de vraag van de aflevering klinken.
- rhetorical_triplet: een driedelige opsomming die als stijlfiguur dient in plaats van inhoud ("snel, simpel en slim"). Inhoudelijke opsommingen uit de stof zijn toegestaan.
- topical_joke: een grap over actualiteit, nieuws, bekende personen of trends.
- generic_joke: een grap die in elke aflevering zou werken omdat hij niet aan de stof hangt.
- visual_joke: een grap waarvoor de luisteraar iets moet zien.
- announcing: een zin die aankondigt wat het gesprek gaat doen in plaats van het te doen ("laten we kijken naar...", "nu gaan we het hebben over...").
- filler: AI-podcastvulling zonder inhoud ("dat is echt een goed punt", "interessant", "zeker weten").
- ignores_previous: de regel gaat voorbij aan wat de ander net zei (een vraag zonder antwoord, een grap waar niemand op reageert, een verhaal dat in het niets valt) en werkt gewoon het volgende punt af.
- stiff: een zin die niemand hardop zo zou zeggen tegen een vriend: schrijftaal, een opsomming uit een leerboek, een definitie als woordenboekregel. Een lange, kronkelende spreekzin met een zijpad is niet stiff; een rijtje kale feiten in losse zinnetjes wel.
- formal: een woord of wending die te plechtig, ambtelijk of boekerig is voor een gesprek aan de keukentafel ("functioneel", "derhalve", "prevalentie", "behoedzaamheid", "er is sprake van"), terwijl er een gewoon woord voor bestaat. Een vakterm die de luisteraar moet leren en die wordt uitgelegd, is geen overtreding.
- forced_image: een vergelijking of beeld dat niets uitlegt, dat niet te volgen is, of dat op een ander beeld wordt gestapeld ("een pleister is geen bewijs van brand", "dan mist de pot zijn randen").
Meld geen stijlvoorkeuren, alleen overtredingen van deze elf regels."""


def check_support(script: Script, plan: ContentPlan, book: Book, llm: LLM, plan_lookup: PlanLookup | None = None,
                  book_lookup: Callable[[str], Book | None] | None = None,
                  progress: Callable[[int, int, str], None] | None = None) -> tuple[list[AuditIssue], SupportReport]:
    issues: list[AuditIssue] = []
    checked = 0
    unsupported = 0

    def resolve_claim(cid: str) -> tuple[KeyClaim | None, Book | None]:
        if ":" in cid:
            chapter_id, _, local = cid.partition(":")
            other = plan_lookup(chapter_id) if plan_lookup else None
            claim = other.claim(local) if other else None
            return claim, (book_lookup(chapter_id) if book_lookup else book) if claim else None
        return plan.claim(cid), book

    guest_ids = {script.guest_id} if script.guest_id else set()
    everything = " ".join(c.claim for c in plan.key_claims)
    for n, seg in enumerate(script.segments, start=1):
        if progress:
            progress(n, len(script.segments), f"bronnencontrole: {seg.title or seg.type}")
        rows: list[str] = []
        ids: list[str] = []
        for line in seg.lines:
            if not line.covers:
                # An expert who gives advice the book never gave is the one voice a listener will not doubt.
                if seg.type == "guest" and line.speaker in guest_ids and len(line.text) >= UNCOVERED_GUEST_MIN_CHARS:
                    rows.append(f"### Regel {line.id} ({line.speaker}, gast, zonder bewering): {line.text}\n"
                                f"  bron: alles wat het hoofdstuk hierover zegt: \"{everything}\"")
                    ids.append(line.id)
                continue
            fragments = []
            for cid in line.covers:
                claim, src_book = resolve_claim(cid)
                if claim is None:
                    continue
                try:
                    source = span_text(src_book or book, claim.source_span, context=150)
                except KeyError:
                    source = claim.source_span.quote or ""
                fragments.append(f"  bewering {cid}: {claim.claim}\n  bron: \"{' '.join(source.split())}\"")
            if not fragments:
                continue
            rows.append(f"### Regel {line.id} ({line.speaker}): {line.text}\n" + "\n".join(fragments))
            ids.append(line.id)
        if not rows:
            continue
        checked += len(rows)
        request = LLMRequest(
            task="support",
            system=SUPPORT_SYSTEM,
            user="\n\n".join(rows) + "\n\nGeef een oordeel voor elke regel: " + ", ".join(ids),
            schema=SupportOut,
        )
        out = llm.generate(request)
        assert isinstance(out, SupportOut)
        for v in out.verdicts:
            if v.line_id in ids and not v.supported:
                unsupported += 1
                issues.append(AuditIssue(check="support", rule="unsupported", line_id=v.line_id,
                                         message=v.problem or "Bewering wordt niet door de bron ondersteund.",
                                         suggestion="Zeg wat het bronfragment zegt, in eigen woorden: de getallen, de namen "
                                                    "en de richting van het verband kloppen ermee."))
    return issues, SupportReport(checked=checked, unsupported=unsupported)


def lint_llm(script: Script, llm: LLM) -> list[AuditIssue]:
    rows = []
    for seg in script.segments:
        rows.append(f"## {seg.type}")
        rows += [f"[{line.id}] {line.speaker}: {line.text}" for line in seg.lines]
    request = LLMRequest(
        task="lint",
        system=LINT_SYSTEM,
        user="\n".join(rows),
        schema=LintOut,
    )
    out = llm.generate(request)
    assert isinstance(out, LintOut)
    valid = {line.id for line in script.lines()}
    return [
        AuditIssue(check="lint", rule=h.rule, line_id=h.line_id, message=h.explanation,
                   severity="warning" if h.rule in CONVERSATION_RULES else "blocking")
        for h in out.hits if h.line_id in valid
    ]


# ---------------------------------------------------------------------------
# The listener: someone who hears it once and knows nothing else
# ---------------------------------------------------------------------------

class ListenerHit(BaseModel):
    line_id: str
    rule: Literal["no_antecedent", "unanswered", "nonsense", "contradicts"]
    explanation: str = Field(description="Wat de luisteraar hier mist of niet kan volgen, in één zin.")


class ListenerOut(BaseModel):
    hits: list[ListenerHit]


LISTENER_SYSTEM = """Je bent een luisteraar die deze Nederlandse studiepodcast voor het eerst hoort. Je kent het boek niet, je kent de hosts niet en je hebt geen eerdere aflevering gehoord. Je weet alleen wat er tot nu toe hardop is gezegd, in deze volgorde.

Je krijgt wat je al hoorde en daarna één nieuw segment. Meld de regels in dat nieuwe segment waar je als luisteraar de draad kwijtraakt. Wees precies: alleen echte gevallen, met regel-id.

- no_antecedent: de regel verwijst naar iemand of iets wat jou nooit is voorgesteld. Een persoon ("Ellen zei"), een onderzoek ("die enquête", "de onderzoekers"), een proef of opstelling ("de diepe kant"), een vakterm die nog niet is uitgelegd, een voorwerp, een eerdere grap of gebeurtenis ("mijn stoofpot", "die pot"). Iets wat in dezelfde regel of vlak ervoor wordt uitgelegd, telt niet.
- unanswered: de ander stelde net een vraag of maakte een bezwaar, en deze regel gaat daar niet op in maar begint over iets anders.
- nonsense: een zin waar je geen betekenis van kunt maken: een woord dat niet bestaat, een beeld dat nergens op slaat, een zin die grammaticaal niet loopt.
- contradicts: de spreker zegt dat de ander fout zit terwijl de ander iets zei wat klopt met wat er eerder is uitgelegd, of de regel spreekt tegen wat dezelfde aflevering eerder beweerde.

Meld geen stijlvoorkeuren en beoordeel niet of de feiten kloppen met de werkelijkheid: jij weet alleen wat er gezegd is."""

LISTENER_HEARD_CHARS = 24000  # how much of the episode so far a call carries


def _rows(lines) -> str:
    return "\n".join(f"[{line.id}] {line.speaker}: {line.text}" for line in lines)


def check_listener(script: Script, llm: LLM, progress: Callable[[int, int, str], None] | None = None) -> list[AuditIssue]:
    """One call per segment, with everything said before it. What a first-time listener cannot follow: a
    name nobody introduced, a question nobody answered, a sentence that means nothing, a correction of
    something that was right. The support check cannot see these (it reads one line against one source
    fragment), and a lint over the whole script in one call reads past them."""
    issues: list[AuditIssue] = []
    heard: list = []
    todo = [seg for seg in script.segments if seg.lines]
    for n, seg in enumerate(todo, start=1):
        if progress:
            progress(n, len(todo), f"luistertest: {seg.title or seg.type}")
        before = _rows(heard)[-LISTENER_HEARD_CHARS:] if heard else "(nog niets: dit is het begin van de aflevering)"
        ids = [line.id for line in seg.lines]
        out = llm.generate(LLMRequest(
            task="listener",
            system=[text_block(LISTENER_SYSTEM), text_block("## Wat je tot nu toe hoorde\n" + before)],
            user=f"## Nieuw segment ({seg.type})\n" + _rows(seg.lines) + "\n\nMeld alleen regels uit dit segment: "
                 + ", ".join(ids),
            schema=ListenerOut,
            metadata={"segment": seg.type},
        ))
        assert isinstance(out, ListenerOut)
        seen: set[tuple[str, str]] = set()
        for hit in out.hits:
            if hit.line_id in ids and (hit.line_id, hit.rule) not in seen:
                seen.add((hit.line_id, hit.rule))
                issues.append(AuditIssue(check="lint", severity="warning", rule=hit.rule, line_id=hit.line_id,
                                         message=hit.explanation, suggestion=LISTENER_FIXES[hit.rule]))
        heard += seg.lines
    return issues


LISTENER_FIXES = {
    "no_antecedent": "Stel het eerst voor (wie of wat is het, en waarom gaat het er nu over) of haal de verwijzing weg.",
    "unanswered": "Laat de volgende beurt eerst antwoord geven op de vraag of het bezwaar.",
    "nonsense": "Zeg in gewone woorden wat er bedoeld is.",
    "contradicts": "Verbeter alleen wat echt fout was, en zeg ja tegen wat klopt.",
}


# ---------------------------------------------------------------------------
# The landing: can a student explain the hard claims after hearing this?
# ---------------------------------------------------------------------------

class LandingVerdict(BaseModel):
    claim_id: str
    landed: bool = Field(description="True als een luisteraar het punt na deze aflevering kan uitleggen, niet alleen herhalen.")
    missing: str = Field(description="Wat er ontbreekt of verkeerd wordt uitgelegd, in één zin. Leeg als landed true is.")


class LandingOut(BaseModel):
    verdicts: list[LandingVerdict]


LANDING_SYSTEM = """Je bent een student die alleen deze aflevering van een studiepodcast heeft gehoord, en je moet morgen tentamen doen. Hieronder staat de hele aflevering.

Daarna krijg je een paar punten uit het hoofdstuk, elk met wat het boek erover zegt. Beoordeel per punt of het in de aflevering is geland:
- landed is true als iemand die alleen de aflevering hoorde het punt in eigen woorden kan uitleggen: wat er gevonden is of hoe het zit, en ook hoe dat komt of wat het betekent, voor zover het boekfragment dat zegt.
- landed is false als het punt alleen wordt genoemd of voorgelezen, als de uitleg waarom ontbreekt terwijl het boek die wel geeft, als de aflevering het tegenstrijdig of verkeerd om vertelt, of als de uitleg schuilgaat achter beeldspraak die niets verklaart.
Het aantal keren dat een punt genoemd wordt, telt niet: vier keer dezelfde zin is niet uitgelegd.
Geef voor elk punt een oordeel."""

LANDING_MIN_DIFFICULTY = 4
LANDING_MAX_CLAIMS = 5


def check_landing(script: Script, plan: ContentPlan, book: Book, llm: LLM) -> list[AuditIssue]:
    """One call for the claims of difficulty four and up: stated is not explained. The first draft of the
    hoogtevrees episode said its hardest finding four times and its reason not once."""
    hard = sorted((c for c in plan.key_claims if c.difficulty >= LANDING_MIN_DIFFICULTY),
                  key=lambda c: (-c.difficulty, -c.exam_relevance))[:LANDING_MAX_CLAIMS]
    if not hard or not any(seg.lines for seg in script.segments):
        return []
    rows = []
    for c in hard:
        try:
            source = " ".join(span_text(book, c.source_span, context=400).split())
        except KeyError:
            source = c.source_span.quote or ""
        rows.append(f"[{c.id}] {c.claim}\n  boek: \"{source}\"")
    out = llm.generate(LLMRequest(
        task="landing",
        system=[text_block(LANDING_SYSTEM), text_block("## De aflevering\n" + _rows(script.lines()))],
        user="\n\n".join(rows) + "\n\nGeef een oordeel voor elk punt: " + ", ".join(c.id for c in hard),
        schema=LandingOut,
    ))
    assert isinstance(out, LandingOut)
    issues: list[AuditIssue] = []
    wanted = {c.id for c in hard}
    for v in out.verdicts:
        cid = v.claim_id.strip().strip("[]")
        if cid not in wanted or v.landed:
            continue
        wanted.discard(cid)
        first = next((line.id for line in script.lines() if cid in line.covers), None)
        issues.append(AuditIssue(
            check="lint", severity="warning", rule="not_landed", line_id=first, claim_id=cid,
            message=f"Bewering {cid} wordt genoemd maar niet uitgelegd: {v.missing or 'het waarom ontbreekt'}",
            suggestion="Leg uit hoe het komt of wat het betekent, met wat het boek erover zegt, in gewone woorden."))
    return issues


# How the dialogue answers itself. These are judgement calls, so they are warnings: quality mode rewrites for them
# and the dialogue edit (Writer.polish) gets them as notes.
CONVERSATION_RULES = frozenset({"ignores_previous", "stiff", "formal", "forced_image"})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _known_words(book: Book, plan: ContentPlan, cast: Cast, glossary: Glossary | None) -> frozenset[str]:
    """What the hosts may say without a dictionary knowing it: the words of the book, the plan and the cast."""
    texts = [ch.full_text() for ch in book.chapters]
    texts += [c.claim for c in plan.key_claims] + [f"{d.term} {d.definition}" for d in plan.definitions]
    for sp in cast.hosts:
        texts += [sp.name, sp.background, *sp.life, *sp.stories, *sp.voice, *sp.tics, *sp.tic_markers]
    texts += [g.name for g in cast.guests] + list(cast.relationship) + [cast.voice_samples]
    texts += [e.surface for e in (glossary.entries if glossary else [])]
    return known_words(*texts)


def audit_script(
    script: Script,
    plan: ContentPlan,
    book: Book,
    cast: Cast,
    banned: BannedRules,
    settings: Settings,
    *,
    glossary: Glossary | None = None,
    llm: LLM | None = None,
    plan_lookup: PlanLookup | None = None,
    llm_lint: bool = True,
    progress: Callable[[int, int, str], None] | None = None,
) -> AuditResult:
    result = AuditResult(episode_id=script.episode_id, script_revision=script.revision)
    result.issues += check_structure(script, cast)
    cov_issues, result.coverage = check_coverage(script, plan)
    result.issues += cov_issues
    result.issues += lint_rules(script, cast, banned, glossary, settings)
    result.issues += lint_repetition(script, cast, plan)
    result.issues += lint_telegraphic(script)
    result.issues += lint_sources(script, plan)
    result.issues += lint_homes(script, plan, cast)
    result.issues += lint_speakable(script)
    result.issues += lint_spelling(script, _known_words(book, plan, cast, glossary))
    if llm is not None:
        sup_issues, result.support = check_support(script, plan, book, llm, plan_lookup, progress=progress)
        result.issues += sup_issues
        if llm_lint:
            result.issues += lint_llm(script, llm)
        if llm_lint and settings.script_listener:
            result.issues += check_listener(script, llm, progress)
            result.issues += check_landing(script, plan, book, llm)
    else:
        result.support = SupportReport(skipped=True)
        result.issues.append(AuditIssue(check="support", severity="warning", rule="support_skipped",
                                        message="Geen LLM beschikbaar: de bronondersteuning is niet gecontroleerd."))
    minutes = script.estimated_seconds(settings.chars_per_second) / 60
    result.stats = {
        "lines": sum(len(s.lines) for s in script.segments),
        "chars": script.char_count,
        "estimated_minutes": round(minutes, 1),
        "interrupts": sum(1 for line in script.lines() if line.overlap.mode == "interrupt"),
        "backchannels": sum(1 for line in script.lines() if line.overlap.mode == "backchannel"),
        "blocking": len(result.blocking()),
        "warnings": len(result.warnings()),
    }
    _ = text_block  # keep import for symmetry with other stages
    return result.finalize()
