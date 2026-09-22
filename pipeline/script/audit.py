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
from typing import Literal

from pydantic import BaseModel, Field

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
from pipeline.script.cast import BannedRules

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
    for seg in script.segments:
        for line in seg.lines:
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
    target = script.target_minutes
    if minutes < target * 0.8 or minutes > target * 1.25:
        issues.append(AuditIssue(check="lint", severity="warning", rule="duration",
                                 message=f"Geschatte duur {minutes:.1f} min bij een streefduur van {target} min."))

    quiz = next((s for s in script.segments if s.type == "quiz"), None)
    if quiz is not None:
        questions = [line for line in quiz.lines if QUESTION.search(line.text) and line.speaker == cast.by_role("skeptic").id]
        if len(questions) < 4:
            issues.append(AuditIssue(check="lint", severity="warning", rule="quiz_questions",
                                     message=f"De quiz heeft {len(questions)} vragen van de skeptic; het format vraagt er vier."))
        if not any(line.pause_after_ms > 0 for line in quiz.lines):
            issues.append(AuditIssue(check="lint", severity="warning", rule="quiz_pause",
                                     message="Geen enkele quizvraag heeft een stilte (pause_after_ms) voor de luisteraar."))
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


SUPPORT_SYSTEM = """Je controleert een podcastscript op feitelijke ondersteuning door de bron. Dit is studiemateriaal; een grappige aflevering die iets verkeerd vertelt is erger dan geen aflevering.

Per regel krijg je de tekst en de bronfragmenten die de regel zou moeten dekken. Beoordeel:
- Wordt elke feitelijke bewering in de regel gedekt door de fragmenten? Getallen, definities, richtingen van verbanden en voorwaarden moeten kloppen.
- Analogieën en beelden mogen simplificeren zolang ze de bron niet tegenspreken.
- Vragen, grappen, reacties en emotie zijn geen beweringen; beoordeel alleen de feitelijke inhoud.
- Een regel die bewust iets fout zegt als onderdeel van een misvatting of quizfout is alleen in orde als dat uit de regel zelf blijkt (de spreker twijfelt, of het is een quizantwoord dat daarna wordt gecorrigeerd).
Geef voor elke regel een oordeel. Wees streng op inhoud, soepel op stijl."""


class LintHit(BaseModel):
    line_id: str
    rule: Literal["summary_sentence", "rhetorical_triplet", "topical_joke", "generic_joke", "visual_joke", "announcing", "filler"]
    explanation: str


class LintOut(BaseModel):
    hits: list[LintHit]


LINT_SYSTEM = """Je bent de linter van een Nederlandse studiepodcast. Je zoekt in het script naar zinnen die de regels overtreden. Wees precies: alleen echte overtredingen, met regel-id.

Regels:
- summary_sentence: een zin die samenvat of verpakt wat er net is besproken ("dus wat we nu weten is...", "kortom", een afsluitende recap van een blok).
- rhetorical_triplet: een driedelige opsomming die als stijlfiguur dient in plaats van inhoud ("snel, simpel en slim"). Inhoudelijke opsommingen uit de stof zijn toegestaan.
- topical_joke: een grap over actualiteit, nieuws, bekende personen of trends.
- generic_joke: een grap die in elke aflevering zou werken omdat hij niet aan de stof hangt.
- visual_joke: een grap waarvoor de luisteraar iets moet zien.
- announcing: een zin die aankondigt wat het gesprek gaat doen in plaats van het te doen ("laten we kijken naar...", "nu gaan we het hebben over...").
- filler: AI-podcastvulling zonder inhoud ("dat is echt een goed punt", "interessant", "zeker weten").
Meld geen stijlvoorkeuren, alleen overtredingen van deze zeven regels."""


def check_support(script: Script, plan: ContentPlan, book: Book, llm: LLM, plan_lookup: PlanLookup | None = None,
                  book_lookup: Callable[[str], Book | None] | None = None) -> tuple[list[AuditIssue], SupportReport]:
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

    for seg in script.segments:
        rows: list[str] = []
        ids: list[str] = []
        for line in seg.lines:
            if not line.covers:
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
            effort="high",
            max_tokens=8000,
            cache_system=False,
        )
        out = llm.generate(request)
        assert isinstance(out, SupportOut)
        for v in out.verdicts:
            if v.line_id in ids and not v.supported:
                unsupported += 1
                issues.append(AuditIssue(check="support", rule="unsupported", line_id=v.line_id,
                                         message=v.problem or "Bewering wordt niet door de bron ondersteund.",
                                         suggestion="Herschrijf trouw aan het bronfragment."))
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
        effort="medium",
        max_tokens=6000,
        cache_system=False,
    )
    out = llm.generate(request)
    assert isinstance(out, LintOut)
    valid = {line.id for line in script.lines()}
    return [
        AuditIssue(check="lint", rule=h.rule, line_id=h.line_id, message=h.explanation)
        for h in out.hits if h.line_id in valid
    ]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

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
) -> AuditResult:
    result = AuditResult(episode_id=script.episode_id, script_revision=script.revision)
    result.issues += check_structure(script, cast)
    cov_issues, result.coverage = check_coverage(script, plan)
    result.issues += cov_issues
    result.issues += lint_rules(script, cast, banned, glossary, settings)
    if llm is not None:
        sup_issues, result.support = check_support(script, plan, book, llm, plan_lookup)
        result.issues += sup_issues
        if llm_lint:
            result.issues += lint_llm(script, llm)
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
