"""Prototype scene: one 60-90 s performance-labelled scene, rendered as labelled A/B variants.

The same saved scene is rendered five ways so you can hear what each part of the
performance layer contributes before anything in the main pipeline changes:
current (today's path), timing (+ response timing, reactions), phrasing
(+ delivery, mood, hybrid phrase re-timing), acoustics (current + shared chain
only) and full.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from pipeline.audio.asr import make_aligner, make_transcriber
from pipeline.audio.performance_render import PerformanceOptions, Tables
from pipeline.audio.reactions import ReactionBank, speakable_text
from pipeline.audio.render import render_episode
from pipeline.audio.synth import make_synth
from pipeline.models import Line, RenderManifest, Script
from pipeline.performance import delivery_rate, phrases_of
from pipeline.script.writer import SegmentBrief, Writer, _chars

SCENE_SECONDS = 75
VARIANTS: dict[str, PerformanceOptions | None] = {
    "current": None,
    "timing": PerformanceOptions(timing=True, reactions=True),
    "phrasing": PerformanceOptions(delivery=True, phrasing=True),
    "acoustics": PerformanceOptions(acoustics=True),
    "full": PerformanceOptions.full(),
}
MOMENTS = {
    "explanation": "een uitleg (delivery explain)",
    "disagreement": "tegenspraak (delivery disagree)",
    "interruption": "een onderbreking (overlap interrupt, de regel ervoor eindigt op —)",
    "realization": "een realisatie (delivery realize, timing deliberate)",
    "laugh": "een lach als reactie (reaction laugh of chuckle)",
    "backchannel": "een backchannel onder de ander door (overlap backchannel)",
    "overlap": "een snelle, licht overlappende wissel (timing immediate)",
    "rates": "minstens drie verschillende soorten voordracht met een ander tempo",
}


def scene_brief(plan, cast, chars_per_second: float) -> SegmentBrief:
    explainer, skeptic = cast.by_role("explainer"), cast.by_role("skeptic")
    top = sorted(plan.key_claims, key=lambda c: (-c.exam_relevance, c.difficulty))[:2]
    claims = "\n".join(f"[{c.id}] {c.claim}" for c in top) or "(geen beweringen; kies iets uit de samenvatting)"
    moments = "\n".join(f"- {text}" for text in MOMENTS.values())
    return SegmentBrief(
        type="body", title="Prototype-scène",
        covers=[c.id for c in top],
        target_chars=_chars(SCENE_SECONDS, chars_per_second),
        speakers=[explainer.id, skeptic.id],
        instructions=(
            f"Een losse scène van ongeveer {SCENE_SECONDS} seconden waarin {explainer.name} iets uitlegt en "
            f"{skeptic.name} het er niet mee eens is, tot het kwartje valt. Geef elke regel performance-labels. "
            f"De scène moet al deze momenten bevatten, elk op een plek waar het gesprek erom vraagt:\n{moments}\n"
            f"Onderwerp:\n{claims}"
        ),
    )


def scene_coverage(script: Script, tables: Tables) -> dict[str, bool]:
    lines = list(script.lines())
    rates = {round(delivery_rate(tables(line.speaker), p.delivery or line.delivery), 3)
             for line in lines for p in phrases_of(line) if (p.delivery or line.delivery)}
    return {
        "explanation": any(line.delivery == "explain" for line in lines),
        "disagreement": any(line.delivery == "disagree" for line in lines),
        "interruption": any(line.overlap.mode == "interrupt" for line in lines),
        "realization": any(line.delivery == "realize" for line in lines),
        "laugh": any(line.reaction in ("laugh", "chuckle") or {"laughs", "laughing"} & set(line.tags) for line in lines),
        "backchannel": any(line.overlap.mode == "backchannel" for line in lines),
        "overlap": any(line.timing == "immediate" for line in lines),
        "rates": len(rates) >= 3,
    }


def _speakable(script: Script) -> Script:
    out = script.model_copy(deep=True)
    for seg in out.segments:
        for line in seg.lines:
            line.text = speakable_text(line)
    return out


def as_current(script: Script) -> Script:
    """What today's pipeline would get: no performance labels; a laugh becomes the existing laughs tag."""
    out = script.model_copy(deep=True)
    for seg in out.segments:
        for line in seg.lines:
            if line.reaction in ("laugh", "chuckle") and "laughs" not in line.tags:
                line.tags = [*line.tags, "laughs"][:2]
            line.delivery = line.mood = line.timing = line.reaction = None
            line.phrases = []
    return out


@dataclass
class PrototypeResult:
    out_dir: Path
    scene: Script
    coverage: dict[str, bool]
    files: dict[str, Path] = field(default_factory=dict)
    manifests: dict[str, RenderManifest] = field(default_factory=dict)
    report: Path | None = None


def run_prototype(pipeline, book_id: str, chapter_id: str, *, synth=None, transcriber=None, aligner=None,
                  variants: dict[str, PerformanceOptions | None] | None = None) -> PrototypeResult:
    settings, cast = pipeline.settings, pipeline.cast
    paths = pipeline.paths(book_id)
    plan = pipeline.load_plan(book_id, chapter_id)
    glossary = pipeline.load_glossary(book_id)
    tables = Tables(cast)
    writer = Writer(pipeline.llm, cast, settings, performance=True)
    brief = scene_brief(plan, cast, settings.chars_per_second)
    episode = f"{chapter_id}-scene"

    scene = writer.write_scene(plan, glossary, brief, episode_id=episode)
    coverage = scene_coverage(scene, tables)
    missing = [k for k, ok in coverage.items() if not ok]
    if missing:
        fix = "Deze momenten ontbreken nog, voeg ze toe waar het gesprek erom vraagt:\n" + "\n".join(
            f"- {MOMENTS[k]}" for k in missing)
        retry = writer.write_scene(plan, glossary, brief, fix=fix, episode_id=episode)
        retry_coverage = scene_coverage(retry, tables)
        if sum(retry_coverage.values()) >= sum(coverage.values()):
            scene, coverage = retry, retry_coverage
    scene = _speakable(scene)

    out_dir = paths.root / "prototype" / chapter_id
    out_dir.mkdir(parents=True, exist_ok=True)
    scene.save(out_dir / "scene.script.json")
    result = PrototypeResult(out_dir=out_dir, scene=scene, coverage=coverage)
    pipeline.emit("prototype", "scene", book_id=book_id, chapter=chapter_id, lines=len(list(scene.lines())),
                  missing=[k for k, ok in coverage.items() if not ok])

    fake = pipeline.fake_audio
    pipeline.release_llm()
    synth = synth or make_synth("null" if fake else "final", settings)
    transcriber = transcriber or make_transcriber(settings, fake=fake)
    aligner = aligner or make_aligner(settings, fake=fake)
    try:
        for name, options in (variants or VARIANTS).items():
            variant = (as_current(scene) if options is None else scene).model_copy(update={"episode_id": f"{episode}-{name}"})
            audio, _, manifest = render_episode(variant, glossary, cast, settings, paths, tier="final", synth=synth,
                                                transcriber=transcriber, aligner=aligner, performance=options,
                                                on_event=lambda kind, data, n=name: pipeline.emit("prototype", kind, variant=n, **data))
            target = out_dir / f"scene_{name}{audio.suffix}"
            shutil.copyfile(audio, target)
            result.files[name] = target
            result.manifests[name] = manifest
    finally:
        close = getattr(synth, "close", None)
        if callable(close):
            close()
    result.report = write_report(result, cast, settings)
    return result


def _labels(line: Line) -> str:
    parts = [f"{k}={v}" for k, v in (("delivery", line.delivery), ("mood", line.mood), ("timing", line.timing),
                                    ("reaction", line.reaction), ("overlap", line.overlap.mode if line.overlap.mode != "none" else None)) if v]
    return ", ".join(parts) or "-"


def write_report(result: PrototypeResult, cast, settings) -> Path:
    out: list[str] = ["# Prototype scene report", ""]
    minutes = result.scene.estimated_seconds(settings.chars_per_second) / 60
    out += [f"Lines: {len(list(result.scene.lines()))}, estimated {minutes * 60:.0f} s.", "", "## Coverage", ""]
    out += [f"- [{'x' if ok else ' '}] {name}: {MOMENTS[name]}" for name, ok in result.coverage.items()]

    out += ["", "## Files", "", "| variant | file | turns | alignment estimated | WER verified |", "|---|---|---|---|---|"]
    for name, path in result.files.items():
        m = result.manifests[name]
        verified, verifiable = m.verification_coverage()
        out.append(f"| {name} | {path.name} | {len(m.turns)} | {m.alignment_fallback_count()} | {verified}/{verifiable} |")

    full = result.manifests.get("full")
    if full is not None:
        out += ["", "## Full variant, per turn", "", "| turn | speaker | labels | exaggeration | phrasing | reaction |",
                "|---|---|---|---|---|---|"]
        by_id = {line.id: line for line in result.scene.lines()}
        for t in full.turns:
            labels = "; ".join(_labels(by_id[i]) for i in t.line_ids if i in by_id)
            perf = t.performance or {}
            phrasing = perf.get("phrasing", "-")
            if isinstance(phrasing, list):
                phrasing = " / ".join(f"{p['delivery'] or '-'} x{p['rate']:.2f}" + (f" +{p['pause_after_s']:.2f}s" if p["pause_after_s"] else "")
                                      for p in phrasing)
            reaction = f"{perf['reaction']} ({perf.get('source')})" if perf.get("reaction") else "-"
            exag = perf.get("exaggeration")
            out.append(f"| {t.turn_id} | {t.speaker} | {labels} | {exag if exag is not None else '-'} | {phrasing} | {reaction} |")
        if full.acoustics:
            out += ["", "## Shared chain: EQ correction per speaker (dB per octave band)", ""]
            out += [f"- {speaker}: " + ", ".join(f"{band} {gain:+.1f}" for band, gain in bands.items())
                    for speaker, bands in full.acoustics.items()]

    speakers = sorted({line.speaker for line in result.scene.lines()})
    out += ["", "## Reaction bank", ""]
    for speaker, labels in ReactionBank(settings.cast_dir).coverage(speakers).items():
        have = ", ".join(f"{k} {v}" for k, v in labels.items() if v) or "empty"
        out.append(f"- {speaker}: {have}")

    out += ["", "## Scene", ""]
    out += [f"- `{line.id}` **{line.speaker}** ({_labels(line)}): {line.text}" for line in result.scene.lines()]
    path = result.out_dir / "report.md"
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return path
