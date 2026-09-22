from pipeline.llm import FakeLLM
from pipeline.models import Glossary, LexiconEntry, Line, Overlap, Script, Segment
from pipeline.plan.content_plan import plan_chapter
from pipeline.script.audit import audit_script, check_coverage, check_structure, lint_rules
from pipeline.script.writer import Writer


def _script(lines, seg_type="body"):
    return Script(episode_id="ch01", target_minutes=1, segments=[Segment(type=seg_type, lines=lines)])


def _rules(script, cast, banned, settings, glossary=None):
    return {(i.rule, i.line_id) for i in lint_rules(script, cast, banned, glossary, settings)}


def test_banned_phrases_and_patterns(cast, banned, settings):
    s = _script([
        Line(id="l001", speaker="tessa", text="Goede vraag, dat is echt fascinerend."),
        Line(id="l002", speaker="joris", text="Dus samengevat: we weten het niet."),
        Line(id="l003", speaker="tessa", text="Het is snel, simpel en slim."),
    ])
    hits = _rules(s, cast, banned, settings)
    assert ("banned_phrase", "l001") in hits and ("summary_opener", "l002") in hits and ("banned_phrase", "l002") in hits
    assert ("three_part_flourish", "l003") in hits
    sev = {i.rule: i.severity for i in lint_rules(s, cast, banned, None, settings)}
    assert sev["three_part_flourish"] == "warning" and sev["banned_phrase"] == "blocking"


def test_tic_leak_and_overlap_rules(cast, banned, settings):
    g = Glossary(book_id="b", entries=[LexiconEntry(surface="entropie", kind="loanword_en", spoken="entropie")])
    s = _script([
        Line(id="l001", speaker="tessa", text="Wacht even, dat klopt niet."),
        Line(id="l002", speaker="joris", text="De kans is 0,5 en dat is de definitie—"),
        Line(id="l003", speaker="tessa", text="Ja precies.", overlap=Overlap(mode="backchannel", target="l002")),
        Line(id="l004", speaker="joris", text="Entropie is rommelig"),
        Line(id="l005", speaker="tessa", text="Nee!", overlap=Overlap(mode="interrupt", target="l004", cut_word="rommelig")),
        Line(id="l006", speaker="joris", text="Dat is heel erg lang voor een backchannel met veel te veel woorden erin", overlap=Overlap(mode="backchannel", target="l005")),
    ])
    hits = _rules(s, cast, banned, settings, g)
    assert ("tic_leak", "l001") in hits
    assert ("overlap_over_content", "l003") in hits  # number in target
    assert ("overlap_over_content", "l005") in hits  # glossary term in target
    assert ("interrupt_without_fragment", "l004") in hits
    assert ("backchannel_too_long", "l006") in hits


def test_interrupt_budget(cast, banned, settings):
    lines = []
    for i in range(1, 15):
        speaker = "tessa" if i % 2 else "joris"
        text = "Ja ja ja—" if i % 2 else "Nee"
        ov = Overlap(mode="interrupt", target=f"l{i - 1:03d}") if (i % 2 == 0) else Overlap()
        lines.append(Line(id=f"l{i:03d}", speaker=speaker, text=text, overlap=ov))
    s = _script(lines)
    assert any(r == "too_many_interrupts" for r, _ in _rules(s, cast, banned, settings))


def test_too_few_overlaps_warns_on_overlap_light_script(cast, banned, settings):
    s = _script([
        Line(id="l001", speaker="tessa", text="Dit is een rustige uitleg zonder enige onderbreking of reactie ertussen."),
        Line(id="l002", speaker="joris", text="En dit is het vervolg, ook gewoon netjes na elkaar zonder overlap."),
    ])
    # force the per-10-min threshold well above what two plain, unoverlapped lines could ever meet
    strict = settings.with_(min_connective_per_10min=100.0)
    issues = lint_rules(s, cast, banned, None, strict)
    hit = next((i for i in issues if i.rule == "too_few_overlaps"), None)
    assert hit is not None and hit.severity == "warning" and hit.line_id is None


def test_too_few_overlaps_not_raised_when_threshold_met(cast, banned, settings):
    s = _script([
        Line(id="l001", speaker="tessa", text="Wat vind jij daarvan?"),
        Line(id="l002", speaker="joris", text="Wacht even.", overlap=Overlap(mode="interrupt", target="l001")),
    ])
    issues = lint_rules(s, cast, banned, None, settings)
    assert not any(i.rule == "too_few_overlaps" for i in issues)


def test_structure_checks(cast):
    s = _script([
        Line(id="l001", speaker="piet", text="Wie ben ik?"),
        Line(id="l002", speaker="tessa", text="Nou.", overlap=Overlap(mode="backchannel", target="l001")),
        Line(id="l003", speaker="tessa", text="Zelf.", overlap=Overlap(mode="interrupt", target="l002")),
        Line(id="l004", speaker="joris", text="Fout doel.", overlap=Overlap(mode="interrupt", target="l001")),
        Line(id="l005", speaker="hanna_vos", text="Gast buiten gastblok."),
    ])
    rules = {(i.rule, i.line_id) for i in check_structure(s, cast)}
    assert ("unknown_speaker", "l001") in rules
    assert ("overlap_same_speaker", "l003") in rules
    assert ("overlap_target", "l004") in rules
    assert ("guest_outside_guest_segment", "l005") in rules
    assert ("no_cold_open", None) in rules


def test_coverage(book, fake_llm, cast):
    plan = plan_chapter(book, "ch01", fake_llm, cast)
    required = [c.id for c in plan.claims_with_relevance(3)]
    s = _script([Line(id="l001", speaker="tessa", text="x", covers=[required[0], "c99"])])
    issues, report = check_coverage(s, plan)
    assert report.missing == required[1:]
    assert sum(1 for i in issues if i.rule == "missing_claim") == len(required) - 1
    assert any(i.rule == "unknown_claim" for i in issues)


def test_full_audit_with_llm_checks(book, fake_llm, cast, banned, settings):
    plan = plan_chapter(book, "ch01", fake_llm, cast)
    script = Writer(fake_llm, cast, settings).write(plan, book, None)
    result = audit_script(script, plan, book, cast, banned, settings, llm=fake_llm)
    assert result.passed and result.support.checked > 0 and not result.coverage.missing
    assert result.stats["lines"] == sum(len(s.lines) for s in script.segments)

    # Unsupported verdict and LLM lint hit become blocking issues.
    target = next(l for l in script.lines() if l.covers)
    strict = FakeLLM(dict(fake_llm.handlers))
    strict.on("support", lambda req: {"verdicts": [{"line_id": target.id, "supported": False, "problem": "getal klopt niet"}]})
    strict.on("lint", lambda req: {"hits": [{"line_id": target.id, "rule": "summary_sentence", "explanation": "vat samen"},
                                            {"line_id": "l999", "rule": "filler", "explanation": "bestaat niet"}]})
    result = audit_script(script, plan, book, cast, banned, settings, llm=strict)
    assert not result.passed
    rules = {(i.rule, i.line_id) for i in result.blocking()}
    assert ("unsupported", target.id) in rules and ("summary_sentence", target.id) in rules
    assert ("filler", "l999") not in rules

    skipped = audit_script(script, plan, book, cast, banned, settings, llm=None)
    assert skipped.support.skipped and any(i.rule == "support_skipped" for i in skipped.warnings())
