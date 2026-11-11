from pipeline.llm import FakeLLM
from pipeline.models import Glossary, LexiconEntry, Line, Overlap, Script, Segment
from pipeline.plan.thorough import plan_chapter_thorough
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


def test_flat_emotional_reaction_warns_when_the_other_host_never_picks_up_the_mood(cast, banned, settings):
    # tessa tags every one of her lines; joris, replying each time, never carries a tag of his own
    lines = []
    for i in range(1, 9):
        speaker = "tessa" if i % 2 else "joris"
        tags = ["excited"] if speaker == "tessa" else []
        lines.append(Line(id=f"l{i:03d}", speaker=speaker, text=f"Dit is regel {i} met genoeg tekst om te tellen.", tags=tags))
    s = _script(lines)
    issues = lint_rules(s, cast, banned, None, settings)
    hit = next((i for i in issues if i.rule == "flat_emotional_reaction"), None)
    assert hit is not None and hit.severity == "warning" and hit.line_id is None


def test_flat_emotional_reaction_not_raised_when_the_other_host_reacts(cast, banned, settings):
    # alternating speakers, every line tagged: 5 reaction opportunities, all of them picked up
    s = _script([
        Line(id="l001", speaker="tessa", text="Dit vind ik echt een geweldig punt over de stof.", tags=["excited"]),
        Line(id="l002", speaker="joris", text="Oké dat is inderdaad best bijzonder.", tags=["surprised"]),
        Line(id="l003", speaker="tessa", text="Precies, en dat verandert alles.", tags=["excited"]),
        Line(id="l004", speaker="joris", text="Nu volg ik het weer.", tags=["deadpan"]),
        Line(id="l005", speaker="tessa", text="Mooi, dan kunnen we verder.", tags=["warm"]),
        Line(id="l006", speaker="joris", text="Eindelijk een punt waar ik het mee eens ben.", tags=["deadpan"]),
    ])
    issues = lint_rules(s, cast, banned, None, settings)
    assert not any(i.rule == "flat_emotional_reaction" for i in issues)


def test_flat_emotional_reaction_not_raised_below_the_opportunity_floor(cast, banned, settings):
    s = _script([
        Line(id="l001", speaker="tessa", text="Dit vind ik echt een geweldig punt.", tags=["excited"]),
        Line(id="l002", speaker="joris", text="Oké, dat is inderdaad wel iets."),
    ])
    issues = lint_rules(s, cast, banned, None, settings)
    assert not any(i.rule == "flat_emotional_reaction" for i in issues)  # only 1 opportunity, below the floor of 4


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
    plan = plan_chapter_thorough(book, "ch01", fake_llm, cast)
    required = [c.id for c in plan.claims_with_relevance(3)]
    s = _script([Line(id="l001", speaker="tessa", text="x", covers=[required[0], "c99"])])
    issues, report = check_coverage(s, plan)
    assert report.missing == required[1:]
    assert sum(1 for i in issues if i.rule == "missing_claim") == len(required) - 1
    assert any(i.rule == "unknown_claim" for i in issues)


def test_full_audit_with_llm_checks(book, fake_llm, cast, banned, settings):
    plan = plan_chapter_thorough(book, "ch01", fake_llm, cast)
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


def _seg(lines, seg_type="body"):
    return Segment(type=seg_type, lines=lines)


def _repetition(segments, cast, plan=None):
    from pipeline.script.audit import lint_repetition

    script = Script(episode_id="ch01", target_minutes=10, segments=segments)
    return [(i.rule, i.line_id) for i in lint_repetition(script, cast, plan)]


def test_autopilot_dialogue_is_blocking(cast):
    tessa = [Line(id=f"t{i}", speaker="tessa", text=f"Oké dus, punt {i} van het verhaal gaat verder.") for i in range(4)]
    joris = [Line(id=f"j{i}", speaker="joris", text=t) for i, t in enumerate([
        "Een deur is geen brood.", "Een brief is geen honger.", "Gereedschap is geen volk.", "Een muur is geen staat.",
        "Er staat dat Alva weg moet.", "De tekst zegt Duitsland.", "Volgens de bron klopt het."])]
    rules = _repetition([_seg(tessa + joris)], cast)
    assert ("tic_overuse", "t1") in rules  # the second 'oké dus' in one segment is one too many
    assert ("stock_phrase", "j0") in rules and ("source_talk", "j4") in rules
    assert sum(r == "repeated_opening" for r, _ in rules) == 0  # the tic is counted once, as a tic


def test_recited_definitions_and_repeated_lines(cast, book, fake_llm):
    from pipeline.models import Definition, SourceSpan

    plan = plan_chapter_thorough(book, "ch01", fake_llm, cast)
    plan.definitions = [Definition(term="protestant", definition="Iemand die zich losmaakte van de katholieke kerk.",
                                   source_span=SourceSpan(section="ch01.1", start=0, end=1))]
    recital = "Oké dus, protestant: Iemand die zich in de zestiende eeuw losmaakte van de katholieke kerk."
    lines = [Line(id="l1", speaker="tessa", text=recital), Line(id="l2", speaker="joris", text="Hm."),
             Line(id="l3", speaker="tessa", text="Protestanten waren mensen die zich van de katholieke kerk losmaakten.")]
    later = [Line(id="l4", speaker="tessa", text=recital)]
    quiz = [Line(id="q1", speaker="tessa", text="Protestanten waren mensen die zich van de katholieke kerk losmaakten.")]
    rules = _repetition([_seg(lines), _seg(later), _seg(quiz, "quiz")], cast, plan)
    assert ("definition_recital", "l1") in rules and ("definition_recital", "l4") in rules
    assert ("duplicate_line", "l4") in rules
    assert ("definition_recital", "l3") not in rules and ("duplicate_line", "q1") not in rules  # a quiz may repeat


def test_varied_dialogue_passes(cast):
    lines = [Line(id="l1", speaker="tessa", text="Oké dus, in 1566 slaan mensen de beelden in de kerken kapot."),
             Line(id="l2", speaker="joris", text="Waarom juist de beelden?"),
             Line(id="l3", speaker="tessa", text="Omdat ze boos zijn op de kerk en op de koning tegelijk."),
             Line(id="l4", speaker="joris", text="Wacht even. Ook omdat ze honger hadden?"),
             Line(id="l5", speaker="tessa", text="Ja, werk, geld en eten waren op.")]
    assert _repetition([_seg(lines)], cast) == []


def test_a_line_lifted_from_the_voice_samples_is_blocking(cast):
    lines = [Line(id="l1", speaker="joris", text="Ik heb een systeem bedacht voor de roulette!"),
             Line(id="l2", speaker="tessa", text="Het balletje in deze opgave rolt nergens heen.")]
    rules = _repetition([_seg(lines)], cast)
    assert ("sample_copy", "l1") in rules and ("sample_copy", "l2") not in rules


def test_conversation_notes_from_the_llm_lint_are_warnings(book, fake_llm, cast, banned, settings):
    from pipeline.script.audit import lint_llm

    plan = plan_chapter_thorough(book, "ch01", fake_llm, cast)
    script = Writer(fake_llm, cast, settings).write(plan, book, None)
    first, second = list(script.lines())[:2]
    picky = FakeLLM(dict(fake_llm.handlers)).on("lint", lambda req: {"hits": [
        {"line_id": first.id, "rule": "stiff", "explanation": "Schrijftaal."},
        {"line_id": second.id, "rule": "filler", "explanation": "Vulling."}]})
    issues = {i.rule: i.severity for i in lint_llm(script, picky)}
    assert issues == {"stiff": "warning", "filler": "blocking"}
