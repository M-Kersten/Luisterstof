from pipeline.models import AuditIssue, AuditResult, Guest
from pipeline.plan.content_plan import plan_chapter
from pipeline.script.cast import pick_guest
from pipeline.script.writer import Writer, build_blueprint


def test_blueprint_shape(book, fake_llm, cast, settings):
    plan1 = plan_chapter(book, "ch01", fake_llm, cast)
    plan2 = plan_chapter(book, "ch02", fake_llm, cast)
    briefs = build_blueprint(plan2, cast, guest=None, previous_plan=plan1, next_chapter_title="X", target_minutes=25, chars_per_second=15)
    types = [b.type for b in briefs]
    assert types[0] == "cold_open" and types[1] == "recap" and types[-1] == "outro"
    assert "reexplain" in types and "quiz" in types and "guest" not in types
    body_covers = [c for b in briefs if b.type == "body" for c in b.covers]
    assert sorted(body_covers) == sorted(c.id for c in plan2.key_claims)
    quiz = next(b for b in briefs if b.type == "quiz")
    assert len(quiz.covers) == 4 and all(plan2.claim(c).exam_relevance >= 4 for c in quiz.covers)
    recap = next(b for b in briefs if b.type == "recap")
    assert recap.covers[0].startswith("ch01:")
    assert sum(b.target_chars for b in briefs) > 15000

    plan2.needs_expert, plan2.expert_domain = True, "statistiek"
    guest = pick_guest(cast, plan2.expert_domain)
    assert isinstance(guest, Guest) and guest.id == "hanna_vos"
    assert pick_guest(cast, "kunstgeschiedenis").id == "sam_de_wit"
    with_guest = build_blueprint(plan2, cast, guest=guest, previous_plan=None, next_chapter_title=None, target_minutes=25, chars_per_second=15)
    assert "guest" in [b.type for b in with_guest] and "recap" not in [b.type for b in with_guest]


def test_write_produces_valid_script(book, fake_llm, cast, settings):
    plan = plan_chapter(book, "ch01", fake_llm, cast)
    script = Writer(fake_llm, cast, settings).write(plan, book, None, next_chapter_title="Verdelingen")
    ids = [l.id for l in script.lines()]
    assert ids == [f"l{i:03d}" for i in range(1, len(ids) + 1)]
    for seg in script.segments:
        for i, line in enumerate(seg.lines):
            if line.overlap.mode != "none":
                assert i > 0 and line.overlap.target == seg.lines[i - 1].id and seg.lines[i - 1].speaker != line.speaker
    assert script.guest_id is None and script.segments[0].type == "cold_open"
    assert any(l.overlap.mode == "interrupt" for l in script.lines())
    covered = {c for l in script.lines() for c in l.covers}
    assert {c.id for c in plan.key_claims} <= covered


def test_revise_only_touches_flagged_segments(book, fake_llm, cast, settings):
    plan = plan_chapter(book, "ch01", fake_llm, cast)
    writer = Writer(fake_llm, cast, settings)
    script = writer.write(plan, book, None)
    quiz = next(s for s in script.segments if s.type == "quiz")
    audit = AuditResult(episode_id="ch01", issues=[AuditIssue(check="lint", rule="banned_phrase", line_id=quiz.lines[0].id, message="x")]).finalize()
    calls_before = len(fake_llm.calls)
    revised = writer.revise(script, audit, plan, None)
    assert len(fake_llm.calls) == calls_before + 1
    assert revised.revision == 2
    for old, new in zip(script.segments, revised.segments, strict=True):
        if old.type == "quiz":
            assert [l.id for l in new.lines] != [l.id for l in old.lines]
            assert all(int(l.id[1:]) > int(script.next_line_id()[1:]) - 1 for l in new.lines)
        else:
            assert new is old
    assert len({l.id for l in revised.lines()}) == sum(len(s.lines) for s in revised.segments)
