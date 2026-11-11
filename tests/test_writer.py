from pipeline.models import AuditIssue, AuditResult, Guest, Misconception
from pipeline.plan.thorough import plan_chapter_thorough
from pipeline.script.cast import pick_guest
from pipeline.script.writer import Writer, build_blueprint, episode_minutes


def test_blueprint_shape(book, fake_llm, cast, settings):
    plan1 = plan_chapter_thorough(book, "ch01", fake_llm, cast)
    plan2 = plan_chapter_thorough(book, "ch02", fake_llm, cast)
    briefs = build_blueprint(plan2, cast, guest=None, previous_plan=plan1, next_chapter_title="X", target_minutes=25, chars_per_second=15)
    types = [b.type for b in briefs]
    assert types[0] == "cold_open" and types[1] == "recap" and types[-1] == "outro"
    assert "quiz" in types and "guest" not in types
    assert "reexplain" not in types  # nothing in this plan is harder than 3: no second route needed
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
    short = build_blueprint(plan2, cast, guest=guest, previous_plan=None, next_chapter_title=None, target_minutes=9, chars_per_second=15)
    assert "guest" not in [b.type for b in short]  # a six-minute guest in a nine-minute episode is padding
    assert len(next(b for b in short if b.type == "quiz").covers) == 2

    plan2.key_claims[0].difficulty = 5
    assert "reexplain" in [b.type for b in build_blueprint(plan2, cast, guest=None, previous_plan=None, next_chapter_title=None,
                                                          target_minutes=25, chars_per_second=15)]


def test_episode_length_follows_the_material(book, fake_llm, cast):
    plan = plan_chapter_thorough(book, "ch01", fake_llm, cast)
    easy = plan.model_copy(deep=True)
    for c in easy.key_claims:
        c.difficulty = 1
    hard = plan.model_copy(deep=True)
    hard.key_claims = [c.model_copy(update={"id": f"c{i}", "difficulty": 5}) for i, c in enumerate(plan.key_claims * 6, 1)]
    middle = plan.model_copy(deep=True)
    middle.key_claims = [c.model_copy(update={"id": f"c{i}"}) for i, c in enumerate(plan.key_claims * 3, 1)]
    assert episode_minutes(easy, 25) == 6  # the floor
    assert episode_minutes(hard, 25) == 25  # the ceiling: STUDIEPODCAST_TARGET_MINUTES
    assert 6 < episode_minutes(middle, 25) < 25


def test_misconceptions_go_to_the_block_they_are_about(book, fake_llm, cast):
    plan = plan_chapter_thorough(book, "ch01", fake_llm, cast)
    opener = max(plan.key_claims, key=lambda c: (c.exam_relevance, -c.difficulty))
    last = [c for c in plan.key_claims if c.id != opener.id][-1]  # not the cold open's own claim
    plan.misconceptions = [Misconception(wrong=f"Het is niet zo dat {last.claim.lower()}", right=last.claim, why_tempting="x")]
    briefs = build_blueprint(plan, cast, guest=None, previous_plan=None, next_chapter_title=None, target_minutes=20,
                             chars_per_second=15)
    with_it = [b for b in briefs if "Loop bewust in deze misvatting" in b.instructions]
    assert len(with_it) == 1 and last.id in with_it[0].covers


def test_write_produces_valid_script(book, fake_llm, cast, settings):
    plan = plan_chapter_thorough(book, "ch01", fake_llm, cast)
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
    plan = plan_chapter_thorough(book, "ch01", fake_llm, cast)
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


def test_the_writer_sees_people_and_how_they_sound(book, fake_llm, cast, settings):
    plan = plan_chapter_thorough(book, "ch01", fake_llm, cast)
    Writer(fake_llm, cast, settings).write(plan, book, None)
    system = "".join(b["text"] for b in next(c for c in fake_llm.calls if c.task == "script_segment").system)
    assert "Zo praat Tessa" in system and "Verhalen die Joris kan vertellen" in system and "### Samen" in system
    assert "pubquiz" in system  # the shared history
    assert "Zo klinken ze" in system and "Ik heb een systeem bedacht voor de roulette." in system


def test_every_block_hangs_on_a_human_moment(book, fake_llm, cast, settings):
    from pipeline.script.writer import human_moment

    tessa, joris = cast.by_role("explainer"), cast.by_role("skeptic")
    kinds = [human_moment(i, tessa, joris)[0] for i in range(1, 6)]
    assert kinds == ["anekdote", "meningsverschil", "plagen", "kwartje", "anekdote"]
    assert human_moment(1, tessa, joris)[1].startswith("Tessa vertelt")
    assert human_moment(5, tessa, joris)[1].startswith("Joris vertelt")  # the anecdotes take turns

    plan = plan_chapter_thorough(book, "ch01", fake_llm, cast)
    briefs = build_blueprint(plan, cast, guest=None, previous_plan=None, next_chapter_title=None, target_minutes=12,
                             chars_per_second=15)
    assert all("Menselijk moment in dit blok" in b.instructions for b in briefs if b.type == "body")
    script = Writer(fake_llm, cast, settings).write(plan, book, None)
    assert all(seg.beats for seg in script.segments)  # the writer's plan is kept on each segment


def test_the_table_read_keeps_every_fact_and_adds_conversation(book, fake_llm, cast, settings):
    plan = plan_chapter_thorough(book, "ch01", fake_llm, cast)
    writer = Writer(fake_llm, cast, settings)
    script = writer.write(plan, book, None)
    polished = writer.polish(script, None, plan, None)
    assert polished.revision == script.revision + 1
    facts = lambda s: [(l.speaker, l.text, l.covers) for l in s.lines() if l.covers]  # noqa: E731
    assert facts(polished) == facts(script)  # every claim line, its words and its ids, untouched by the fake edit
    assert len(list(polished.lines())) > len(list(script.lines()))  # a reaction under a line was added
    assert all(seg.beats for seg in polished.segments)
    edit_prompts = [c.user for c in fake_llm.calls if c.task == "script_polish"]
    assert len(edit_prompts) == len(script.segments) and all("Tafellezing" in u for u in edit_prompts)
    assert any("covers: c" in u for u in edit_prompts)  # the edit sees which lines carry which claims
