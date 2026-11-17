from pipeline.models import AuditIssue, AuditResult, Guest, Misconception
from pipeline.plan.thorough import plan_chapter_thorough
from pipeline.script.cast import pick_guest
from pipeline.script.writer import (
    HUMAN_MOMENT_SECONDS,
    SECONDS_PER_MISCONCEPTION,
    Writer,
    build_blueprint,
    claim_seconds,
    episode_minutes,
    plan_shape,
    source_passages,
)


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
    assert briefs[0].covers == []  # the opening sets the scene and asks the question; it carries no claim

    plan2.needs_expert, plan2.expert_domain = True, "statistiek"
    guest = pick_guest(cast, plan2.expert_domain)
    assert isinstance(guest, Guest) and guest.id == "hanna_vos"
    assert pick_guest(cast, "kunstgeschiedenis").id == "sam_de_wit"
    assert pick_guest(cast, "kunstgeschiedenis", generic=False) is None  # the writer takes no stand-in expert
    with_guest = build_blueprint(plan2, cast, guest=guest, previous_plan=None, next_chapter_title=None, target_minutes=25, chars_per_second=15)
    assert "guest" in [b.type for b in with_guest] and "recap" not in [b.type for b in with_guest]
    short = build_blueprint(plan2, cast, guest=guest, previous_plan=None, next_chapter_title=None, target_minutes=9, chars_per_second=15)
    assert "guest" not in [b.type for b in short]  # a guest in a nine-minute episode is padding
    assert len(next(b for b in short if b.type == "quiz").covers) == 2

    plan2.key_claims[0].difficulty = 5
    assert "reexplain" in [b.type for b in build_blueprint(plan2, cast, guest=None, previous_plan=None, next_chapter_title=None,
                                                          target_minutes=25, chars_per_second=15)]


def test_a_claim_has_one_home(book, fake_llm, cast):
    """What the guest tells, the hosts do not tell first: a claim told twice is heard twice in the same words."""
    plan = plan_chapter_thorough(book, "ch02", fake_llm, cast)
    plan.needs_expert, plan.expert_domain = True, "statistiek"
    guest = pick_guest(cast, plan.expert_domain, generic=False)
    briefs = build_blueprint(plan, cast, guest=guest, previous_plan=None, next_chapter_title=None, target_minutes=25,
                             chars_per_second=15)
    told = [c for b in briefs if b.type in ("body", "guest") for c in b.covers]
    assert sorted(told) == sorted(c.id for c in plan.key_claims)  # every claim once, none twice
    guest_brief = next(b for b in briefs if b.type == "guest")
    assert guest_brief.covers and "nog niet verteld" in guest_brief.instructions


def test_episode_length_follows_the_material(book, fake_llm, cast):
    plan = plan_chapter_thorough(book, "ch01", fake_llm, cast)
    easy = plan.model_copy(deep=True)
    easy.key_claims = [c.model_copy(update={"difficulty": 1}) for c in plan.key_claims[:3]]
    hard = plan.model_copy(deep=True)
    hard.key_claims = [c.model_copy(update={"id": f"c{i}", "difficulty": 5}) for i, c in enumerate(plan.key_claims * 6, 1)]
    middle = plan.model_copy(deep=True)
    middle.key_claims = [c.model_copy(update={"id": f"c{i}"}) for i, c in enumerate(plan.key_claims * 3, 1)]
    assert episode_minutes(easy, 25) == 6  # the floor
    assert episode_minutes(hard, 25) == 25  # the ceiling: STUDIEPODCAST_TARGET_MINUTES
    assert 6 < episode_minutes(middle, 25) < 25


def test_the_claims_get_their_time_before_the_extras(hoogtevrees, cast):
    """The first draft of this chapter gave its 25 claims 570 of the 1215 seconds they were sized for: the
    guest and the second route came out of the same total. Now the body is served first."""
    plan = hoogtevrees.plan
    guest = pick_guest(cast, plan.expert_domain, generic=False)
    assert guest is None  # nobody in the cast has this field, and a stand-in is not a guest
    minutes = episode_minutes(plan, 25)
    shape = plan_shape(plan, guest=None, target_minutes=minutes, moments=2)
    need = claim_seconds(plan.key_claims) + SECONDS_PER_MISCONCEPTION * len(plan.misconceptions) + 2 * HUMAN_MOMENT_SECONDS
    assert shape.seconds["body"] >= need
    assert shape.reexplain is not None and shape.reexplain.id == "c21"  # there is room for the second route

    squeezed = plan_shape(plan, guest=None, target_minutes=18, moments=2)
    assert squeezed.reexplain is None  # no room: the extra goes, the claims keep what is left
    hanna = pick_guest(cast, "statistiek")
    assert plan_shape(plan, guest=hanna, target_minutes=18, moments=2).guest_claims == []

    briefs = build_blueprint(plan, cast, guest=None, previous_plan=None, next_chapter_title="Het evenwicht en de ogen",
                             target_minutes=minutes, chars_per_second=15, book=hoogtevrees.book)
    assert [b.type for b in briefs].count("guest") == 0
    body = [b for b in briefs if b.type == "body"]
    assert sum(b.target_chars for b in body) > 14000  # the five blocks had 7695 between them
    quiz = next(b for b in briefs if b.type == "quiz")
    assert "c21" in quiz.covers  # the quiz asks about the hard claim, where it used to ask the four easiest


def test_a_block_reads_the_book_around_its_claims(hoogtevrees, cast):
    """One source sentence per claim is a sentence to read out. The paragraph around it has who did the
    study and why the outcome is what it is."""
    plan, book = hoogtevrees.plan, hoogtevrees.book
    c21 = plan.claim("c21")
    rows = source_passages(book, [c21])
    assert len(rows) == 1 and "Dunedin-studie" in rows[0] and "Nieuw-Zeeland" in rows[0]  # who and where
    assert "niet-associatief model" in rows[0]  # and the why that the single quote lacks
    joined = source_passages(book, [plan.claim("c21"), plan.claim("c22"), plan.claim("c23")])
    assert len(joined) == 1 and joined[0].startswith("  bij c21, c22, c23:")  # one experiment, told once
    assert source_passages(None, [c21]) == [f'  bij c21: "{c21.source_span.quote}"']  # no book: the bare quote

    briefs = build_blueprint(plan, cast, guest=None, previous_plan=None, next_chapter_title=None, target_minutes=25,
                             chars_per_second=15, book=book)
    again = next(b for b in briefs if b.type == "reexplain")
    assert "het waarom" in again.instructions and "niet-associatief model" in again.instructions
    assert "Volgens het niet-associatieve model" in again.instructions  # the neighbouring claims that hold the why


def test_misconceptions_go_to_the_block_they_are_about(book, fake_llm, cast):
    plan = plan_chapter_thorough(book, "ch01", fake_llm, cast)
    last = plan.key_claims[-1]
    plan.misconceptions = [Misconception(wrong=f"Het is niet zo dat {last.claim.lower()}", right=last.claim, why_tempting="x")]
    briefs = build_blueprint(plan, cast, guest=None, previous_plan=None, next_chapter_title=None, target_minutes=20,
                             chars_per_second=15)
    with_it = [b for b in briefs if "Misvatting in dit blok" in b.instructions]
    assert len(with_it) == 1 and last.id in with_it[0].covers
    # the host who has not read the chapter is the one who believes it; the one who has, puts it right
    assert "Joris denkt hardop" in with_it[0].instructions and "Tessa trapt er zelf niet in" in with_it[0].instructions


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


def test_style_notes_ride_along_with_a_block_that_is_rewritten_anyway(book, fake_llm, cast, settings):
    """A rewritten block is new text that the next check can find new faults in. So with something really wrong
    in the quiz, only the quiz is rewritten; the note that a body block sounds stiff, and the one about the whole
    episode, go along with it. Only a script with nothing wrong in it is rewritten for how it sounds."""
    plan = plan_chapter_thorough(book, "ch01", fake_llm, cast)
    writer = Writer(fake_llm, cast, settings)
    script = writer.write(plan, book, None)
    quiz = next(s for s in script.segments if s.type == "quiz")
    body = next(s for s in script.segments if s.type == "body")
    stiff = AuditIssue(check="lint", severity="warning", rule="stiff", line_id=body.lines[0].id, message="klinkt als een boek")
    flat = AuditIssue(check="lint", severity="warning", rule="too_few_overlaps", message="niemand valt elkaar in de rede")
    wrong = AuditIssue(check="lint", rule="banned_phrase", line_id=quiz.lines[0].id, message="verboden zin")

    calls_before = len(fake_llm.calls)
    revised = writer.revise(script, AuditResult(episode_id="ch01", issues=[wrong, stiff, flat]).finalize(), plan, None,
                            also_naturalness=True)
    rewrites = fake_llm.calls[calls_before:]
    assert len(rewrites) == 1 and "verboden zin" in rewrites[0].user and "niemand valt elkaar" in rewrites[0].user
    assert next(s for s in revised.segments if s.type == "body") is body

    calls_before = len(fake_llm.calls)
    writer.revise(script, AuditResult(episode_id="ch01", issues=[stiff, flat]).finalize(), plan, None, also_naturalness=True)
    rewrites = fake_llm.calls[calls_before:]
    assert any("klinkt als een boek" in c.user for c in rewrites)  # nothing wrong: then it is rewritten for its sound
    assert all("niemand valt elkaar" in c.user for c in rewrites if "klinkt als een boek" not in c.user)


def test_the_writer_sees_how_they_sound_and_their_lives_only_where_a_story_is_told(book, fake_llm, cast, settings):
    plan = plan_chapter_thorough(book, "ch01", fake_llm, cast)
    Writer(fake_llm, cast, settings).write(plan, book, None)
    calls = [c for c in fake_llm.calls if c.task == "script_segment"]
    system = "".join(b["text"] for b in calls[0].system)
    assert "Zo praat Tessa" in system and "Wat Joris wil" in system
    assert "Zo klinken ze" in system and "Ik heb een systeem bedacht voor de roulette." in system
    # courgettes, the stew and the pub quiz in every call end up in every segment, set up or not
    assert "Verhalen die Joris kan vertellen" not in system and "### Samen" not in system
    assert "stoofpot" not in system.split("## Zo klinken ze")[0] and "pubquiz" not in system
    with_story = [c.user for c in calls if "Verhalen van Tessa" in c.user]
    assert len(with_story) == 1 and "zak drop" in with_story[0]  # the one block the outline gave a story


def test_the_outline_gives_the_episode_a_thread(book, fake_llm, cast, settings):
    plan = plan_chapter_thorough(book, "ch01", fake_llm, cast)
    script = Writer(fake_llm, cast, settings).write(plan, book, None, next_chapter_title="Verdelingen")
    outline = script.outline
    assert outline is not None and outline.planned and outline.question == "Waar gaat dit hoofdstuk eigenlijk over?"
    told = [cid for b in outline.blocks for cid in b.claims]
    assert sorted(told) == sorted(c.id for c in plan.key_claims)  # the claim the model forgot was put back
    assert outline.blocks[-1].bridge == "" and sum(1 for b in outline.blocks if b.moment) <= 2
    assert all(seg.beats for seg in script.segments)  # the writer's own plan is kept on each segment

    calls = [c for c in fake_llm.calls if c.task == "script_segment"]
    system = "".join(b["text"] for b in calls[0].system)
    assert "Hoofdvraag van de aflevering: Waar gaat dit hoofdstuk eigenlijk over?" in system  # every call sees the thread
    opening, first, second, outro = calls[0].user, calls[1].user, calls[2].user, calls[-1].user
    assert "Begin daarom niet midden in een discussie" in opening and outline.opening in opening
    assert f'"{outline.question}"' in opening and "'covers' blijft leeg" in opening
    assert f'Dit blok beantwoordt één vraag: "{outline.blocks[0].question}"' in first
    assert outline.blocks[0].bridge in first and outline.blocks[0].bridge in second  # one block hands over to the next
    assert "Menselijk moment in dit blok" in first and "Menselijk moment in dit blok" not in calls[4].user
    assert outline.question in outro and outline.takeaway in outro and "Verdelingen" in outro


def test_an_outline_is_repaired_or_replaced(book, fake_llm, cast):
    from pipeline.models import OutlineBlock
    from pipeline.script.outline import OutlineOut, fallback_outline, plan_outline, repair_outline

    plan = plan_chapter_thorough(book, "ch01", fake_llm, cast)
    claims = list(plan.key_claims)
    ids = [c.id for c in claims]
    messy = OutlineOut.model_validate({"question": "Wat is een kans?", "opening": "", "takeaway": "", "blocks": [
        {"question": "a", "claims": [ids[2], "c99", ids[0]], "bridge": "en dan?", "moment": "anekdote", "moment_note": ""},
        {"question": "b", "claims": [ids[0], f"[{ids[3]}]"], "bridge": "x", "moment": "plagen", "moment_note": ""},
        {"question": "c", "claims": [], "bridge": "", "moment": "kwartje", "moment_note": ""},
        {"question": "d", "claims": ids[4:5], "bridge": "blijft hangen", "moment": "kwartje", "moment_note": ""}]})
    fixed = repair_outline(messy, plan, claims, 3)
    told = [cid for b in fixed.blocks for cid in b.claims]
    assert sorted(told) == sorted(ids) and "c99" not in told  # every claim once, the invented id gone
    assert fixed.blocks[0].claims[:3] == [ids[2], ids[0], ids[1]]  # the forgotten c2 lands after its neighbour c1
    assert len(fixed.blocks) == 3 and fixed.blocks[-1].bridge == ""  # the empty block is gone, the last one leads nowhere
    assert [b.moment for b in fixed.blocks] == ["anekdote", "plagen", ""]  # no more than two moments

    plain = fallback_outline(plan, claims, 3)
    assert [cid for b in plain.blocks for cid in b.claims] == ids and not plain.planned
    fake_llm.on("script_outline", lambda req: {"question": "", "opening": "", "takeaway": "", "blocks": []})
    assert plan_outline(plan, claims, cast, fake_llm, n_blocks=3).planned is False  # nothing usable: book order
    del fake_llm.handlers["script_outline"]
    assert plan_outline(plan, claims, cast, fake_llm, n_blocks=3).planned is False  # no answer at all: the same

    # an outline made for other claims still gives blocks over exactly the claims of this episode
    briefs = build_blueprint(plan, cast, guest=None, previous_plan=None, next_chapter_title=None, target_minutes=20,
                             chars_per_second=15,
                             outline=fixed.model_copy(update={"blocks": [OutlineBlock(claims=["c1", "zz"])]}))
    assert sorted(c for b in briefs if b.type == "body" for c in b.covers) == sorted(ids)


def test_a_human_moment_only_where_the_outline_puts_one(cast):
    from pipeline.script.writer import human_moment

    tessa, joris = cast.by_role("explainer"), cast.by_role("skeptic")
    assert human_moment("", tessa, joris) == ""
    assert human_moment("anekdote", tessa, joris).startswith("Tessa vertelt")
    told_by_joris = human_moment("anekdote", tessa, joris, teller=joris, note="Joris over de grafiek bij de bank.")
    assert told_by_joris.startswith("Joris vertelt") and "Tessa reageert" in told_by_joris
    assert told_by_joris.endswith("Uit het draaiboek: Joris over de grafiek bij de bank.")
    assert "eerder in deze aflevering" in human_moment("plagen", tessa, joris)  # no teasing with what was never told


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
