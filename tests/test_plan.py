from pipeline.plan.content_plan import ClaimOut, PlanOut, build_plan, plan_chapter, resolve_span, span_text


def test_plan_resolves_spans_and_marks_sections(book, fake_llm, cast):
    plan = plan_chapter(book, "ch01", fake_llm, cast)
    assert plan.chapter_id == "ch01" and len(plan.key_claims) >= 4
    assert all(c.source_span.match_score == 100.0 for c in plan.key_claims)
    assert not plan.warnings
    for c in plan.key_claims:
        assert span_text(book, c.source_span) == c.source_span.quote
    assert plan.claims_with_relevance(3)
    assert plan.hardest_claim() is not None
    assert plan.section_difficulty


def test_fuzzy_quote_and_fallback(book):
    section = book.section("ch01.1")
    hit = resolve_span(section.text, "kans is een getal tusen 0 en 1 dat uitdrukt")  # typo
    assert hit is not None and hit[2] >= 80
    assert resolve_span(section.text, "dit staat nergens in de tekst van het boek zeker niet") is None


def test_expert_rule_and_unresolved_quote(book):
    chapter = book.chapter("ch02")
    out = PlanOut(
        summary="s", learning_objectives=["a"],
        key_claims=[
            ClaimOut(claim="x", section="ch02.1", source_quote="De binomiale verdeling beschrijft het aantal successen", difficulty=5, exam_relevance=5),
            ClaimOut(claim="y", section="ch02.1", source_quote="niet aanwezig in deze tekst helemaal nergens", difficulty=5, exam_relevance=9),
            ClaimOut(claim="z", section="ch02.9", source_quote="Een z-score geeft aan hoeveel standaardafwijkingen", difficulty=1, exam_relevance=0),
        ],
        definitions=[], misconceptions=[], worked_example=None, expert_domain=None, expert_reason=None,
        formula_dense_sections=["ch02.2", "nope"],
    )
    plan = build_plan(chapter, out)
    assert plan.needs_expert and plan.expert_domain == chapter.title
    assert plan.key_claims[1].exam_relevance == 5 and plan.key_claims[2].exam_relevance == 1
    assert plan.key_claims[1].source_span.match_score == 0.0 and plan.key_claims[1].source_span.end == len(chapter.section("ch02.1").text)
    assert plan.key_claims[2].source_span.section == "ch02.2"  # quote found in another section
    assert plan.formula_dense_sections == ["ch02.2"]
    assert len(plan.warnings) == 2
