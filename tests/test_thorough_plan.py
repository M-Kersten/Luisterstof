"""Thorough content plan and lexicon: per section, quotes verified, merged, reviewed."""

import pytest
from typer.testing import CliRunner

from pipeline.config import Settings
from pipeline.fake_handlers import default_fake_llm
from pipeline.models import Chapter, Glossary, Section
from pipeline.plan.thorough import plan_chapter_thorough, propose_lexicon_thorough, section_parts


def _content_sections(chapter):
    return {s.id for s in chapter.sections if len(s.text.strip()) >= 400}


def test_thorough_plan_has_only_literal_quotes_and_covers_every_section(book):
    llm = default_fake_llm()
    chapter = book.episode_chapters()[0]
    plan = plan_chapter_thorough(book, chapter.id, llm, None)
    assert plan.key_claims and all(c.source_span.match_score == 100.0 for c in plan.key_claims)
    assert _content_sections(chapter) <= {c.source_span.section for c in plan.key_claims}
    tasks = [c.task for c in llm.calls]
    assert tasks.count("plan_section") == len([s for s in chapter.sections if s.text.strip()])
    assert "plan_quotes" in tasks  # the fake paraphrases one quote per section on purpose
    assert tasks[-2:] == ["plan_merge", "plan_review"]
    assert any("review 1" in w for w in plan.warnings)  # the review put back the candidate the merge dropped


def test_unfixable_quote_drops_the_claim_and_an_empty_section_gets_a_second_pass(book):
    llm = default_fake_llm()
    seen = {"sections": 0}
    original = llm.handlers["plan_section"]

    def first_try_empty(req):
        seen["sections"] += 1
        if seen["sections"] == 1:
            return {"claims": [], "definitions": [], "misconceptions": [], "worked_example": None, "formula_dense": False}
        return original(req)

    llm.on("plan_section", first_try_empty).on("plan_quotes", lambda req: {"fixes": []})
    chapter = book.episode_chapters()[0]
    plan = plan_chapter_thorough(book, chapter.id, llm, None, review_rounds=0)
    assert [c.task for c in llm.calls][:2] == ["plan_section", "plan_section"]  # the empty harvest was retried
    assert any("geschrapt" in w for w in plan.warnings)
    assert all(not c.claim.startswith("Volgens het boek") and c.source_span.match_score == 100.0 for c in plan.key_claims)


def test_merge_references_to_unknown_candidates_are_ignored(book):
    llm = default_fake_llm()
    merge = llm.handlers["plan_merge"]

    def with_ghost(req):
        out = merge(req)
        out["key_claims"].append({"candidate": "k999", "claim": "Verzonnen.", "difficulty": 5, "exam_relevance": 5})
        return out

    llm.on("plan_merge", with_ghost)
    plan = plan_chapter_thorough(book, book.episode_chapters()[0].id, llm, None, review_rounds=0)
    assert not any(c.claim == "Verzonnen." for c in plan.key_claims)
    assert any("k999" in w for w in plan.warnings)


def test_long_sections_are_read_in_parts_on_paragraph_boundaries():
    paragraphs = [f"Alinea {i}. " + "woord " * 300 for i in range(12)]
    section = Section(id="s1", title="Lang", text="\n\n".join(paragraphs))
    parts = section_parts(section, max_chars=5000)
    assert len(parts) > 1 and all(len(p) <= 5000 for p in parts)
    assert "\n\n".join(parts) == section.text  # nothing lost, nothing duplicated
    assert section_parts(Section(id="s2", title="Kort", text="Kort."), max_chars=5000) == ["Kort."]


def test_thorough_lexicon_asks_per_section_and_later_sections_see_earlier_entries():
    llm = default_fake_llm()
    chapter = Chapter(id="ch01", title="T", pages=(1, 2), sections=[
        Section(id="s1", title="A", text="Het CBS meldt dat P(A|B) groter is dan 1/2 volgens het CBS."),
        Section(id="s2", title="B", text="Ook hier noemt het CBS de breuk 1/2 opnieuw, naast NATO."),
    ])
    entries = propose_lexicon_thorough(chapter, Glossary(book_id="b"), llm)
    lexicon_calls = [c for c in llm.calls if c.task == "lexicon"]
    assert len(lexicon_calls) == 2
    assert "CBS" in lexicon_calls[1].user  # the second section is told CBS is already in the lexicon
    surfaces = [e.surface for e in entries]
    assert len(surfaces) == len({s.casefold() for s in surfaces})  # one entry per term


def test_plan_mode_defaults_follow_the_backend(monkeypatch):
    for key in ("STUDIEPODCAST_PLAN_MODE", "STUDIEPODCAST_OFFLINE", "STUDIEPODCAST_LLM_BACKEND"):
        monkeypatch.delenv(key, raising=False)
    assert Settings.from_env(dotenv=None).plan_mode == "single"
    monkeypatch.setenv("STUDIEPODCAST_LLM_BACKEND", "local")
    assert Settings.from_env(dotenv=None).plan_mode == "thorough"
    monkeypatch.setenv("STUDIEPODCAST_PLAN_MODE", "single")
    assert Settings.from_env(dotenv=None).plan_mode == "single"


def test_pipeline_uses_thorough_mode_and_compare_writes_both_plans(tmp_path, sample_pdf, cast_dir):
    from pipeline import cli

    cli.state.clear()
    common = ["--data-dir", str(tmp_path / "data"), "--cast-dir", str(cast_dir), "--fake-llm", "--fake-audio"]
    runner = CliRunner()
    assert runner.invoke(cli.app, [*common, "ingest", str(sample_pdf), "--book-id", "demo"]).exit_code == 0
    result = runner.invoke(cli.app, [*common, "plan", "demo", "ch01", "--compare"])
    assert result.exit_code == 0, result.output
    plans = tmp_path / "data" / "books" / "demo" / "plans"
    assert (plans / "ch01.plan.single.json").is_file() and (plans / "ch01.plan.thorough.json").is_file()
    assert (plans / "ch01.plan.json").is_file()
    report = (plans / "ch01.plan-compare.md").read_text(encoding="utf-8")
    assert "| | single | thorough |" in report and "quotes exact / fuzzy / not in source" in report


@pytest.mark.parametrize("mode", ["thorough"])
def test_run_chapter_with_thorough_plan_reaches_a_script(ingested, mode):
    p = ingested
    p.settings = p.settings.with_(plan_mode=mode)
    out = p.run_chapter("demo", "ch01", upto="script")
    assert out["plan"].key_claims and out["script"].segments
    assert any(c.task == "plan_section" for c in p.llm.calls)
