"""Quality over speed: what the knob raises, what you can still pin yourself, and what the
writer does with the extra rounds."""

from __future__ import annotations

import pytest

from pipeline.config import QUALITY_PASSES, Settings
from pipeline.models import AuditIssue, AuditResult
from pipeline.plan.content_plan import plan_chapter
from pipeline.runner import Pipeline
from pipeline.script.writer import FLAT_SEGMENT_TARGETS, Writer, naturalness_issues

PASS_ENVS = ("STUDIEPODCAST_QUALITY_MODE", "STUDIEPODCAST_PLAN_REVIEW_ROUNDS",
             "STUDIEPODCAST_PLAN_PART_CHARS", "STUDIEPODCAST_REVISE_ROUNDS", "OPENCODE_TIMEOUT")


@pytest.fixture
def clean_env(monkeypatch):
    for name in PASS_ENVS:
        monkeypatch.delenv(name, raising=False)


def test_quality_mode_raises_the_writing_passes(clean_env):
    plain = Settings.from_env(dotenv=None)
    assert not plain.quality_mode
    assert (plain.plan_review_rounds, plain.plan_part_chars, plain.script_revise_rounds) == (1, 8000, 1)

    thorough = Settings.from_env(dotenv=None, quality_mode=True)
    assert thorough.quality_mode
    assert thorough.plan_review_rounds == QUALITY_PASSES["plan_review_rounds"][0] == 3
    assert thorough.plan_part_chars == QUALITY_PASSES["plan_part_chars"][0] == 4000  # smaller pieces, read better
    assert thorough.script_revise_rounds == QUALITY_PASSES["script_revise_rounds"][0] == 3
    assert thorough.opencode_timeout_s == QUALITY_PASSES["opencode_timeout_s"][0] == 3600.0


def test_env_wins_over_quality_mode(clean_env, monkeypatch):
    """The knob raises what you left alone; a value you wrote yourself always stands."""
    monkeypatch.setenv("STUDIEPODCAST_QUALITY_MODE", "1")
    monkeypatch.setenv("STUDIEPODCAST_REVISE_ROUNDS", "0")      # "no rewrite rounds today"
    monkeypatch.setenv("STUDIEPODCAST_PLAN_REVIEW_ROUNDS", "5")  # even above the quality default
    monkeypatch.setenv("OPENCODE_TIMEOUT", "900")
    s = Settings.from_env(dotenv=None)
    assert s.quality_mode
    assert s.script_revise_rounds == 0
    assert s.plan_review_rounds == 5
    assert s.opencode_timeout_s == 900.0


def test_with_quality_only_moves_towards_thorough():
    s = Settings().with_quality()
    assert s.quality_mode and s.plan_review_rounds == 3 and s.plan_part_chars == 4000
    # already more thorough than the mode asks for: nothing is ever relaxed
    thorough = Settings(plan_review_rounds=5, plan_part_chars=2000, script_revise_rounds=7,
                        opencode_timeout_s=7200.0).with_quality()
    assert (thorough.plan_review_rounds, thorough.plan_part_chars) == (5, 2000)
    assert (thorough.script_revise_rounds, thorough.opencode_timeout_s) == (7, 7200.0)
    assert thorough.with_quality(False).quality_mode is False


# --------------------------------------------------------------------------- writer side

def _warning_audit(*rules: str) -> AuditResult:
    return AuditResult(episode_id="ch01", issues=[
        AuditIssue(check="lint", severity="warning", rule=r, message=f"{r} voor de test") for r in rules
    ]).finalize()


def test_naturalness_issues_only_the_shape_rules():
    audit = _warning_audit("too_few_overlaps", "flat_emotional_reaction", "duration", "quiz_questions")
    assert [i.rule for i in naturalness_issues(audit)] == ["too_few_overlaps", "flat_emotional_reaction"]


def test_flat_script_without_quality_mode_is_left_alone(book, fake_llm, cast, settings):
    plan = plan_chapter(book, "ch01", fake_llm, cast)
    writer = Writer(fake_llm, cast, settings)
    script = writer.write(plan, book, None)
    audit = _warning_audit("too_few_overlaps")
    calls_before = len(fake_llm.calls)
    assert writer.revise(script, audit, plan, None) is script  # warnings alone never cost a call
    assert len(fake_llm.calls) == calls_before


def test_quality_mode_hands_flat_segments_to_the_writer(book, fake_llm, cast, settings):
    plan = plan_chapter(book, "ch01", fake_llm, cast)
    writer = Writer(fake_llm, cast, settings.with_quality())
    script = writer.write(plan, book, None)
    audit = _warning_audit("too_few_overlaps")  # episode-wide: nobody owns it, the flattest segments do
    flat = writer._flat_segments(script)
    assert 0 < len(flat) <= FLAT_SEGMENT_TARGETS
    assert all(script.segments[i].type == "body" for i in flat)

    calls_before = len(fake_llm.calls)
    revised = writer.revise(script, audit, plan, None, also_naturalness=True)
    assert len(fake_llm.calls) == calls_before + len(flat)
    assert revised.revision == script.revision + 1
    for i, (old, new) in enumerate(zip(script.segments, revised.segments, strict=True)):
        assert (new is not old) == (i in flat)  # only the flat ones were touched


def test_needs_rewrite_chases_flat_scripts_only_in_quality_mode(settings, fake_llm):
    flat = _warning_audit("too_few_overlaps")
    assert flat.passed  # the audit is happy, the dialogue still reads as two monologues
    in_a_hurry = Pipeline(settings, llm=fake_llm, fake_audio=True)
    unhurried = Pipeline(settings.with_quality(), llm=fake_llm, fake_audio=True)
    assert not in_a_hurry._needs_rewrite(flat)
    assert unhurried._needs_rewrite(flat)

    broken = AuditResult(episode_id="ch01", issues=[
        AuditIssue(check="lint", rule="banned_phrase", message="x")]).finalize()
    assert in_a_hurry._needs_rewrite(broken)  # blocking work is never skipped
