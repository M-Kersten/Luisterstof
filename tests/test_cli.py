"""CLI smoke tests, in-process via Typer's CliRunner."""

from typer.testing import CliRunner

from pipeline import cli

runner = CliRunner()

PASS_ENVS = ("STUDIEPODCAST_QUALITY_MODE", "STUDIEPODCAST_PLAN_REVIEW_ROUNDS",
             "STUDIEPODCAST_PLAN_PART_CHARS", "STUDIEPODCAST_REVISE_ROUNDS", "OPENCODE_TIMEOUT")


def test_inspect_lists_turns_with_wer_and_alignment_marks(tmp_path, sample_pdf, cast_dir):
    cli.state.clear()  # module-level cache from an earlier in-process invocation must not leak in
    data_dir = tmp_path / "data"
    common = ["--data-dir", str(data_dir), "--cast-dir", str(cast_dir), "--fake-llm", "--fake-audio"]

    result = runner.invoke(cli.app, [*common, "run", "demo", "--pdf", str(sample_pdf), "--upto", "render", "--chapters", "ch01"])
    assert result.exit_code == 0, result.output

    result = runner.invoke(cli.app, [*common, "inspect", "demo", "ch01"])
    assert result.exit_code == 0, result.output
    assert "beurten" in result.output and "verificatie" in result.output
    assert "wer=" in result.output

    result = runner.invoke(cli.app, [*common, "inspect", "demo", "ch01", "--only-suspect"])
    assert result.exit_code == 0, result.output


def test_inspect_without_a_render_yet_exits_with_a_clear_message(tmp_path, sample_pdf, cast_dir):
    cli.state.clear()
    data_dir = tmp_path / "data"
    common = ["--data-dir", str(data_dir), "--cast-dir", str(cast_dir), "--fake-llm", "--fake-audio"]

    result = runner.invoke(cli.app, [*common, "ingest", str(sample_pdf), "--book-id", "demo"])
    assert result.exit_code == 0, result.output

    result = runner.invoke(cli.app, [*common, "inspect", "demo", "ch01"])
    assert result.exit_code == 1
    assert "eerst renderen" in result.output


def test_quality_and_pass_flags_reach_the_settings(tmp_path, sample_pdf, cast_dir, monkeypatch):
    """--quality and the per-run pass flags must end up in the Settings the Pipeline runs on."""
    cli.state.clear()
    for name in PASS_ENVS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)  # so no stray .env of the developer joins the test
    common = ["--data-dir", str(tmp_path / "data"), "--cast-dir", str(cast_dir), "--fake-llm", "--fake-audio"]

    runner.invoke(cli.app, [*common, "--quality", "inspect", "demo", "ch01"])  # fails, the settings still landed
    settings = cli.state["settings"]
    assert settings.quality_mode is True
    assert (settings.plan_review_rounds, settings.plan_part_chars, settings.script_revise_rounds) == (3, 4000, 3)
    assert settings.opencode_timeout_s == 3600.0

    cli.state.clear()
    runner.invoke(cli.app, [*common, "run", "demo", "--revise-rounds", "0", "--reviews", "2", "--part-chars", "3000"])
    settings = cli.state["settings"]
    assert settings.quality_mode is False       # a one-off flag is not the mode
    assert settings.script_revise_rounds == 0   # and your number stands, below the quality default
    assert settings.plan_review_rounds == 2
    assert settings.plan_part_chars == 3000
