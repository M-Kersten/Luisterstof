"""CLI smoke tests, in-process via Typer's CliRunner."""

from typer.testing import CliRunner

from pipeline import cli

runner = CliRunner()


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
