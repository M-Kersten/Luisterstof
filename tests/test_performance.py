"""Performance prototype: labels, phrase re-timing, timing gaps, reactions, shared acoustics, prototype command."""

import numpy as np
import pytest
from typer.testing import CliRunner

from pipeline.audio.asr import UniformAligner, WordTiming
from pipeline.audio.performance_render import PerformanceOptions, PhraseSeg, Tables, edit_phrases, split_hook
from pipeline.audio.reactions import ReactionBank, speakable_text
from pipeline.audio.render import render_episode
from pipeline.audio.synth import AudioClip, NullSynth
from pipeline.audio.timeline import assemble
from pipeline.audio.turns import group_turns
from pipeline.fake_handlers import default_fake_llm
from pipeline.llm import FakeLLM
from pipeline.models import ContentPlan, Host, Line, Script, Segment
from pipeline.paths import BookPaths
from pipeline.performance import merged_table, timing_gap
from pipeline.script.writer import SegmentBrief, Writer


class RealTimingAligner(UniformAligner):
    """Uniform spacing, but reported as real alignment so phrasing is allowed to cut."""

    last_used_fallback = False


def _scene(lines):
    return Script(episode_id="ch01", segments=[Segment(type="body", lines=lines)])


def test_old_lines_load_without_performance_fields_and_bad_phrases_are_dropped():
    line = Line.model_validate({"id": "l001", "speaker": "tessa", "text": "Gewoon een regel."})
    assert line.delivery is None and line.mood is None and line.timing is None and line.phrases == [] and line.reaction is None
    ok = Line(id="l002", speaker="tessa", text="Ik snap het... maar wacht even—",
              phrases=[{"text": "Ik snap het..."}, {"text": "maar wacht even—", "delivery": "interrupt"}])
    assert len(ok.phrases) == 2
    bad = Line(id="l003", speaker="tessa", text="Ik snap het.", phrases=[{"text": "iets heel anders"}])
    assert bad.phrases == []


def test_host_table_overrides_merge_over_defaults():
    host = Host(id="tessa", name="Tessa", role="explainer",
                performance={"delivery": {"excite": {"rate": 1.2}}, "mood": {"amused": 0.3}})
    table = merged_table(host)
    assert table["delivery"]["excite"] == {"rate": 1.2, "exaggeration": 0.15}  # rate overridden, exaggeration kept
    assert table["delivery"]["think"]["rate"] == 0.88 and table["mood"]["amused"] == 0.3
    line = Line(id="l001", speaker="tessa", text="x", timing="hesitate")
    gaps = {timing_gap(table, line, seed) for seed in range(20)}
    assert all(0.35 <= g <= 0.6 for g in gaps) and len(gaps) > 1  # label picks the range, seed the spot
    assert timing_gap(table, line, 3) == timing_gap(table, line, 3)  # reproducible


def test_writer_performance_mode_keeps_known_labels_only(cast, settings):
    def handler(req):
        return {"lines": [
            {"speaker": "tessa", "text": "Dit is uitleg.", "tags": [], "covers": [], "overlap": "none", "pause_after_ms": 0,
             "delivery": "EXPLAIN", "mood": "furious", "timing": "whenever", "phrases": [], "reaction": ""},
            {"speaker": "joris", "text": "(lacht)", "tags": [], "covers": [], "overlap": "backchannel", "pause_after_ms": 0,
             "delivery": "", "mood": "", "timing": "", "phrases": [], "reaction": "laugh"},
        ]}

    writer = Writer(FakeLLM({"script_scene": handler}), cast, settings, performance=True)
    brief = SegmentBrief(type="body", title="t", covers=[], target_chars=500, instructions="x", speakers=["tessa", "joris"])
    lines = list(writer.write_scene(ContentPlan(chapter_id="ch01"), None, brief).lines())
    assert (lines[0].delivery, lines[0].mood, lines[0].timing) == ("explain", None, None)
    assert lines[1].reaction == "laugh" and lines[1].overlap.mode == "backchannel"
    assert speakable_text(lines[1]) == "Haha."


def test_normal_writer_prompt_has_no_performance_section(cast, settings):
    from pipeline.script.writer import PERFORMANCE_RULES

    plain = Writer(default_fake_llm(), cast, settings)._system(ContentPlan(chapter_id="c"), None, None, [])
    assert PERFORMANCE_RULES.strip() not in plain[0]["text"]


def _clip_with_words(words, seconds=3.0, sr=24000):
    samples = (0.2 * np.sin(2 * np.pi * 220 * np.arange(int(seconds * sr)) / sr)).astype(np.float32)
    return AudioClip(samples, sr), [WordTiming(w, i * 0.5, i * 0.5 + 0.4) for i, w in enumerate(words)]


def test_edit_phrases_inserts_pauses_and_retimes_words():
    clip, words = _clip_with_words(["een", "twee", "drie", "vier"])
    segs = [PhraseSeg(2, 1.0, 0.3, "explain"), PhraseSeg(2, 1.0, 0.0, "excite")]
    edited, new_words, applied = edit_phrases(clip, words, segs)
    assert edited.duration_s == pytest.approx(clip.duration_s + 0.3, abs=0.01)
    assert new_words[2].start == pytest.approx(words[2].start + 0.3, abs=0.01)  # after the pause, shifted by it
    assert [w.start for w in new_words] == sorted(w.start for w in new_words) and applied == [1.0, 1.0]
    with pytest.raises(ValueError):
        edit_phrases(clip, words, [PhraseSeg(3, 1.0, 0.0, None)])


def test_edit_phrases_stretches_per_phrase():
    pytest.importorskip("librosa")
    clip, words = _clip_with_words(["een", "twee", "drie", "vier"], seconds=2.0)
    edited, new_words, applied = edit_phrases(clip, words, [PhraseSeg(2, 0.5, 0.0, "think"), PhraseSeg(2, 1.0, 0.0, None)])
    assert applied == [0.5, 1.0] and edited.duration_s > clip.duration_s + 0.7  # the first half got twice as long
    assert new_words[1].end > words[1].end


def test_split_hook_isolates_reactions_and_big_mood_jumps(cast):
    tables = Tables(cast)
    lines = [Line(id="l001", speaker="tessa", text="Rustig uitgelegd.", delivery="think", mood="thoughtful"),
             Line(id="l002", speaker="tessa", text="En dan ineens dit!", delivery="excite", mood="surprised"),
             Line(id="l003", speaker="tessa", text="Nog een zin.", delivery="excite", mood="surprised")]
    assert len(group_turns(_scene(lines))) == 1
    split = group_turns(_scene(lines), split_between=split_hook(tables, PerformanceOptions(delivery=True)))
    assert [t.line_ids for t in split] == [["l001"], ["l002", "l003"]]


def test_timing_label_sets_the_gap_and_counts_as_written(cast):
    lines = [Line(id="l001", speaker="tessa", text="Wat denk jij?"),
             Line(id="l002", speaker="joris", text="Even denken.", timing="deliberate")]
    turns = group_turns(_scene(lines))
    synth = NullSynth(jitter=0.0)
    rendered = {t.turn_id: (c := synth.render(t.text, _voice(t.speaker), exaggeration=0.5, seed=1), UniformAligner().align(c, t.text))
                for t in turns}
    tl = assemble(turns, rendered, sr=24000, timing_gap=lambda turn: 1.25 if turn.first.timing else None)
    a, b = tl.placements
    assert b.start - a.end == pytest.approx(1.25)
    assert not tl.qa  # a labelled beat is not reported as an unwritten gap
    plain = assemble(turns, rendered, sr=24000)
    assert plain.placements[1].start - plain.placements[0].end < 0.42  # without the hook: today's rules


def _voice(speaker):
    from pipeline.audio.synth import VoiceSpec

    return VoiceSpec(speaker_id=speaker)


def test_render_phrasing_applies_on_real_timing_and_skips_on_estimated(settings, cast):
    line = Line(id="l001", speaker="tessa", text="Oh. Dus zo werkt het eigenlijk.", delivery="realize",
                phrases=[{"text": "Oh.", "delivery": "realize", "pause_after": "beat"},
                         {"text": "Dus zo werkt het eigenlijk.", "delivery": "realize"}])
    paths = BookPaths(settings.data_dir, "demo").ensure()
    opts = PerformanceOptions(delivery=True, phrasing=True)
    _, _, estimated = render_episode(_scene([line]), None, cast, settings, paths, tier="draft", synth=NullSynth(jitter=0.0),
                                     transcriber=None, aligner=None, performance=opts)
    assert str(estimated.turns[0].performance["phrasing"]).startswith("skipped: estimated")
    other = _scene([line]).model_copy(update={"episode_id": "ch02"})
    _, _, real = render_episode(other, None, cast, settings, paths, tier="draft", synth=NullSynth(jitter=0.0),
                                transcriber=None, aligner=RealTimingAligner(), performance=opts)
    phrasing = real.turns[0].performance["phrasing"]
    assert isinstance(phrasing, list) and len(phrasing) == 2 and 0.35 <= phrasing[0]["pause_after_s"] <= 0.55


def test_reaction_bank_clip_replaces_synthesis(settings, cast):
    folder = settings.cast_dir / "reactions" / "joris" / "laugh"
    folder.mkdir(parents=True)
    for i in range(3):
        AudioClip(np.full(4800, 0.1 * (i + 1), dtype=np.float32), 24000).write(folder / f"lach{i}.wav")
    bank = ReactionBank(settings.cast_dir)
    assert bank.pick("joris", "laugh", 7, "l002") == bank.pick("joris", "laugh", 7, "l002")
    assert bank.pick("tessa", "laugh", 7, "l002") is None
    lines = [Line(id="l001", speaker="tessa", text="En toen viel hij van zijn stoel."),
             Line(id="l002", speaker="joris", text="Haha.", reaction="laugh", overlap={"mode": "backchannel", "target": "l001"})]
    paths = BookPaths(settings.data_dir, "demo").ensure()
    _, _, manifest = render_episode(_scene(lines), None, cast, settings, paths, tier="draft", synth=NullSynth(),
                                    transcriber=None, aligner=None, performance=PerformanceOptions(reactions=True))
    laugh = manifest.turns[1]
    assert laugh.performance == {"reaction": "laugh", "source": "bank"} and laugh.chosen_take.reason == "reaction bank"
    assert manifest.verification_coverage()[1] == 1  # the bank clip is not counted as an unverified take


def test_shared_chain_pulls_two_differently_coloured_voices_closer(cast):
    pytest.importorskip("pedalboard")
    from pipeline.audio.acoustics import Acoustics, AcousticsConfig, band_levels, shape

    sr, rng = 24000, np.random.default_rng(1)

    def voice(tilt):
        x = rng.standard_normal(sr * 4)
        spectrum = np.fft.rfft(x) * (np.maximum(np.fft.rfftfreq(len(x), 1 / sr), 50) / 1000) ** tilt
        y = np.fft.irfft(spectrum, len(x)).astype(np.float32)
        return y / np.abs(y).max() * 0.3

    dark, bright = voice(-0.6), voice(0.3)
    acoustics = Acoustics.fit({"tessa": [dark], "joris": [bright]}, sr, cast, AcousticsConfig())
    before = np.abs(shape(band_levels(dark, sr)) - shape(band_levels(bright, sr))).mean()
    after = np.abs(shape(band_levels(acoustics.process_speaker("tessa", dark, sr), sr))
                   - shape(band_levels(acoustics.process_speaker("joris", bright, sr), sr))).mean()
    assert after < before * 0.7  # closer, not identical: timbres stay their own
    assert after > 0.5
    bus = acoustics.process_bus(dark, sr)
    assert bus.shape == dark.shape and np.isfinite(bus).all()


def test_prototype_scene_command_writes_every_variant_and_a_report(tmp_path, sample_pdf, cast_dir):
    from pipeline import cli

    cli.state.clear()
    common = ["--data-dir", str(tmp_path / "data"), "--cast-dir", str(cast_dir), "--fake-llm", "--fake-audio"]
    runner = CliRunner()
    result = runner.invoke(cli.app, [*common, "run", "demo", "--pdf", str(sample_pdf), "--upto", "glossary", "--chapters", "ch01"])
    assert result.exit_code == 0, result.output
    result = runner.invoke(cli.app, [*common, "prototype-scene", "demo", "ch01"])
    assert result.exit_code == 0, result.output
    out = tmp_path / "data" / "books" / "demo" / "prototype" / "ch01"
    for name in ("current", "timing", "phrasing", "acoustics", "full"):
        assert (out / f"scene_{name}.mp3").is_file()
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "## Coverage" in report and "[ ]" not in report  # the fake scene covers every required moment
    scene = Script.load(out / "scene.script.json")
    assert any(line.reaction == "laugh" for line in scene.lines()) and "(lacht)" not in report
    result = runner.invoke(cli.app, [*common, "reactions", "list"])
    assert result.exit_code == 0 and "laugh=0" in result.output


def test_reactions_generate_all_labels_for_both_hosts(tmp_path, cast_dir):
    from pipeline import cli

    cli.state.clear()
    common = ["--data-dir", str(tmp_path / "data"), "--cast-dir", str(cast_dir), "--fake-audio"]
    result = CliRunner().invoke(cli.app, [*common, "reactions", "generate", "--n", "2"])
    assert result.exit_code == 0, result.output
    from pipeline.performance import REACTIONS

    for speaker in ("tessa", "joris"):
        for label in REACTIONS:
            assert len(list((cast_dir / "reactions" / "_candidates" / speaker / label).glob("*.wav"))) == 2
    bad = CliRunner().invoke(cli.app, [*common, "reactions", "generate", "--label", "giggle"])
    assert bad.exit_code != 0
