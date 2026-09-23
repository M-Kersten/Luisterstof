import pytest

from pipeline.models import EpisodeTranscript, RenderManifest, Script
from pipeline.runner import StageError
from pipeline.script.cast import load_continuity


def test_run_book_to_render_with_fakes(ingested):
    p = ingested
    results = p.run_book("demo", upto="render", chapters=["ch01", "ch02"])
    assert set(results) == {"ch01", "ch02"}
    for ch, out in results.items():
        assert out["audit"].passed and out["draft"].is_file() and out["render"].is_file()
        paths = p.paths("demo")
        assert paths.plan(ch).is_file() and paths.script(ch).is_file() and paths.audit(ch).is_file()
        transcript = EpisodeTranscript.load(paths.transcript(ch, "final"))
        assert transcript.duration_s > 60 and transcript.blocks and all(b.path for b in transcript.blocks)
        manifest = RenderManifest.load(paths.manifest(ch, "final"))
        assert manifest.turns and not manifest.flagged_turns()
    # continuity: episode 2 was written with episode 1's log available
    entries = load_continuity(p.settings.cast_dir)
    assert [e.episode for e in entries] == ["ch01", "ch02"]
    ch2_calls = [c for c in p.llm.calls if c.task == "script_segment" and "Aflevering ch01" in "".join(b["text"] for b in c.system)]
    assert ch2_calls, "chapter 2 prompts should carry chapter 1's continuity"
    recap = next(s for s in Script.load(p.paths("demo").script("ch02")).segments if s.type == "recap")
    assert any(c.startswith("ch01:") for l in recap.lines for c in l.covers)
    status = p.status("demo")
    assert status["chapters"][0]["final"] and status["chapters"][0]["approved"] and not status["chapters"][2]["plan"]
    stages = [s for s, st, _ in p.events if st == "start"]
    assert stages[:4] == ["ingest", "plan", "glossary", "script"]


def test_approval_gate(ingested):
    p = ingested
    p.run_chapter("demo", "ch03", upto="draft")
    with pytest.raises(StageError):
        p.render("demo", "ch03")
    p.approve("demo", "ch03")
    assert p.is_approved("demo", "ch03")
    script = p.load_script("demo", "ch03")
    script.segments[0].lines[0].text = "Iets anders."
    saved = p.save_script("demo", "ch03", script)
    assert saved.revision == 2 and not p.is_approved("demo", "ch03")
    audit = p.audit("demo", "ch03")
    assert audit.script_revision == 2
    p.approve("demo", "ch03")
    audio, transcript, manifest = p.render("demo", "ch03")
    assert audio.is_file()
    # second render: everything from cache
    _, _, manifest2 = p.render("demo", "ch03")
    assert all(t.from_cache for t in manifest2.turns)


def test_render_manifest_tracks_alignment_fallback_per_turn(ingested):
    """End-to-end: a flaky aligner's per-call fallback flag ends up correctly attributed to
    each turn in the saved manifest, and survives an all-cached re-render without re-aligning."""
    p = ingested
    p.run_chapter("demo", "ch01", upto="draft")
    p.approve("demo", "ch01")

    class AlternatingAligner:
        def __init__(self):
            self.calls = 0
            self.last_used_fallback = False

        def align(self, clip, text, language="nl"):
            from pipeline.audio.asr import WordTiming, tokenize

            self.last_used_fallback = self.calls % 2 == 1
            self.calls += 1
            return [WordTiming(w, i * 0.3, i * 0.3 + 0.2) for i, w in enumerate(tokenize(text))]

    aligner = AlternatingAligner()
    _, _, manifest = p.render("demo", "ch01", aligner=aligner)

    with_take = [t for t in manifest.turns if t.chosen_take is not None]
    assert len(with_take) >= 2, "need at least 2 rendered turns to exercise the alternation"
    # duplicate turn text (e.g. short backchannels) can reuse a prior turn's cached alignment
    # instead of calling align() again, so this only checks both values actually occur, not strict order
    fallback_flags = {t.aligned_with_fallback for t in with_take}
    assert fallback_flags == {True, False}, "expected both real and estimated alignment to appear across turns"
    assert manifest.alignment_fallback_count() == sum(1 for t in with_take if t.aligned_with_fallback)
    verified, verifiable = manifest.verification_coverage()
    assert verified == verifiable == len(with_take)  # EchoTranscriber always matches exactly

    calls_before = aligner.calls
    _, _, manifest2 = p.render("demo", "ch01", aligner=aligner, force=True)
    assert aligner.calls == calls_before  # cache hit: the saved fallback flag is reused, not recomputed
    assert [t.aligned_with_fallback for t in manifest2.turns] == [t.aligned_with_fallback for t in manifest.turns]


def test_eleven_blocks_spliced(ingested):
    import numpy as np

    from pipeline.audio.synth import AudioClip

    p = ingested
    p.run_chapter("demo", "ch01", upto="render")
    calls = []

    def fake_render(inputs):
        calls.append(inputs)
        assert all(i["voice_id"] for i in inputs)
        return AudioClip(np.random.default_rng(1).standard_normal(24000 * 20).astype(np.float32) * 0.05, 24000)

    for h in p.cast.hosts:
        h.voice_id_final = f"voice-{h.id}"
    manifest = p.eleven("demo", "ch01", ["b001"], render_fn=fake_render)
    assert "b001" in manifest.block_overrides and calls and calls[0][0]["text"]
    audio, transcript_path, manifest2 = p.render("demo", "ch01", force=True)
    assert "b001" in manifest2.block_overrides
    transcript = EpisodeTranscript.load(transcript_path)
    assert transcript.duration_s > 0
