import pytest

from pipeline.models import EpisodeTranscript, RenderManifest, Script
from pipeline.runner import StageError
from pipeline.script.cast import load_continuity


def test_run_book_to_render_with_fakes(ingested):
    p = ingested
    results = p.run_book("demo", upto="render", chapters=["ch01", "ch02"])
    assert set(results) == {"ch01", "ch02"}
    for ch, out in results.items():
        assert out["audit"].passed and out["render"].is_file()
        paths = p.paths("demo")
        assert paths.plan(ch).is_file() and paths.script(ch).is_file() and paths.audit(ch).is_file()
        transcript = EpisodeTranscript.load(paths.transcript(ch))
        assert transcript.duration_s > 60 and transcript.blocks and all(b.path for b in transcript.blocks)
        manifest = RenderManifest.load(paths.manifest(ch))
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
    assert status["chapters"][0]["summary"] and status["chapters"][2]["summary"] is None  # what the episode is about
    stages = [s for s, st, _ in p.events if st == "start"]
    assert stages[:4] == ["ingest", "plan", "glossary", "script"]


def test_approval_gate(ingested):
    p = ingested
    p.run_chapter("demo", "ch03", upto="script")
    with pytest.raises(StageError):
        p.render("demo", "ch03")
    p.approve("demo", "ch03")
    assert p.is_approved("demo", "ch03")
    script = p.load_script("demo", "ch03")
    before = script.revision
    script.segments[0].lines[0].text = "Iets anders."
    saved = p.save_script("demo", "ch03", script)
    assert saved.revision == before + 1 and not p.is_approved("demo", "ch03")
    audit = p.audit("demo", "ch03")
    assert audit.script_revision == saved.revision
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
    p.run_chapter("demo", "ch01", upto="script")
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

    class FixedAligner(AlternatingAligner):
        def align(self, clip, text, language="nl"):
            words = super().align(clip, text, language)
            self.last_used_fallback = False  # the aligner works now
            return words

    fixed = FixedAligner()
    _, _, manifest2 = p.render("demo", "ch01", aligner=fixed, force=True)
    # real alignments come from cache; only the estimated ones are retried, and a working aligner heals them
    assert 0 < fixed.calls <= manifest.alignment_fallback_count()
    assert manifest2.alignment_fallback_count() == 0


def test_a_rewrite_that_makes_it_worse_is_undone(ingested):
    from pipeline.models import AuditIssue, AuditResult

    p = ingested
    p.run_chapter("demo", "ch01", upto="script")
    first = p.load_script("demo", "ch01")
    worse = first.model_copy(deep=True)
    worse.revision = first.revision + 1
    worse.segments[0].lines[0].text = "Oké dus, de tekst zegt dit."

    def audit_with(blocking: int, revision: int) -> AuditResult:
        audit = AuditResult(episode_id="ch01", script_revision=revision)
        audit.issues = [AuditIssue(check="lint", rule="x", message="x") for _ in range(blocking)]
        audit.passed = not audit.issues
        return audit

    kept, audit = p._keep_best("demo", "ch01", [(first, audit_with(1, first.revision)), (worse, audit_with(4, worse.revision))])
    assert kept.revision == worse.revision + 1  # the better text, saved as the newest revision
    assert kept.segments[0].lines[0].text == first.segments[0].lines[0].text
    on_disk = p.load_script("demo", "ch01")
    assert on_disk.revision == kept.revision and on_disk.segments[0].lines[0].text == first.segments[0].lines[0].text
    assert AuditResult.load(p.paths("demo").audit("ch01")).script_revision == kept.revision
    assert ("script", "kept_best") in {(s, st) for s, st, _ in p.events}

    better = worse.model_copy(deep=True)
    better.revision = kept.revision + 1
    same, _ = p._keep_best("demo", "ch01", [(first, audit_with(2, 1)), (better, audit_with(0, better.revision))])
    assert same is better  # the last one was best: nothing to restore


def test_the_dialogue_edit_stays_unless_it_breaks_a_fact(ingested):
    p = ingested
    p.run_chapter("demo", "ch02", upto="script")
    assert ("script", "polished") in {(s, st) for s, st, _ in p.events}
    assert any(l.overlap.mode == "backchannel" and l.text == "Hm, ja." for l in p.load_script("demo", "ch02").lines())

    original = p.llm.handlers["script_polish"]

    def careless(req):  # an edit that drops the claim ids: the coverage check now misses claims
        out = original(req)
        for line in out["lines"]:
            line["covers"] = []
        return out

    p.llm.on("script_polish", careless)
    events = len(p.events)
    out = p.run_chapter("demo", "ch03", upto="script")
    script, audit = out["script"], out["audit"]
    assert ("script", "kept_best") in {(s, st) for s, st, _ in p.events[events:]}
    assert audit.passed and any(l.covers for l in script.lines())


def test_the_dialogue_edit_can_be_switched_off(ingested):
    p = ingested
    p.settings = p.settings.with_(script_polish=False)
    p.run_chapter("demo", "ch01", upto="script")
    assert not any(c.task == "script_polish" for c in p.llm.calls)
