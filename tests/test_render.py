"""Emotional continuity between hosts: same-speaker carry and the cross-speaker bleed-through."""

from pipeline.audio.render import render_episode
from pipeline.audio.synth import NullSynth
from pipeline.models import EpisodeTranscript, Line, Script, Segment
from pipeline.paths import BookPaths


def test_untagged_line_picks_up_some_of_the_other_hosts_preceding_mood(settings, cast):
    """Joris's line carries no tag of its own, but Tessa's excited line just before it should still
    nudge his delivery up from his own neutral baseline: emotion is not isolated per speaker."""
    script = Script(episode_id="ch01", segments=[
        Segment(type="cold_open", lines=[
            Line(id="l001", speaker="tessa", text="Dit vind ik echt een geweldig punt.", tags=["excited"]),
            Line(id="l002", speaker="joris", text="Oké, dat is inderdaad wel iets."),
        ]),
    ])
    paths = BookPaths(settings.data_dir, "demo").ensure()
    _, _, manifest = render_episode(script, None, cast, settings, paths, synth=NullSynth(jitter=0.0),
                                    transcriber=None, aligner=None)

    tessa_take = manifest.turns[0].chosen_take
    joris_take = manifest.turns[1].chosen_take
    joris_host = next(h for h in cast.hosts if h.id == "joris")

    assert tessa_take.exaggeration == 0.75  # tessa's base 0.55 + the excited tag's +0.20 delta
    # joris's own untagged baseline (0.35) alone would give 0.35; the 20% cross-speaker bleed from
    # tessa's 0.75 pulls it up to 0.8*0.35 + 0.2*0.75 = 0.43, audibly picking up on her mood
    assert joris_take.exaggeration > joris_host.exaggeration
    assert joris_take.exaggeration == 0.43


def test_first_turn_of_the_episode_has_no_predecessor_to_bleed_from(settings, cast):
    script = Script(episode_id="ch01", segments=[
        Segment(type="cold_open", lines=[Line(id="l001", speaker="tessa", text="We beginnen bij het begin.")]),
    ])
    paths = BookPaths(settings.data_dir, "demo2").ensure()
    _, _, manifest = render_episode(script, None, cast, settings, paths, synth=NullSynth(jitter=0.0),
                                    transcriber=None, aligner=None)
    tessa_host = next(h for h in cast.hosts if h.id == "tessa")
    assert manifest.turns[0].chosen_take.exaggeration == tessa_host.exaggeration


def test_without_the_opening_the_music_goes_under_the_cold_open(settings, cast):
    """Opening off: a cold open still gets the jingle and the voices wait one bar for it. Any other start does not."""
    from dataclasses import replace

    from pipeline.audio.mixer import HOOK_LEAD_S

    bare = replace(settings, opening=False)

    def first_word(segment_type, book_id, s=bare):
        script = Script(episode_id="ch01", segments=[
            Segment(type=segment_type, lines=[Line(id="l001", speaker="tessa", text="We beginnen bij het begin.")]),
        ])
        _, transcript_path, manifest = render_episode(script, None, cast, s, BookPaths(s.data_dir, book_id).ensure(),
                                                      synth=NullSynth(jitter=0.0), transcriber=None, aligner=None)
        assert manifest.opening == {} and [t.turn_id for t in manifest.turns] == ["t001"]
        return EpisodeTranscript.load(transcript_path).lines[0].start

    plain = first_word("body", "plain")
    assert abs(first_word("cold_open", "hook") - (plain + HOOK_LEAD_S)) < 0.01
    assert abs(first_word("cold_open", "off", replace(bare, hook_music=False)) - plain) < 0.01
