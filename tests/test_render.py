"""Emotional continuity between hosts: same-speaker carry and the cross-speaker bleed-through."""

from pipeline.audio.render import render_episode
from pipeline.audio.synth import NullSynth
from pipeline.models import Line, Script, Segment
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
    _, _, manifest = render_episode(script, None, cast, settings, paths, tier="draft", synth=NullSynth(jitter=0.0),
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
    _, _, manifest = render_episode(script, None, cast, settings, paths, tier="draft", synth=NullSynth(jitter=0.0),
                                    transcriber=None, aligner=None)
    tessa_host = next(h for h in cast.hosts if h.id == "tessa")
    assert manifest.turns[0].chosen_take.exaggeration == tessa_host.exaggeration
