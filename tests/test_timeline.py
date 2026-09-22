import numpy as np
import pytest

from pipeline.audio.asr import UniformAligner
from pipeline.audio.synth import AudioClip, NullSynth, VoiceSpec
from pipeline.audio.timeline import (
    GAP_RANGE,
    INTERRUPT_LEAD,
    QUICK_GAP_RANGE,
    SEGMENT_GAP_RANGE,
    assemble,
    is_quick_handoff,
)
from pipeline.audio.turns import group_turns
from pipeline.models import Line, Overlap, Script, Segment

SR = 24000


def _render(turns):
    synth, aligner = NullSynth(sample_rate=SR, jitter=0.0), UniformAligner()
    out = {}
    for t in turns:
        clip = synth.render(t.text, VoiceSpec(speaker_id=t.speaker), exaggeration=0.5, seed=1)
        out[t.turn_id] = (clip, aligner.align(clip, t.text))
    return out


def _script():
    return Script(episode_id="ch01", segments=[
        Segment(type="cold_open", lines=[
            Line(id="l001", speaker="joris", text="Nee dat accepteer ik niet, zo werkt dat niet."),
            Line(id="l002", speaker="tessa", text="Het werkt wel als je kijkt naar de hele—"),
            Line(id="l003", speaker="joris", text="Wacht even, welke hele?", overlap=Overlap(mode="interrupt", target="l002", cut_word="hele")),
            Line(id="l004", speaker="tessa", text="De hele verzameling natuurlijk, dat zei ik al drie keer."),
            Line(id="l005", speaker="joris", text="Ja precies.", overlap=Overlap(mode="backchannel", target="l004")),
            Line(id="l006", speaker="tessa", text="En dan gaan we verder met de rest van het verhaal.", pause_after_ms=2000),
        ]),
        Segment(type="body", lines=[
            Line(id="l007", speaker="joris", text="Nieuw segment, nieuwe vraag over de stof."),
            Line(id="l008", speaker="joris", text="Tweede regel van dezelfde spreker."),
        ]),
    ])


def test_turn_grouping():
    turns = group_turns(_script())
    assert [t.line_ids for t in turns] == [["l001"], ["l002"], ["l003"], ["l004"], ["l005"], ["l006"], ["l007", "l008"]]
    assert turns[-1].word_counts() == [7, 5]


def test_assembly_rules():
    turns = group_turns(_script())
    tl = assemble(turns, _render(turns), sr=SR, seed=3)
    by = tl.by_line()
    a, b = by["l002"], by["l003"]
    # interrupt: B starts INTERRUPT_LEAD before the cut word onset, A is ducked and cut short
    cut = next(w for w in reversed(a.words) if w.word.startswith("hele"))
    assert abs(b.start - (cut.start - INTERRUPT_LEAD)) < 1e-6
    assert a.duck_at == b.start and a.duck_db == 12.0 and a.cut_at is not None and a.cut_at <= a.start + a.clip.duration_s
    assert b.start < a.start + a.clip.duration_s
    # backchannel: placed past 60% of the target, does not advance the timeline
    c, d, e = by["l004"], by["l005"], by["l006"]
    assert d.start >= c.start + 0.6 * c.clip.duration_s - 1e-6 and d.gain_db == -8.0 and not d.advances
    gap = e.start - c.end
    assert GAP_RANGE[0] - 1e-6 <= gap <= GAP_RANGE[1] + 1e-6
    # deliberate pause + segment gap before the next segment
    f = by["l007"]
    assert f.start - e.end >= 2.0 + SEGMENT_GAP_RANGE[0] - 1e-6
    assert tl.qa == []
    # normal gaps inside a segment are sampled, never constant
    gap_ab = a.start - by["l001"].end
    assert GAP_RANGE[0] <= gap_ab <= GAP_RANGE[1]


def test_unwritten_gap_is_flagged():
    turns = group_turns(_script())
    rendered = _render(turns)
    tl = assemble(turns, rendered, sr=SR, seed=3)
    # simulate a long silence by shifting the last placement
    tl.placements[-1].start += 3.0
    from pipeline.audio.timeline import unwritten_gaps
    issues = unwritten_gaps(tl, 1.2)
    assert issues and "l008" not in issues[0] and "l007" in issues[0]


def test_missing_audio_and_bad_overlap_reported():
    script = _script()
    turns = group_turns(script)
    rendered = _render(turns)
    del rendered[turns[2].turn_id]  # the interrupting line has no audio
    tl = assemble(turns, rendered, sr=SR)
    assert any("has no audio" in q for q in tl.qa)
    assert isinstance(tl.placements[0].clip, AudioClip) and tl.duration > 0
    assert np.isfinite(tl.duration)


@pytest.mark.parametrize("text,expected", [
    ("Wat bedoel je daarmee?", True),  # question
    ("Ja precies.", True),  # short reactive line
    ("Nee.", True),  # very short
    ("Dat is een lang antwoord met behoorlijk wat woorden erin, geen snelle reactie.", False),
])
def test_is_quick_handoff(text, expected):
    assert is_quick_handoff(Line(id="l001", speaker="tessa", text=text)) is expected


def test_is_quick_handoff_guards_blank_text():
    # Line's own validator rejects empty text on construction or assignment; reach the function's
    # blank-text branch by writing the attribute directly, bypassing pydantic's validate_assignment.
    line = Line(id="l001", speaker="tessa", text="x")
    object.__setattr__(line, "text", "")
    assert is_quick_handoff(line) is False


def _quick_handoff_script(first_text: str) -> Script:
    return Script(episode_id="ch01", segments=[Segment(type="body", lines=[
        Line(id="l001", speaker="joris", text=first_text),
        Line(id="l002", speaker="tessa", text="Dat leg ik zo uit, geef me even de ruimte om het rustig op te bouwen."),
    ])])


def test_quick_handoff_uses_tight_gap_range_and_ducks_on_overlap():
    turns = group_turns(_quick_handoff_script("Hoe zit dat dan precies?"))
    tl = assemble(turns, _render(turns), sr=SR, seed=1)
    a, b = tl.by_line()["l001"], tl.by_line()["l002"]
    gap = b.start - a.end
    assert QUICK_GAP_RANGE[0] - 1e-6 <= gap <= QUICK_GAP_RANGE[1] + 1e-6
    assert gap < GAP_RANGE[0]  # unambiguously tighter than a normal handoff would ever sample
    # seed=1 is verified to draw a negative (overlapping) value first from QUICK_GAP_RANGE
    assert gap < 0
    assert a.duck_at == b.start and a.duck_db == 4.0 and a.duck_db < 12.0  # gentle, not an interrupt-strength duck
    # the ducked line still plays out in full - a quick handoff never buries a word the way interrupt does
    assert a.cut_at is None


def test_normal_handoff_after_a_long_declarative_line_is_unaffected():
    turns = group_turns(_quick_handoff_script("Dat is precies waarom ik het daar niet mee eens ben, en ik zal uitleggen waarom."))
    tl = assemble(turns, _render(turns), sr=SR, seed=1)
    a, b = tl.by_line()["l001"], tl.by_line()["l002"]
    gap = b.start - a.end
    assert GAP_RANGE[0] - 1e-6 <= gap <= GAP_RANGE[1] + 1e-6
    assert a.duck_at is None


def test_quick_handoff_suppressed_by_explicit_pause_or_segment_change():
    # an explicit written beat overrides the quick-handoff shortcut even after a question
    paused = Script(episode_id="ch01", segments=[Segment(type="body", lines=[
        Line(id="l001", speaker="joris", text="Wat denk jij dat het antwoord is?", pause_after_ms=1500),
        Line(id="l002", speaker="tessa", text="Nou..."),
    ])])
    turns = group_turns(paused)
    tl = assemble(turns, _render(turns), sr=SR, seed=1)
    a, b = tl.by_line()["l001"], tl.by_line()["l002"]
    assert b.start - a.end >= 1.5 + GAP_RANGE[0] - 1e-6

    # a segment boundary always gets the deliberate long gap, even after a question
    across_segments = Script(episode_id="ch01", segments=[
        Segment(type="cold_open", lines=[Line(id="l001", speaker="joris", text="Klaar voor het volgende deel?")]),
        Segment(type="body", lines=[Line(id="l002", speaker="tessa", text="Helemaal.")]),
    ])
    turns2 = group_turns(across_segments)
    tl2 = assemble(turns2, _render(turns2), sr=SR, seed=1)
    a2, b2 = tl2.by_line()["l001"], tl2.by_line()["l002"]
    assert b2.start - a2.end >= SEGMENT_GAP_RANGE[0] - 1e-6
