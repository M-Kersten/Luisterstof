import numpy as np

from pipeline.audio.asr import UniformAligner
from pipeline.audio.chunker import chunk_script
from pipeline.audio.mixer import build_transcript, duck, export_blocks, measure_lufs, mix, normalise_loudness
from pipeline.audio.synth import AudioClip, NullSynth, VoiceSpec
from pipeline.audio.timeline import assemble
from pipeline.audio.turns import group_turns
from pipeline.models import Line, Script, Segment

SR = 24000


def _tone(seconds=2.0, amp=0.1):
    t = np.arange(int(SR * seconds)) / SR
    return (amp * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def test_loudness_and_duck():
    tone = _tone()
    out = normalise_loudness(tone, SR, -18.0)
    assert abs(measure_lufs(out, SR) + 18.0) < 0.5
    ducked = duck(tone, SR, 1.0, 12.0, 0.08)
    head, tail = np.sqrt(np.mean(ducked[: SR // 2] ** 2)), np.sqrt(np.mean(ducked[-SR // 2:] ** 2))
    assert abs(20 * np.log10(head / tail) - 12.0) < 0.5


def test_mix_and_transcript(tmp_path):
    script = Script(episode_id="ch01", segments=[Segment(type="body", lines=[
        Line(id="l001", speaker="tessa", text="Dit is de eerste regel van de test met genoeg woorden erin."),
        Line(id="l002", speaker="joris", text="En dit de tweede regel, ook met genoeg woorden."),
        Line(id="l003", speaker="tessa", text="Derde regel."),
    ])])
    turns = group_turns(script)
    synth, aligner = NullSynth(sample_rate=SR), UniformAligner()
    rendered = {}
    for t in turns:
        clip = synth.render(t.text, VoiceSpec(speaker_id=t.speaker), exaggeration=0.5, seed=1)
        rendered[t.turn_id] = (clip, aligner.align(clip, t.text))
    tl = assemble(turns, rendered, sr=SR)
    intro = AudioClip(_tone(1.0, 0.2), 44100)
    mixed, offset = mix(tl, sr=SR, target_lufs=-16.0, intro=intro)
    assert abs(measure_lufs(mixed.samples, SR) + 16.0) < 1.0
    assert offset > 0.3 and mixed.duration_s > tl.duration + offset
    assert float(np.max(np.abs(mixed.samples))) <= 0.985
    transcript = build_transcript(tl, script, tier="draft", offset=offset, duration_s=mixed.duration_s, blocks=chunk_script(script))
    assert [l.line_id for l in transcript.lines] == ["l001", "l002", "l003"]
    assert all(l.words for l in transcript.lines) and transcript.lines[0].start >= offset
    assert transcript.lines[1].start > transcript.lines[0].end
    assert len(transcript.blocks) == 1 and transcript.blocks[0].line_ids == ["l001", "l002", "l003"]
    paths = export_blocks(mixed, transcript, tmp_path / "blocks")
    assert (tmp_path / "blocks" / "b001.mp3").is_file() and transcript.blocks[0].path == paths["b001"]
    out = mixed.write(tmp_path / "ep.mp3")
    assert AudioClip.read(out).duration_s > 0
