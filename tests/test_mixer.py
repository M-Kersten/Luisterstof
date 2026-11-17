import numpy as np

from pipeline.audio.asr import UniformAligner
from pipeline.audio.chunker import chunk_script
from pipeline.audio.mixer import (
    HOOK_DUCK_DB,
    HOOK_LEAD_S,
    build_transcript,
    duck,
    even_out,
    export_blocks,
    hook_music_path,
    load_music,
    measure_lufs,
    mix,
    normalise_loudness,
    under_voices,
)
from pipeline.audio.synth import AudioClip, NullSynth, VoiceSpec
from pipeline.audio.timeline import Timeline, assemble
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
    transcript = build_transcript(tl, script, offset=offset, duration_s=mixed.duration_s, blocks=chunk_script(script))
    assert [l.line_id for l in transcript.lines] == ["l001", "l002", "l003"]
    assert all(l.words for l in transcript.lines) and transcript.lines[0].start >= offset
    assert transcript.lines[1].start > transcript.lines[0].end
    assert len(transcript.blocks) == 1 and transcript.blocks[0].line_ids == ["l001", "l002", "l003"]
    paths = export_blocks(mixed, transcript, tmp_path / "blocks")
    assert (tmp_path / "blocks" / "b001.mp3").is_file() and transcript.blocks[0].path == paths["b001"]
    out = mixed.write(tmp_path / "ep.mp3")
    assert AudioClip.read(out).duration_s > 0


def _rms_db(x):
    return 20 * np.log10(float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2))) + 1e-12)


def _timeline(lines):
    script = Script(episode_id="ch01", segments=[Segment(type="cold_open", lines=lines)])
    turns = group_turns(script)
    synth, aligner = NullSynth(sample_rate=SR), UniformAligner()
    rendered = {}
    for t in turns:
        clip = synth.render(t.text, VoiceSpec(speaker_id=t.speaker), exaggeration=0.5, seed=1)
        rendered[t.turn_id] = (clip, aligner.align(clip, t.text))
    return assemble(turns, rendered, sr=SR)


def test_music_goes_under_the_voices_and_comes_up_between_them():
    music = _tone(20.0)
    bed = under_voices(music, SR, [(2.0, 6.0), (9.0, 12.0)], until=16.0)
    alone, under = _rms_db(bed[: SR]), _rms_db(bed[int(2.2 * SR): int(5.8 * SR)])
    assert abs(alone - _rms_db(music[: SR])) < 0.1 and abs((alone - under) - HOOK_DUCK_DB) < 0.2
    assert abs(_rms_db(bed[int(7.2 * SR): int(8.2 * SR)]) - alone) < 0.2  # up again between the two voices
    assert abs(_rms_db(bed[int(9.2 * SR): int(11.8 * SR)]) - under) < 0.2
    assert len(bed) == 16 * SR and _rms_db(bed[-SR // 10:]) < under - 10  # faded out by the time it has to be gone
    tight = under_voices(music, SR, [(2.0, 6.0), (6.5, 9.0)])  # half a second of silence is no place for a swell
    assert abs(_rms_db(tight[int(6.1 * SR): int(6.4 * SR)]) - under) < 0.2
    assert len(tight) == len(music) and abs(_rms_db(tight[-SR:]) - alone) < 0.2  # no end given: it plays out
    stereo = under_voices(np.stack([music, music * 0.5], axis=1), SR, [(2.0, 6.0)], until=8.0)
    assert stereo.shape == (8 * SR, 2) and abs(_rms_db(stereo[: SR, 0]) - _rms_db(stereo[: SR, 1]) - 6.0) < 0.1


def test_music_that_grows_is_evened_out_by_half():
    soft, loud = _tone(8.0, 0.2 * 10 ** -0.5), _tone(8.0, 0.2)  # 10 dB apart
    out = even_out(np.concatenate([soft, loud]), SR)
    gap = _rms_db(out[-4 * SR:]) - _rms_db(out[: 4 * SR])
    assert 4.5 < gap < 5.5
    assert np.array_equal(even_out(np.zeros(SR, dtype=np.float32), SR), np.zeros(SR, dtype=np.float32))


def test_mix_with_music_under_the_voices():
    tl = _timeline([
        Line(id="l001", speaker="tessa", text="Dit is de eerste regel van de test met genoeg woorden erin."),
        Line(id="l002", speaker="joris", text="En dit de tweede regel, ook met genoeg woorden."),
    ])
    plain, plain_offset = mix(tl, sr=SR)
    later = Timeline(placements=[p.moved(HOOK_LEAD_S) for p in tl.placements], sr=SR)  # the music is heard first
    t = np.arange(44100 * 20) / 44100
    music = AudioClip(np.stack([0.1 * np.sin(2 * np.pi * 330 * t), 0.1 * np.sin(2 * np.pi * 495 * t)], axis=1).astype(np.float32), 44100)
    mixed, offset = mix(later, sr=SR, music=music)
    assert mixed.samples.ndim == 1  # a mono episode stays mono, whatever the music is
    assert offset == plain_offset  # the timeline says where the voices are; the music starts with it
    assert abs(measure_lufs(mixed.samples, SR) + 16.0) < 1.0 and float(np.max(np.abs(mixed.samples))) <= 0.985
    first_word = offset + later.placements[0].start
    lead = mixed.samples[int(0.5 * SR): int((first_word - 0.8) * SR)]
    assert _rms_db(lead) > _rms_db(plain.samples[: int(0.25 * SR)]) + 30  # music before the first word, not room tone
    assert mixed.duration_s < offset + later.duration + 2.5  # 20 s of music does not outlast the voices
    assert _rms_db(mixed.samples[-SR // 20:]) < _rms_db(lead) - 30
    transcript = build_transcript(later, Script(episode_id="ch01", segments=[]), offset=offset, duration_s=mixed.duration_s)
    assert abs(transcript.lines[0].start - first_word) < 0.1


def test_hook_music_file(tmp_path):
    import soundfile as sf

    shipped = hook_music_path(tmp_path)
    assert shipped is not None and shipped.name == "hook_jingle.wav"  # the jingle that comes with the project
    clip = load_music(shipped)
    assert clip.samples.ndim == 2 and abs(clip.duration_s - 30.0) < 0.01
    own = tmp_path / "stings" / "hook.wav"
    own.parent.mkdir()
    sf.write(own, _tone(1.0), SR)
    assert hook_music_path(tmp_path) == own and load_music(own).samples.ndim == 1
    assert load_music(tmp_path / "nothing.wav") is None


def test_the_music_level_is_set_against_the_voices():
    tl = _timeline([Line(id="l001", speaker="tessa", text="Dit is de eerste regel van de test met genoeg woorden erin.")])
    later = Timeline(placements=[p.moved(HOOK_LEAD_S) for p in tl.placements], sr=SR)
    t = np.arange(SR * 20) / SR
    music = AudioClip((0.1 * np.sin(2 * np.pi * 330 * t)).astype(np.float32), SR)

    def music_against_voice(music_db):
        mixed, offset = mix(later, sr=SR, music=music, music_db=music_db, room_tone_db=None)
        first = offset + later.placements[0].start
        return _rms_db(mixed.samples[int(0.3 * SR): int((first - 0.8) * SR)]) - _rms_db(mixed.samples[int((first + 0.8) * SR): int((first + 2.5) * SR)])

    assert abs((music_against_voice(0.0) - music_against_voice(-2.5)) - 2.5) < 0.3
