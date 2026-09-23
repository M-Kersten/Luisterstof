"""Pacing controls: cfg_weight (generation-side) and speech_rate (deterministic post-render)."""

from pathlib import Path

import pytest

from pipeline.audio.audition import audition
from pipeline.audio.synth import NullSynth, VoiceSpec, apply_speech_rate, voice_spec_for
from pipeline.models import Guest, Host


def test_voice_spec_defaults_and_threading_from_host(tmp_path):
    assert VoiceSpec(speaker_id="x").cfg_weight == 0.5
    assert VoiceSpec(speaker_id="x").speech_rate == 1.0
    assert VoiceSpec(speaker_id="x").temperature == 0.8

    host = Host(id="tessa", name="Tessa", role="explainer", cfg_weight=0.35, speech_rate=0.85, temperature=0.6)
    spec = voice_spec_for(host, tmp_path)
    assert spec.cfg_weight == 0.35 and spec.speech_rate == 0.85 and spec.temperature == 0.6

    guest = Guest(id="hanna", name="Hanna", cfg_weight=0.4, speech_rate=0.9, temperature=1.0)
    spec2 = voice_spec_for(guest, tmp_path)
    assert spec2.cfg_weight == 0.4 and spec2.speech_rate == 0.9 and spec2.temperature == 1.0


def test_apply_speech_rate_is_noop_at_one():
    librosa = pytest.importorskip("librosa")
    import numpy as np

    from pipeline.audio.synth import AudioClip

    clip = AudioClip(np.zeros(2400, dtype=np.float32), 24000, meta={"text": "x"})
    out = apply_speech_rate(clip, 1.0)
    assert out is clip  # short-circuited, not even imported librosa's stretch path
    _ = librosa


def test_apply_speech_rate_changes_duration_as_expected():
    pytest.importorskip("librosa")
    import numpy as np

    from pipeline.audio.synth import AudioClip

    sr = 24000
    t = np.arange(int(sr * 2.0)) / sr
    samples = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    clip = AudioClip(samples, sr, meta={"speaker": "tessa"})

    slower = apply_speech_rate(clip, 0.5)
    faster = apply_speech_rate(clip, 2.0)
    assert slower.duration_s == pytest.approx(clip.duration_s * 2, rel=0.02)
    assert faster.duration_s == pytest.approx(clip.duration_s / 2, rel=0.02)
    assert slower.sr == clip.sr and slower.meta["speaker"] == "tessa"  # metadata preserved


def test_apply_speech_rate_empty_clip_does_not_crash():
    import numpy as np

    from pipeline.audio.synth import AudioClip

    clip = AudioClip(np.zeros(0, dtype=np.float32), 24000)
    assert apply_speech_rate(clip, 0.8) is clip


def test_audition_sweeps_every_combination(tmp_path):
    ref = tmp_path / "tessa.wav"
    ref.write_bytes(b"not a real wav, NullSynth ignores it")
    rows = audition("Dit is een test paragraaf voor de audition.", {"tessa": ref}, tmp_path / "out", NullSynth(),
                    exaggerations=(0.4, 0.6), cfg_weights=(0.3, 0.5), speech_rates=(1.0,), temperatures=(0.8, 1.0))
    assert len(rows) == 8  # 1 speaker x 2 exaggerations x 2 cfg_weights x 1 speech_rate x 2 temperatures
    combos = {(r["exaggeration"], r["cfg_weight"], r["speech_rate"], r["temperature"]) for r in rows}
    assert combos == {(0.4, 0.3, 1.0, 0.8), (0.4, 0.3, 1.0, 1.0), (0.4, 0.5, 1.0, 0.8), (0.4, 0.5, 1.0, 1.0),
                      (0.6, 0.3, 1.0, 0.8), (0.6, 0.3, 1.0, 1.0), (0.6, 0.5, 1.0, 0.8), (0.6, 0.5, 1.0, 1.0)}
    assert len({r["path"] for r in rows}) == 8  # distinct filenames, nothing overwritten
    assert all(Path(r["path"]).is_file() for r in rows)
