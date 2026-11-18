"""Reference clip processing on synthetic speech: voiced 'words' with a pitch contour, gaps and pauses."""

import numpy as np
import pytest
import soundfile as sf
from scipy import signal

from pipeline.audio.refclip import ClipError, _clipping, _f0, _semitones, process

SR = 24000


def _word(seconds, f0_from, f0_to, rng):
    n = int(seconds * SR)
    f0 = np.linspace(f0_from, f0_to, n)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    tone = sum(np.sin(k * phase) / k for k in range(1, 6))
    env = np.sin(np.linspace(0, np.pi, n)) ** 0.5
    return 0.25 * tone * env + rng.normal(0, 0.002, n)


def _speech(parts, rng, noise=0.0005):
    """parts: ("flat"|"lively", seconds) or ("pause", seconds)."""
    out = []
    for kind, seconds in parts:
        if kind == "pause":
            out.append(rng.normal(0, noise, int(seconds * SR)))
            continue
        t = 0.0
        while t < seconds:
            if kind == "flat":
                out.append(_word(0.35, 118, 122, rng))
            else:
                lo, hi = rng.uniform(95, 140), rng.uniform(180, 260)
                out.append(_word(0.35, *(lo, hi) if rng.random() < 0.5 else (hi, lo), rng))
            out.append(rng.normal(0, noise, int(0.07 * SR)))
            t += 0.42
        out.append(rng.normal(0, noise, int(0.3 * SR)))  # a pause between phrases
    return np.concatenate(out)


def test_a_flat_start_is_moved_back_and_the_clip_is_cleaned(tmp_path):
    rng = np.random.default_rng(1)
    x = _speech([("pause", 1.5), ("flat", 9), ("pause", 1.2), ("lively", 14), ("pause", 0.4), ("lively", 14)], rng)
    src = tmp_path / "raw.wav"
    sf.write(src, np.stack([x, x], axis=1) * 0.5, SR)
    report = process(src, tmp_path / "clip.wav").as_dict()
    info = sf.info(tmp_path / "clip.wav")
    assert info.channels == 1 and info.subtype == "PCM_24"
    before, after = report["before"], report["after"]
    assert before["lead_silence_s"] > 1.2 and after["lead_silence_s"] < 0.3
    assert after["opening_semitones"] > before["opening_semitones"] + 3
    assert after["long_pauses"] == 0
    assert abs(after["lufs"] - -20.0) < 0.6 and after["peak_dbfs"] <= -1.4
    assert any("levendigste stuk" in c for c in report["changes"])
    assert any("Stereo naar mono" in c for c in report["changes"])
    assert after["seconds"] > 30  # nothing of the material was thrown away


def test_a_lively_clip_keeps_its_order(tmp_path):
    rng = np.random.default_rng(2)
    x = _speech([("lively", 18), ("lively", 18)], rng)
    sf.write(tmp_path / "raw.wav", x, SR)
    report = process(tmp_path / "raw.wav", tmp_path / "clip.wav").as_dict()
    assert not any("levendigste stuk" in c for c in report["changes"])
    assert report["verdict"] == "ok"


def test_what_cannot_be_fixed_is_reported(tmp_path):
    rng = np.random.default_rng(3)
    noisy = _speech([("lively", 35)], rng, noise=0.01)  # a fan in the room
    sf.write(tmp_path / "noisy.wav", noisy, SR)
    report = process(tmp_path / "noisy.wav", tmp_path / "a.wav").as_dict()
    assert any(f["level"] in ("warn", "bad") and "achtergrondgeluid" in f["text"] for f in report["findings"])

    loud = np.clip(_speech([("lively", 35)], rng) * 8, -1, 1)
    sf.write(tmp_path / "loud.wav", loud, SR)
    report = process(tmp_path / "loud.wav", tmp_path / "b.wav").as_dict()
    assert report["verdict"] == "bad" and any("oversturd" in f["text"] for f in report["findings"])
    assert report["before"]["pops"] == 0 and not any("Ploffen" in f["text"] for f in report["findings"])


def test_too_short_or_not_audio_is_refused(tmp_path):
    rng = np.random.default_rng(4)
    sf.write(tmp_path / "short.wav", _speech([("lively", 4)], rng), SR)
    with pytest.raises(ClipError):
        process(tmp_path / "short.wav", tmp_path / "x.wav")
    (tmp_path / "junk.wav").write_bytes(b"this is not audio")
    with pytest.raises(ClipError):
        process(tmp_path / "junk.wav", tmp_path / "y.wav")


# ---------------------------------------------------------------- a low voice
def _resonance(x, hz, bw):
    """A formant: a two-pole resonance at hz, bw wide."""
    r = np.exp(-np.pi * bw / SR)
    return signal.lfilter([1 - r], [1, -2 * r * np.cos(2 * np.pi * hz / SR), r * r], x)


def _pulses(seconds, f0, fall=0.06):
    """Glottal pulses at about f0, falling a little over the stretch as a voice does towards the end of a word."""
    n = int(seconds * SR)
    pitch = f0 * np.linspace(1 + fall / 2, 1 - fall / 2, n)
    out = np.zeros(n)
    out[np.flatnonzero(np.diff(np.floor(np.cumsum(pitch) / SR)) > 0)] = 1.0
    return out


def _low_speech(parts, rng, f0=100.0, noise=0.0005):
    """A monotone voice. parts: (first formants, seconds), the words taking the formants in turn, or ("pause", s)."""
    out = []
    for formants, seconds in parts:
        if formants == "pause":
            out.append(rng.normal(0, noise, int(seconds * SR)))
            continue
        for i in range(int(seconds / 0.42) + 1):
            y = _pulses(0.35, f0 * rng.uniform(0.98, 1.02))
            y = _resonance(y, formants[i % len(formants)], 60.0) + 0.2 * _resonance(y, 1500.0, 150.0)
            env = np.sin(np.linspace(0, np.pi, len(y))) ** 0.5
            out.append(0.25 * y / np.max(np.abs(y)) * env + rng.normal(0, 0.002, len(y)))
            out.append(rng.normal(0, noise, int(0.07 * SR)))
        out.append(rng.normal(0, noise, int(0.3 * SR)))
    return np.concatenate(out)


def test_a_low_voice_is_not_read_at_three_times_its_pitch():
    # 100 Hz pulses through a resonance at 300 Hz, the oe of a low male voice: the third harmonic dominates, and the
    # tracker used to take it for the pitch (about 300 Hz, an octave and a half too high).
    y = _resonance(_pulses(6, 100.0), 300.0, 60.0)
    f0 = _f0(0.3 * y / np.max(np.abs(y)), SR)
    voiced = f0[~np.isnan(f0)]
    assert len(voiced) > 0.9 * len(f0)
    assert 97 < np.median(voiced) < 103 and np.percentile(voiced, 90) < 110
    assert _semitones(f0) < 2


@pytest.mark.parametrize("f0, formants", [(85, (300, 700)), (105, (315, 600)), (120, (250, 600)), (220, (700, 450))])
def test_the_pitch_is_the_voice_and_not_a_harmonic(f0, formants):
    found = _f0(_low_speech([(formants, 6)], np.random.default_rng(3), f0=f0), SR)
    voiced = found[~np.isnan(found)]
    assert abs(np.median(voiced) / f0 - 1) < 0.04  # a high voice is not halved either
    assert _semitones(found) < 3


def test_a_monotone_low_voice_is_not_taken_for_a_lively_one(tmp_path):
    # The oe's later on used to read as leaps of an octave and a half, so the clip was turned around to start there
    # and the report praised a lively opening.
    rng = np.random.default_rng(7)
    x = _low_speech([("pause", 1.0), ((600, 700), 9), ("pause", 1.0), ((315, 600), 14), ("pause", 0.4),
                     ((315, 600), 14)], rng, f0=105)
    sf.write(tmp_path / "raw.wav", x, SR)
    report = process(tmp_path / "raw.wav", tmp_path / "clip.wav").as_dict()
    assert report["before"]["semitones"] < 3 and report["after"]["opening_semitones"] < 3
    assert not any("levendigste stuk" in c for c in report["changes"])
    assert any(f["level"] == "warn" and "vlak" in f["text"] for f in report["findings"])
    assert not any("Levendig begin" in f["text"] for f in report["findings"])


# ---------------------------------------------------------------- pops
def _pop(x, at):
    """The air of a p or b on the microphone: one push of low pressure that runs into the ceiling."""
    t = np.arange(int(0.06 * SR)) / SR
    i = int(at * SR)
    x[i:i + len(t)] += 1.6 * (t / 0.008) * np.exp(1 - t / 0.008)
    return x


def _popped_recording(path, pops):
    x = _speech([("pause", 0.5), ("lively", 9), ("pause", 1.0), ("lively", 14), ("pause", 0.4), ("lively", 14)],
                np.random.default_rng(11))
    for t in pops:
        _pop(x, t)
    sf.write(path, np.clip(x, -1, 1), SR)
    return sf.read(path)[0]


def test_pops_are_not_clipping_and_stay_out_of_the_first_ten_seconds(tmp_path):
    pops = [2.0, 5.5]
    x = _popped_recording(tmp_path / "raw.wav", pops)
    assert np.sum(np.abs(x) >= 0.999) > 500  # this used to be reported as an overdriven recording
    report = process(tmp_path / "raw.wav", tmp_path / "clip.wav").as_dict()
    assert report["before"]["pops"] == 2 and report["before"]["clipped"] == 0
    assert not any("oversturd" in f["text"] for f in report["findings"])
    assert any(f["level"] == "warn" and f["text"].startswith("Ploffen: gebruik een popfilter of spreek langs de "
                                                             "microfoon") for f in report["findings"])
    assert any("geen plof in de eerste 10 seconden" in c for c in report["changes"])
    assert report["after"]["seconds"] > 30  # moved back, not cut out
    # The clip is the recording after the 70 Hz low cut, sample for sample: find where each pop landed.
    clip = sf.read(tmp_path / "clip.wav")[0]
    low_cut = signal.sosfiltfilt(signal.butter(4, 70.0, "highpass", fs=SR, output="sos"), x - np.mean(x))
    for t in pops:
        piece = low_cut[int((t - 0.05) * SR):int((t + 0.1) * SR)]
        landed = np.argmax(signal.correlate(clip, piece, mode="valid", method="fft")) / SR + 0.05
        assert landed > 10.5


def test_pops_all_through_the_recording_are_bad(tmp_path):
    _popped_recording(tmp_path / "raw.wav", [2.0, 5.5, 14.0, 22.0, 30.0, 37.0])
    report = process(tmp_path / "raw.wav", tmp_path / "clip.wav").as_dict()
    assert report["before"]["pops"] == 6 and report["verdict"] == "bad"
    assert any(f["level"] == "bad" and "nergens begint 10 seconden spraak zonder plof" in f["text"]
               for f in report["findings"])
    assert not any("oversturd" in f["text"] for f in report["findings"])


def test_a_voice_clipped_on_one_side_is_still_overdriven():
    # A low voice with a lot of bass that only hits the ceiling on its peaks: one side, but the energy is in the voice.
    phase = 2 * np.pi * 110.0 * np.arange(SR) / SR
    voice = sum(np.cos(k * phase) / k for k in range(1, 11))
    x = np.clip(1.3 * voice / voice.max(), -1, 1)
    assert x.min() > -0.9
    runs = _clipping(x[:, None], x - x.mean(), SR)
    assert runs and not any(r.pop for r in runs)
