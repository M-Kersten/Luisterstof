"""Reference clip processing on synthetic speech: voiced 'words' with a pitch contour, gaps and pauses."""

import numpy as np
import pytest
import soundfile as sf

from pipeline.audio.refclip import ClipError, process

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


def test_too_short_or_not_audio_is_refused(tmp_path):
    rng = np.random.default_rng(4)
    sf.write(tmp_path / "short.wav", _speech([("lively", 4)], rng), SR)
    with pytest.raises(ClipError):
        process(tmp_path / "short.wav", tmp_path / "x.wav")
    (tmp_path / "junk.wav").write_bytes(b"this is not audio")
    with pytest.raises(ClipError):
        process(tmp_path / "junk.wav", tmp_path / "y.wav")
