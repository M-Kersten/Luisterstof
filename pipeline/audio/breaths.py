"""Breaths: the inhale a speaker takes before a longer sentence.

Chatterbox does not breathe, and a conversation without a single breath sounds made. The breaths come from the
speaker's own recordings (the reference clip, and the original upload it was made from when that is still there),
so they have that voice's mouth, nose and room. An inhale looks like this in a recording: after a pause, before the
voice starts, a soft stretch of noise without pitch, 0.12 to 0.7 seconds, well below the level of speech, with a
small gap before the first word. That last part and the level keep a Dutch "g" or "ch" out: those are loud and run
straight into the vowel. A breath also swells and fades smoothly; mouth clicks and creak jump from one 10 ms frame to
the next, and those stretches are left alone.

When a recording has no usable breath (a clip cut tight, a noisy room), a soft one is made: shaped noise with the
two broad resonances of an open mouth. ``source`` says which one a speaker gets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

HOP = 0.01
MIN_S, MAX_S = 0.12, 0.7
MAX_BREATHS = 8


@dataclass
class Breaths:
    clips: list[np.ndarray] = field(default_factory=list)
    sr: int = 24000
    source: str = "none"  # "opname": cut from the speaker's recordings; "nagemaakt": made up


def _frames(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per 10 ms: level in dB, spectral centroid in Hz, spectral flatness (0 tonal .. 1 noise)."""
    n, h = int(0.02 * sr), int(HOP * sr)
    if len(x) < n:
        return np.array([-120.0]), np.array([0.0]), np.array([0.0])
    starts = np.arange(0, len(x) - n, h)
    frames = np.lib.stride_tricks.sliding_window_view(x, n)[starts] * np.hanning(n)
    spec = np.abs(np.fft.rfft(frames, axis=1)) + 1e-9
    freqs = np.fft.rfftfreq(n, 1 / sr)
    power = spec**2
    db = 10 * np.log10(power.mean(axis=1) + 1e-12)
    centroid = (spec * freqs).sum(axis=1) / spec.sum(axis=1)
    band = (freqs > 300) & (freqs < 8000)
    flat = np.exp(np.log(spec[:, band]).mean(axis=1)) / spec[:, band].mean(axis=1)
    return db, centroid, flat


def find_spans(x: np.ndarray, sr: int, limit: int = MAX_BREATHS) -> list[tuple[int, int]]:
    """Where the clearest inhales are, as (start, end) samples, best first."""
    from pipeline.audio.refclip import _f0

    if len(x) < sr:
        return []
    db, centroid, flat = _frames(x, sr)
    f0 = _f0(x, sr)
    m = min(len(db), len(f0))
    db, centroid, flat, voiced = db[:m], centroid[:m], flat[:m], ~np.isnan(f0[:m])
    speech = float(np.percentile(db, 95))
    floor = float(np.percentile(db, 5))
    loud = db > speech - 18  # where a word is
    soft = (db > max(floor + 6, speech - 48)) & (db < speech - 18)
    candidate = soft & ~voiced & (centroid > 500) & (centroid < 4500) & (flat > 0.08)
    found: list[tuple[float, int, int]] = []
    i = 0
    while i < m:
        if not candidate[i]:
            i += 1
            continue
        j = i
        while j + 1 < m and (candidate[j + 1] or (j + 2 < m and candidate[j + 2])):  # bridge one-frame dips
            j += 1
        a, b = i, j + 1
        i = b
        dur = (b - a) * HOP
        if not MIN_S <= dur <= MAX_S:
            continue
        before = loud[max(0, a - 15):a]
        if before.any():  # a breath comes after a pause, not in the middle of a word
            continue
        after = np.flatnonzero(loud[b:b + 35] & voiced[b:b + 35])
        if not len(after) or after[0] < 2:  # the voice starts within 0.35 s, and not straight out of it
            continue
        level = float(np.median(db[a:b]))
        if float(np.std(np.diff(db[a:b]))) > 3.0:  # clicks and creak jump from frame to frame, a breath swells
            continue
        score = -abs(dur - 0.35) * 2 - abs(level - (speech - 28)) / 10 + float(np.median(flat[a:b]))
        found.append((score, a, b, b + int(after[0])))  # the frame where the voice comes in
    found.sort(reverse=True)
    spans = []
    pad, h = int(0.015 * sr), int(HOP * sr)
    for _, a, b, voice in found[:limit]:
        end = min(b * h + pad, voice * h - int(0.01 * sr))  # never into the word that follows
        spans.append((max(0, a * h - pad), min(len(x), end)))
    return spans


def find(x: np.ndarray, sr: int, limit: int = MAX_BREATHS) -> list[np.ndarray]:
    """The clearest inhales in a recording, best first, each with short fades."""
    clips = []
    for a, b in find_spans(x, sr, limit):
        seg = x[a:b].astype(np.float32).copy()
        nf = max(1, min(int(0.012 * sr), len(seg) // 4))
        seg[:nf] *= np.linspace(0, 1, nf, dtype=np.float32)
        seg[-nf:] *= np.linspace(1, 0, nf, dtype=np.float32)
        clips.append(seg)
    return clips


def synthetic(sr: int, seed: int = 3, seconds: float = 0.38) -> np.ndarray:
    """A soft inhale made up: noise through two broad mouth resonances, rising slowly and ending quickly."""
    import pedalboard  # type: ignore
    from scipy.signal import butter, sosfilt

    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    noise = sosfilt(butter(2, [300, min(3800, sr * 0.45)], btype="bandpass", fs=sr, output="sos"),
                    rng.standard_normal(n)).astype(np.float32)
    board = pedalboard.Pedalboard([pedalboard.PeakFilter(cutoff_frequency_hz=950, gain_db=6, q=1.2),
                                   pedalboard.PeakFilter(cutoff_frequency_hz=2200, gain_db=4, q=1.4)])
    shaped = board(noise.reshape(1, -1), sr)[0]
    t = np.linspace(0, 1, n)
    env = np.where(t < 0.65, np.sin(np.pi / 2 * t / 0.65) ** 1.5, np.cos(np.pi / 2 * (t - 0.65) / 0.35) ** 2)
    out = (shaped * env).astype(np.float32)
    return out / (np.sqrt(np.mean(out**2)) + 1e-9) * 0.01


def _recordings(cast_dir: Path, data_dir: Path, speaker_id: str, voice_ref: str | None) -> list[Path]:
    paths = []
    if voice_ref and (cast_dir / voice_ref).is_file():
        paths.append(cast_dir / voice_ref)
    uploads = sorted((data_dir / "voicelab" / speaker_id / "clips").glob("*/original.*"))
    paths += uploads[-2:]  # the newest uploads: unprocessed, with their pauses (and breaths) intact
    return paths


@lru_cache(maxsize=32)
def _from_file(path: str, mtime: float, sr: int) -> tuple[np.ndarray, ...]:
    from pipeline.audio.synth import AudioClip, resample

    clip = AudioClip.read(path)
    x = resample(clip.samples, clip.sr, sr) if clip.sr != sr else clip.samples
    return tuple(find(x, sr))


def for_speaker(cast_dir: Path | str, data_dir: Path | str, speaker_id: str, voice_ref: str | None, sr: int) -> Breaths:
    """This speaker's breaths at ``sr``: from their recordings, else a made-up one."""
    clips: list[np.ndarray] = []
    for path in _recordings(Path(cast_dir), Path(data_dir), speaker_id, voice_ref):
        try:
            clips += list(_from_file(str(path), path.stat().st_mtime, sr))
        except Exception:  # noqa: BLE001 - an unreadable recording just gives no breaths
            continue
    if clips:
        return Breaths(clips[:MAX_BREATHS], sr, "opname")
    return Breaths([synthetic(sr, seed) for seed in (3, 5, 8)], sr, "nagemaakt")
