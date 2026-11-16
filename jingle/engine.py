"""A small sample player for string patches: sustained bows, short bows, plucks; buses, a hall, a master."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, fftconvolve, resample_poly, sosfilt, sosfiltfilt
from sfz import Region, parse

SR = 44100
NOTE_INDEX = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def midi(name: str | int) -> int:
    """'F#5' -> 78 (C4 = 60)."""
    if isinstance(name, int):
        return name
    letter, rest = name[0].upper(), name[1:]
    acc = 0
    while rest and rest[0] in "#b":
        acc += 1 if rest[0] == "#" else -1
        rest = rest[1:]
    return 12 * (int(rest) + 1) + NOTE_INDEX[letter] + acc


def db(x: float) -> float:
    return float(10.0 ** (x / 20.0))


def _smooth(x: np.ndarray, n: int) -> np.ndarray:
    n = max(1, n)
    c = np.cumsum(np.concatenate([[0.0], x.astype(np.float64)]))
    out = (c[n:] - c[:-n]) / n
    return np.concatenate([np.full(n - 1, out[0] if len(out) else 0.0), out]) if len(out) else np.zeros_like(x)


def _cents_off(mono: np.ndarray, f0: float) -> float | None:
    """How far the recording sits from f0, from its first harmonics. None when it cannot be told."""
    if len(mono) < SR // 8:
        return None
    nfft = 1 << int(np.ceil(np.log2(len(mono) * 8)))
    spec = np.abs(np.fft.rfft(mono * np.hanning(len(mono)), n=nfft))
    freqs = np.fft.rfftfreq(nfft, 1 / SR)
    found = []
    for h in (1, 2, 3, 4):
        target = f0 * h
        if target > SR / 2.5:
            break
        band = np.where((freqs > target * 0.955) & (freqs < target * 1.045))[0]
        if len(band) < 5:
            continue
        k = band[np.argmax(spec[band])]
        if k in (band[0], band[-1]):
            continue  # the peak sits on the edge of the window: not this harmonic
        a, b, c = spec[k - 1], spec[k], spec[k + 1]
        shift = 0.5 * (a - c) / (a - 2 * b + c + 1e-12)
        fpk = (k + shift) * SR / nfft
        found.append((1200 * np.log2(fpk / target), b))
    if not found:
        return None
    cents = np.array([f[0] for f in found])
    weights = np.array([f[1] for f in found])
    centre = float(np.sum(cents * weights) / np.sum(weights))
    if np.max(np.abs(cents - centre)) > 18 and len(found) > 1:  # harmonics disagree: trust the strongest
        centre = float(cents[np.argmax(weights)])
    return centre


@dataclass
class Sample:
    data: np.ndarray  # (n, 2) float32: the near microphone, and what the far one adds as width
    onset: int  # where the note starts to sound
    env: np.ndarray  # 50 ms RMS envelope of the mono sum
    steady: float  # level of the held part
    cents: float  # tuning error of the recording against its nominal key


class Patch:
    """One articulation of one instrument."""

    def __init__(self, root: Path, name: str, kind: str, near: int = 0):
        # kind: 'sus' or 'short'. near: the channel of the microphone closest to the players. The two microphones
        # of these recordings stand far apart, and added together they cancel part of the sound; so the near one
        # is the instrument, and the difference with the other is kept apart as width that vanishes in mono.
        self.root, self.name, self.kind, self.near = root, name, kind, near
        self.regions = parse(root / f"{name}.sfz")
        self._rr: dict[tuple[int, int], int] = {}
        self._cache: dict[str, Sample] = {}

    def pick(self, note: int, vel: int) -> Region:
        fits = [r for r in self.regions if r.lokey <= note <= r.hikey]
        if not fits:  # outside the patch: the nearest recording, stretched
            nearest = min(self.regions, key=lambda r: abs(r.center - note)).center
            fits = [r for r in self.regions if r.center == nearest]
        layer = [r for r in fits if r.lovel <= vel <= r.hivel] or fits
        narrow = min(r.hivel - r.lovel for r in layer)
        layer = [r for r in layer if r.hivel - r.lovel == narrow]
        layer.sort(key=lambda r: (r.seq_position, r.sample))
        key = (layer[0].center, layer[0].lovel)
        turn = self._rr.get(key, 0)
        self._rr[key] = turn + 1
        return layer[turn % len(layer)]

    def sample(self, region: Region) -> Sample:
        hit = self._cache.get(region.sample)
        if hit is not None:
            return hit
        data, sr = sf.read(self.root / region.sample, dtype="float32", always_2d=True)
        assert sr == SR, (region.sample, sr)
        if data.shape[1] == 1:
            data = np.repeat(data, 2, axis=1)
        mono, far = data[:, self.near], data[:, 1 - self.near]
        shared = float(np.dot(far, mono) / (np.dot(mono, mono) + 1e-12))
        data = np.stack([mono, far - shared * mono], axis=1)  # the width holds nothing the near microphone has
        fast = np.sqrt(_smooth(mono**2, int(0.002 * SR)))
        onset = int(np.argmax(fast > 0.06 * fast.max()))
        env = np.sqrt(_smooth(mono**2, int(0.05 * SR)))
        f0 = 440.0 * 2 ** ((region.center - 69) / 12)
        if self.kind == "sus":
            a, b = int(len(mono) * 0.3), int(len(mono) * 0.7)
            steady = float(np.median(env[a:b]))
            cents = _cents_off(mono[a:b], f0)
        else:
            steady = float(env.max())
            a = onset + int(0.03 * SR)
            cents = _cents_off(mono[a: a + int(0.35 * SR)], f0)
        cents = 0.0 if cents is None else float(np.clip(cents, -35, 35))
        smp = Sample(data, onset, env, steady, cents)
        self._cache[region.sample] = smp
        return smp


def _pitch(seg: np.ndarray, ratio: float) -> np.ndarray:
    """Play ``seg`` faster by ``ratio`` (the tape way: pitch and length move together)."""
    if abs(ratio - 1.0) < 2e-4:
        return seg
    frac = Fraction(1.0 / ratio).limit_denominator(1500)
    return resample_poly(seg, frac.numerator, frac.denominator, axis=0).astype(np.float32)


def vel_gain(vel: float) -> float:
    return db((vel - 90.0) * 0.2)


@dataclass
class Note:
    start: float  # seconds
    dur: float  # seconds (sustains); shorts ring out
    pitch: int
    vel: float = 90.0
    attack: float = 0.06
    release: float = 0.2
    speak: float = 0.5  # how far into the bow stroke the recording starts (share of the held level)
    swell: float = 0.0  # dB the note grows and falls back over its length
    fall: float = 0.0  # dB the note has lost by its end
    damp: float | None = None  # shorts: stop the ring after this many seconds
    layer: float | None = None  # the written dynamic: picks soft or loud recordings, whatever vel ends up being


def render_sustain(patch: Patch, n: Note, cents: float = 0.0) -> np.ndarray:
    region = patch.pick(n.pitch, int(n.layer if n.layer is not None else n.vel))
    smp = patch.sample(region)
    ratio = 2 ** (((n.pitch - region.center) * 100 - smp.cents + cents) / 1200)
    start = int(np.argmax(smp.env >= n.speak * smp.steady))
    start = max(0, start - int(0.03 * SR))
    want = int((n.dur + n.release + 0.05) * SR)
    src = smp.data[start: start + int(want * ratio) + 64]
    seg = _pitch(src, ratio)[:want]
    if len(seg) < want:  # the bow ran out: the note ends when the recording does
        want = len(seg)
    t = np.arange(want) / SR
    body = max(1e-3, want / SR - n.release)
    env = np.ones(want, dtype=np.float64)
    a = min(int(n.attack * SR), want)
    env[:a] = 0.5 - 0.5 * np.cos(np.pi * np.arange(a) / max(1, a))
    r = want - int(body * SR)
    if r > 0:
        env[-r:] *= (0.5 + 0.5 * np.cos(np.pi * np.arange(r) / r)) ** 1.5
    shape = np.zeros(want)
    if n.swell:
        shape += n.swell * np.sin(np.pi * np.clip(t / body, 0, 1)) ** 1.2 - n.swell * 0.5
    if n.fall:
        shape -= n.fall * np.clip(t / body, 0, 1)
    env *= 10 ** (shape / 20)
    held = seg[int(0.03 * SR): max(int(0.03 * SR) + 1, int(body * SR)), 0]
    level = float(np.sqrt(np.mean(held.astype(np.float64) ** 2)) + 1e-9)
    gain = min(0.05 / level, db(40)) * vel_gain(n.vel)
    return (seg * (env * gain)[:, None]).astype(np.float32)


def render_short(patch: Patch, n: Note, cents: float = 0.0) -> tuple[np.ndarray, float]:
    """The note, and the seconds it takes to speak (a short bow needs longer than a pluck)."""
    region = patch.pick(n.pitch, int(n.layer if n.layer is not None else n.vel))
    smp = patch.sample(region)
    ratio = 2 ** (((n.pitch - region.center) * 100 - smp.cents + cents) / 1200)
    start = max(0, smp.onset - int(0.004 * SR))
    seg = _pitch(smp.data[start: start + int(3.5 * SR * ratio)], ratio).copy()
    env = np.sqrt(_smooth(seg[:, 0].astype(np.float64) ** 2, int(0.02 * SR)))
    peak = float(env.max()) + 1e-9
    # the ring is over where the envelope has fallen 46 dB: fade there so room noise does not pile up
    below = np.where(env[int(np.argmax(env)):] < peak * db(-46))[0]
    end = int(np.argmax(env)) + (int(below[0]) if len(below) else len(seg))
    end = min(len(seg), end + int(0.08 * SR))
    if n.damp is not None:
        end = min(end, int(n.damp * SR))
    seg = seg[:end]
    f = min(len(seg), int(0.09 * SR))
    seg[-f:] *= np.linspace(1.0, 0.0, f, dtype=np.float32)[:, None] ** 2
    a = min(len(seg), int(0.003 * SR))
    seg[:a] *= np.linspace(0.0, 1.0, a, dtype=np.float32)[:, None]
    first = seg[: int(0.3 * SR), 0].astype(np.float64)
    loud = float(np.sqrt(np.mean(first**2)) + 1e-9)
    gain = min(0.04 / loud, db(40)) * vel_gain(n.vel)
    quick = np.sqrt(_smooth(seg[: int(0.3 * SR), 0].astype(np.float64) ** 2, int(0.008 * SR)))
    speaks = float(np.argmax(quick >= 0.5 * quick.max())) / SR if len(quick) else 0.0
    return (seg * gain).astype(np.float32), min(speaks, 0.12)


@dataclass
class Part:
    name: str
    patch: Patch
    gain_db: float = 0.0
    pan: float = 0.0  # -1 left .. 1 right
    width: float = 0.5  # how much of the far microphone is heard as width
    send_db: float = -12.0  # how much of it goes to the hall
    lead_s: float = 0.0  # play this much early (slow bows speak late)
    jitter_s: float = 0.006
    vel_jitter: float = 5.0
    detune: float = 3.0  # cents, per note
    notes: list[Note] = field(default_factory=list)

    def add(self, *notes: Note) -> None:
        self.notes.extend(notes)


def render_part(part: Part, total_s: float, rng: np.random.Generator) -> np.ndarray:
    out = np.zeros((int(total_s * SR), 2), dtype=np.float32)
    for n in sorted(part.notes, key=lambda n: n.start):
        played = Note(**{**n.__dict__, "layer": n.vel, "vel": float(np.clip(n.vel + rng.normal(0, part.vel_jitter), 20, 127))})
        cents = float(rng.normal(0, part.detune))
        if part.patch.kind == "sus":
            seg, speaks = render_sustain(part.patch, played, cents), part.lead_s
        else:
            seg, speaks = render_short(part.patch, played, cents)
        at = int((n.start - speaks + rng.normal(0, part.jitter_s)) * SR)
        if at < 0:
            seg, at = seg[-at:], 0
        end = min(len(out), at + len(seg))
        if end > at:
            out[at:end] += seg[: end - at]
    left = np.cos((part.pan + 1) * np.pi / 4) * np.sqrt(2)
    right = np.sin((part.pan + 1) * np.pi / 4) * np.sqrt(2)
    mid, side = out[:, 0], out[:, 1] * part.width
    if part.patch.near == 0:
        side = -side  # the far microphone stood on the other side of the players
    stereo = np.stack([mid * left + side, mid * right - side], axis=1)  # in mono the width cancels and the mid stays
    return (stereo * db(part.gain_db)).astype(np.float32)


def hall(seconds: float = 2.4, seed: int = 11) -> np.ndarray:
    """A medium hall: a few early reflections and a tail that loses its top first."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    bands = [(None, 250, 2.0), (250, 1000, 1.8), (1000, 3500, 1.4), (3500, 9000, 0.9), (9000, None, 0.5)]
    ir = np.zeros((n, 2))
    for ch in range(2):
        noise = rng.standard_normal(n)
        for lo, hi, rt in bands:
            if lo is None:
                sos = butter(2, hi, "lowpass", fs=SR, output="sos")
            elif hi is None:
                sos = butter(2, lo, "highpass", fs=SR, output="sos")
            else:
                sos = butter(2, [lo, hi], "bandpass", fs=SR, output="sos")
            ir[:, ch] += sosfiltfilt(sos, noise) * 10 ** (-3 * t / rt)
    build = np.clip(t / 0.06, 0, 1) ** 1.5  # the tail needs a moment to become dense
    ir *= build[:, None]
    ir /= np.sqrt((ir**2).sum(axis=0, keepdims=True))
    early = np.zeros((n, 2))
    for i, (ms, g) in enumerate([(13, 0.55), (19, 0.48), (27, 0.42), (34, 0.36), (43, 0.3), (55, 0.25), (68, 0.2)]):
        ch = i % 2
        early[int(ms / 1000 * SR), ch] += g
        early[int((ms + 3.1) / 1000 * SR), 1 - ch] += g * 0.45
    sos = butter(1, 6000, "lowpass", fs=SR, output="sos")
    early = sosfilt(sos, early, axis=0)
    early /= np.sqrt((early**2).sum(axis=0, keepdims=True)) * 2.2
    ir = ir + early
    pre = int(0.012 * SR)
    ir = np.concatenate([np.zeros((pre, 2)), ir])[:n]
    return ir.astype(np.float32)


def convolve(x: np.ndarray, ir: np.ndarray) -> np.ndarray:
    out = np.stack([fftconvolve(x[:, c], ir[:, c]) for c in range(2)], axis=1)
    return out[: len(x)].astype(np.float32)


def limit(x: np.ndarray, ceiling_db: float = -1.0, lookahead_s: float = 0.004, release_s: float = 0.12) -> tuple[np.ndarray, float]:
    """Look-ahead limiter on both channels together. Returns the audio and the deepest reduction in dB."""
    ceiling = db(ceiling_db)
    peak = np.abs(x).max(axis=1)
    need = np.minimum(1.0, ceiling / np.maximum(peak, 1e-9))
    look = int(lookahead_s * SR)
    # hold the lowest gain over the look-ahead window, then let it recover slowly
    from scipy.ndimage import minimum_filter1d

    held = minimum_filter1d(need, size=2 * look + 1, mode="nearest")
    gain = np.empty_like(held)
    coeff = np.exp(-1.0 / (release_s * SR))
    g = 1.0
    for i, h in enumerate(held):
        g = h if h < g else 1.0 - (1.0 - g) * coeff if h >= 1.0 else min(h, 1.0 - (1.0 - g) * coeff)
        gain[i] = g
    gain = np.minimum(_smooth(gain, look), held)
    return (x * gain[:, None]).astype(np.float32), float(20 * np.log10(gain.min()))
