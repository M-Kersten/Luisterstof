"""The hook jingle: 30 seconds, D major, 100 beats a minute. Violins carry the tune over plucked strings.

    python jingle/make_jingle.py

writes hook_jingle.wav and hook_jingle.mp3 next to this file. The instruments are recordings of real players from
VSCO 2 Community Edition (Versilian Studios, CC0); the first run fetches them into jingle/.samples (about 400 MB).
Needs scipy next to what the pipeline already installs. The same seed gives the same performance every time.

Twelve bars. Bars 1-2 state the tune in the open, bars 3-8 settle into a bed that leaves room for voices,
bars 9-11 bring the tune back and build, bar 12 lands on a held chord and a plucked button.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from engine import SR, Note, Part, Patch, convolve, db, hall, limit, midi, render_part
from scipy.signal import butter, resample_poly, sosfilt
from sfz import fetch

HERE = Path(__file__).resolve().parent
PATCHES = ["ViolinEnsSusVib", "ViolinEnsSpic", "ViolinEnsPizz", "SViolinVib", "SViolinSpic", "ViolaEnsSusVib", "ViolaEnsSpic",
           "ViolaEnsPizz", "CelloEnsSusVib", "CelloEnsPizz", "ContrabassPizz"]

BPM = 100.0
BEAT = 60.0 / BPM
T0 = 0.12  # air before the first note: a bow starts moving before the beat
LENGTH = 30.0
SEED = 4

# one chord per half bar: name -> (bass, fifth below/above for the bass, chug low note, chug pair, arpeggio)
CHORDS = {
    "D":   dict(bass="D2", fifth="A2", low="A3", pair=("D4", "F#4"), arp=("D4", "A4", "F#4", "A4", "D5", "A4", "F#4", "A4")),
    "G":   dict(bass="G2", fifth="D3", low="G3", pair=("B3", "D4"), arp=("D4", "B4", "G4", "B4", "D5", "B4", "G4", "B4")),
    "Bm":  dict(bass="B2", fifth="F#2", low="F#3", pair=("B3", "D4"), arp=("D4", "B4", "F#4", "B4", "D5", "B4", "F#4", "B4")),
    "A":   dict(bass="A2", fifth="E2", low="A3", pair=("C#4", "E4"), arp=("C#4", "A4", "E4", "A4", "C#5", "A4", "E4", "A4")),
    "Em":  dict(bass="E2", fifth="B2", low="G3", pair=("B3", "E4"), arp=("E4", "B4", "G4", "B4", "E5", "B4", "G4", "B4")),
    "D/A": dict(bass="A2", fifth="E2", low="A3", pair=("D4", "F#4"), arp=("D4", "A4", "F#4", "A4", "D5", "A4", "F#4", "A4")),
    "A7":  dict(bass="A2", fifth="E2", low="A3", pair=("C#4", "G4"), arp=("C#4", "A4", "G4", "A4", "C#5", "A4", "E4", "A4")),
}
# bars 1..12, two chords per bar
BARS = [("D", "D"), ("G", "G"), ("Bm", "Bm"), ("A", "A"),
        ("D", "D"), ("G", "G"), ("Em", "Em"), ("A", "A"),
        ("Bm", "Bm"), ("G", "G"), ("A", "D/A"), ("D", "D")]
# the last eighth of each bar in the bass: a step into the next root
APPROACH = ["F#2", "A2", "B2", "C#2", "F#2", "F#2", "G2", "C#3", "A2", "B2", "C#2", None]

# the tune: (bar, beat, beats, pitch, bowing, velocity). 's' short bow, 'l' long bow, '-' long bow slurred from the last
TUNE = [
    (1, 0.0, 0.5, "A4", "s", 88), (1, 0.5, 0.5, "D5", "s", 92), (1, 1.0, 1.5, "F#5", "l", 96), (1, 2.5, 0.5, "E5", "-", 88), (1, 3.0, 1.0, "D5", "-", 88),
    (2, 0.0, 0.5, "B4", "s", 88), (2, 0.5, 0.5, "D5", "s", 92), (2, 1.0, 1.5, "G5", "l", 98), (2, 2.5, 0.5, "F#5", "-", 90), (2, 3.0, 1.0, "D5", "-", 86),
    (3, 0.0, 0.5, "B4", "s", 84), (3, 0.5, 0.5, "D5", "s", 88), (3, 1.0, 1.0, "F#5", "l", 92), (3, 2.0, 1.0, "A5", "-", 94), (3, 3.0, 1.0, "F#5", "-", 88),
    (4, 0.0, 3.0, "E5", "-", 86),
    # the bed: long notes, soft, out of the way
    (5, 0.0, 2.0, "A5", "l", 58), (5, 2.0, 2.0, "F#5", "-", 56),
    (6, 0.0, 3.0, "G5", "-", 58), (6, 3.0, 1.0, "E5", "-", 55),
    (7, 0.0, 2.0, "E5", "l", 57), (7, 2.0, 2.0, "G5", "-", 60),
    (8, 0.0, 3.0, "E5", "-", 61),
    # the tune again, growing
    (9, 0.0, 0.5, "B4", "s", 90), (9, 0.5, 0.5, "D5", "s", 94), (9, 1.0, 1.5, "F#5", "l", 98), (9, 2.5, 0.5, "E5", "-", 92), (9, 3.0, 1.0, "D5", "-", 92),
    (10, 0.0, 0.5, "B4", "s", 92), (10, 0.5, 0.5, "D5", "s", 96), (10, 1.0, 1.5, "G5", "l", 100), (10, 2.5, 0.5, "A5", "-", 98), (10, 3.0, 1.0, "B5", "-", 102),
    (11, 0.0, 1.5, "A5", "-", 104), (11, 1.5, 0.5, "G5", "-", 98), (11, 2.0, 1.0, "F#5", "-", 100), (11, 3.0, 1.0, "E5", "-", 98),
    (12, 0.0, 1.9, "D5", "-", 102),
]
# a second voice under the tune where it grows (violas)
SECOND = [
    (9, 1.0, 1.5, "A4", 80), (9, 2.5, 0.5, "G4", 76), (9, 3.0, 1.0, "F#4", 78),
    (10, 1.0, 1.5, "B4", 84), (10, 2.5, 1.5, "D5", 86),
    (11, 0.0, 1.5, "C#5", 88), (11, 1.5, 0.5, "B4", 84), (11, 2.0, 1.0, "A4", 86), (11, 3.0, 1.0, "G4", 84),
    (12, 0.0, 1.9, "F#4", 88),
]
# held chords under everything from bar 3: (bar, cello low, cello high, viola)
PAD = [
    (3, "B2", "F#3", "D4"), (4, "A2", "E3", "C#4"), (5, "D3", "A3", "F#4"), (6, "G2", "D3", "G4"),
    (7, "E3", "B3", "G4"), (8, "A2", "E3", "E4"), (9, "B2", "F#3", "D4"), (10, "G2", "D3", "D4"), (11, "A2", "E3", "E4"),
]


def at(bar: int, beat: float) -> float:
    return T0 + ((bar - 1) * 4 + beat) * BEAT


def level(bar: int) -> float:
    """How loud the accompaniment plays, bar by bar (velocity offset)."""
    return {1: 0, 2: 0, 3: -2, 4: -2, 5: -6, 6: -6, 7: -5, 8: -3, 9: 0, 10: 3, 11: 6, 12: 6}[bar]


def build(lib: Path) -> list[Part]:
    vln_sus = Part("violins long", Patch(lib, "ViolinEnsSusVib", "sus"), gain_db=0.0, pan=-0.25, send_db=-9, lead_s=0.035)
    vln_spic = Part("violins short", Patch(lib, "ViolinEnsSpic", "short"), gain_db=-2.0, pan=-0.25, send_db=-10)
    solo_sus = Part("first desk long", Patch(lib, "SViolinVib", "sus"), gain_db=-7.0, pan=-0.1, send_db=-8, lead_s=0.03, detune=2.0)
    solo_spic = Part("first desk short", Patch(lib, "SViolinSpic", "short"), gain_db=-9.0, pan=-0.1, send_db=-9)
    vln_pizz = Part("violins plucked", Patch(lib, "ViolinEnsPizz", "short"), gain_db=-1.0, pan=-0.35, send_db=-10, jitter_s=0.007)
    vla_spic = Part("violas short", Patch(lib, "ViolaEnsSpic", "short", near=1), gain_db=-6.0, pan=0.25, send_db=-11, jitter_s=0.007)
    vla_sus = Part("violas long", Patch(lib, "ViolaEnsSusVib", "sus", near=1), gain_db=-8.0, pan=0.2, send_db=-9, lead_s=0.04)
    vla_pizz = Part("violas plucked", Patch(lib, "ViolaEnsPizz", "short", near=1), gain_db=-4.0, pan=0.2, send_db=-10)
    vc_sus = Part("cellos long", Patch(lib, "CelloEnsSusVib", "sus", near=1), gain_db=-10.0, pan=0.3, send_db=-10, lead_s=0.04)
    vc_pizz = Part("cellos plucked", Patch(lib, "CelloEnsPizz", "short", near=1), gain_db=-5.0, pan=0.3, send_db=-12)
    bass = Part("bass plucked", Patch(lib, "ContrabassPizz", "short"), gain_db=-2.0, pan=0.1, send_db=-16, jitter_s=0.004)

    # the tune
    for i, (bar, beat, beats, pitch, bow, vel) in enumerate(TUNE):
        start, p = at(bar, beat), midi(pitch)
        if bow == "s":
            vln_spic.add(Note(start, 0, p, vel))
            solo_spic.add(Note(start, 0, p, vel))
            continue
        nxt = TUNE[i + 1] if i + 1 < len(TUNE) else None
        slur_out = nxt is not None and nxt[4] == "-" and abs(at(nxt[0], nxt[1]) - (start + beats * BEAT)) < 1e-6
        dur = beats * BEAT + (0.03 if slur_out else -0.05)
        fresh = bow == "l"
        long = beats >= 2
        vln_sus.add(Note(start, dur, p, vel, attack=0.075 if fresh else 0.055, release=0.17 if slur_out else 0.3,
                         speak=0.5 if fresh else 0.6, swell=2.0 if long else 1.0, fall=0.0 if slur_out else 3.0))
        if vel >= 63:  # in the bed the section plays alone
            solo_sus.add(Note(start, dur, p, vel, attack=0.06 if fresh else 0.045, release=0.15 if slur_out else 0.28,
                              speak=0.45 if fresh else 0.6, swell=1.5 if long else 0.5, fall=0.0 if slur_out else 3.0))
    for bar, beat, beats, pitch, vel in SECOND:
        vla_sus.add(Note(at(bar, beat), beats * BEAT + 0.03, midi(pitch), vel + 8, attack=0.07, release=0.2, speak=0.55, swell=1.0))

    for bar in range(1, 12):
        first, second = BARS[bar - 1]
        lv = level(bar)
        # plucked violins: eighth notes through the chord
        for k in range(8):
            chord = CHORDS[first if k < 4 else second]
            vel = 74 + lv + (9 if k == 0 else 4 if k == 4 else 0) - (3 if k % 2 else 0)
            vln_pizz.add(Note(at(bar, k * 0.5), 0, midi(chord["arp"][k]), vel))
        # bass: on one, a push before three, the fifth on three, a step into the next bar
        chord, late = CHORDS[first], CHORDS[second]
        root, fifth = midi(chord["bass"]), midi(late["fifth"])
        line = [(0.0, root, 98), (1.5, root, 70), (2.0, fifth if first == second else midi(late["bass"]), 86)]
        if APPROACH[bar - 1]:
            line.append((3.5, midi(APPROACH[bar - 1]), 72))
        if bar == 11:
            line.insert(3, (3.0, root, 84))
        for beat, p, vel in line:
            bass.add(Note(at(bar, beat), 0, p, vel + lv, damp=0.9))
            vc_pizz.add(Note(at(bar, beat), 0, p + 12, vel + lv - 6, damp=0.8))
        # violas, short bows: low note on the accents (3+3+2), the pair in between
        if bar >= 3:
            for k in range(8):
                chord = CHORDS[first if k < 4 else second]
                accent = k in (0, 3, 6)
                vel = (80 if accent else 58) + lv
                if accent:
                    vla_spic.add(Note(at(bar, k * 0.5), 0, midi(chord["low"]), vel))
                else:
                    for pitch in chord["pair"]:
                        vla_spic.add(Note(at(bar, k * 0.5), 0, midi(pitch), vel - 3))

    # held chords
    for i, (bar, low, high, alto) in enumerate(PAD):
        last = i == len(PAD) - 1
        dur = 4 * BEAT + (0.0 if last else 0.08)
        vel = 55 + level(bar)  # soft bows
        for pitch, part in ((low, vc_sus), (high, vc_sus), (alto, vla_sus)):
            part.add(Note(at(bar, 0), dur, midi(pitch), vel, attack=0.25, release=0.35, speak=0.35, swell=1.5))

    # bar 12: the chord, held and let go, then one pluck
    hit, button = at(12, 0), at(12, 2)
    for pitch, part, vel in (("D3", vc_sus, 96), ("A3", vc_sus, 92), ("A4", vla_sus, 92), ("F#5", vln_sus, 90)):
        part.add(Note(hit, 1.9 * BEAT, midi(pitch), vel, attack=0.06, release=0.3, speak=0.5, swell=1.0, fall=4.0))
    bass.add(Note(hit, 0, midi("D2"), 96, damp=1.1))
    vc_pizz.add(Note(hit, 0, midi("D3"), 98, damp=1.0))
    for pitch in ("D4", "A4", "D5"):
        vln_pizz.add(Note(hit, 0, midi(pitch), 92))
    for pitch, part, vel in (("D2", bass, 98), ("D3", vc_pizz, 100), ("A3", vc_pizz, 94), ("A3", vla_pizz, 96), ("D4", vla_pizz, 96),
                             ("F#4", vln_pizz, 98), ("A4", vln_pizz, 98), ("D5", vln_pizz, 102)):
        part.add(Note(button, 0, midi(pitch), vel, damp=1.3 if part is bass else 1.6 if part is vc_pizz else None))
    return [vln_sus, solo_sus, vln_spic, solo_spic, vln_pizz, vla_spic, vla_sus, vla_pizz, vc_sus, vc_pizz, bass]


def lufs(x: np.ndarray) -> float:
    return float(pyln.Meter(SR).integrated_loudness(x.astype(np.float64)))


def true_peak_db(x: np.ndarray) -> float:
    return float(20 * np.log10(np.abs(resample_poly(x, 4, 1, axis=0)).max() + 1e-12))


def master(parts: list[Part], stems: dict[str, np.ndarray], target_lufs: float = -16.0) -> tuple[np.ndarray, dict]:
    dry = np.sum([stems[p.name] for p in parts], axis=0)
    send = np.sum([stems[p.name] * db(p.send_db) for p in parts], axis=0)
    mix = dry + convolve(send, hall())
    mix = sosfilt(butter(2, 48, "highpass", fs=SR, output="sos"), mix, axis=0)
    n = int(LENGTH * SR)
    mix = mix[:n]
    out = np.zeros((n, 2), dtype=np.float32)
    out[: len(mix)] = mix
    f = int(0.5 * SR)  # the hall dies away before the file ends
    out[-f:] *= (np.linspace(1.0, 0.0, f, dtype=np.float32) ** 1.5)[:, None]
    out *= db(target_lufs - lufs(out))
    out, reduction = limit(out, ceiling_db=-1.5)
    return out, {"lufs": lufs(out), "lufs_mono": lufs(out.mean(axis=1)), "true_peak_db": true_peak_db(out),
                 "limiter_db": reduction, "seconds": len(out) / SR}


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    cli.add_argument("--samples", type=Path, default=HERE / ".samples", help="where the sample library lives")
    cli.add_argument("--out", type=Path, default=HERE, help="folder for hook_jingle.wav and hook_jingle.mp3")
    cli.add_argument("--stems", action="store_true", help="also write each instrument on its own (stems.npz)")
    args = cli.parse_args()
    fetch(args.samples, PATCHES)
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    parts = build(args.samples)
    stems = {p.name: render_part(p, LENGTH + 3.0, rng) for p in parts}
    out, report = master(parts, stems)
    rng = np.random.default_rng(SEED)
    dither = (rng.random(out.shape) - rng.random(out.shape)) / 32768.0
    sf.write(args.out / "hook_jingle.wav", out + dither.astype(np.float32), SR, subtype="PCM_16")
    sf.write(args.out / "hook_jingle.mp3", out, SR, format="MP3")
    if args.stems:
        np.savez_compressed(args.out / "stems.npz", **{k: v[: int(LENGTH * SR)] for k, v in stems.items()})
    for key, value in report.items():
        print(f"{key}: {value:.2f}")


if __name__ == "__main__":
    main()
