"""The hook jingle: 30 seconds of suspense in D minor, 100 beats a minute. It asks a question and leaves it open.

    python jingle/make_jingle.py

writes hook_jingle.wav and hook_jingle.mp3 next to this file. The instruments are recordings of real players from
VSCO 2 Community Edition (Versilian Studios, CC0); the first run fetches them into jingle/.samples (about 500 MB).
Needs scipy next to what the pipeline already installs. The same seed gives the same performance every time.

Twelve bars. A low pulse in cellos and basses runs through all of it, three plus three plus two, under a ticking
pluck. Bars 1-2 are the pulse alone and a plucked question (A, D, E, upwards). Bars 3-8 the violins ask it with the
bow, each time a step higher, over tremolo violas, and stop on the dominant. Bars 9-11 everything plays and climbs
towards the home chord. Bar 12 does not give it: the basses land on A, the violins hang on the note above, and the
plucked question is the last thing heard.
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
PATCHES = ["ViolinEnsSusVib", "ViolinEnsSpic", "ViolinEnsPizz", "ViolinEnsTrem", "SViolinVib", "SViolinSpic",
           "ViolaEnsSusVib", "ViolaEnsSpic", "ViolaEnsTrem", "CelloEnsSusVib", "CelloEnsSpic", "CelloEnsPizz",
           "ContrabassSusNV", "ContrabassSpic", "ContrabassPizz"]

BPM = 100.0
BEAT = 60.0 / BPM
T0 = 0.12  # air before the first note: a bow starts moving before the beat
LENGTH = 30.0
SEED = 4

# per chord: the pulse (cellos: a low note on the accents, a higher one between; violas: a pair between the accents),
# the note the basses pluck and bow, and the two notes the violins tick on
CHORDS = {
    "Dm":   dict(low="D2", mid="D3", pair=("A3", "D4"), bass="D2", tick=("A4", "D5")),
    "Dm3":  dict(low="D2", mid="D3", pair=("A3", "F4"), bass="D2", tick=("A4", "D5")),
    "Bb":   dict(low="Bb2", mid="F3", pair=("Bb3", "D4"), bass="Bb1", tick=("Bb4", "D5")),
    "C":    dict(low="C3", mid="G3", pair=("C4", "E4"), bass="C2", tick=("C5", "E5")),
    "Gm":   dict(low="G2", mid="D3", pair=("G3", "Bb3"), bass="G1", tick=("Bb4", "D5")),
    "Asus": dict(low="A2", mid="E3", pair=("A3", "D4"), bass="A1", tick=("A4", "D5")),
    "A":    dict(low="A2", mid="E3", pair=("A3", "C#4"), bass="A1", tick=("A4", "C#5")),
}
# bars 1..12, two chords per bar
BARS = [("Dm", "Dm"), ("Dm", "Dm"), ("Dm3", "Dm3"), ("Dm3", "Dm3"),
        ("Bb", "Bb"), ("Bb", "C"), ("Gm", "Gm"), ("Asus", "A"),
        ("Dm3", "Dm3"), ("Bb", "Bb"), ("C", "C"), ("Asus", "Asus")]
ACCENTS = (0, 3, 6)  # of the eight eighths in a bar: three, three, two

# the question, plucked: (beat, pitch, velocity)
QUESTION = [(2.0, "A4", 80), (2.5, "D5", 86), (3.0, "E5", 92)]

# the tune: (bar, beat, beats, pitch, bowing, velocity). 's' short bow, 'l' long bow, '-' long bow slurred from the last
TUNE = [
    (3, 0.5, 0.5, "D5", "s", 84), (3, 1.0, 0.5, "E5", "s", 88), (3, 1.5, 1.5, "F5", "l", 90), (3, 3.0, 1.0, "A5", "-", 94),
    (4, 0.0, 1.5, "G5", "-", 90), (4, 1.5, 2.2, "E5", "-", 84),
    (5, 0.5, 0.5, "D5", "s", 86), (5, 1.0, 0.5, "F5", "s", 90), (5, 1.5, 1.5, "G5", "l", 92), (5, 3.0, 1.0, "Bb5", "-", 96),
    (6, 0.0, 2.0, "A5", "-", 94), (6, 2.0, 1.0, "G5", "-", 90), (6, 3.0, 0.9, "E5", "-", 88),
    (7, 0.5, 0.5, "G5", "s", 90), (7, 1.0, 0.5, "A5", "s", 94), (7, 1.5, 2.5, "Bb5", "l", 98),
    (8, 0.0, 1.0, "A5", "-", 98), (8, 1.0, 1.0, "G5", "-", 96), (8, 2.0, 1.9, "E5", "-", 98),
    (9, 0.5, 0.5, "D5", "s", 98), (9, 1.0, 0.5, "E5", "s", 100), (9, 1.5, 1.5, "F5", "l", 102), (9, 3.0, 1.0, "A5", "-", 104),
    (10, 0.0, 2.0, "Bb5", "-", 104), (10, 2.0, 1.0, "A5", "-", 100), (10, 3.0, 1.0, "F5", "-", 98),
    (11, 0.0, 1.0, "G5", "-", 100), (11, 1.0, 1.0, "A5", "-", 104), (11, 2.0, 2.0, "C6", "-", 108),
    (12, 0.0, 3.4, "D6", "-", 100),
]
GROW = {(4, 1.5): 5.0, (8, 2.0): -4.0, (11, 2.0): -5.0, (12, 0.0): 14.0}  # dB a note loses over its length (minus: gains)

# violas, tremolo: the unrest under the question. (bar, beat, beats, pitches)
SHIVER = [
    (3, 0, 4, ("A3", "F4")), (4, 0, 2, ("A3", "F4")), (4, 2, 2, ("A3", "E4")),
    (5, 0, 4, ("Bb3", "F4")), (6, 0, 2, ("Bb3", "F4")), (6, 2, 2, ("C4", "G4")),
    (7, 0, 4, ("Bb3", "G4")), (8, 0, 2, ("A3", "D4")), (8, 2, 2, ("A3", "C#4")),
]
# violas, long bows, when everything plays
INNER = [(9, 0, 4, ("A3", "F4")), (10, 0, 4, ("Bb3", "F4")), (11, 0, 4, ("C4", "G4"))]
# cellos, long bows: (bar, beat, beats, pitch, velocity)
CELLO_LINE = [
    (5, 0, 4, "Bb2", 80), (6, 0, 2, "Bb2", 80), (6, 2, 2, "C3", 84), (7, 0, 4, "G2", 84), (8, 0, 4, "A2", 90),
    (9, 0, 2, "D3", 96), (9, 2, 1, "F3", 94), (9, 3, 1, "A3", 96), (10, 0, 2, "Bb2", 98), (10, 2, 1, "D3", 96), (10, 3, 1, "F3", 98),
    (11, 0, 2, "C3", 100), (11, 2, 1, "E3", 100), (11, 3, 1, "G3", 104),
]
# basses, bowed: the floor. (bar, beat, beats, pitch, velocity)
FLOOR = [
    (1, 0, 8, "D2", 56), (3, 0, 8, "D2", 60), (5, 0, 6, "Bb1", 70), (6, 2, 2, "C2", 72), (7, 0, 4, "G1", 74), (8, 0, 4, "A1", 80),
    (9, 0, 4, "D2", 90), (10, 0, 4, "Bb1", 90), (11, 0, 4, "C2", 94),
]


def at(bar: int, beat: float) -> float:
    return T0 + ((bar - 1) * 4 + beat) * BEAT


def level(bar: int) -> float:
    """How loud the pulse plays, bar by bar (velocity offset): it never stops growing until the last bar."""
    return {1: -5, 2: -4, 3: -4, 4: -3, 5: -3, 6: -2, 7: 0, 8: 3, 9: 5, 10: 5, 11: 8, 12: 4}[bar]


def build(lib: Path) -> list[Part]:
    vln_sus = Part("violins long", Patch(lib, "ViolinEnsSusVib", "sus"), gain_db=0.0, pan=-0.25, send_db=-8, lead_s=0.035)
    vln_spic = Part("violins short", Patch(lib, "ViolinEnsSpic", "short"), gain_db=-2.0, pan=-0.25, send_db=-10)
    solo_sus = Part("first desk long", Patch(lib, "SViolinVib", "sus"), gain_db=-7.0, pan=-0.1, send_db=-8, lead_s=0.03, detune=2.0)
    solo_spic = Part("first desk short", Patch(lib, "SViolinSpic", "short"), gain_db=-9.0, pan=-0.1, send_db=-9)
    vln_pizz = Part("violins plucked", Patch(lib, "ViolinEnsPizz", "short"), gain_db=-3.0, pan=-0.35, send_db=-8, jitter_s=0.007)
    vln_trem = Part("violins tremolo", Patch(lib, "ViolinEnsTrem", "sus"), gain_db=-9.0, pan=-0.3, send_db=-7, lead_s=0.04)
    vla_spic = Part("violas short", Patch(lib, "ViolaEnsSpic", "short", near=1), gain_db=-7.0, pan=0.2, send_db=-11, jitter_s=0.007)
    vla_trem = Part("violas tremolo", Patch(lib, "ViolaEnsTrem", "sus"), gain_db=-8.0, pan=0.2, send_db=-8, lead_s=0.04)
    vla_sus = Part("violas long", Patch(lib, "ViolaEnsSusVib", "sus", near=1), gain_db=-7.0, pan=0.2, send_db=-9, lead_s=0.04)
    vc_spic = Part("cellos short", Patch(lib, "CelloEnsSpic", "short", near=1), gain_db=-2.0, pan=0.3, send_db=-12, jitter_s=0.005)
    vc_sus = Part("cellos long", Patch(lib, "CelloEnsSusVib", "sus", near=1), gain_db=-6.0, pan=0.3, send_db=-10, lead_s=0.04)
    vc_pizz = Part("cellos plucked", Patch(lib, "CelloEnsPizz", "short", near=1), gain_db=-4.0, pan=0.3, send_db=-11)
    cb_sus = Part("basses long", Patch(lib, "ContrabassSusNV", "sus", near=1), gain_db=-5.0, pan=0.1, send_db=-14, lead_s=0.05)
    cb_spic = Part("basses short", Patch(lib, "ContrabassSpic", "short", near=1), gain_db=-4.0, pan=0.1, send_db=-14, jitter_s=0.004)
    cb_pizz = Part("basses plucked", Patch(lib, "ContrabassPizz", "short"), gain_db=-2.0, pan=0.1, send_db=-14, jitter_s=0.004)

    # the pulse
    for bar in range(1, 12):
        lv = level(bar)
        for k in range(8):
            chord = CHORDS[BARS[bar - 1][0 if k < 4 else 1]]
            start = at(bar, k * 0.5)
            if k in ACCENTS:
                strength = (100, 0, 0, 86, 0, 0, 92, 0)[k]
                vc_spic.add(Note(start, 0, midi(chord["low"]), strength + lv, damp=0.5))
                cb_spic.add(Note(start, 0, midi(chord["bass"]), strength + lv - 4, damp=0.5))
                if k == 0:
                    cb_pizz.add(Note(start, 0, midi(chord["bass"]), 96 + lv, damp=1.0))
            else:
                vc_spic.add(Note(start, 0, midi(chord["mid"]), 66 + lv, damp=0.4))
                if bar >= 3:
                    for pitch in chord["pair"]:
                        vla_spic.add(Note(start, 0, midi(pitch), 64 + lv))
        if bar % 2 == 0:  # every second bar the cellos push into the next one with two quick notes
            chord = CHORDS[BARS[bar - 1][1]]
            vc_spic.add(Note(at(bar, 3.75), 0, midi(chord["mid"]), 70 + lv, damp=0.3))
        # the tick: plucked violins on every beat, the off beats a little louder
        for beat in range(4):
            if bar == 2 and beat >= 2:
                continue  # the question takes their place
            chord = CHORDS[BARS[bar - 1][0 if beat < 2 else 1]]
            vln_pizz.add(Note(at(bar, beat), 0, midi(chord["tick"][beat % 2]), (66, 80)[beat % 2] + lv))
    for beat, pitch, vel in QUESTION:
        vln_pizz.add(Note(at(2, beat), 0, midi(pitch), vel))
        vln_pizz.add(Note(at(12, beat), 0, midi(pitch), vel - 8))

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
        fall = GROW.get((bar, beat), 0.0 if slur_out else 3.0)
        release = 0.17 if slur_out else 0.6 if bar == 12 else 0.3
        vln_sus.add(Note(start, dur, p, vel, attack=0.075 if fresh else 0.055, release=release,
                         speak=0.5 if fresh else 0.6, swell=2.0 if long and not fall else 1.0, fall=fall))
        solo_sus.add(Note(start, dur, p, vel, attack=0.06 if fresh else 0.045, release=release,
                          speak=0.45 if fresh else 0.6, swell=1.5 if long and not fall else 0.5, fall=fall))

    # under the tune
    for bar, beat, beats, pitches in SHIVER:
        for pitch in pitches:
            vla_trem.add(Note(at(bar, beat), beats * BEAT + 0.06, midi(pitch), 72 + level(bar), attack=0.2, release=0.3,
                              speak=0.4, skip=0.7, swell=2.0, fall=-3.0 if bar == 8 else 0.0))
    for bar, beat, beats, pitches in INNER:
        for pitch in pitches:
            vla_sus.add(Note(at(bar, beat), beats * BEAT + 0.06, midi(pitch), 84 + level(bar), attack=0.12, release=0.3,
                             speak=0.5, swell=1.5))
    for bar, beat, beats, pitch, vel in CELLO_LINE:
        long = beats >= 2
        vc_sus.add(Note(at(bar, beat), beats * BEAT + 0.04, midi(pitch), vel, attack=0.12 if long else 0.07, release=0.25,
                        speak=0.45, swell=1.5 if long else 0.5, fall=-3.0 if bar in (8, 11) and beat >= 2 or bar == 8 else 0.0))
    for bar, beat, beats, pitch, vel in FLOOR:
        cb_sus.add(Note(at(bar, beat), beats * BEAT + 0.25, midi(pitch), vel, attack=0.35 if bar < 5 else 0.2, release=0.4,
                        speak=0.4, swell=3.0 if beats >= 6 else 1.5, fall=-3.0 if bar in (8, 11) else 0.0))
    # high violins, tremolo: at the start, into the halt on the dominant, and into the end
    vln_trem.add(Note(at(1, 0), 8 * BEAT, midi("A5"), 62, attack=0.9, release=0.8, speak=0.4, swell=3.0))
    vln_trem.add(Note(at(8, 0), 4 * BEAT, midi("A5"), 76, attack=0.4, release=0.25, speak=0.4, skip=0.7, fall=-7.0))
    vln_trem.add(Note(at(11, 0), 4 * BEAT + 0.05, midi("E5"), 78, attack=0.4, release=0.25, speak=0.4, skip=0.7, fall=-7.0))

    # bar 1: one firm pluck to open on
    vc_pizz.add(Note(at(1, 0), 0, midi("D3"), 96, damp=1.2))
    for pitch in ("D4", "A4"):
        vln_pizz.add(Note(at(1, 0), 0, midi(pitch), 90))
    # bar 9: everyone arrives together
    vc_pizz.add(Note(at(9, 0), 0, midi("D3"), 100, damp=1.2))
    for pitch in ("D4", "A4"):
        vln_pizz.add(Note(at(9, 0), 0, midi(pitch), 92))

    # bar 12: the basses land on A, not on D. The chord above it hangs and thins out; the question again.
    hit = at(12, 0)
    cb_pizz.add(Note(hit, 0, midi("A1"), 104, damp=1.6))
    cb_spic.add(Note(hit, 0, midi("A1"), 100, damp=0.7))
    vc_pizz.add(Note(hit, 0, midi("A2"), 100, damp=1.4))
    vc_spic.add(Note(hit, 0, midi("A2"), 102, damp=0.6))
    cb_sus.add(Note(hit, 3.4 * BEAT, midi("A1"), 88, attack=0.1, release=0.7, speak=0.45, fall=12.0))
    for pitch in ("A2", "E3"):
        vc_sus.add(Note(hit, 3.4 * BEAT, midi(pitch), 92, attack=0.08, release=0.7, speak=0.45, fall=12.0))
    for pitch in ("D4", "E4"):
        vla_trem.add(Note(hit, 3.4 * BEAT, midi(pitch), 86, attack=0.08, release=0.7, speak=0.45, fall=12.0))
    for pitch in ("A4", "E5"):
        vln_trem.add(Note(hit, 3.4 * BEAT, midi(pitch), 84, attack=0.08, release=0.7, speak=0.45, fall=12.0))
    return [vln_sus, solo_sus, vln_spic, solo_spic, vln_pizz, vln_trem, vla_spic, vla_trem, vla_sus,
            vc_spic, vc_sus, vc_pizz, cb_sus, cb_spic, cb_pizz]


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
