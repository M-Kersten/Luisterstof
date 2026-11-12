"""Turn an uploaded recording into the best reference clip it can give.

Chatterbox clones everything in a reference clip: the voice, the mood, the room, the noise. It takes the speaking
style from the first 6 seconds and the timbre from the first 10; the whole clip only shapes the speaker profile.
So what makes a good reference is a lively opening, clean edges, no long silences, and the same loudness for every
voice. This module measures a recording and makes that clip from it, without touching the voice itself:

- stereo to mono (or the better channel, when the two differ),
- rumble under 70 Hz filtered out, DC removed,
- silence at the start and the end trimmed,
- the liveliest stretch that starts after a pause put first (the rest follows it, so nothing is lost),
- pauses longer than 0.45 s shortened, the clip cut to at most 55 s at a pause,
- loudness set to -20 LUFS with the peaks below -1.5 dBFS, short fades, 24-bit WAV.

No noise reduction and no equaliser: their artifacts would be cloned along with the voice. What the processing
cannot fix (clipping, echo, noise, a phone-quality recording) is reported, with what to do about it.
Every cut lands at the quietest point inside a pause, so no word is clipped.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

TARGET_LUFS = -20.0
PEAK_CEILING_DB = -1.5
HIGHPASS_HZ = 70.0
MAX_PAUSE_S = 0.45
TARGET_S = 45.0  # the length to aim for when there is more material
MAX_S = 55.0
MIN_GOOD_S = 30.0
MIN_S = 8.0
OPENING_S = 6.0  # what Chatterbox takes the speaking style from
ROTATE_MARGIN = 1.5  # semitones of extra pitch range an opening must win by before the clip is reordered
FADE_S, PAD_S = 0.015, 0.12
HOP = 0.01


class ClipError(ValueError):
    """The upload cannot become a reference clip (unreadable, silent, far too short)."""


@dataclass
class Finding:
    level: str  # ok | warn | bad
    text: str

    def as_dict(self) -> dict:
        return {"level": self.level, "text": self.text}


@dataclass
class ClipReport:
    findings: list[Finding] = field(default_factory=list)
    changes: list[str] = field(default_factory=list)
    before: dict = field(default_factory=dict)
    after: dict = field(default_factory=dict)

    @property
    def verdict(self) -> str:
        levels = {f.level for f in self.findings}
        return "bad" if "bad" in levels else "warn" if "warn" in levels else "ok"

    def as_dict(self) -> dict:
        return {"verdict": self.verdict, "findings": [f.as_dict() for f in self.findings], "changes": self.changes,
                "before": self.before, "after": self.after}


# ---------------------------------------------------------------- measuring
def _frames_db(x: np.ndarray, sr: int, win: float = 0.02) -> np.ndarray:
    """Level per 10 ms step over a 20 ms window, in dB."""
    n, h = int(win * sr), int(HOP * sr)
    if len(x) < n:
        return np.array([-120.0])
    c = np.concatenate([[0.0], np.cumsum(x.astype(np.float64) ** 2)])
    starts = np.arange(0, len(x) - n, h)
    power = (c[starts + n] - c[starts]) / n
    return 10 * np.log10(power + 1e-12)


def _f0(x: np.ndarray, sr: int) -> np.ndarray:
    """Pitch per 10 ms (NaN where unvoiced), by autocorrelation at 16 kHz."""
    from scipy import signal

    y = signal.resample_poly(x, 16000, sr) if sr != 16000 else x
    n, h = 640, 160
    if len(y) < n:
        return np.full(1, np.nan)
    idx = np.arange(0, len(y) - n, h)
    frames = np.stack([y[i:i + n] for i in idx]) * np.hanning(n)
    loud = np.sqrt(np.mean(frames ** 2, axis=1)) > 1e-3
    spec = np.fft.rfft(frames, 2 * n)
    ac = np.fft.irfft(np.abs(spec) ** 2)[:, :n]
    ac = ac / (ac[:, :1] + 1e-12)
    lo, hi = 16000 // 400, 16000 // 65
    lag = lo + np.argmax(ac[:, lo:hi], axis=1)
    strength = ac[np.arange(len(lag)), lag]
    f0 = 16000 / lag
    f0[(strength < 0.45) | ~loud] = np.nan
    return f0


def _semitones(f0: np.ndarray) -> float | None:
    v = f0[~np.isnan(f0)]
    if len(v) < 10:
        return None
    return float(12 * np.log2(np.percentile(v, 90) / np.percentile(v, 10)))


def _bandwidth(x: np.ndarray, sr: int) -> int | None:
    from scipy import signal

    f, p = signal.welch(x, sr, nperseg=min(8192, len(x)))
    pdb = 10 * np.log10(p + 1e-20)
    ref = np.max(pdb[(f > 200) & (f < 4000)]) if np.any((f > 200) & (f < 4000)) else np.max(pdb)
    above = f[(pdb > ref - 60) & (f > 2000)]
    return int(above.max()) if len(above) else None


@dataclass
class _Speech:
    db: np.ndarray
    quiet: np.ndarray
    speech_db: float
    noise_db: float
    pauses: list[tuple[float, float]]  # (start, end) of quiet runs of at least 0.15 s


def _speech(x: np.ndarray, sr: int) -> _Speech:
    db = _frames_db(x, sr)
    speech = float(np.percentile(db, 95))
    floor = float(np.percentile(db, 5))
    quiet = db < max(floor + 10, speech - 38)
    pauses, run = [], 0
    for i, q in enumerate(np.append(quiet, False)):
        if q:
            run += 1
        else:
            if run * HOP >= 0.15:
                pauses.append(((i - run) * HOP, i * HOP))
            run = 0
    # The background: the low end (10th percentile) of the pauses between words. A pause still holds the tail of
    # the word before it, and someone who talks fast hardly pauses, so its middle says more about the words than
    # about the room; a fan or a hiss runs through every pause and lifts the low end too. Digital zeros are left
    # out, and so is the silence before and after the speech, which is often edited.
    loud = np.where(~quiet)[0]
    first, last = (loud[0] * HOP, loud[-1] * HOP) if len(loud) else (0.0, len(db) * HOP)
    inside = [db[int(a / HOP):int(b / HOP)] for a, b in pauses if a > first and b < last]
    inside = np.concatenate(inside) if inside else db
    inside = inside[inside > -100]
    noise = float(np.percentile(inside, 10)) if len(inside) else floor
    return _Speech(db, quiet, speech, noise, pauses)


def measure(x: np.ndarray, sr: int) -> dict:
    """The numbers the report and the page show for one mono signal."""
    import pyloudnorm as pyln

    sp = _speech(x, sr)
    f0 = _f0(x, sr)
    first, last = _first_speech(sp), _last_speech(sp)
    lead = first
    open_f0 = f0[int(first / HOP):int((first + OPENING_S) / HOP)]
    try:
        lufs = float(pyln.Meter(sr).integrated_loudness(x))
    except ValueError:
        lufs = float("nan")
    return {
        "seconds": round(len(x) / sr, 2),
        "sample_rate": sr,
        "peak_dbfs": round(float(20 * np.log10(np.max(np.abs(x)) + 1e-12)), 1),
        "lufs": None if np.isnan(lufs) else round(lufs, 1),
        "noise_db": round(sp.noise_db, 1),
        "speech_db": round(sp.speech_db, 1),
        "lead_silence_s": round(lead, 2),
        "long_pauses": sum(1 for a, b in sp.pauses if b - a > 0.6 and a > first and b < last),
        "opening_semitones": None if _semitones(open_f0) is None else round(_semitones(open_f0), 1),
        "semitones": None if _semitones(f0) is None else round(_semitones(f0), 1),
        "voiced_share": round(float(np.mean(~np.isnan(f0))), 2),
        "bandwidth_hz": _bandwidth(x, sr),
    }


def _first_speech(sp: _Speech) -> float:
    loud = np.where(~sp.quiet)[0]
    return float(loud[0] * HOP) if len(loud) else 0.0


def _last_speech(sp: _Speech) -> float:
    loud = np.where(~sp.quiet)[0]
    return float((loud[-1] + 2) * HOP) if len(loud) else 0.0


def _decay_db(db: np.ndarray, speech: float) -> float | None:
    """How far the level falls in the 60 ms after a word ends: little fall means echo."""
    drops = []
    for i in range(len(db) - 40):
        if db[i] > speech - 15 and np.all(db[i + 6:i + 36] < db[i] - 12) and np.all(db[i + 1:i + 30] < db[i] + 1):
            drops.append(db[i] - db[i + 6])
    return float(np.median(drops)) if len(drops) >= 5 else None


def _quietest(x: np.ndarray, sr: int, a: float, b: float) -> float:
    """The quietest 10 ms point between a and b (seconds): where a cut lands between sounds."""
    lo, hi = max(0, int(a * sr)), min(len(x), int(b * sr))
    if hi - lo < int(0.02 * sr):
        return (a + b) / 2
    n = int(0.01 * sr)
    c = np.concatenate([[0.0], np.cumsum(x[lo:hi].astype(np.float64) ** 2)])
    e = c[n:] - c[:-n]
    return (lo + int(np.argmin(e)) + n // 2) / sr


# ---------------------------------------------------------------- processing
def _read(path: Path) -> tuple[np.ndarray, int]:
    try:
        data, sr = sf.read(str(path), dtype="float64", always_2d=True)
    except Exception as exc:  # noqa: BLE001 - libsndfile says what it could not read
        raise ClipError(f"Dit bestand is niet te lezen als audio ({exc}). Exporteer het als WAV of FLAC.") from exc
    if not len(data):
        raise ClipError("Het bestand bevat geen geluid.")
    return data, int(sr)


def _to_mono(data: np.ndarray, sr: int, report: ClipReport) -> np.ndarray:
    if data.shape[1] == 1:
        return data[:, 0]
    left, right = data[:, 0], data[:, 1]
    corr = float(np.corrcoef(left, right)[0, 1]) if np.std(left) > 0 and np.std(right) > 0 else 1.0
    if corr > 0.9:
        report.changes.append("Stereo naar mono: beide kanalen waren gelijk, er gaat niets verloren.")
        return data.mean(axis=1)
    snr = [_speech(ch, sr).speech_db - _speech(ch, sr).noise_db for ch in (left, right)]
    pick = int(np.argmax(snr))
    report.changes.append(f"Alleen het {'linker' if pick == 0 else 'rechter'} kanaal gebruikt: de kanalen verschilden, "
                          "en samenvoegen zou de stem hol laten klinken.")
    return data[:, pick]


def _opening_score(f0: np.ndarray, quiet: np.ndarray, start: float) -> float:
    seg = f0[int(start / HOP):int((start + OPENING_S) / HOP)]
    q = quiet[int(start / HOP):int((start + OPENING_S) / HOP)]
    st = _semitones(seg)
    if st is None or not len(q):
        return -1e9
    return st + 4 * float(np.mean(~np.isnan(seg))) - 6 * float(np.mean(q))


def _shorten_pauses(x: np.ndarray, sr: int, report: ClipReport) -> np.ndarray:
    sp = _speech(x, sr)
    first, last = _first_speech(sp), _last_speech(sp)
    long = [(a, b) for a, b in sp.pauses if b - a > MAX_PAUSE_S and a > first and b < last]
    if not long:
        return x
    keep = MAX_PAUSE_S / 2
    pieces, at = [], 0
    for a, b in long:
        pieces.append(x[at:int((a + keep) * sr)])
        at = int((b - keep) * sr)
    pieces.append(x[at:])
    fade = int(0.008 * sr)
    out = pieces[0]
    for p in pieces[1:]:
        ramp = np.linspace(1, 0, fade)
        joined = out[-fade:] * ramp + p[:fade] * (1 - ramp)
        out = np.concatenate([out[:-fade], joined, p[fade:]])
    report.changes.append(f"{len(long)} {'lange pauze' if len(long) == 1 else 'lange pauzes'} ingekort tot "
                          f"{MAX_PAUSE_S:.2f} s".replace(".", ","))
    return out


def process(src: Path, dst: Path) -> ClipReport:
    """Read an upload, write the reference clip to ``dst`` (24-bit WAV) and return what was found and done."""
    from scipy import signal

    report = ClipReport()
    data, sr = _read(src)
    x = _to_mono(data, sr, report)
    report.before = measure(x, sr)

    clipped = int(np.sum(np.abs(data) >= 0.999))
    x = x - np.mean(x)
    if sr >= 2 * HIGHPASS_HZ * 4:
        x = signal.sosfiltfilt(signal.butter(4, HIGHPASS_HZ, "highpass", fs=sr, output="sos"), x)
        report.changes.append("Lage brom onder 70 Hz weggefilterd; de stem zelf zit daarboven.")

    sp = _speech(x, sr)
    first, last = _first_speech(sp), _last_speech(sp)
    if last - first < MIN_S:
        raise ClipError(f"Er staat maar {last - first:.0f} seconden spraak in. Een referentieclip heeft er minstens 30 nodig.")
    if first > 0.3:
        report.changes.append(f"{first:.1f} s stilte aan het begin weggehaald.".replace(".", ",", 1))

    # The opening: the liveliest stretch that starts after a pause, when it clearly beats the current start.
    f0 = _f0(x, sr)
    onsets = [first] + [b for a, b in sp.pauses if b - a >= 0.25 and first < b < last - OPENING_S]
    scores = {o: _opening_score(f0, sp.quiet, o) for o in onsets}
    best = max(scores, key=scores.get)
    start = first
    if best != first and scores[best] - scores[first] >= ROTATE_MARGIN:
        start = best

    # Cut points sit in the quietest spot of a pause, close to the speech they border: just before the speech that
    # starts after it, or just after the speech that ends before it.
    def cut_before(t: float) -> float:
        pause = next(((a, b) for a, b in sp.pauses if abs(b - t) < 0.02), None)
        return _quietest(x, sr, max(pause[0], pause[1] - 0.15), pause[1] - 0.02) if pause else max(0.0, t - 0.05)

    begin = cut_before(start) if start != first else max(0.0, first - 0.05)
    end = min(len(x) / sr, last + 0.15)

    def pause_cut_near(t: float, lo: float) -> float:
        cands = [(a, b) for a, b in sp.pauses if lo < a < t]
        if not cands:
            return t
        a, b = max(cands, key=lambda p: p[0])
        return _quietest(x, sr, a + 0.02, min(b, a + 0.25))

    if start != first:
        tail = x[int(begin * sr):int(end * sr)]
        if len(tail) / sr >= TARGET_S:
            stop = pause_cut_near(begin + MAX_S, begin + MIN_GOOD_S)
            body = x[int(begin * sr):int(stop * sr)]
            report.changes.append(f"Het levendigste stuk (vanaf {start:.1f} s) gebruikt als begin; Chatterbox haalt de "
                                  "spreekstijl uit de eerste 6 seconden.".replace(".", ",", 1))
        else:
            head_end = pause_cut_near(begin, first + 1.0) if begin - first > 1.0 else begin
            gap = np.zeros(int(0.35 * sr))
            head = x[int(max(0.0, first - 0.05) * sr):int(head_end * sr)]
            f = int(0.01 * sr)
            tail = tail.copy()
            head = head.copy()
            tail[-f:] *= np.linspace(1, 0, f)
            head[:f] *= np.linspace(0, 1, f)
            body = np.concatenate([tail, gap, head])
            report.changes.append(f"Het levendigste stuk (vanaf {start:.1f} s) vooraan gezet, de rest erachter; "
                                  "Chatterbox haalt de spreekstijl uit de eerste 6 seconden.".replace(".", ",", 1))
    else:
        stop = end
        if end - begin > MAX_S + 5:
            stop = pause_cut_near(begin + MAX_S, begin + MIN_GOOD_S)
        body = x[int(begin * sr):int(stop * sr)]

    body = _shorten_pauses(body, sr, report)
    if len(body) / sr > MAX_S + 5:
        sp2 = _speech(body, sr)
        stop = max((a for a, b in sp2.pauses if MIN_GOOD_S < a < MAX_S), default=MAX_S)
        body = body[:int(_quietest(body, sr, stop, stop + 0.15) * sr)]
    if report.before["seconds"] - len(body) / sr > 8:
        report.changes.append(f"Ingekort tot {len(body) / sr:.0f} s, afgesneden in een pauze.")

    f = int(FADE_S * sr)
    body = body.copy()
    body[:f] *= np.linspace(0, 1, f)
    body[-f:] *= np.linspace(1, 0, f)
    import pyloudnorm as pyln

    loud = pyln.Meter(sr).integrated_loudness(body)
    peak = 20 * np.log10(np.max(np.abs(body)) + 1e-12)
    gain = min(TARGET_LUFS - loud, PEAK_CEILING_DB - peak)
    body = body * 10 ** (gain / 20)
    if abs(gain) >= 0.3:
        word = "harder" if gain > 0 else "zachter"
        report.changes.append(f"{abs(gain):.1f} dB {word} gezet: elke stem even luid, pieken onder {PEAK_CEILING_DB} dB."
                              .replace(".", ",", 1))
    pad = np.zeros(int(PAD_S * sr))
    out = np.concatenate([pad, body, pad]).astype(np.float32)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), out, sr, subtype="PCM_24")
    report.changes.append("Opgeslagen als mono WAV, 24 bit.")
    report.after = measure(out.astype(np.float64), sr)

    _judge(report, clipped, sp, sr, x)
    return report


def _judge(report: ClipReport, clipped: int, sp: _Speech, sr: int, x: np.ndarray) -> None:
    """What the processing could not fix, and what it means for the clone."""
    add = report.findings.append
    b, a = report.before, report.after
    if clipped > 10:
        add(Finding("bad", f"De opname is oversturd ({clipped} samples op het maximum). Dat kraakt, en de kloon kraakt "
                           "mee. Neem opnieuw op met de microfoon zachter of verder weg."))
    if sp.noise_db > -42:
        add(Finding("bad", f"Veel achtergrondgeluid in de stiltes ({sp.noise_db:.0f} dB). De kloon neemt dat over. "
                           "Neem opnieuw op in een stillere ruimte, dichter op de microfoon."))
    elif sp.noise_db > -52:
        add(Finding("warn", f"Hoorbaar achtergrondgeluid in de stiltes ({sp.noise_db:.0f} dB). Luister of je het in de "
                            "takes terughoort; zo ja, neem opnieuw op in een stillere ruimte."))
    else:
        add(Finding("ok", f"Stil op de achtergrond ({sp.noise_db:.0f} dB in de pauzes)."))
    decay = _decay_db(sp.db, sp.speech_db)
    if decay is not None and decay < 12:
        add(Finding("warn", "Het geluid sterft langzaam weg na een woord: er zit galm in de ruimte. Demp de ruimte "
                            "met dekens of kleding, of neem dichter op de microfoon op."))
    bw = b.get("bandwidth_hz")
    if bw and bw < 7000:
        add(Finding("warn", f"De opname loopt maar tot {bw / 1000:.0f} kHz, zoals een telefoon of een sterk "
                            "gecomprimeerd bestand. De kloon klinkt dan dof. Gebruik een WAV-opname."))
    if sr < 22050:
        add(Finding("warn", f"Lage samplefrequentie ({sr} Hz). Neem op met 44,1 of 48 kHz."))
    secs = a.get("seconds", 0)
    if secs < 20:
        add(Finding("bad", f"Maar {secs:.0f} s bruikbaar geluid. Een referentie heeft 30 tot 60 seconden nodig."))
    elif secs < MIN_GOOD_S:
        add(Finding("warn", f"{secs:.0f} s bruikbaar geluid. Met 30 tot 60 seconden wordt het stemprofiel stabieler."))
    else:
        add(Finding("ok", f"{secs:.0f} seconden bruikbaar geluid."))
    st = a.get("opening_semitones")
    if st is not None and st < 7:
        add(Finding("warn", f"Het begin is vlak ({st:.1f} halve tonen bereik in de eerste 6 seconden). Chatterbox haalt "
                            "daar de spreekstijl uit, dus de stem klinkt rustig. Neem een levendiger stuk op als je "
                            "meer expressie wilt."))
    elif st is not None:
        add(Finding("ok", f"Levendig begin ({st:.1f} halve tonen bereik in de eerste 6 seconden)."))


def process_to_folder(src: Path, folder: Path) -> dict:
    """Process ``src`` into ``folder/clip.wav`` and keep the report next to it as ``report.json``."""
    report = process(src, folder / "clip.wav")
    data = report.as_dict()
    (folder / "report.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return data
