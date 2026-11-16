"""The sound of the show: what makes dry Chatterbox takes sound recorded.

Chatterbox renders every line dry: no room, no microphone, no level riding. Next to each other the voices sound like
they come from nowhere. A real podcast has two people in one room on similar microphones through one chain, and that
is what this adds, the same for every voice:

- the same microphone: a low cut against rumble;
- warmth: a little low end and a very light saturation, the way a microphone and preamp colour a voice;
- tamed s-sounds: synthetic voices hiss on s, z and sch; a de-esser takes the edge off only where it spikes;
- a gentle compressor over the conversation, so quiet and loud lines sit closer together;
- one small shared room, so both hosts sound like they sit at the same table;
- a limiter at the end that catches peaks instead of turning the whole episode down.

Five knobs between 0 and 1 set how much of each; four presets set them all at once. ``droog`` (dry) switches it
all off and gives the episode exactly as before. The settings live in ``cast/klank.yaml`` and apply to every voice,
every take in the voice lab (to listen to, the take itself stays dry on disk) and every recorded episode.

Nothing here needs the GPU: a take takes a fraction of a second, an episode a few seconds.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from pydantic import BaseModel, field_validator

KNOBS = ("room", "size", "warmth", "deess", "compression")
PRESETS: dict[str, dict[str, float]] = {
    "droog": {"room": 0.0, "size": 0.0, "warmth": 0.0, "deess": 0.0, "compression": 0.0},
    "studio": {"room": 0.25, "size": 0.2, "warmth": 0.3, "deess": 0.5, "compression": 0.35},
    "huiskamer": {"room": 0.45, "size": 0.45, "warmth": 0.45, "deess": 0.4, "compression": 0.25},
    "radio": {"room": 0.1, "size": 0.15, "warmth": 0.35, "deess": 0.6, "compression": 0.7},
}
DEFAULT_PRESET = "studio"
FILE = "klank.yaml"
HEADER = """# De klank van de show: dezelfde microfoon en dezelfde kamer voor elke stem.
# Instellen en beluisteren op de pagina Stemmen; dit bestand schrijft die pagina.
# preset: droog | studio | huiskamer | radio. Droog is precies het geluid van Chatterbox, zonder nabewerking.
# De knoppen gaan van 0 (niets) tot 1 (veel):
#   room: hoeveel kamer je hoort   size: hoe groot de kamer klinkt   warmth: warmte van microfoon en voorversterker
#   deess: hoe sterk scherpe s-klanken getemperd worden   compression: hoe dicht zachte en harde zinnen bij elkaar liggen
"""


class Finish(BaseModel):
    preset: str = DEFAULT_PRESET
    room: float = PRESETS[DEFAULT_PRESET]["room"]
    size: float = PRESETS[DEFAULT_PRESET]["size"]
    warmth: float = PRESETS[DEFAULT_PRESET]["warmth"]
    deess: float = PRESETS[DEFAULT_PRESET]["deess"]
    compression: float = PRESETS[DEFAULT_PRESET]["compression"]

    @field_validator(*KNOBS)
    @classmethod
    def _unit(cls, v: float) -> float:
        v = float(v)
        if not 0.0 <= v <= 1.0:
            raise ValueError("tussen 0 en 1")
        return round(v, 3)

    @field_validator("preset")
    @classmethod
    def _preset(cls, v: str) -> str:
        if v not in PRESETS and v != "eigen":
            raise ValueError(f"onbekende preset {v!r}")
        return v

    @classmethod
    def from_preset(cls, name: str) -> Finish:
        if name not in PRESETS:
            raise ValueError(f"onbekende preset {name!r}")
        return cls(preset=name, **PRESETS[name])

    @property
    def dry(self) -> bool:
        return all(getattr(self, k) == 0 for k in KNOBS)

    def knobs(self) -> dict[str, float]:
        return {k: getattr(self, k) for k in KNOBS}


def load(cast_dir: Path | str) -> Finish:
    path = Path(cast_dir) / FILE
    if not path.is_file():
        return Finish.from_preset(DEFAULT_PRESET)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return Finish.model_validate(data)


def save(cast_dir: Path | str, finish: Finish) -> Path:
    path = Path(cast_dir) / FILE
    body = yaml.safe_dump(finish.model_dump(), sort_keys=False, allow_unicode=True)
    path.write_text(HEADER + body, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- the pieces
def _board(*plugins):
    import pedalboard  # type: ignore

    return pedalboard.Pedalboard(list(plugins))


def _run(board, x: np.ndarray, sr: int) -> np.ndarray:
    return board(x.reshape(1, -1).astype(np.float32), sr)[0].astype(np.float32)


def _deess(x: np.ndarray, sr: int, amount: float) -> np.ndarray:
    """Turn the 4.5-10 kHz band down only while it rises above the voice underneath it (below 4 kHz): an s, z or
    sch does, a vowel never does."""
    if amount <= 0 or len(x) < sr // 50:
        return x
    from scipy.signal import butter, sosfiltfilt

    hi = min(10000.0, sr * 0.45)
    band = sosfiltfilt(butter(4, [4500.0, hi], btype="bandpass", fs=sr, output="sos"), x).astype(np.float32)
    body = sosfiltfilt(butter(4, 4000.0, btype="low", fs=sr, output="sos"), x).astype(np.float32)
    win = max(1, int(0.005 * sr))
    kernel = np.ones(win, dtype=np.float32) / win
    env_band = np.sqrt(np.convolve(band**2, kernel, mode="same") + 1e-10)
    env_body = np.sqrt(np.convolve(body**2, kernel, mode="same") + 1e-10)
    over_db = 20 * np.log10(env_band / env_body) - (2.0 + 8.0 * (1 - amount))
    reduction_db = np.clip(over_db * (0.5 + 0.5 * amount), 0.0, 4.0 + 8.0 * amount)
    gain = 10 ** (-_hold_release(reduction_db, sr, hold_s=0.001, release_s=0.03) / 20)
    return (x - band * (1 - gain)).astype(np.float32)


def _hold_release(reduction_db: np.ndarray, sr: int, *, hold_s: float, release_s: float) -> np.ndarray:
    """Gain reduction that comes in at once (a short look-around, so it is there before the peak) and lets go
    slowly, without a per-sample loop: an episode has tens of millions of samples."""
    from scipy.ndimage import maximum_filter1d
    from scipy.signal import lfilter

    held = maximum_filter1d(reduction_db.astype(np.float64), size=2 * max(1, int(hold_s * sr)) + 1)
    a = np.exp(-1.0 / (release_s * sr))
    released = lfilter([1 - a], [1, -a], held)
    return np.maximum(held, released)


def _warmth(x: np.ndarray, sr: int, amount: float) -> np.ndarray:
    if amount <= 0:
        return x
    import pedalboard  # type: ignore

    y = _run(_board(pedalboard.LowShelfFilter(cutoff_frequency_hz=160, gain_db=2.5 * amount, q=0.7),
                    pedalboard.HighShelfFilter(cutoff_frequency_hz=9000, gain_db=-1.5 * amount, q=0.7)), x, sr)
    peak = float(np.max(np.abs(y))) or 1.0
    drive = 1.0 + 1.5 * amount
    saturated = np.tanh(drive * y / peak) / np.tanh(drive) * peak
    mix = 0.5 * amount
    return ((1 - mix) * y + mix * saturated).astype(np.float32)


def voice(x: np.ndarray, sr: int, f: Finish) -> np.ndarray:
    """Per line, before the line's loudness is set: the shared microphone, warmth and tamed s-sounds."""
    if f.dry or len(x) == 0:
        return x
    import pedalboard  # type: ignore

    y = _run(_board(pedalboard.HighpassFilter(cutoff_frequency_hz=75)), x, sr)
    y = _warmth(y, sr, f.warmth)
    return _deess(y, sr, f.deess)


def room_ir(sr: int, size: float, seed: int = 7) -> np.ndarray:
    """A small room's impulse response, made up: a few early reflections off table and walls, then a diffuse tail
    that gets darker as it dies away. Seeded, so every episode sits in the same room."""
    from scipy.signal import butter, sosfilt

    rt60 = 0.12 + 0.5 * size  # 0.12 s (treated booth) to 0.62 s (living room)
    n = int(sr * rt60 * 1.2)
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sr
    tail = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)
    bright = sosfilt(butter(2, 6000, btype="low", fs=sr, output="sos"), tail)
    dark = sosfilt(butter(2, 1800, btype="low", fs=sr, output="sos"), tail)
    fade = np.clip(t / max(rt60 * 0.6, 1e-3), 0, 1)
    ir = (1 - fade) * bright + fade * dark
    ir[: int(0.004 * sr)] = 0.0  # pre-delay: the first reflection arrives a few ms after the voice
    for ms, gain in ((5.5, 0.5), (9.0, 0.35), (13.0, 0.28), (19.0, 0.2)):
        i = int(ms * (0.7 + 0.8 * size) * sr / 1000)
        if i < n:
            ir[i] += gain * (1 if rng.random() > 0.5 else -1)
    return (ir / np.sqrt(np.sum(ir**2))).astype(np.float32)


def _room(x: np.ndarray, sr: int, room: float, size: float) -> np.ndarray:
    if room <= 0 or len(x) == 0:
        return x
    from scipy.signal import fftconvolve

    wet = fftconvolve(x, room_ir(sr, size))[: len(x)].astype(np.float32)
    return (x + 10 ** ((-34 + 20 * room) / 20) * wet).astype(np.float32)


def bus(x: np.ndarray, sr: int, f: Finish) -> np.ndarray:
    """Over the whole conversation: one compressor, one room."""
    if f.dry or len(x) == 0:
        return x
    import pedalboard  # type: ignore

    y = x
    if f.compression > 0:
        y = _run(_board(pedalboard.Compressor(threshold_db=-14 - 14 * f.compression, ratio=1.5 + 2.5 * f.compression,
                                              attack_ms=8, release_ms=120)), y, sr)
    return _room(y, sr, f.room, f.size)


def limit(x: np.ndarray, sr: int, f: Finish, ceiling_db: float = -1.0) -> np.ndarray:
    """Peaks above the ceiling are turned down, nothing else (pedalboard's Limiter adds make-up gain)."""
    if f.dry or len(x) == 0:
        return x
    ceiling = 10 ** (ceiling_db / 20)
    over_db = 20 * np.log10(np.maximum(1.0, np.abs(x) / ceiling))
    if not over_db.any():
        return x
    gain = 10 ** (-_hold_release(over_db, sr, hold_s=0.002, release_s=0.05) / 20)
    return (x * gain).astype(np.float32)


def take(x: np.ndarray, sr: int, f: Finish) -> np.ndarray:
    """One take with the whole chain, at the take's own loudness, so dry and treated compare fairly (the louder of
    two always sounds better)."""
    if f.dry or len(x) == 0:
        return x
    from pipeline.audio.mixer import measure_lufs, normalise_loudness

    before = measure_lufs(x, sr)
    tail = np.zeros(int(sr * (0.15 + 0.6 * f.size)) if f.room > 0 else 0, dtype=np.float32)  # let the room ring out
    y = bus(voice(np.concatenate([x, tail]), sr, f), sr, f)
    if np.isfinite(before):
        y = normalise_loudness(y, sr, before)
    return limit(y, sr, f)
