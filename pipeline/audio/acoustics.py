"""Shared acoustic environment: different speakers, same microphone, room and chain.

Per speaker, the long-term average spectrum (octave bands, voiced frames only)
is measured over everything that speaker says in the episode. The target is the
mean *shape* across speakers; each speaker gets a partial correction toward it
(``strength``), clamped, so chain differences (boxiness, tilt, dull or harsh
top) shrink while the voices keep their own timbre. Manual EQ bands from
hosts.yaml go on top. Then every speaker gets the same recording profile, and
the summed dialogue bus gets one shared compressor and one shared small room.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from pydantic import BaseModel, Field

from pipeline.models import Cast, EqBand

OCTAVE_CENTERS = (125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0)
BAND_Q = 1.4  # roughly one octave wide, so adjacent bells overlap smoothly instead of rippling
FRAME = 2048
VOICED_DB = -45.0


class MatchConfig(BaseModel):
    strength: float = 0.5
    max_db: float = 6.0


class ProfileConfig(BaseModel):
    highpass_hz: float = 80.0
    presence: EqBand = Field(default_factory=lambda: EqBand(freq_hz=4000.0, gain_db=1.5, q=0.8))


class CompressorConfig(BaseModel):
    threshold_db: float = -20.0
    ratio: float = 2.5
    attack_ms: float = 10.0
    release_ms: float = 120.0


class RoomConfig(BaseModel):
    room_size: float = 0.12
    damping: float = 0.6
    wet_level: float = 0.06
    dry_level: float = 1.0
    width: float = 0.0


class BusConfig(BaseModel):
    compressor: CompressorConfig = Field(default_factory=CompressorConfig)
    room: RoomConfig = Field(default_factory=RoomConfig)


class AcousticsConfig(BaseModel):
    match: MatchConfig = Field(default_factory=MatchConfig)
    profile: ProfileConfig = Field(default_factory=ProfileConfig)
    bus: BusConfig = Field(default_factory=BusConfig)

    @classmethod
    def load(cls, cast_dir: Path | str) -> AcousticsConfig:
        path = Path(cast_dir) / "acoustics.yaml"
        if not path.is_file():
            return cls()
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")) or {})


def _pedalboard():
    try:
        import pedalboard  # type: ignore
    except ImportError as exc:  # pragma: no cover - declared dependency, message for broken installs
        raise RuntimeError("the shared acoustic chain needs pedalboard: uv pip install pedalboard") from exc
    return pedalboard


def band_levels(samples: np.ndarray, sr: int) -> np.ndarray:
    """Octave-band levels in dB of the voiced frames (silence would drag every band down equally)."""
    if len(samples) < FRAME:
        samples = np.pad(samples, (0, FRAME - len(samples)))
    n = len(samples) // FRAME
    frames = samples[: n * FRAME].reshape(n, FRAME).astype(np.float64)
    rms_db = 20 * np.log10(np.sqrt(np.mean(frames**2, axis=1)) + 1e-12)
    voiced = frames[rms_db > max(VOICED_DB, float(rms_db.max()) - 40.0)] if n else frames
    if len(voiced) == 0:
        voiced = frames
    power = np.mean(np.abs(np.fft.rfft(voiced * np.hanning(FRAME), axis=1)) ** 2, axis=0)
    freqs = np.fft.rfftfreq(FRAME, 1.0 / sr)
    levels = []
    for c in OCTAVE_CENTERS:
        band = (freqs >= c / np.sqrt(2)) & (freqs < c * np.sqrt(2))
        levels.append(10 * np.log10(power[band].sum() + 1e-18) if band.any() else -180.0)
    return np.array(levels)


def shape(levels: np.ndarray) -> np.ndarray:
    """Spectral shape relative to its own mean: overall level is loudness normalisation's job."""
    return levels - levels.mean()


def corrections(ltas: dict[str, np.ndarray], config: MatchConfig) -> dict[str, np.ndarray]:
    if not ltas:
        return {}
    shapes = {s: shape(v) for s, v in ltas.items()}
    target = np.mean(list(shapes.values()), axis=0)
    target = np.convolve(np.pad(target, 1, mode="edge"), np.ones(3) / 3, mode="valid")  # smooth the target
    return {s: np.clip(config.strength * (target - v), -config.max_db, config.max_db) for s, v in shapes.items()}


class Acoustics:
    """Per-speaker EQ + shared profile, and the shared dialogue bus, fitted to one episode."""

    def __init__(self, config: AcousticsConfig, gains: dict[str, np.ndarray], manual: dict[str, list[EqBand]]):
        self.config = config
        self.gains = gains
        self.manual = manual

    @classmethod
    def fit(cls, speech: dict[str, list[np.ndarray]], sr: int, cast: Cast, config: AcousticsConfig) -> Acoustics:
        ltas = {s: band_levels(np.concatenate(chunks), sr) for s, chunks in speech.items() if chunks}
        manual = {s.id: list(s.eq) for s in list(cast.hosts) + list(cast.guests)}
        return cls(config, corrections(ltas, config.match), manual)

    def speaker_board(self, speaker: str):
        pb = _pedalboard()
        plugins = []
        for center, gain in zip(OCTAVE_CENTERS, self.gains.get(speaker, np.zeros(len(OCTAVE_CENTERS))), strict=True):
            if abs(gain) >= 0.25:
                plugins.append(pb.PeakFilter(cutoff_frequency_hz=center, gain_db=float(gain), q=BAND_Q))
        for band in self.manual.get(speaker, []):
            plugins.append(pb.PeakFilter(cutoff_frequency_hz=band.freq_hz, gain_db=band.gain_db, q=band.q))
        profile = self.config.profile
        plugins.append(pb.HighpassFilter(cutoff_frequency_hz=profile.highpass_hz))
        plugins.append(pb.PeakFilter(cutoff_frequency_hz=profile.presence.freq_hz, gain_db=profile.presence.gain_db,
                                     q=profile.presence.q))
        return pb.Pedalboard(plugins)

    def process_speaker(self, speaker: str, samples: np.ndarray, sr: int) -> np.ndarray:
        if len(samples) == 0:
            return samples
        return self.speaker_board(speaker)(samples.reshape(1, -1).astype(np.float32), sr)[0].astype(np.float32)

    def process_bus(self, samples: np.ndarray, sr: int) -> np.ndarray:
        pb = _pedalboard()
        c, r = self.config.bus.compressor, self.config.bus.room
        board = pb.Pedalboard([
            pb.Compressor(threshold_db=c.threshold_db, ratio=c.ratio, attack_ms=c.attack_ms, release_ms=c.release_ms),
            pb.Reverb(room_size=r.room_size, damping=r.damping, wet_level=r.wet_level, dry_level=r.dry_level, width=r.width),
        ])
        return board(samples.reshape(1, -1).astype(np.float32), sr)[0].astype(np.float32)

    def report(self) -> dict[str, dict[str, float]]:
        return {s: {f"{int(c)}Hz": round(float(g), 2) for c, g in zip(OCTAVE_CENTERS, gains, strict=True)}
                for s, gains in self.gains.items()}
