"""Mixing: loudness, fades, ducking, room tone, stings, final render, transcript."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from pipeline.audio.chunker import Block
from pipeline.audio.synth import AudioClip, resample
from pipeline.audio.timeline import Placement, Timeline
from pipeline.models import BlockSpan, EpisodeTranscript, Script, TranscriptLine, TranscriptWord

log = logging.getLogger(__name__)


def db_to_gain(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def measure_lufs(samples: np.ndarray, sr: int) -> float:
    """Integrated loudness; RMS proxy for clips too short for the gating block."""
    if len(samples) == 0:
        return -70.0
    if len(samples) >= int(sr * 0.5):
        try:
            import pyloudnorm as pyln

            value = float(pyln.Meter(sr).integrated_loudness(samples.astype(np.float64)))
            if np.isfinite(value):
                return value
        except Exception as exc:  # pragma: no cover
            log.debug("pyloudnorm failed: %s", exc)
    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2) + 1e-12))
    return 20 * np.log10(rms + 1e-9) - 0.691


def normalise_loudness(samples: np.ndarray, sr: int, target_lufs: float, max_gain_db: float = 30.0) -> np.ndarray:
    current = measure_lufs(samples, sr)
    if current < -60:
        return samples
    gain_db = max(-max_gain_db, min(max_gain_db, target_lufs - current))
    return (samples * db_to_gain(gain_db)).astype(np.float32)


def fade(samples: np.ndarray, sr: int, fade_in: float = 0.0, fade_out: float = 0.0) -> np.ndarray:
    out = samples.astype(np.float32, copy=True)
    n_in = min(len(out), int(fade_in * sr))
    n_out = min(len(out), int(fade_out * sr))
    if n_in > 0:
        out[:n_in] *= np.linspace(0.0, 1.0, n_in, dtype=np.float32)
    if n_out > 0:
        out[-n_out:] *= np.linspace(1.0, 0.0, n_out, dtype=np.float32)
    return out


def duck(samples: np.ndarray, sr: int, from_s: float, depth_db: float, fade_s: float) -> np.ndarray:
    out = samples.astype(np.float32, copy=True)
    start = max(0, int(from_s * sr))
    if start >= len(out):
        return out
    n_fade = max(1, int(fade_s * sr))
    ramp_end = min(len(out), start + n_fade)
    ramp = np.linspace(1.0, db_to_gain(-depth_db), ramp_end - start, dtype=np.float32)
    out[start:ramp_end] *= ramp
    out[ramp_end:] *= db_to_gain(-depth_db)
    return out


def room_tone(n: int, sr: int, level_db: float = -60.0, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(n).astype(np.float32)
    # one-pole low-pass for a brownish, unobtrusive floor
    out = np.empty_like(white)
    acc = 0.0
    alpha = 0.02
    for i in range(n):
        acc += alpha * (white[i] - acc)
        out[i] = acc
    rms = float(np.sqrt(np.mean(out**2)) + 1e-9)
    return out * (db_to_gain(level_db) / rms)


def _prepare(placement: Placement, sr: int, line_lufs: float, acoustics=None) -> np.ndarray:
    clip = placement.clip
    samples = resample(clip.samples, clip.sr, sr) if clip.sr != sr else clip.samples
    if acoustics is not None:
        samples = acoustics.process_speaker(placement.turn.speaker, samples, sr)  # before loudness, so levels stay even
    samples = normalise_loudness(samples, sr, line_lufs)
    samples = samples * db_to_gain(placement.gain_db)
    if placement.duck_at is not None:
        samples = duck(samples, sr, placement.duck_at - placement.start, placement.duck_db, placement.duck_fade)
    if placement.cut_at is not None:
        keep = int((placement.cut_at - placement.start + placement.cut_fade) * sr)
        samples = samples[: max(1, min(len(samples), keep))]
        samples = fade(samples, sr, 0.0, placement.cut_fade)
    return fade(samples, sr, 0.005, 0.01)


def mix(
    timeline: Timeline,
    *,
    sr: int,
    target_lufs: float = -16.0,
    line_lufs: float = -18.0,
    intro: AudioClip | None = None,
    outro: AudioClip | None = None,
    room_tone_db: float | None = -60.0,
    tail_s: float = 1.0,
    head_s: float = 0.3,
    acoustics=None,
) -> tuple[AudioClip, float]:
    """Returns the mixed episode and the offset (seconds) the dialogue was shifted by.

    ``acoustics`` (pipeline.audio.acoustics.Acoustics, performance prototype) adds per-speaker
    EQ + the shared recording profile per line, and a shared compressor and room on the dialogue bus.
    """
    offset = head_s
    intro_samples = None
    if intro is not None:
        intro_samples = resample(intro.samples, intro.sr, sr) if intro.sr != sr else intro.samples
        intro_samples = normalise_loudness(intro_samples, sr, target_lufs - 2)
        offset = head_s + max(0.0, len(intro_samples) / sr - 0.5)
    total = offset + timeline.duration + tail_s
    outro_samples = None
    if outro is not None:
        outro_samples = resample(outro.samples, outro.sr, sr) if outro.sr != sr else outro.samples
        outro_samples = normalise_loudness(outro_samples, sr, target_lufs - 2)
        total += len(outro_samples) / sr
    buffer = np.zeros(int(total * sr) + sr, dtype=np.float32)

    if intro_samples is not None:
        _add(buffer, fade(intro_samples, sr, 0.01, 0.5), int(head_s * sr))
    if acoustics is None:
        for placement in timeline.placements:
            _add(buffer, _prepare(placement, sr, line_lufs), int((placement.start + offset) * sr))
    else:
        dialogue = np.zeros_like(buffer)
        for placement in timeline.placements:
            _add(dialogue, _prepare(placement, sr, line_lufs, acoustics), int((placement.start + offset) * sr))
        buffer += acoustics.process_bus(dialogue, sr)
    if outro_samples is not None:
        _add(buffer, fade(outro_samples, sr, 0.3, 0.5), int((offset + timeline.duration + 0.4) * sr))
    if room_tone_db is not None:
        buffer += room_tone(len(buffer), sr, room_tone_db)

    end = int((offset + timeline.duration + tail_s) * sr) + (len(outro_samples) if outro_samples is not None else 0)
    buffer = buffer[: min(len(buffer), end)]
    buffer = normalise_loudness(buffer, sr, target_lufs)
    peak = float(np.max(np.abs(buffer))) if len(buffer) else 0.0
    if peak > 0.98:
        buffer = (buffer / peak * 0.98).astype(np.float32)
        timeline.qa.append(f"peak limited by {20 * np.log10(peak / 0.98):.1f} dB to avoid clipping")
    return AudioClip(buffer.astype(np.float32), sr), offset


def _add(buffer: np.ndarray, samples: np.ndarray, at: int) -> None:
    at = max(0, at)
    end = min(len(buffer), at + len(samples))
    if end > at:
        buffer[at:end] += samples[: end - at]


def load_sting(path: Path | None) -> AudioClip | None:
    if path is None or not Path(path).is_file():
        return None
    try:
        return AudioClip.read(path)
    except Exception as exc:  # pragma: no cover
        log.warning("could not read sting %s: %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# Transcript
# ---------------------------------------------------------------------------

def build_transcript(
    timeline: Timeline,
    script: Script,
    *,
    tier: str,
    offset: float,
    duration_s: float,
    blocks: list[Block] | None = None,
    block_paths: dict[str, str] | None = None,
) -> EpisodeTranscript:
    seg_of = {line.id: seg.type for seg in script.segments for line in seg.lines}
    lines: list[TranscriptLine] = []
    for p in timeline.placements:
        counts = p.turn.word_counts()
        words = p.words
        idx = 0
        for line, n_words, spoken in zip(p.turn.lines, counts, p.turn.spoken_lines, strict=True):
            chunk = words[idx: idx + n_words] if words else []
            idx += n_words
            if chunk:
                start, end = chunk[0].start, chunk[-1].end
            else:
                share_start = sum(counts[: p.turn.lines.index(line)]) / max(1, sum(counts))
                share_end = (sum(counts[: p.turn.lines.index(line) + 1])) / max(1, sum(counts))
                start = p.start + p.clip.duration_s * share_start
                end = p.start + p.clip.duration_s * share_end
            lines.append(TranscriptLine(
                line_id=line.id, speaker=line.speaker, text=spoken,
                start=round(start + offset, 3), end=round(min(end, p.end) + offset, 3),
                words=[TranscriptWord(word=w.word, start=round(w.start + offset, 3), end=round(w.end + offset, 3)) for w in chunk],
                overlap=line.overlap.mode, segment=seg_of.get(line.id),
            ))
    lines.sort(key=lambda l: l.start)
    spans: list[BlockSpan] = []
    if blocks:
        by_line = {l.line_id: l for l in lines}
        for b in blocks:
            present = [by_line[lid] for lid in b.line_ids if lid in by_line]
            if not present:
                continue
            spans.append(BlockSpan(block_id=b.id, line_ids=b.line_ids, start=min(l.start for l in present),
                                   end=max(l.end for l in present), path=(block_paths or {}).get(b.id)))
    return EpisodeTranscript(episode_id=script.episode_id, tier=tier, duration_s=round(duration_s, 3), lines=lines,  # type: ignore[arg-type]
                             blocks=spans, qa=list(timeline.qa))


def export_blocks(mixed: AudioClip, transcript: EpisodeTranscript, out_dir: Path, fmt: str = "mp3") -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for span in transcript.blocks:
        a = max(0, int((span.start - 0.15) * mixed.sr))
        b = min(len(mixed.samples), int((span.end + 0.3) * mixed.sr))
        clip = AudioClip(fade(mixed.samples[a:b], mixed.sr, 0.02, 0.05), mixed.sr)
        path = out_dir / f"{span.block_id}.{fmt}"
        clip.write(path)
        span.path = str(path)
        paths[span.block_id] = str(path)
    return paths
