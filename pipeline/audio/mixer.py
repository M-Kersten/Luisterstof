"""Mixing: loudness, fades, ducking, room tone, stings, final render, transcript."""

from __future__ import annotations

import logging
import zlib
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
    """Fade in and out; mono, or (n, 2) with both channels together."""
    out = samples.astype(np.float32, copy=True)
    n_in = min(len(out), int(fade_in * sr))
    n_out = min(len(out), int(fade_out * sr))
    shape = (-1,) + (1,) * (out.ndim - 1)
    if n_in > 0:
        out[:n_in] *= np.linspace(0.0, 1.0, n_in, dtype=np.float32).reshape(shape)
    if n_out > 0:
        out[-n_out:] *= np.linspace(1.0, 0.0, n_out, dtype=np.float32).reshape(shape)
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
    from scipy.signal import lfilter

    rng = np.random.default_rng(seed)
    white = rng.standard_normal(n).astype(np.float32)
    alpha = 0.02  # one-pole low-pass for a brownish, unobtrusive floor
    out = lfilter([alpha], [1.0, -(1.0 - alpha)], white.astype(np.float64)).astype(np.float32)
    rms = float(np.sqrt(np.mean(out**2)) + 1e-9)
    return out * (db_to_gain(level_db) / rms)


def _prepare(placement: Placement, sr: int, line_lufs: float, acoustics=None, finish=None) -> np.ndarray:
    clip = placement.clip
    samples = resample(clip.samples, clip.sr, sr) if clip.sr != sr else clip.samples
    if acoustics is not None:
        samples = acoustics.process_speaker(placement.turn.speaker, samples, sr)  # before loudness, so levels stay even
    elif finish is not None:
        from pipeline.audio import finish as sound

        samples = sound.voice(samples, sr, finish)  # the shared microphone, before the line's loudness is set
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
    finish=None,
    pans: dict[str, float] | None = None,
    breaths: dict | None = None,
    seed: int = 7,
) -> tuple[AudioClip, float]:
    """Returns the mixed episode and the offset (seconds) the dialogue was shifted by.

    ``acoustics`` (pipeline.audio.acoustics.Acoustics, performance prototype) adds per-speaker
    EQ + the shared recording profile per line, and a shared compressor and room on the dialogue bus.
    ``finish`` (pipeline.audio.finish.Finish, the sound of the show from cast/klank.yaml) does the same with the
    show's own settings: microphone, warmth and tamed s-sounds per line, compressor and room on the dialogue, a
    limiter at the end. With width, the episode comes out stereo with each speaker at ``pans[speaker]``; with
    breath, ``breaths[speaker]`` (pipeline.audio.breaths.Breaths) gives the inhales before longer lines. A dry
    Finish leaves the episode exactly as without it.
    """
    if finish is not None and finish.dry:
        finish = None
    if acoustics is not None:
        finish = None
    stereo = finish is not None and finish.stereo
    pans = pans or {}
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
    n = int(total * sr) + sr
    buffer = np.zeros((n, 2) if stereo else n, dtype=np.float32)

    if intro_samples is not None:
        _add(buffer, fade(intro_samples, sr, 0.01, 0.5), int(head_s * sr))
    if finish is not None:
        from pipeline.audio import finish as sound

        dialogue = np.zeros_like(buffer)
        for placement in timeline.placements:
            line = _prepare(placement, sr, line_lufs, finish=finish)
            pan = pans.get(placement.turn.speaker, 0.0)
            _add(dialogue, sound.place(line, pan) if stereo else line, int((placement.start + offset) * sr))
        if finish.breath > 0 and breaths:
            for at, inhale, speaker in _breaths(timeline, breaths, finish, sr, line_lufs, seed):
                inhale = sound.voice(inhale, sr, finish)
                pan = pans.get(speaker, 0.0)
                _add(dialogue, sound.place(inhale, pan) if stereo else inhale, int((at + offset) * sr))
        buffer += sound.bus(dialogue, sr, finish)  # one compressor, one room for everyone
    elif acoustics is None:
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
        if stereo:  # two quiet floors that do not match: the room is around you, not in the middle of your head
            buffer += np.stack([room_tone(len(buffer), sr, room_tone_db, seed=1),
                                room_tone(len(buffer), sr, room_tone_db, seed=2)], axis=1)
        else:
            buffer += room_tone(len(buffer), sr, room_tone_db)

    end = int((offset + timeline.duration + tail_s) * sr) + (len(outro_samples) if outro_samples is not None else 0)
    buffer = buffer[: min(len(buffer), end)]
    buffer = normalise_loudness(buffer, sr, target_lufs)
    if finish is not None:
        from pipeline.audio import finish as sound

        buffer = sound.limit(buffer, sr, finish)  # peaks down, not the whole episode
    peak = float(np.max(np.abs(buffer))) if len(buffer) else 0.0
    if peak > 0.98:
        buffer = (buffer / peak * 0.98).astype(np.float32)
        timeline.qa.append(f"peak limited by {20 * np.log10(peak / 0.98):.1f} dB to avoid clipping")
    return AudioClip(buffer.astype(np.float32), sr), offset


def _add(buffer: np.ndarray, samples: np.ndarray, at: int) -> None:
    """Mix ``samples`` in at ``at``; a mono clip in a stereo buffer goes to the middle, equally loud."""
    at = max(0, at)
    end = min(len(buffer), at + len(samples))
    if end <= at:
        return
    part = samples[: end - at]
    if buffer.ndim == 2 and part.ndim == 1:
        part = part[:, None] * np.float32(np.sqrt(0.5))
    buffer[at:end] += part


MIN_BREATH_S = 0.12
BREATH_MIN_LINE_S = 2.5
BREATH_MIN_GAP_S = 0.22
BREATH_AFTER_PREVIOUS_S = 0.05  # never right on top of the last word of the line before


def _breaths(timeline: Timeline, breaths: dict, finish, sr: int, line_lufs: float, seed: int):
    """Where the speakers breathe: before a longer line that moves the conversation on, when there is room for it.
    The same script gives the same breaths every time. Yields (start in seconds, samples, speaker)."""
    from pipeline.audio import finish as sound

    order = sorted(timeline.placements, key=lambda p: p.start)
    prev_end = 0.0
    for p in order:
        bank = breaths.get(p.turn.speaker)
        line_s = p.end - p.start
        gap = p.start - prev_end
        prev_end = max(prev_end, p.end)
        if bank is None or not bank.clips or not p.advances or p.turn.first.reaction or line_s < BREATH_MIN_LINE_S:
            continue
        if gap < BREATH_MIN_GAP_S:
            continue
        roll = zlib.crc32(f"{seed}:{p.turn.turn_id}".encode()) / 2**32
        if roll >= sound.breath_chance(finish, line_s):
            continue
        clip = bank.clips[zlib.crc32(f"{p.turn.turn_id}:{seed}".encode()) % len(bank.clips)]
        clip = resample(clip, bank.sr, sr) if bank.sr != sr else clip
        room = gap - sound.BREATH_LEAD_S - BREATH_AFTER_PREVIOUS_S
        n = min(len(clip), int(room * sr))
        if n < int(MIN_BREATH_S * sr):
            continue
        inhale = fade(clip[len(clip) - n:], sr, 0.015, 0.0)  # cut from the front: the end of a breath leads into the word
        inhale = sound.breath_level(inhale, line_lufs + p.gain_db, finish)
        yield p.start - sound.BREATH_LEAD_S - n / sr, inhale, p.turn.speaker


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
    return EpisodeTranscript(episode_id=script.episode_id, duration_s=round(duration_s, 3), lines=lines,
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
