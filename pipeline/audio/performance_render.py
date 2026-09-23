"""Performance prototype, render side: labels -> exaggeration, phrase re-timing, timing gaps, reactions.

Phrasing is hybrid: Chatterbox renders the whole turn in one pass (it has no
context between calls, so a separately generated fragment ends on sentence-final
intonation), then the audio is cut between aligned words and each phrase gets
its own rate and pause. That only works on real word timing; on estimated
(uniform) timing a cut lands mid-word, so the turn is left untouched.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pipeline.audio.asr import WordTiming, tokenize
from pipeline.audio.synth import AudioClip, apply_speech_rate
from pipeline.audio.turns import Turn, spoken_text
from pipeline.cues import reaction_of
from pipeline.models import Cast, Glossary, Line
from pipeline.performance import exaggeration_delta, merged_table, phrase_values, phrases_of, timing_gap

SPLIT_DELTA = 0.15  # a mood/delivery jump this large between two lines gets its own generation call
EDGE_FADE_S = 0.008


@dataclass(frozen=True)
class PerformanceOptions:
    delivery: bool = False
    phrasing: bool = False
    timing: bool = False
    reactions: bool = False
    acoustics: bool = False

    @classmethod
    def full(cls) -> PerformanceOptions:
        return cls(True, True, True, True, True)

    def label(self) -> str:
        on = [k for k in ("delivery", "phrasing", "timing", "reactions", "acoustics") if getattr(self, k)]
        return "+".join(on) or "none"


class Tables:
    """Merged label tables per speaker, built once per render."""

    def __init__(self, cast: Cast):
        self._by_id = {s.id: merged_table(s) for s in list(cast.hosts) + list(cast.guests)}
        self._default = merged_table(None)

    def __call__(self, speaker: str) -> dict:
        return self._by_id.get(speaker, self._default)


def split_hook(tables: Tables, options: PerformanceOptions):
    def split(prev: Line, line: Line) -> bool:
        if options.reactions and (reaction_of(prev) or reaction_of(line)):
            return True
        if options.timing and line.timing:
            return True  # a response timing is a gap before this line, so it needs a turn boundary
        if options.delivery:
            table = tables(line.speaker)
            return abs(exaggeration_delta(table, prev) - exaggeration_delta(table, line)) >= SPLIT_DELTA
        return False

    return split


def turn_exaggeration_delta(tables: Tables, turn: Turn) -> float:
    table = tables(turn.speaker)
    deltas = [exaggeration_delta(table, line) for line in turn.lines]
    return sum(deltas) / len(deltas) if deltas else 0.0


def turn_timing_gap(tables: Tables, seed: int):
    def gap(turn: Turn) -> float | None:
        return timing_gap(tables(turn.speaker), turn.first, seed)

    return gap


@dataclass
class PhraseSeg:
    words: int
    rate: float
    pause_after_s: float
    delivery: str | None


def phrase_plan(turn: Turn, tables: Tables, glossary: Glossary | None, seed: int) -> list[PhraseSeg]:
    table = tables(turn.speaker)
    segs: list[PhraseSeg] = []
    for line in turn.lines:
        phrases = phrases_of(line)
        for i, phrase in enumerate(phrases):
            probe = line.model_copy(update={"text": phrase.text, "phrases": []})
            spoken = spoken_text(probe, glossary) if i == len(phrases) - 1 else spoken_text_mid(probe, glossary)
            values = phrase_values(table, line, phrase, i, seed)
            segs.append(PhraseSeg(len(tokenize(spoken)), values.rate, values.pause_after_s, phrase.delivery or line.delivery))
    return [s for s in segs if s.words > 0]


def spoken_text_mid(line: Line, glossary: Glossary | None) -> str:
    """Like spoken_text, but a mid-line phrase keeps its own punctuation (only line ends lose a dash)."""
    from pipeline.cues import strip_cues
    from pipeline.plan.glossary import apply_lexicon

    text = strip_cues(line.text) or line.text.strip()
    return apply_lexicon(text, glossary) if glossary else text


def _stretch(clip: AudioClip, rate: float) -> tuple[AudioClip, float]:
    if abs(rate - 1.0) < 1e-6:
        return clip, 1.0
    try:
        return apply_speech_rate(clip, rate), rate
    except ImportError:  # librosa ships with the chatterbox install, not the lean dev one
        return clip, 1.0


def _edge_fade(samples: np.ndarray, sr: int, fade_in: bool, fade_out: bool) -> np.ndarray:
    out = samples.copy()
    n = min(len(out) // 2, int(EDGE_FADE_S * sr))
    if n > 0:
        ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
        if fade_in:
            out[:n] *= ramp
        if fade_out:
            out[-n:] *= ramp[::-1]
    return out


def edit_phrases(clip: AudioClip, words: list[WordTiming], segs: list[PhraseSeg]) -> tuple[AudioClip, list[WordTiming], list[float]]:
    """Cut between aligned words at phrase boundaries, re-time each piece, insert pauses.

    Returns the new clip, re-timed words, and the rate actually applied per phrase.
    """
    if sum(s.words for s in segs) != len(words):
        raise ValueError(f"phrases cover {sum(s.words for s in segs)} words, alignment has {len(words)}")
    if all(abs(s.rate - 1.0) < 1e-6 for s in segs) and all(s.pause_after_s <= 0 for s in segs[:-1]):
        return clip, words, [1.0] * len(segs)
    sr = clip.sr
    bounds = [0.0]
    idx = 0
    for seg in segs[:-1]:
        idx += seg.words
        bounds.append((words[idx - 1].end + words[idx].start) / 2)
    bounds.append(clip.duration_s)

    pieces: list[np.ndarray] = []
    new_words: list[WordTiming] = []
    applied: list[float] = []
    cursor = 0.0
    idx = 0
    for k, seg in enumerate(segs):
        a, b = bounds[k], bounds[k + 1]
        piece = AudioClip(clip.samples[int(a * sr):int(b * sr)], sr)
        stretched, rate = _stretch(piece, seg.rate)
        applied.append(rate)
        for w in words[idx:idx + seg.words]:
            new_words.append(WordTiming(w.word, cursor + (w.start - a) / rate, cursor + (w.end - a) / rate))
        idx += seg.words
        pieces.append(_edge_fade(stretched.samples, sr, fade_in=k > 0, fade_out=k < len(segs) - 1))
        cursor += stretched.duration_s
        if k < len(segs) - 1 and seg.pause_after_s > 0:
            pieces.append(np.zeros(int(seg.pause_after_s * sr), dtype=np.float32))
            cursor += seg.pause_after_s
    samples = np.concatenate(pieces).astype(np.float32) if pieces else clip.samples
    return AudioClip(samples, sr, meta=dict(clip.meta)), new_words, applied
