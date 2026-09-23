"""Timeline assembly: where every rendered turn starts, how overlaps land.

* Normal turn: gap sampled from 180-420 ms. A fixed gap is the clearest tell.
* Quick handoff: a normal turn whose prior line reads as expecting an
  immediate reply (ends in "?", or is short and reactive) gets a much
  tighter, sometimes genuinely overlapping gap instead, with a gentle duck
  on the outgoing tail. Every line is still synthesized in isolation (the
  model has no cross-speaker context to draw on), so without this the
  timing itself is the only thing that can read as conversational momentum
  rather than turn-taking narration; a uniform 180-420ms silence before
  every single handoff is what makes back-and-forth exchanges sound like
  separate monologues even when the words are right.
* interrupt: line B starts 250 ms before the cut word's onset in A; A is
  ducked 12 dB with an 80 ms fade and its buried fragment runs 400-600 ms
  underneath before fading out.
* backchannel: placed at the nearest word boundary past 60% of A, mixed at
  -8 dB, does not advance the timeline. Laughter goes slightly earlier.
* Segment boundaries get a longer, deliberate gap. Quiz pauses come from
  ``pause_after_ms`` on the line.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from pipeline.audio.asr import WordTiming
from pipeline.audio.synth import AudioClip
from pipeline.audio.turns import Turn
from pipeline.audio.wer import normalise
from pipeline.cues import reaction_of
from pipeline.models import Line

GAP_RANGE = (0.18, 0.42)
SEGMENT_GAP_RANGE = (0.6, 0.9)
QUICK_GAP_RANGE = (-0.06, 0.09)  # a "yes, and-" handoff: often a touch of real overlap, never a long silence
QUICK_DUCK_DB = 4.0  # gentle, nowhere near INTERRUPT_DUCK_DB: blends the handoff, never buries a word
QUICK_DUCK_FADE = 0.05
QUICK_WORD_LIMIT = 6
INTERRUPT_LEAD = 0.25
INTERRUPT_DUCK_DB = 12.0
INTERRUPT_DUCK_FADE = 0.08
FRAGMENT_RANGE = (0.4, 0.6)
FRAGMENT_FADE = 0.12
BACKCHANNEL_GAIN_DB = -8.0
BACKCHANNEL_POINT = 0.6
LAUGH_POINT = 0.5


def is_quick_handoff(line: Line) -> bool:
    """True when the line reads as expecting an immediate reply, not a considered pause.

    A question invites an answer right away; a short reactive line ("Ja, precies.",
    "Wacht even.") is itself the kind of thing that gets said quickly in response to
    something, and tends to be replied to just as quickly in turn.
    """
    text = line.text.strip()
    if not text:
        return False
    if text.endswith("?"):
        return True
    return len(re.findall(r"\w+", text)) <= QUICK_WORD_LIMIT


@dataclass
class Placement:
    turn: Turn
    clip: AudioClip
    start: float
    words: list[WordTiming]  # absolute times
    gain_db: float = 0.0
    duck_at: float | None = None  # absolute time where ducking starts
    duck_db: float = 0.0
    duck_fade: float = INTERRUPT_DUCK_FADE
    cut_at: float | None = None  # absolute time where the clip is faded out
    cut_fade: float = FRAGMENT_FADE
    advances: bool = True
    deliberate_gap_after: float = 0.0

    @property
    def end(self) -> float:
        end = self.start + self.clip.duration_s
        return min(end, self.cut_at + self.cut_fade) if self.cut_at is not None else end

    @property
    def line_ids(self) -> list[str]:
        return self.turn.line_ids


@dataclass
class Timeline:
    placements: list[Placement] = field(default_factory=list)
    sr: int = 24000
    qa: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max((p.end for p in self.placements), default=0.0)

    def by_line(self) -> dict[str, Placement]:
        return {lid: p for p in self.placements for lid in p.line_ids}


def _find_cut_onset(prev: Placement, cut_word: str | None) -> float:
    words = prev.words
    if not words:
        return prev.start + prev.clip.duration_s * 0.85
    if cut_word:
        target = normalise(cut_word)
        if target:
            for w in reversed(words):
                if normalise(w.word) == target:
                    return w.start
    return words[-1].start


def _word_boundary_after(prev: Placement, point: float) -> float:
    for w in prev.words:
        if w.start >= point:
            return w.start
    return point


def assemble(
    turns: list[Turn],
    rendered: dict[str, tuple[AudioClip, list[WordTiming]]],
    *,
    sr: int,
    seed: int = 7,
    max_unwritten_gap: float = 1.2,
    timing_gap: Callable[[Turn], float | None] | None = None,
) -> Timeline:
    """``timing_gap`` (performance prototype) returns the response gap a line's timing label asks
    for, or None; it replaces the sampled gap on normal transitions only. Interrupts, backchannels
    and segment boundaries keep their own rules."""
    rng = random.Random(seed)
    tl = Timeline(sr=sr)
    cursor = 0.0
    prev_advancing: Placement | None = None
    prev_any: Placement | None = None
    pending_gap = 0.0  # deliberate gap requested by the previous line/segment

    for turn in turns:
        if turn.turn_id not in rendered:
            tl.qa.append(f"turn {turn.turn_id} ({turn.speaker}) has no audio")
            continue
        clip, rel_words = rendered[turn.turn_id]
        mode = turn.first.overlap.mode
        target_ok = prev_any is not None and turn.first.overlap.target in prev_any.line_ids and prev_any.turn.speaker != turn.speaker

        if mode == "interrupt" and target_ok and prev_any is not None:
            onset = _find_cut_onset(prev_any, turn.first.overlap.cut_word)
            start = max(prev_any.start + 0.1, onset - INTERRUPT_LEAD)
            prev_any.duck_at = start
            prev_any.duck_db = INTERRUPT_DUCK_DB
            fragment = rng.uniform(*FRAGMENT_RANGE)
            prev_any.cut_at = min(prev_any.start + prev_any.clip.duration_s, onset + fragment)
            placement = Placement(turn, clip, start, [w.shifted(start) for w in rel_words])
            tl.placements.append(placement)
            cursor = placement.end
            prev_advancing = placement
            prev_any = placement
            pending_gap = turn.pause_after_ms / 1000.0
            continue

        if mode == "backchannel" and target_ok and prev_any is not None:
            laugh = any(t in ("laughs", "laughing") for t in turn.tags) or reaction_of(turn.first) in ("laugh", "chuckle")
            point_share = LAUGH_POINT if laugh else BACKCHANNEL_POINT
            point = prev_any.start + prev_any.clip.duration_s * point_share
            start = _word_boundary_after(prev_any, point)
            placement = Placement(turn, clip, start, [w.shifted(start) for w in rel_words], gain_db=BACKCHANNEL_GAIN_DB, advances=False)
            tl.placements.append(placement)
            # does not advance: cursor stays at the previous advancing end
            prev_any = prev_advancing or placement
            continue

        if mode != "none" and not target_ok:
            tl.qa.append(f"overlap on {turn.first.id} ignored: target not the previous turn or same speaker")

        segment_change = prev_advancing is not None and prev_advancing.turn.segment_index != turn.segment_index
        quick = (
            prev_advancing is not None
            and not segment_change
            and pending_gap == 0.0
            and prev_advancing.turn.speaker != turn.speaker
            and is_quick_handoff(prev_advancing.turn.last)
        )
        labelled = timing_gap(turn) if (timing_gap and prev_advancing is not None and not segment_change) else None
        if prev_advancing is None:
            gap = 0.0
        elif segment_change:
            gap = rng.uniform(*SEGMENT_GAP_RANGE)
        elif labelled is not None:
            gap = labelled
            quick = gap < 0
        elif quick:
            gap = rng.uniform(*QUICK_GAP_RANGE)
        else:
            gap = rng.uniform(*GAP_RANGE)
        gap += pending_gap
        start = cursor + gap
        placement = Placement(turn, clip, start, [w.shifted(start) for w in rel_words])
        if quick and gap < 0 and prev_advancing is not None:
            # a genuine brief overlap: duck the outgoing tail just enough to blend the
            # handoff, never enough to bury a word the way an interrupt's duck does
            prev_advancing.duck_at = start
            prev_advancing.duck_db = QUICK_DUCK_DB
            prev_advancing.duck_fade = QUICK_DUCK_FADE
        if prev_advancing is not None:
            written = segment_change or labelled is not None  # a timing label is a written beat, not an accident
            prev_advancing.deliberate_gap_after = pending_gap + (gap - pending_gap if written else 0.0)
        tl.placements.append(placement)
        cursor = placement.end
        prev_advancing = placement
        prev_any = placement
        pending_gap = turn.pause_after_ms / 1000.0

    tl.qa.extend(unwritten_gaps(tl, max_unwritten_gap))
    return tl


def unwritten_gaps(tl: Timeline, max_gap: float) -> list[str]:
    issues = []
    advancing = [p for p in tl.placements if p.advances]
    for a, b in zip(advancing, advancing[1:], strict=False):
        silence = b.start - a.end
        allowed = a.deliberate_gap_after + a.turn.pause_after_ms / 1000.0 + 0.45
        if silence > max_gap and silence > allowed:
            issues.append(f"gap of {silence:.2f}s between {a.turn.last.id} and {b.turn.first.id} was not written as a beat")
    return issues
