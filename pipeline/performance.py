"""Performance labels -> numbers. The writer picks labels; this maps them per character.

Defaults live here as a fallback; cast/hosts.yaml carries the tuned table per
host under ``performance:`` and is merged over these. Inside a timing or pause
range the exact value comes from a seeded hash, so a rerun is identical and
the label, not free randomness, decides the variation.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from typing import get_args

from pipeline.models import Delivery, Guest, Host, Line, Mood, Phrase, PhrasePause, Reaction, Timing

DELIVERIES: tuple[str, ...] = get_args(Delivery)
MOODS: tuple[str, ...] = get_args(Mood)
TIMINGS: tuple[str, ...] = get_args(Timing)
PHRASE_PAUSES: tuple[str, ...] = get_args(PhrasePause)
REACTIONS: tuple[str, ...] = get_args(Reaction)

DEFAULT_PERFORMANCE: dict = {
    "delivery": {
        "explain": {"rate": 0.94, "exaggeration": -0.05},
        "think": {"rate": 0.88, "exaggeration": -0.10},
        "excite": {"rate": 1.08, "exaggeration": 0.15},
        "react": {"rate": 1.02, "exaggeration": 0.05},
        "interrupt": {"rate": 1.10, "exaggeration": 0.10},
        "realize": {"rate": 0.90, "exaggeration": 0.05},
        "disagree": {"rate": 1.05, "exaggeration": 0.08},
        "setup": {"rate": 0.93, "exaggeration": 0.0},
        "punchline": {"rate": 1.06, "exaggeration": 0.12},
    },
    "mood": {"confident": 0.0, "challenged": 0.05, "surprised": 0.15, "amused": 0.10, "thoughtful": -0.10, "calm": -0.08},
    "timing": {"immediate": [-0.08, 0.10], "hesitate": [0.35, 0.6], "search": [0.6, 0.9], "deliberate": [0.9, 1.3]},
    "phrase_pause": {"none": [0.0, 0.0], "short": [0.12, 0.2], "beat": [0.35, 0.55]},
}


def merged_table(speaker: Host | Guest | None) -> dict:
    table = copy.deepcopy(DEFAULT_PERFORMANCE)
    for key, value in ((speaker.performance if speaker else None) or {}).items():
        if isinstance(value, dict) and isinstance(table.get(key), dict):
            for label, v in value.items():
                if isinstance(v, dict) and isinstance(table[key].get(label), dict):
                    table[key][label].update(v)
                else:
                    table[key][label] = v
        else:
            table[key] = value
    return table


def seeded_fraction(*parts: object) -> float:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def pick_in_range(bounds, *seed_parts: object) -> float:
    lo, hi = float(bounds[0]), float(bounds[1])
    return lo + (hi - lo) * seeded_fraction(*seed_parts)


@dataclass(frozen=True)
class PhraseValues:
    rate: float
    pause_after_s: float


def exaggeration_delta(table: dict, line: Line) -> float:
    delta = 0.0
    if line.delivery:
        delta += float(table["delivery"].get(line.delivery, {}).get("exaggeration", 0.0))
    if line.mood:
        delta += float(table["mood"].get(line.mood, 0.0))
    return delta


def delivery_rate(table: dict, delivery: str | None) -> float:
    if not delivery:
        return 1.0
    return float(table["delivery"].get(delivery, {}).get("rate", 1.0))


def phrases_of(line: Line) -> list[Phrase]:
    """The line as phrases: its own split when it has one, else the whole line as one phrase."""
    if line.phrases:
        return list(line.phrases)
    return [Phrase(text=line.text, delivery=line.delivery)]


def phrase_values(table: dict, line: Line, phrase: Phrase, index: int, seed: int) -> PhraseValues:
    rate = delivery_rate(table, phrase.delivery or line.delivery)
    bounds = table["phrase_pause"].get(phrase.pause_after, [0.0, 0.0])
    return PhraseValues(rate=rate, pause_after_s=pick_in_range(bounds, seed, line.id, index, "pause"))


def timing_gap(table: dict, line: Line, seed: int) -> float | None:
    if not line.timing or line.timing not in table["timing"]:
        return None
    return pick_in_range(table["timing"][line.timing], seed, line.id, "timing")
