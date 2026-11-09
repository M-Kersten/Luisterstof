"""Line tag vocabulary shared by the writer, the linter and the audio stage.

Each tag maps to an exaggeration delta for Chatterbox.
"""

from __future__ import annotations

TAGS: dict[str, dict[str, float]] = {
    "excited": {"exaggeration": 0.20},
    "laughs": {"exaggeration": 0.25},
    "laughing": {"exaggeration": 0.25},
    "deadpan": {"exaggeration": -0.20},
    "skeptical": {"exaggeration": -0.10},
    "thinking": {"exaggeration": -0.10},
    "sighs": {"exaggeration": -0.05},
    "serious": {"exaggeration": -0.10},
    "warm": {"exaggeration": 0.05},
    "annoyed": {"exaggeration": 0.10},
    "surprised": {"exaggeration": 0.20},
    "whisper": {"exaggeration": -0.25},
    "fast": {"exaggeration": 0.10},
    "slow": {"exaggeration": -0.10},
    "question": {"exaggeration": 0.05},
}


def known_tags() -> list[str]:
    return sorted(TAGS)


def exaggeration_for(tags: list[str], base: float, lo: float = 0.15, hi: float = 0.95) -> float:
    delta = sum(float(TAGS[t]["exaggeration"]) for t in tags if t in TAGS)  # type: ignore[arg-type]
    return max(lo, min(hi, base + delta))
