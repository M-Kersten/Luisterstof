"""Line tag vocabulary shared by the writer, the linter and the audio stage.

Each tag maps to an exaggeration delta for Chatterbox and to an ElevenLabs v3
audio tag (English words, even inside Dutch text).
"""

from __future__ import annotations

TAGS: dict[str, dict[str, float | str | None]] = {
    "excited": {"exaggeration": 0.20, "eleven": "[excited]"},
    "laughs": {"exaggeration": 0.25, "eleven": "[laughs]"},
    "laughing": {"exaggeration": 0.25, "eleven": "[laughing]"},
    "deadpan": {"exaggeration": -0.20, "eleven": "[deadpan]"},
    "skeptical": {"exaggeration": -0.10, "eleven": "[skeptical]"},
    "thinking": {"exaggeration": -0.10, "eleven": "[thoughtful]"},
    "sighs": {"exaggeration": -0.05, "eleven": "[sighs]"},
    "serious": {"exaggeration": -0.10, "eleven": "[serious]"},
    "warm": {"exaggeration": 0.05, "eleven": "[warmly]"},
    "annoyed": {"exaggeration": 0.10, "eleven": "[annoyed]"},
    "surprised": {"exaggeration": 0.20, "eleven": "[surprised]"},
    "whisper": {"exaggeration": -0.25, "eleven": "[whispers]"},
    "fast": {"exaggeration": 0.10, "eleven": None},
    "slow": {"exaggeration": -0.10, "eleven": None},
    "question": {"exaggeration": 0.05, "eleven": None},
}

OVERLAP_ELEVEN = {"interrupt": "[interrupting]", "backchannel": "[overlapping]"}


def known_tags() -> list[str]:
    return sorted(TAGS)


def exaggeration_for(tags: list[str], base: float, lo: float = 0.15, hi: float = 0.95) -> float:
    delta = sum(float(TAGS[t]["exaggeration"]) for t in tags if t in TAGS)  # type: ignore[arg-type]
    return max(lo, min(hi, base + delta))


def eleven_tags(tags: list[str], overlap_mode: str = "none") -> list[str]:
    out: list[str] = []
    if overlap_mode in OVERLAP_ELEVEN:
        out.append(OVERLAP_ELEVEN[overlap_mode])
    for t in tags:
        tag = TAGS.get(t, {}).get("eleven")
        if tag and tag not in out:
            out.append(str(tag))
    return out
