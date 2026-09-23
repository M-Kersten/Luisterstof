"""Word alignments are stored next to the take they belong to.

Alongside the words, ``fallback`` records whether they came from real forced
alignment or from UniformAligner's estimate, so that provenance survives a
cache hit on a later render (without it, fixing a broken WhisperX/MLX
install would silently keep reporting "aligned" for turns whose cached
timing was actually only ever estimated).
"""

from __future__ import annotations

import json

from pipeline.audio.asr import WordTiming
from pipeline.audio.cache import RenderCache


def _path(cache: RenderCache, take_key: str):
    return cache.root / take_key[:2] / f"{take_key}.align.json"


def load_alignment(cache: RenderCache, take_key: str) -> tuple[list[WordTiming], bool] | None:
    """Returns (words, was_fallback), or None on a cache miss / unreadable entry."""
    p = _path(cache, take_key)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            words_raw, fallback = data["words"], bool(data.get("fallback", False))
        else:
            words_raw, fallback = data, False  # pre-existing cache entry, format predates fallback tracking
        words = [WordTiming(str(w["word"]), float(w["start"]), float(w["end"])) for w in words_raw]
        return words, fallback
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def save_alignment(cache: RenderCache, take_key: str, words: list[WordTiming], fallback: bool = False) -> None:
    p = _path(cache, take_key)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"words": [{"word": w.word, "start": round(w.start, 4), "end": round(w.end, 4)} for w in words],
              "fallback": fallback}
    p.write_text(json.dumps(payload), encoding="utf-8")
