"""Word alignments are stored next to the take they belong to."""

from __future__ import annotations

import json

from pipeline.audio.asr import WordTiming
from pipeline.audio.cache import RenderCache


def _path(cache: RenderCache, take_key: str):
    return cache.root / take_key[:2] / f"{take_key}.align.json"


def load_alignment(cache: RenderCache, take_key: str) -> list[WordTiming] | None:
    p = _path(cache, take_key)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return [WordTiming(str(w["word"]), float(w["start"]), float(w["end"])) for w in data]
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def save_alignment(cache: RenderCache, take_key: str, words: list[WordTiming]) -> None:
    p = _path(cache, take_key)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps([{"word": w.word, "start": round(w.start, 4), "end": round(w.end, 4)} for w in words]),
                 encoding="utf-8")
