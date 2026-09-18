"""Content-hash cache so unchanged turns never re-render.

Two layers:
* takes: keyed on sha256(text + voice_ref + exaggeration + seed + synth)
* selections: keyed on the turn (text + voice_ref + base exaggeration + synth
  + loop parameters), pointing at the chosen take.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pipeline.audio.synth import AudioClip


class RenderCache:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def take_key(text: str, voice_ref_hash: str, exaggeration: float, seed: int, synth: str, extra: str = "") -> str:
        raw = "|".join([text, voice_ref_hash, f"{exaggeration:.3f}", str(seed), synth, extra])
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def selection_key(text: str, voice_ref_hash: str, base_exaggeration: float, synth: str, params: str) -> str:
        raw = "|".join(["sel", text, voice_ref_hash, f"{base_exaggeration:.3f}", synth, params])
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.wav"

    def meta_path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def has(self, key: str) -> bool:
        return self.path(key).is_file()

    def get(self, key: str) -> tuple[AudioClip, dict[str, Any]] | None:
        p = self.path(key)
        if not p.is_file():
            return None
        meta: dict[str, Any] = {}
        mp = self.meta_path(key)
        if mp.is_file():
            try:
                meta = json.loads(mp.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                meta = {}
        clip = AudioClip.read(p)
        clip.meta = {**meta, "cache_key": key}
        return clip, meta

    def put(self, key: str, clip: AudioClip, meta: dict[str, Any] | None = None) -> Path:
        p = self.path(key)
        clip.write(p)
        payload = {k: v for k, v in (meta or {}).items() if _jsonable(v)}
        payload.update({k: v for k, v in clip.meta.items() if _jsonable(v) and k not in payload})
        self.meta_path(key).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        return p

    def update_meta(self, key: str, **fields: Any) -> None:
        mp = self.meta_path(key)
        data = {}
        if mp.is_file():
            try:
                data = json.loads(mp.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                data = {}
        data.update({k: v for k, v in fields.items() if _jsonable(v)})
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    # Selections -----------------------------------------------------------
    def selection_path(self, key: str) -> Path:
        return self.root / "selections" / f"{key}.json"

    def get_selection(self, key: str) -> dict[str, Any] | None:
        p = self.selection_path(key)
        if not p.is_file():
            return None
        try:
            record = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        if not self.has(record.get("take_key", "")):
            return None
        return record

    def put_selection(self, key: str, record: dict[str, Any]) -> Path:
        p = self.selection_path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
        return p

    def stats(self) -> dict[str, int]:
        takes = sum(1 for _ in self.root.glob("*/*.wav"))
        selections = sum(1 for _ in (self.root / "selections").glob("*.json")) if (self.root / "selections").exists() else 0
        return {"takes": takes, "selections": selections}


def _jsonable(value: Any) -> bool:
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False
