"""Characters: make a guest, edit a host or a guest, give them a reference clip.

The cast lives in ``cast/hosts.yaml`` and ``cast/guests.yaml``, files people also edit by hand and annotate with
comments. An edit from the web app therefore rewrites only the fields that changed: a character's block is split
into one chunk per field (with the comment lines above it), changed chunks are written again, everything else
stays byte for byte. The file is read back as a cast before it is kept; one that does not load is put back.

The writer works with exactly two hosts, one explainer and one skeptic, so a new character is a guest (an expert
the writer brings in when a chapter needs one). Hosts are edited, not added.
"""

from __future__ import annotations

import re
import shutil
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from pipeline.models import Cast, Guest, Host
from pipeline.paths import validate_id

HOST_FIELDS: dict[str, type] = {
    "name": str, "background": str, "strength": str, "weakness": str, "humour": str, "wants": str,
    "voice": list, "life": list, "stories": list, "behaviour": list, "never": list, "opinions": list,
    "tics": list, "tic_markers": list,
}
GUEST_FIELDS: dict[str, type] = {"name": str, "domains": list, "persona": str, "tics": list, "tic_markers": list}


class _Dumper(yaml.SafeDumper):
    """Lists indented under their key, like the hand-written files; text with line breaks as a block."""

    def increase_indent(self, flow=False, indentless=False):  # noqa: ARG002 - pyyaml's signature
        return super().increase_indent(flow, False)


def _str(dumper, data: str):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|" if "\n" in data else None)


_Dumper.add_representer(str, _str)


def _dump(value: Any, indent: int) -> list[str]:
    text = yaml.dump(value, Dumper=_Dumper, allow_unicode=True, sort_keys=False, width=110, default_flow_style=False)
    return [(" " * indent + ln if ln.strip() else ln) + "\n" for ln in text.splitlines()]


def find(cast: Cast, character_id: str) -> tuple[str, Host | Guest]:
    for h in cast.hosts:
        if h.id == character_id:
            return "host", h
    for g in cast.guests:
        if g.id == character_id:
            return "guest", g
    raise KeyError(character_id)


def describe(cast: Cast, character_id: str) -> dict[str, Any]:
    kind, ch = find(cast, character_id)
    fields = HOST_FIELDS if kind == "host" else GUEST_FIELDS
    data = ch.model_dump()
    return {"kind": kind, "id": ch.id, "role": getattr(ch, "role", None), "voice_ref": ch.voice_ref,
            "fields": {k: data.get(k) for k in fields}}


def clean_changes(kind: str, raw: dict[str, Any]) -> dict[str, Any]:
    """Only the fields a character of this kind has, as the right type: text trimmed, lists without empty lines."""
    fields = HOST_FIELDS if kind == "host" else GUEST_FIELDS
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if key not in fields:
            continue
        if fields[key] is list:
            items = value.splitlines() if isinstance(value, str) else list(value or [])
            out[key] = [str(i).strip() for i in items if str(i).strip()]
        else:
            out[key] = " ".join(str(value or "").split()) if key == "name" else str(value or "").strip()
    if "name" in out and not out["name"]:
        raise ValueError("Een personage heeft een naam nodig.")
    return out


# ------------------------------------------------------------------ the file
def _entry_span(lines: list[str], character_id: str) -> tuple[int, int, int]:
    """(first line, end, dash indent) of the list item ``- id: <character_id>``."""
    pattern = re.compile(rf"^(\s*)-\s+id:\s*['\"]?{re.escape(character_id)}['\"]?\s*(#.*)?$")
    start = next((i for i, ln in enumerate(lines) if pattern.match(ln.rstrip("\n"))), None)
    if start is None:
        raise KeyError(character_id)
    dash = len(lines[start]) - len(lines[start].lstrip())
    end = len(lines)
    for i in range(start + 1, len(lines)):
        stripped = lines[i].strip()
        if not stripped or stripped.startswith("#"):
            continue
        if len(lines[i]) - len(lines[i].lstrip()) <= dash:
            end = i
            break
    # Comment lines right before the next item or top-level key belong to that, not to this character.
    while end - 1 > start and (not lines[end - 1].strip() or lines[end - 1].lstrip().startswith("#")):
        indent = len(lines[end - 1]) - len(lines[end - 1].lstrip())
        if lines[end - 1].strip() and indent > dash + 2:
            break
        end -= 1
    return start, end, dash


def _chunks(block: list[str], key_indent: int) -> list[dict]:
    """Split a character's block (without its ``- id:`` line) into one chunk per field."""
    key_line = re.compile(rf"^ {{{key_indent}}}([A-Za-z_][\w-]*):")
    chunks: list[dict] = []
    pending: list[str] = []
    for line in block:
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())
        if not stripped or (stripped.startswith("#") and indent <= key_indent):
            pending.append(line)
            continue
        m = key_line.match(line)
        if m and indent == key_indent:
            chunks.append({"key": m.group(1), "pre": pending, "lines": [line]})
            pending = []
        elif chunks:
            chunks[-1]["lines"].extend(pending + [line])
            pending = []
        else:
            pending.append(line)
    if pending:
        chunks.append({"key": None, "pre": pending, "lines": []})
    return chunks


def _write_checked(path: Path, text: str, cast_dir: Path, check) -> None:
    from pipeline.script.cast import load_cast

    backup = path.with_suffix(path.suffix + ".bak")
    had = path.is_file()
    if had:
        shutil.copyfile(path, backup)
    path.write_text(text, encoding="utf-8")
    try:
        check(load_cast(cast_dir))
    except Exception:
        if had:
            shutil.copyfile(backup, path)
        else:
            path.unlink(missing_ok=True)
        backup.unlink(missing_ok=True)
        raise
    backup.unlink(missing_ok=True)


def _file_for(cast_dir: Path, kind: str) -> Path:
    return cast_dir / ("hosts.yaml" if kind == "host" else "guests.yaml")


def update(cast_dir: Path, cast: Cast, character_id: str, changes: dict[str, Any]) -> dict[str, Any]:
    """Write the changed fields of one character; returns what was written."""
    kind, current = find(cast, character_id)
    # voice_ref only comes from install_clip; everything else is what a person may edit.
    internal = {"voice_ref": changes["voice_ref"]} if "voice_ref" in changes else {}
    changes = clean_changes(kind, changes) | internal
    data = current.model_dump()
    changes = {k: v for k, v in changes.items() if data.get(k) != v}
    if not changes:
        return {}
    path = _file_for(cast_dir, kind)
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    start, end, dash = _entry_span(lines, character_id)
    key_indent = dash + 2
    chunks = _chunks(lines[start + 1:end], key_indent)
    seen = set()
    for chunk in chunks:
        if chunk["key"] in changes:
            chunk["lines"] = _dump({chunk["key"]: changes[chunk["key"]]}, key_indent)
            seen.add(chunk["key"])
    tail = chunks.pop() if chunks and chunks[-1]["key"] is None else None
    for key, value in changes.items():
        if key not in seen:
            chunks.append({"key": key, "pre": [], "lines": _dump({key: value}, key_indent)})
    if tail:
        chunks.append(tail)
    block = [ln for c in chunks for ln in c["pre"] + c["lines"]]
    text = "".join(lines[:start + 1] + block + lines[end:])

    def check(cast2: Cast) -> None:
        _, ch = find(cast2, character_id)
        for key, value in changes.items():
            if getattr(ch, key) != value:
                raise ValueError(f"{key} kwam niet goed terug na het opslaan")

    _write_checked(path, text, cast_dir, check)
    return changes


def slug(name: str, taken: set[str]) -> str:
    base = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    base = re.sub(r"[^a-z0-9]+", "_", base).strip("_")[:40] or "gast"
    out, n = base, 2
    while out in taken:
        out, n = f"{base}_{n}", n + 1
    return validate_id(out)


def create_guest(cast_dir: Path, cast: Cast, fields: dict[str, Any]) -> str:
    """Add a guest to guests.yaml; returns its id."""
    changes = clean_changes("guest", fields)
    if not changes.get("name"):
        raise ValueError("Een personage heeft een naam nodig.")
    taken = {h.id for h in cast.hosts} | {g.id for g in cast.guests}
    new_id = slug(changes["name"], taken)
    entry = {"id": new_id, "name": changes["name"], "domains": changes.get("domains", []),
             "voice_ref": f"refs/{new_id}.wav", "persona": changes.get("persona", ""),
             "tics": changes.get("tics", []), "tic_markers": changes.get("tic_markers", []), "exaggeration": 0.4}
    path = cast_dir / "guests.yaml"
    text = path.read_text(encoding="utf-8") if path.is_file() else "guests:\n"
    if not text.endswith("\n"):
        text += "\n"
    if not re.search(r"^guests:\s*$", text, re.M):
        text += "guests:\n"
    text += "".join(_dump([entry], 2))

    def check(cast2: Cast) -> None:
        find(cast2, new_id)

    _write_checked(path, text, cast_dir, check)
    return new_id


def install_clip(cast_dir: Path, cast: Cast, character_id: str, clip: Path) -> Path:
    """Make ``clip`` the character's reference clip. The old one is kept in ``cast/refs/backup/``."""
    _, ch = find(cast, character_id)
    rel = ch.voice_ref or f"refs/{character_id}.wav"
    target = cast_dir / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        backup = cast_dir / "refs" / "backup" / f"{character_id}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}.wav"
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(target, backup)
    shutil.copyfile(clip, target)
    if ch.voice_ref != rel:
        update(cast_dir, cast, character_id, {"voice_ref": rel})
    return target
