"""The sample library behind the jingle: VSCO 2 Community Edition (Versilian Studios, CC0), its string patches.

Reads the patch files (SFZ: which recording covers which keys, at which dynamic, in which turn) and fetches the
patches and their recordings on first use.
"""

from __future__ import annotations

import re
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

BASE = "https://raw.githubusercontent.com/sgossner/VSCO-2-CE/SFZ"


@dataclass
class Region:
    sample: str  # path relative to the library root, forward slashes
    lokey: int
    hikey: int
    center: int
    lovel: int = 0
    hivel: int = 127
    volume: float = 0.0
    tune: float = 0.0
    seq_position: int = 1
    seq_length: int = 1


def parse(path: Path) -> list[Region]:
    text = re.sub(r"//[^\n]*", "", path.read_text(errors="replace"))
    default_path = ""
    m = re.search(r"default_path=([^\n\r]+)", text)
    if m:
        default_path = m.group(1).strip().replace("\\", "/")
    regions: list[Region] = []
    scope: dict[str, dict[str, str]] = {"global": {}, "group": {}}
    for header, body in re.findall(r"<(\w+)>([^<]*)", text):
        opcodes = dict(re.findall(r"(\w+)=([^\n\r]+?)(?=\s+\w+=|\s*$)", body, flags=re.M))
        opcodes = {k: v.strip() for k, v in opcodes.items()}
        if header == "global":
            scope["global"] = opcodes
            scope["group"] = {}
        elif header == "group":
            scope["group"] = opcodes
        elif header == "region":
            op = {**scope["global"], **scope["group"], **opcodes}
            if "sample" not in op:
                continue
            key = op.get("key")
            center = int(op.get("pitch_keycenter", key or 60))
            regions.append(Region(
                sample=default_path + op["sample"].replace("\\", "/"),
                lokey=int(op.get("lokey", key or center)),
                hikey=int(op.get("hikey", key or center)),
                center=center,
                lovel=int(op.get("lovel", 0)),
                hivel=int(op.get("hivel", 127)),
                volume=float(op.get("volume", 0)),
                tune=float(op.get("tune", 0)),
                seq_position=int(op.get("seq_position", 1)),
                seq_length=int(op.get("seq_length", 1)),
            ))
    return regions


def _download(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for _ in range(3):
        try:
            urllib.request.urlretrieve(url, part)
            part.replace(dest)
            return True
        except OSError:
            continue
    part.unlink(missing_ok=True)
    return False


def fetch(root: Path, patches: list[str]) -> None:
    """Make sure the patch files and every recording they name are in ``root`` (about 400 MB the first time)."""
    for name in patches:
        patch = root / f"{name}.sfz"
        if not patch.is_file() and not _download(f"{BASE}/{name}.sfz", patch):
            raise SystemExit(f"could not fetch {name}.sfz from {BASE}")
        todo = [r for r in parse(patch) if not (root / r.sample).is_file()]
        if todo:
            print(f"fetching {len(todo)} recordings for {name}", file=sys.stderr)
        for r in todo:
            if not _download(f"{BASE}/{quote(r.sample)}", root / r.sample):
                raise SystemExit(f"could not fetch {r.sample} from {BASE}")
