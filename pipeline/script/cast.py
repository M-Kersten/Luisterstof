"""Cast bible and continuity log.

Loads ``cast/hosts.yaml``, ``cast/guests.yaml`` and the last N entries of
``cast/continuity.jsonl``; renders them for the writer prompt; picks a guest
for a plan's expert domain.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from rapidfuzz import fuzz

from pipeline.models import Cast, ContinuityEntry, Guest, Host

CONTINUITY_FILE = "continuity.jsonl"
BANNED_FILE = "banned_phrases.yaml"


def load_cast(cast_dir: Path | str) -> Cast:
    cast_dir = Path(cast_dir)
    hosts_raw = yaml.safe_load((cast_dir / "hosts.yaml").read_text(encoding="utf-8")) or {}
    guests_path = cast_dir / "guests.yaml"
    guests_raw = yaml.safe_load(guests_path.read_text(encoding="utf-8")) if guests_path.is_file() else {}
    hosts = [Host.model_validate(h) for h in hosts_raw.get("hosts", [])]
    guests = [Guest.model_validate(g) for g in (guests_raw or {}).get("guests", [])]
    if len(hosts) < 2:
        raise ValueError("hosts.yaml must define at least two hosts")
    roles = {h.role for h in hosts}
    if "explainer" not in roles or "skeptic" not in roles:
        raise ValueError("hosts.yaml needs one explainer and one skeptic")
    return Cast(hosts=hosts, guests=guests)


def continuity_path(cast_dir: Path | str) -> Path:
    return Path(cast_dir) / CONTINUITY_FILE


def load_continuity(cast_dir: Path | str, last_n: int = 10) -> list[ContinuityEntry]:
    path = continuity_path(cast_dir)
    if not path.is_file():
        return []
    latest: dict[str, ContinuityEntry] = {}
    order: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            entry = ContinuityEntry.model_validate(json.loads(raw))
        except Exception:
            continue
        if entry.episode in latest:
            order.remove(entry.episode)
        latest[entry.episode] = entry
        order.append(entry.episode)
    entries = [latest[e] for e in order]
    return entries[-last_n:] if last_n else entries


def append_continuity(cast_dir: Path | str, entry: ContinuityEntry) -> Path:
    path = continuity_path(cast_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(entry.model_dump_json() + "\n")
    return path


def pick_guest(cast: Cast, expert_domain: str | None) -> Guest | None:
    if not cast.guests:
        return None
    if expert_domain:
        best, best_score = None, 55.0
        for g in cast.guests:
            for domain in g.domains:
                score = fuzz.token_set_ratio(domain.casefold(), expert_domain.casefold())
                if score > best_score:
                    best, best_score = g, score
        if best is not None:
            return best
    generic = next((g for g in cast.guests if not g.domains), None)
    return generic or cast.guests[0]


def _bullets(items: list[str]) -> str:
    return "\n".join(f"  - {it}" for it in items) if items else "  - (geen)"


def cast_bible_text(cast: Cast, guest: Guest | None = None) -> str:
    parts = ["## De cast"]
    for h in cast.hosts:
        parts.append(
            f"### {h.name} (spreker-id: {h.id}, rol: {h.role})\n"
            f"Achtergrond: {h.background.strip()}\n"
            f"Sterk in: {h.strength}\n"
            f"Zwak in: {h.weakness}\n"
            f"Verbale tics (alleen van {h.name}):\n{_bullets(h.tics)}\n"
            f"Humor: {h.humour}\n"
            f"Meningen:\n{_bullets(h.opinions)}"
        )
    if guest is not None:
        parts.append(
            f"### Gast: {guest.name} (spreker-id: {guest.id})\n"
            f"Vakgebied: {', '.join(guest.domains) or 'algemeen'}\n"
            f"Persona: {guest.persona.strip()}"
        )
    return "\n\n".join(parts)


def continuity_text(entries: list[ContinuityEntry]) -> str:
    if not entries:
        return "## Continuïteit\nDit is de eerste aflevering. Er zijn nog geen callbacks."
    parts = ["## Continuïteit (eerdere afleveringen, oud naar nieuw)"]
    for e in entries:
        parts.append(
            f"### Aflevering {e.episode}" + (f" (gast: {e.guest})" if e.guest else "") + "\n"
            f"Callbacks:\n{_bullets(e.callbacks)}\n"
            f"Running jokes:\n{_bullets(e.running_jokes)}\n"
            f"Fouten:\n{_bullets(e.mistakes)}\n"
            f"Open draden:\n{_bullets(e.open_threads)}"
        )
    return "\n\n".join(parts)


@dataclass
class BannedRules:
    phrases: list[str] = field(default_factory=list)
    patterns: list[tuple[str, str, re.Pattern]] = field(default_factory=list)  # (name, severity, regex)


def load_banned(cast_dir: Path | str) -> BannedRules:
    path = Path(cast_dir) / BANNED_FILE
    if not path.is_file():
        return BannedRules()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    phrases = [str(p).strip().casefold() for p in data.get("phrases", []) if str(p).strip()]
    patterns = []
    for p in data.get("patterns", []):
        try:
            patterns.append((str(p["name"]), str(p.get("severity", "blocking")), re.compile(p["regex"], re.IGNORECASE)))
        except (KeyError, re.error):
            continue
    return BannedRules(phrases=phrases, patterns=patterns)
