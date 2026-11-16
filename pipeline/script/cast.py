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
VOICE_SAMPLES_FILE = "voice_samples.md"


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
    samples = cast_dir / VOICE_SAMPLES_FILE
    return Cast(hosts=hosts, guests=guests, relationship=[str(r) for r in hosts_raw.get("relationship", []) or []],
                voice_samples=samples.read_text(encoding="utf-8").strip() if samples.is_file() else "")


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


def pick_guest(cast: Cast, expert_domain: str | None, *, generic: bool = True) -> Guest | None:
    """The guest whose field matches. With ``generic`` off there is no stand-in: an expert without a field
    has nothing to add that the hosts cannot say themselves, and a guest block then only repeats the episode."""
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
    if not generic:
        return None
    stand_in = next((g for g in cast.guests if not g.domains), None)
    return stand_in or cast.guests[0]


def _bullets(items: list[str]) -> str:
    return "\n".join(f"  - {it}" for it in items) if items else "  - (geen)"


def _section(title: str, items: list[str] | str) -> str:
    if not items:
        return ""
    if isinstance(items, str):
        return f"{title}: {items.strip()}\n"
    return f"{title}:\n{_bullets(items)}\n"


def cast_bible_text(cast: Cast, guest: Guest | None = None, *, personal: bool = False) -> str:
    """How the cast talks and what they want. Their lives, their stories and what they share stay out unless
    ``personal`` is set: a writer that sees the courgettes and the stew in every call puts them in every
    segment, set up or not. A block that tells a story gets that one story in its own brief (``story_seeds``)."""
    parts = ["## De cast",
             "Dit zijn mensen, geen rollen. Schrijf ze zoals ze hieronder staan: hoe ze praten, wat ze willen, "
             "hoe ze reageren. Iets uit hun eigen leven komt alleen voor waar de opdracht van een segment erom "
             "vraagt, en dan zo verteld dat een luisteraar die hen niet kent het kan volgen."]
    for h in cast.hosts:
        parts.append(
            f"### {h.name} (spreker-id: {h.id}, rol: {h.role})\n"
            f"Achtergrond: {h.background.strip()}\n"
            + _section("Zo praat " + h.name, h.voice)
            + (_section("Leven", h.life) + _section("Verhalen die " + h.name + " kan vertellen", h.stories)
               if personal else "")
            + _section("Wat " + h.name + " wil", h.wants)
            + _section("Zo reageert " + h.name, h.behaviour)
            + _section("Nooit", h.never)
            + f"Sterk in: {h.strength}\n"
            f"Zwak in: {h.weakness}\n"
            f"Verbale tics (alleen van {h.name}; spaarzaam, hooguit één keer per segment):\n{_bullets(h.tics)}\n"
            f"Humor: {h.humour}\n"
            f"Meningen:\n{_bullets(h.opinions)}"
        )
    if cast.relationship and personal:
        parts.append("### Samen\n" + _bullets(cast.relationship))
    if guest is not None:
        parts.append(
            f"### Gast: {guest.name} (spreker-id: {guest.id})\n"
            f"Vakgebied: {', '.join(guest.domains) or 'algemeen'}\n"
            f"Persona: {guest.persona.strip()}"
        )
    return "\n\n".join(parts)


def story_seeds(host) -> str:
    """What one host can draw on for a story of their own: the people in their life and the stories they
    tell. Handed to the one block that asks for it."""
    return _section("Mensen en gewoontes van " + host.name, host.life) + _section("Verhalen van " + host.name, host.stories)


_SAMPLE_LINE = re.compile(r"^\s*[A-ZÀ-Ý_]+\s*(?:\[[^\]]*\]|\([^)]*\))?\s*:\s*(?P<text>\S.*)$")


def voice_sample_lines(cast: Cast) -> list[str]:
    """The spoken lines in the voice samples, for the check that keeps them out of the episodes."""
    return [m.group("text").strip() for line in cast.voice_samples.splitlines() if (m := _SAMPLE_LINE.match(line))]


def voice_samples_text(cast: Cast) -> str:
    if not cast.voice_samples:
        return ""
    return ("## Zo klinken ze (voorbeelden: neem het register over, niet de inhoud)\n"
            "Lees dit als een opname van hoe deze mensen praten. Schrijf niets ervan over: geen zin, geen grap, "
            "geen verhaal en geen onderwerp. Het audit blokkeert een regel die op een voorbeeldregel lijkt.\n\n"
            + cast.voice_samples)


def continuity_text(entries: list[ContinuityEntry]) -> str:
    if not entries:
        return "## Continuïteit\nDit is de eerste aflevering. Er zijn nog geen callbacks."
    parts = ["## Continuïteit (eerdere afleveringen, oud naar nieuw)\n"
             "De luisteraar van deze aflevering heeft de vorige niet per se gehoord. Gebruik hier hooguit één ding "
             "uit, en vertel het dan opnieuw alsof de luisteraar het niet kent. Een running joke die je niet opnieuw "
             "opzet, laat je liggen."]
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
