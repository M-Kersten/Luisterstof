"""Data contracts for every stage. Every stage writes one of these to disk as JSON.

The shapes follow the build brief. Fields beyond the brief are additive and
documented inline; nothing here is required by a stage that does not own it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ModelT = TypeVar("ModelT", bound=BaseModel)


class Contract(BaseModel):
    """Base with JSON round-tripping helpers."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2, exclude_none=False), encoding="utf-8")
        return path

    @classmethod
    def load(cls: type[ModelT], path: Path | str) -> ModelT:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    @classmethod
    def load_or_none(cls: type[ModelT], path: Path | str) -> ModelT | None:
        p = Path(path)
        return cls.load(p) if p.is_file() else None


def now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


# ---------------------------------------------------------------------------
# Stage 1: book.json
# ---------------------------------------------------------------------------

class Section(Contract):
    id: str
    title: str
    text: str
    pages: tuple[int, int] | None = None
    formula_density: float = 0.0  # share of "notation-like" characters, 0..1
    figure_count: int = 0
    table_count: int = 0

    @property
    def char_count(self) -> int:
        return len(self.text)


class Chapter(Contract):
    id: str
    title: str
    level: int = 1  # 0 = front matter, 1 = chapter
    pages: tuple[int, int]
    sections: list[Section] = Field(default_factory=list)

    def full_text(self) -> str:
        return "\n\n".join(s.text for s in self.sections)

    def section(self, section_id: str) -> Section | None:
        return next((s for s in self.sections if s.id == section_id), None)

    @property
    def char_count(self) -> int:
        return sum(s.char_count for s in self.sections)


class Book(Contract):
    book_id: str
    title: str
    language: Literal["en", "nl"] = "nl"
    chapters: list[Chapter] = Field(default_factory=list)
    page_count: int | None = None
    structure_method: str | None = None  # fonts | toc | llm | single
    warnings: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=now_iso)

    def chapter(self, chapter_id: str) -> Chapter:
        for ch in self.chapters:
            if ch.id == chapter_id:
                return ch
        raise KeyError(f"unknown chapter {chapter_id}")

    def section(self, section_id: str) -> Section:
        for ch in self.chapters:
            s = ch.section(section_id)
            if s is not None:
                return s
        raise KeyError(f"unknown section {section_id}")

    def episode_chapters(self) -> list[Chapter]:
        """Chapters that become episodes (front matter is skipped)."""
        return [c for c in self.chapters if c.level >= 1 and c.char_count > 0]

    def previous_chapter(self, chapter_id: str) -> Chapter | None:
        eps = self.episode_chapters()
        for i, ch in enumerate(eps):
            if ch.id == chapter_id:
                return eps[i - 1] if i > 0 else None
        return None

    def next_chapter(self, chapter_id: str) -> Chapter | None:
        eps = self.episode_chapters()
        for i, ch in enumerate(eps):
            if ch.id == chapter_id:
                return eps[i + 1] if i + 1 < len(eps) else None
        return None


# ---------------------------------------------------------------------------
# Stage 2: chXX.plan.json
# ---------------------------------------------------------------------------

class SourceSpan(Contract):
    section: str
    start: int
    end: int
    quote: str | None = None  # verbatim source text the span was resolved from
    match_score: float | None = None  # 0..100 fuzzy match confidence when resolved


class KeyClaim(Contract):
    id: str
    claim: str
    source_span: SourceSpan
    difficulty: int = Field(ge=1, le=5)
    exam_relevance: int = Field(ge=1, le=5)


class Definition(Contract):
    term: str
    definition: str
    source_span: SourceSpan


class Misconception(Contract):
    wrong: str
    right: str
    why_tempting: str


class WorkedExample(Contract):
    setup: str
    steps: list[str]
    answer: str


class ContentPlan(Contract):
    chapter_id: str
    chapter_title: str = ""
    summary: str = ""  # one paragraph, Dutch, no jokes
    learning_objectives: list[str] = Field(default_factory=list)
    key_claims: list[KeyClaim] = Field(default_factory=list)
    definitions: list[Definition] = Field(default_factory=list)
    misconceptions: list[Misconception] = Field(default_factory=list)
    worked_example: WorkedExample | None = None
    needs_expert: bool = False
    expert_domain: str | None = None
    expert_reason: str | None = None
    formula_dense_sections: list[str] = Field(default_factory=list)
    section_difficulty: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=now_iso)

    def claim(self, claim_id: str) -> KeyClaim | None:
        return next((c for c in self.key_claims if c.id == claim_id), None)

    def claims_with_relevance(self, minimum: int) -> list[KeyClaim]:
        return [c for c in self.key_claims if c.exam_relevance >= minimum]

    def hardest_claim(self) -> KeyClaim | None:
        if not self.key_claims:
            return None
        return max(self.key_claims, key=lambda c: (c.difficulty, c.exam_relevance))


# ---------------------------------------------------------------------------
# Stage 2b: glossary.json (lexicon)
# ---------------------------------------------------------------------------

LexiconKind = Literal["loanword_en", "notation", "abbreviation"]


class LexiconEntry(Contract):
    surface: str
    kind: LexiconKind
    spoken: str
    lock: bool = False
    note: str | None = None
    source_chapter: str | None = None

    @field_validator("surface", "spoken")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("empty")
        return v


class Glossary(Contract):
    book_id: str
    entries: list[LexiconEntry] = Field(default_factory=list)
    updated_at: str = Field(default_factory=now_iso)

    def find(self, surface: str) -> LexiconEntry | None:
        key = surface.strip().casefold()
        return next((e for e in self.entries if e.surface.casefold() == key), None)

    def merge(self, new_entries: list[LexiconEntry]) -> int:
        """Add entries; locked or existing surfaces are kept. Returns the number added."""
        added = 0
        for entry in new_entries:
            existing = self.find(entry.surface)
            if existing is None:
                self.entries.append(entry)
                added += 1
            elif not existing.lock and existing.spoken != entry.spoken and entry.lock:
                existing.spoken = entry.spoken
                existing.lock = True
        self.entries.sort(key=lambda e: (-len(e.surface), e.surface.casefold()))
        self.updated_at = now_iso()
        return added


# ---------------------------------------------------------------------------
# Stage 3: chXX.script.json
# ---------------------------------------------------------------------------

SegmentType = Literal["cold_open", "recap", "body", "guest", "reexplain", "quiz", "outro"]
OverlapMode = Literal["interrupt", "backchannel", "none"]

LINE_ID_RE = re.compile(r"^l(\d{3,})$")


class Overlap(Contract):
    mode: OverlapMode = "none"
    target: str | None = None  # line id this line overlaps (always an earlier line)
    cut_word: str | None = None  # for interrupt: word in the target line where this line starts


class Line(Contract):
    id: str
    speaker: str
    text: str
    tags: list[str] = Field(default_factory=list)
    covers: list[str] = Field(default_factory=list)
    overlap: Overlap = Field(default_factory=Overlap)
    pause_after_ms: int = 0  # deliberate beat after this line (quiz pauses)

    @field_validator("text")
    @classmethod
    def _text_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("line text is empty")
        return v.strip()

    @property
    def ends_with_fragment(self) -> bool:
        return self.text.rstrip().endswith(("—", "…", "-", "..."))


class Segment(Contract):
    type: SegmentType
    covers: list[str] = Field(default_factory=list)
    lines: list[Line] = Field(default_factory=list)
    title: str | None = None
    brief: str | None = None  # what the writer was asked to do in this segment

    @property
    def char_count(self) -> int:
        return sum(len(line.text) for line in self.lines)


class Script(Contract):
    episode_id: str
    target_minutes: int = 25
    title: str | None = None
    guest_id: str | None = None
    segments: list[Segment] = Field(default_factory=list)
    revision: int = 1
    created_at: str = Field(default_factory=now_iso)

    def lines(self) -> Iterator[Line]:
        for seg in self.segments:
            yield from seg.lines

    def line(self, line_id: str) -> Line | None:
        return next((line for line in self.lines() if line.id == line_id), None)

    def segment_of(self, line_id: str) -> Segment | None:
        for seg in self.segments:
            if any(line.id == line_id for line in seg.lines):
                return seg
        return None

    @property
    def char_count(self) -> int:
        return sum(seg.char_count for seg in self.segments)

    def estimated_seconds(self, chars_per_second: float = 15.0, gap_s: float = 0.3) -> float:
        n_lines = sum(len(seg.lines) for seg in self.segments)
        pauses = sum(line.pause_after_ms for line in self.lines()) / 1000.0
        return self.char_count / chars_per_second + n_lines * gap_s + pauses

    def next_line_id(self) -> str:
        highest = 0
        for line in self.lines():
            m = LINE_ID_RE.match(line.id)
            if m:
                highest = max(highest, int(m.group(1)))
        return f"l{highest + 1:03d}"

    def renumber(self) -> Script:
        """Assign sequential ids and rewrite overlap targets accordingly."""
        mapping: dict[str, str] = {}
        counter = 0
        for line in self.lines():
            counter += 1
            mapping[line.id] = f"l{counter:03d}"
        for line in self.lines():
            line.id = mapping[line.id]
            if line.overlap.target:
                line.overlap.target = mapping.get(line.overlap.target, line.overlap.target)
        return self

    @model_validator(mode="after")
    def _unique_ids(self) -> Script:
        seen: set[str] = set()
        for line in self.lines():
            if line.id in seen:
                raise ValueError(f"duplicate line id {line.id}")
            seen.add(line.id)
        return self


# ---------------------------------------------------------------------------
# Stage 3b: chXX.audit.json
# ---------------------------------------------------------------------------

AuditCheck = Literal["support", "coverage", "lint", "structure"]
Severity = Literal["blocking", "warning"]


class AuditIssue(Contract):
    check: AuditCheck
    severity: Severity = "blocking"
    rule: str
    message: str
    line_id: str | None = None
    claim_id: str | None = None
    suggestion: str | None = None


class CoverageReport(Contract):
    required: list[str] = Field(default_factory=list)
    covered: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)


class SupportReport(Contract):
    checked: int = 0
    unsupported: int = 0
    skipped: bool = False  # true when no LLM was available for the support check


class AuditResult(Contract):
    episode_id: str
    passed: bool = False
    issues: list[AuditIssue] = Field(default_factory=list)
    coverage: CoverageReport = Field(default_factory=CoverageReport)
    support: SupportReport = Field(default_factory=SupportReport)
    stats: dict[str, float | int | str] = Field(default_factory=dict)
    script_revision: int | None = None
    created_at: str = Field(default_factory=now_iso)

    def blocking(self) -> list[AuditIssue]:
        return [i for i in self.issues if i.severity == "blocking"]

    def warnings(self) -> list[AuditIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    def finalize(self) -> AuditResult:
        self.passed = not self.blocking()
        return self


# ---------------------------------------------------------------------------
# Cast and continuity
# ---------------------------------------------------------------------------

class Host(Contract):
    id: str
    name: str
    role: Literal["explainer", "skeptic"]
    voice_id_final: str | None = None
    voice_id_draft: str | None = None
    voice_ref: str | None = None  # path to frozen reference clip (relative to cast dir)
    background: str = ""
    strength: str = ""
    weakness: str = ""
    tics: list[str] = Field(default_factory=list)
    tic_markers: list[str] = Field(default_factory=list)  # literal phrases only this host may use
    humour: str = ""
    opinions: list[str] = Field(default_factory=list)
    exaggeration: float = 0.5  # Chatterbox base value
    cfg_weight: float = 0.5  # Chatterbox classifier-free-guidance weight; lowering it tends to slow delivery
    speech_rate: float = 1.0  # post-render pitch-preserving time-stretch; <1.0 slower, >1.0 faster, 1.0 = off
    chars_per_second: float | None = None


class Guest(Contract):
    id: str
    name: str
    domains: list[str] = Field(default_factory=list)
    voice_id_final: str | None = None
    voice_id_draft: str | None = None
    voice_ref: str | None = None
    persona: str = ""
    tics: list[str] = Field(default_factory=list)
    tic_markers: list[str] = Field(default_factory=list)
    exaggeration: float = 0.4
    cfg_weight: float = 0.5
    speech_rate: float = 1.0
    chars_per_second: float | None = None


class Cast(Contract):
    hosts: list[Host] = Field(default_factory=list)
    guests: list[Guest] = Field(default_factory=list)

    @property
    def host_ids(self) -> list[str]:
        return [h.id for h in self.hosts]

    def host(self, host_id: str) -> Host:
        for h in self.hosts:
            if h.id == host_id:
                return h
        raise KeyError(f"unknown host {host_id}")

    def guest(self, guest_id: str) -> Guest:
        for g in self.guests:
            if g.id == guest_id:
                return g
        raise KeyError(f"unknown guest {guest_id}")

    def speaker(self, speaker_id: str) -> Host | Guest:
        for h in self.hosts:
            if h.id == speaker_id:
                return h
        for g in self.guests:
            if g.id == speaker_id:
                return g
        raise KeyError(f"unknown speaker {speaker_id}")

    def by_role(self, role: str) -> Host:
        for h in self.hosts:
            if h.role == role:
                return h
        raise KeyError(f"no host with role {role}")


class ContinuityEntry(Contract):
    episode: str
    callbacks: list[str] = Field(default_factory=list)
    running_jokes: list[str] = Field(default_factory=list)
    mistakes: list[str] = Field(default_factory=list)
    open_threads: list[str] = Field(default_factory=list)
    guest: str | None = None
    recorded_at: str = Field(default_factory=now_iso)


# ---------------------------------------------------------------------------
# Audio: render manifest and transcript
# ---------------------------------------------------------------------------

class TakeRecord(Contract):
    seed: int
    exaggeration: float
    path: str
    duration_s: float
    wer: float | None = None
    transcript: str | None = None
    accepted: bool = False
    reason: str | None = None


class TurnRender(Contract):
    """One rendered speaker turn: consecutive lines by the same speaker."""

    turn_id: str
    speaker: str
    line_ids: list[str]
    text_spoken: str  # after lexicon application
    takes: list[TakeRecord] = Field(default_factory=list)
    chosen: int | None = None
    from_cache: bool = False
    flagged: bool = False
    flag_reason: str | None = None

    @property
    def chosen_take(self) -> TakeRecord | None:
        if self.chosen is None or self.chosen >= len(self.takes):
            return None
        return self.takes[self.chosen]


class RenderManifest(Contract):
    episode_id: str
    tier: Literal["draft", "final"]
    synth: str
    turns: list[TurnRender] = Field(default_factory=list)
    block_overrides: dict[str, str] = Field(default_factory=dict)  # block id -> audio path (accent tier)
    created_at: str = Field(default_factory=now_iso)

    def flagged_turns(self) -> list[TurnRender]:
        return [t for t in self.turns if t.flagged]


class TranscriptWord(Contract):
    word: str
    start: float
    end: float


class TranscriptLine(Contract):
    line_id: str
    speaker: str
    text: str
    start: float
    end: float
    words: list[TranscriptWord] = Field(default_factory=list)
    overlap: OverlapMode = "none"
    segment: SegmentType | None = None


class BlockSpan(Contract):
    block_id: str
    line_ids: list[str]
    start: float
    end: float
    path: str | None = None


class EpisodeTranscript(Contract):
    episode_id: str
    tier: Literal["draft", "final"]
    duration_s: float
    lines: list[TranscriptLine] = Field(default_factory=list)
    blocks: list[BlockSpan] = Field(default_factory=list)
    qa: list[str] = Field(default_factory=list)  # mixer warnings (unwritten gaps etc.)
    created_at: str = Field(default_factory=now_iso)


class Approval(Contract):
    episode_id: str
    script_revision: int
    audit_passed: bool
    approved_at: str = Field(default_factory=now_iso)
    note: str | None = None


def dump_json(data: object, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
