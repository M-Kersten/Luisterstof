"""Orchestration: runs stages per book and chapter, writes artifacts, emits events.

The CLI and the web app both drive this class. Every stage reads its inputs
from disk and writes its output to disk, so any stage can be re-run alone.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pipeline.audio.render_draft import render_draft
from pipeline.audio.render_final import render_eleven_blocks, render_final
from pipeline.audio.synth import NullSynth, Synth
from pipeline.config import Settings
from pipeline.ingest.figures import LLMCaptioner
from pipeline.ingest.run import ingest_book
from pipeline.llm import LLM, make_llm
from pipeline.models import Approval, AuditResult, Book, Cast, ContentPlan, Glossary, RenderManifest, Script
from pipeline.paths import BookPaths, list_books
from pipeline.plan.content_plan import plan_chapter
from pipeline.plan.glossary import load_or_create_glossary, propose_lexicon
from pipeline.script.audit import audit_script
from pipeline.script.cast import append_continuity, load_banned, load_cast, load_continuity
from pipeline.script.continuity import extract_continuity
from pipeline.script.writer import Writer

log = logging.getLogger(__name__)

EventFn = Callable[[str, str, dict[str, Any]], None]  # (stage, status, data)

STAGES = ["ingest", "plan", "glossary", "script", "audit", "draft", "approve", "render", "mix"]
AUTO_STAGES = ["plan", "glossary", "script", "draft"]


class StageError(RuntimeError):
    pass


class Pipeline:
    def __init__(
        self,
        settings: Settings,
        *,
        llm: LLM | None = None,
        fake_llm: bool = False,
        fake_audio: bool = False,
        cast: Cast | None = None,
        on_event: EventFn | None = None,
    ):
        from pipeline.offline import apply_env

        apply_env(settings)  # before anything imports huggingface_hub/transformers
        self.settings = settings
        self.fake_llm = fake_llm
        self.fake_audio = fake_audio
        self._llm = llm
        self._llm_fake_requested = fake_llm
        self.cast = cast or load_cast(settings.cast_dir)
        self.banned = load_banned(settings.cast_dir)
        self.on_event = on_event

    # ------------------------------------------------------------------
    @property
    def llm(self) -> LLM:
        if self._llm is None:
            self._llm = make_llm(self.settings, fake=self._llm_fake_requested)
        return self._llm

    def release_llm(self) -> None:
        """Hand the GPU back before rendering: a local model server would otherwise sit on its memory."""
        release = getattr(self._llm, "release", None)
        if callable(release):
            release()

    def emit(self, stage: str, status: str, **data: Any) -> None:
        log.info("%s %s %s", stage, status, data if data else "")
        if self.on_event:
            self.on_event(stage, status, data)

    def paths(self, book_id: str) -> BookPaths:
        return BookPaths(self.settings.data_dir, book_id)

    def books(self) -> list[str]:
        return list_books(self.settings.data_dir)

    def load_book(self, book_id: str) -> Book:
        path = self.paths(book_id).book_json
        if not path.is_file():
            raise StageError(f"book {book_id} has not been ingested yet")
        return Book.load(path)

    def load_plan(self, book_id: str, chapter_id: str) -> ContentPlan:
        path = self.paths(book_id).plan(chapter_id)
        if not path.is_file():
            raise StageError(f"no content plan for {book_id}/{chapter_id}")
        return ContentPlan.load(path)

    def load_script(self, book_id: str, chapter_id: str) -> Script:
        path = self.paths(book_id).script(chapter_id)
        if not path.is_file():
            raise StageError(f"no script for {book_id}/{chapter_id}")
        return Script.load(path)

    def load_glossary(self, book_id: str) -> Glossary:
        return load_or_create_glossary(self.paths(book_id).glossary_json, book_id, self.settings.cast_dir)

    def plan_lookup(self, book_id: str) -> Callable[[str], ContentPlan | None]:
        def lookup(chapter_id: str) -> ContentPlan | None:
            return ContentPlan.load_or_none(self.paths(book_id).plan(chapter_id))
        return lookup

    # ------------------------------------------------------------------
    # Stage 1
    def ingest(self, pdf_path: Path | str, book_id: str, *, method: str = "auto", captions: bool = True,
               title: str | None = None) -> Book:
        paths = self.paths(book_id).ensure()
        src = Path(pdf_path)
        if src.resolve() != paths.source_pdf.resolve():
            shutil.copyfile(src, paths.source_pdf)
        self.emit("ingest", "start", book_id=book_id, pdf=str(src))
        if captions and not getattr(self.llm, "supports_images", True):
            self.emit("ingest", "note", book_id=book_id,
                      message="figure captions skipped: the local model has no vision (LOCAL_LLM_VISION=0)")
            captions = False
        captioner = LLMCaptioner(self.llm) if captions else None
        book = ingest_book(paths.source_pdf, book_id, llm=self.llm, captioner=captioner, figures_dir=paths.figures_dir,
                           method=method, title=title)
        book.save(paths.book_json)
        self.emit("ingest", "done", book_id=book_id, chapters=len(book.episode_chapters()), method=book.structure_method,
                  warnings=book.warnings)
        return book

    # Stage 2
    def plan(self, book_id: str, chapter_id: str) -> ContentPlan:
        book = self.load_book(book_id)
        self.emit("plan", "start", book_id=book_id, chapter=chapter_id)
        plan = plan_chapter(book, chapter_id, self.llm, self.cast)
        plan.save(self.paths(book_id).plan(chapter_id))
        self.emit("plan", "done", book_id=book_id, chapter=chapter_id, claims=len(plan.key_claims),
                  needs_expert=plan.needs_expert, warnings=plan.warnings)
        return plan

    # Stage 2b
    def glossary(self, book_id: str, chapter_id: str) -> Glossary:
        book = self.load_book(book_id)
        glossary = self.load_glossary(book_id)
        self.emit("glossary", "start", book_id=book_id, chapter=chapter_id)
        added = glossary.merge(propose_lexicon(book.chapter(chapter_id), glossary, self.llm))
        glossary.save(self.paths(book_id).glossary_json)
        self.emit("glossary", "done", book_id=book_id, chapter=chapter_id, added=added, total=len(glossary.entries))
        return glossary

    def save_glossary(self, book_id: str, glossary: Glossary) -> Glossary:
        glossary.save(self.paths(book_id).glossary_json)
        return glossary

    # Stage 3 + 3b
    def script(self, book_id: str, chapter_id: str, *, revise_rounds: int = 1, llm_checks: bool = True) -> tuple[Script, AuditResult]:
        book = self.load_book(book_id)
        plan = self.load_plan(book_id, chapter_id)
        glossary = self.load_glossary(book_id)
        prev = book.previous_chapter(chapter_id)
        previous_plan = ContentPlan.load_or_none(self.paths(book_id).plan(prev.id)) if prev else None
        nxt = book.next_chapter(chapter_id)
        continuity = load_continuity(self.settings.cast_dir, last_n=10)
        continuity = [e for e in continuity if e.episode != chapter_id]
        writer = Writer(self.llm, self.cast, self.settings, continuity)
        self.emit("script", "start", book_id=book_id, chapter=chapter_id, guest=plan.needs_expert)
        script = writer.write(plan, book, glossary, previous_plan=previous_plan, next_chapter_title=nxt.title if nxt else None)
        script.save(self.paths(book_id).script(chapter_id))
        self.emit("script", "done", book_id=book_id, chapter=chapter_id, lines=sum(len(s.lines) for s in script.segments),
                  estimated_minutes=round(script.estimated_seconds(self.settings.chars_per_second) / 60, 1))
        audit = self.audit(book_id, chapter_id, script=script, llm_checks=llm_checks)
        rounds = 0
        while not audit.passed and rounds < revise_rounds:
            rounds += 1
            self.emit("script", "revise", book_id=book_id, chapter=chapter_id, round=rounds, blocking=len(audit.blocking()))
            script = writer.revise(script, audit, plan, glossary)
            script.save(self.paths(book_id).script(chapter_id))
            audit = self.audit(book_id, chapter_id, script=script, llm_checks=llm_checks)
        self.continuity(book_id, chapter_id, script=script)
        return script, audit

    def audit(self, book_id: str, chapter_id: str, *, script: Script | None = None, llm_checks: bool = True) -> AuditResult:
        book = self.load_book(book_id)
        plan = self.load_plan(book_id, chapter_id)
        script = script or self.load_script(book_id, chapter_id)
        glossary = self.load_glossary(book_id)
        self.emit("audit", "start", book_id=book_id, chapter=chapter_id, revision=script.revision)
        audit = audit_script(script, plan, book, self.cast, self.banned, self.settings, glossary=glossary,
                             llm=self.llm if llm_checks else None, plan_lookup=self.plan_lookup(book_id))
        audit.save(self.paths(book_id).audit(chapter_id))
        self.emit("audit", "done", book_id=book_id, chapter=chapter_id, passed=audit.passed,
                  blocking=len(audit.blocking()), warnings=len(audit.warnings()), coverage_missing=audit.coverage.missing)
        return audit

    def continuity(self, book_id: str, chapter_id: str, *, script: Script | None = None):
        script = script or self.load_script(book_id, chapter_id)
        previous = [e for e in load_continuity(self.settings.cast_dir, last_n=10) if e.episode != chapter_id]
        entry = extract_continuity(script, self.cast, self.llm, previous)
        append_continuity(self.settings.cast_dir, entry)
        self.emit("continuity", "done", book_id=book_id, chapter=chapter_id, callbacks=len(entry.callbacks))
        return entry

    def save_script(self, book_id: str, chapter_id: str, script: Script) -> Script:
        existing = Script.load_or_none(self.paths(book_id).script(chapter_id))
        script.revision = (existing.revision + 1) if existing else 1
        script.save(self.paths(book_id).script(chapter_id))
        approval = self.paths(book_id).approval(chapter_id)
        if approval.is_file():
            approval.unlink()  # an edited script must be re-approved
        return script

    # Stage 4
    def _draft_synth(self, synth: Synth | None) -> Synth | None:
        if synth is not None:
            return synth
        if self.fake_audio:
            return NullSynth(sample_rate=22050, chars_per_second=self.settings.chars_per_second)
        return None

    def draft(self, book_id: str, chapter_id: str, *, synth: Synth | None = None) -> tuple[Path, Path, RenderManifest]:
        script = self.load_script(book_id, chapter_id)
        glossary = self.load_glossary(book_id)
        self.emit("draft", "start", book_id=book_id, chapter=chapter_id)
        result = render_draft(script, glossary, self.cast, self.settings, self.paths(book_id), synth=self._draft_synth(synth),
                              on_event=lambda kind, data: self.emit("draft", kind, **data))
        self.emit("draft", "done", book_id=book_id, chapter=chapter_id, audio=str(result[0]))
        return result

    def approve(self, book_id: str, chapter_id: str, *, note: str | None = None, force: bool = False) -> Approval:
        script = self.load_script(book_id, chapter_id)
        audit = AuditResult.load_or_none(self.paths(book_id).audit(chapter_id))
        if audit is None or audit.script_revision != script.revision:
            audit = self.audit(book_id, chapter_id, script=script)
        if not audit.passed and not force:
            raise StageError(f"audit has {len(audit.blocking())} blocking issue(s); fix them or approve with force")
        approval = Approval(episode_id=chapter_id, script_revision=script.revision, audit_passed=audit.passed, note=note)
        approval.save(self.paths(book_id).approval(chapter_id))
        self.emit("approve", "done", book_id=book_id, chapter=chapter_id, forced=not audit.passed)
        return approval

    def is_approved(self, book_id: str, chapter_id: str) -> bool:
        approval = Approval.load_or_none(self.paths(book_id).approval(chapter_id))
        script = Script.load_or_none(self.paths(book_id).script(chapter_id))
        return bool(approval and script and approval.script_revision == script.revision)

    def render(self, book_id: str, chapter_id: str, *, force: bool = False, synth: Synth | None = None,
               transcriber=None, aligner=None) -> tuple[Path, Path, RenderManifest]:
        if not force and not self.is_approved(book_id, chapter_id):
            raise StageError("script is not approved for this revision; listen to the draft and approve first")
        script = self.load_script(book_id, chapter_id)
        glossary = self.load_glossary(book_id)
        self.release_llm()
        self.emit("render", "start", book_id=book_id, chapter=chapter_id, fake=self.fake_audio)
        result = render_final(script, glossary, self.cast, self.settings, self.paths(book_id), synth=synth,
                              transcriber=transcriber, aligner=aligner, fake=self.fake_audio,
                              on_event=lambda kind, data: self.emit("render", kind, **data))
        self.emit("render", "done", book_id=book_id, chapter=chapter_id, audio=str(result[0]),
                  flagged=[t.turn_id for t in result[2].flagged_turns()])
        return result

    def eleven(self, book_id: str, chapter_id: str, block_ids: list[str], *, render_fn=None) -> RenderManifest:
        script = self.load_script(book_id, chapter_id)
        glossary = self.load_glossary(book_id)
        self.emit("eleven", "start", book_id=book_id, chapter=chapter_id, blocks=block_ids)
        manifest = render_eleven_blocks(script, glossary, self.cast, self.settings, self.paths(book_id), block_ids,
                                        render_fn=render_fn, on_event=lambda kind, data: self.emit("eleven", kind, **data))
        self.emit("eleven", "done", book_id=book_id, chapter=chapter_id, overrides=list(manifest.block_overrides))
        return manifest

    # ------------------------------------------------------------------
    def run_chapter(self, book_id: str, chapter_id: str, *, upto: str = "draft", llm_checks: bool = True) -> dict[str, Any]:
        order = ["plan", "glossary", "script", "draft", "approve", "render"]
        if upto not in order:
            raise ValueError(f"upto must be one of {order}")
        stop = order.index(upto)
        out: dict[str, Any] = {}
        if stop >= 0:
            out["plan"] = self.plan(book_id, chapter_id)
        if stop >= 1:
            out["glossary"] = self.glossary(book_id, chapter_id)
        if stop >= 2:
            out["script"], out["audit"] = self.script(book_id, chapter_id, llm_checks=llm_checks)
        if stop >= 3:
            out["draft"] = self.draft(book_id, chapter_id)[0]
        if stop >= 4:
            out["approve"] = self.approve(book_id, chapter_id, force=upto == "render" and not out["audit"].passed)
        if stop >= 5:
            out["render"] = self.render(book_id, chapter_id)[0]
        return out

    def run_book(self, book_id: str, *, upto: str = "draft", chapters: list[str] | None = None, llm_checks: bool = True) -> dict[str, dict]:
        book = self.load_book(book_id)
        results = {}
        for ch in book.episode_chapters():
            if chapters and ch.id not in chapters:
                continue
            results[ch.id] = self.run_chapter(book_id, ch.id, upto=upto, llm_checks=llm_checks)
        return results

    def status(self, book_id: str) -> dict[str, Any]:
        paths = self.paths(book_id)
        book = Book.load_or_none(paths.book_json)
        chapters = []
        if book is not None:
            for ch in book.episode_chapters():
                audit = AuditResult.load_or_none(paths.audit(ch.id))
                script = Script.load_or_none(paths.script(ch.id))
                manifest = RenderManifest.load_or_none(paths.manifest(ch.id, "final"))
                chapters.append({
                    "id": ch.id, "title": ch.title, "pages": list(ch.pages), "chars": ch.char_count,
                    "plan": paths.plan(ch.id).is_file(),
                    "script": script is not None, "script_revision": script.revision if script else None,
                    "audit": audit is not None, "audit_passed": audit.passed if audit else None,
                    "audit_blocking": len(audit.blocking()) if audit else None,
                    "draft": paths.out_audio(ch.id, "draft").is_file(),
                    "approved": self.is_approved(book_id, ch.id),
                    "final": paths.out_audio(ch.id, "final").is_file(),
                    "flagged_turns": len(manifest.flagged_turns()) if manifest else 0,
                })
        return {
            "book_id": book_id,
            "title": book.title if book else None,
            "ingested": book is not None,
            "structure_method": book.structure_method if book else None,
            "glossary_entries": len(self.load_glossary(book_id).entries) if paths.glossary_json.is_file() else 0,
            "chapters": chapters,
        }
