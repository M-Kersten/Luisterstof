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

from pipeline.audio.render_final import render_final
from pipeline.audio.synth import Synth
from pipeline.cancel import Cancelled
from pipeline.config import Settings
from pipeline.ingest.run import ingest_book
from pipeline.llm import LLM, make_llm
from pipeline.models import Approval, AuditResult, Book, Cast, ContentPlan, Glossary, RenderManifest, Script
from pipeline.paths import BookPaths, list_books
from pipeline.plan.glossary import load_or_create_glossary
from pipeline.script.audit import audit_script
from pipeline.script.cast import append_continuity, load_banned, load_cast, load_continuity
from pipeline.script.continuity import extract_continuity
from pipeline.script.writer import Writer, naturalness_issues

log = logging.getLogger(__name__)

EventFn = Callable[[str, str, dict[str, Any]], None]  # (stage, status, data)

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
        self._aborted = False

    # ------------------------------------------------------------------
    @property
    def llm(self) -> LLM:
        if self._llm is None:
            self._llm = make_llm(self.settings, fake=self._llm_fake_requested)
        if hasattr(self._llm, "on_tokens") and self._llm.on_tokens is None:
            self._llm.on_tokens = lambda task, tokens, seconds: self.emit("llm", "tokens", task=task, tokens=tokens,
                                                                          seconds=round(seconds))
        return self._llm

    def abort(self) -> None:
        """Stop at the next checkpoint: the next event, or the next answer of a model call in flight."""
        self._aborted = True
        stop = getattr(self._llm, "abort", None)
        if stop:
            stop()

    def release_llm(self) -> None:
        """Let the writing model go before rendering; the OpenCode server decides what to keep loaded."""
        release = getattr(self._llm, "release", None)
        if callable(release):
            release()

    def emit(self, stage: str, status: str, **data: Any) -> None:
        if self._aborted:
            raise Cancelled()
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
    def ingest(self, pdf_path: Path | str, book_id: str, *, method: str = "auto", title: str | None = None) -> Book:
        paths = self.paths(book_id).ensure()
        src = Path(pdf_path)
        if src.resolve() != paths.source_pdf.resolve():
            shutil.copyfile(src, paths.source_pdf)
        self.emit("ingest", "start", book_id=book_id, pdf=str(src))
        book = ingest_book(paths.source_pdf, book_id, llm=self.llm, method=method, title=title)
        book.save(paths.book_json)
        self.emit("ingest", "done", book_id=book_id, chapters=len(book.episode_chapters()), method=book.structure_method,
                  warnings=book.warnings)
        return book

    # ------------------------------------------------------------ llm accounting
    def _llm_usage(self) -> tuple[int, int, float]:
        u = getattr(self._llm, "usage", None) if self._llm is not None else None
        return (u.calls, u.output_tokens, u.seconds) if u is not None else (0, 0, 0.0)

    def _llm_spent(self, before: tuple[int, int, float]) -> dict:
        """Calls, output tokens, model seconds and speed since ``before``: where a stage's time went."""
        calls, out, secs = (a - b for a, b in zip(self._llm_usage(), before, strict=True))
        if not calls:
            return {}
        spent = {"llm_calls": calls, "llm_seconds": round(secs), "llm_output_tokens": out}
        if out and secs:
            spent["tokens_per_s"] = round(out / secs, 1)
        return spent

    # Stage 2
    def plan(self, book_id: str, chapter_id: str) -> ContentPlan:
        from pipeline.plan.thorough import plan_chapter_thorough

        book = self.load_book(book_id)
        self.emit("plan", "start", book_id=book_id, chapter=chapter_id)
        before = self._llm_usage()
        plan = plan_chapter_thorough(book, chapter_id, self.llm, self.cast, review_rounds=self.settings.plan_review_rounds,
                                     part_chars=self.settings.plan_part_chars, progress=lambda kind, data: self.emit("plan", kind, chapter=chapter_id, **data))
        plan.save(self.paths(book_id).plan(chapter_id))
        self.emit("plan", "done", book_id=book_id, chapter=chapter_id, claims=len(plan.key_claims),
                  needs_expert=plan.needs_expert, warnings=plan.warnings, **self._llm_spent(before))
        return plan

    # Stage 2b
    def glossary(self, book_id: str, chapter_id: str) -> Glossary:
        from pipeline.plan.thorough import propose_lexicon_thorough

        book = self.load_book(book_id)
        glossary = self.load_glossary(book_id)
        self.emit("glossary", "start", book_id=book_id, chapter=chapter_id)
        before = self._llm_usage()
        chapter = book.chapter(chapter_id)
        entries = propose_lexicon_thorough(chapter, glossary, self.llm, part_chars=self.settings.plan_part_chars,
                                           progress=lambda kind, data: self.emit("glossary", kind, chapter=chapter_id, **data))
        added = glossary.merge(entries)
        glossary.save(self.paths(book_id).glossary_json)
        self.emit("glossary", "done", book_id=book_id, chapter=chapter_id, added=added, total=len(glossary.entries),
                  **self._llm_spent(before))
        return glossary

    def save_glossary(self, book_id: str, glossary: Glossary) -> Glossary:
        glossary.save(self.paths(book_id).glossary_json)
        return glossary

    # Stage 3 + 3b
    def script(self, book_id: str, chapter_id: str, *, revise_rounds: int | None = None,
               llm_checks: bool = True) -> tuple[Script, AuditResult]:
        book = self.load_book(book_id)
        plan = self.load_plan(book_id, chapter_id)
        glossary = self.load_glossary(book_id)
        prev = book.previous_chapter(chapter_id)
        previous_plan = ContentPlan.load_or_none(self.paths(book_id).plan(prev.id)) if prev else None
        nxt = book.next_chapter(chapter_id)
        continuity = load_continuity(self.settings.cast_dir, last_n=10)
        continuity = [e for e in continuity if e.episode != chapter_id]
        writer = Writer(self.llm, self.cast, self.settings, continuity)
        writer.progress = lambda n, total, title: self.emit("script", "segment", chapter=chapter_id, index=n, total=total,
                                                            label=title)
        rounds_left = max(0, self.settings.script_revise_rounds if revise_rounds is None else revise_rounds)
        self.emit("script", "start", book_id=book_id, chapter=chapter_id, guest=plan.needs_expert,
                  revise_rounds=rounds_left)
        script = writer.write(plan, book, glossary, previous_plan=previous_plan, next_chapter_title=nxt.title if nxt else None)
        script.save(self.paths(book_id).script(chapter_id))
        self.emit("script", "done", book_id=book_id, chapter=chapter_id, lines=sum(len(s.lines) for s in script.segments),
                  estimated_minutes=round(script.estimated_seconds(self.settings.chars_per_second) / 60, 1))
        audit = self.audit(book_id, chapter_id, script=script, llm_checks=llm_checks)
        rounds = 0
        while rounds < rounds_left and self._needs_rewrite(audit):
            rounds += 1
            self.emit("script", "revise", book_id=book_id, chapter=chapter_id, round=rounds,
                      blocking=len(audit.blocking()), naturalness=len(naturalness_issues(audit)))
            script = writer.revise(script, audit, plan, glossary, also_naturalness=self.settings.quality_mode)
            script.save(self.paths(book_id).script(chapter_id))
            audit = self.audit(book_id, chapter_id, script=script, llm_checks=llm_checks)
        self.continuity(book_id, chapter_id, script=script)
        return script, audit

    def _needs_rewrite(self, audit: AuditResult) -> bool:
        """Whether another rewrite round is warranted.

        A blocking issue always is. In quality mode a script that passes but speaks flat is not done
        either: interruptions and reactions have to be written in, and a run that may take all night
        should spend some of it on the part the listener actually hears.
        """
        if not audit.passed:
            return True
        return self.settings.quality_mode and bool(naturalness_issues(audit))

    def audit(self, book_id: str, chapter_id: str, *, script: Script | None = None, llm_checks: bool = True) -> AuditResult:
        book = self.load_book(book_id)
        plan = self.load_plan(book_id, chapter_id)
        script = script or self.load_script(book_id, chapter_id)
        glossary = self.load_glossary(book_id)
        self.emit("audit", "start", book_id=book_id, chapter=chapter_id, revision=script.revision)
        audit = audit_script(script, plan, book, self.cast, self.banned, self.settings, glossary=glossary,
                             llm=self.llm if llm_checks else None, plan_lookup=self.plan_lookup(book_id),
                             progress=lambda n, total, label: self.emit("audit", "segment", chapter=chapter_id, index=n,
                                                                        total=total, label=label))
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
            raise StageError("script is not approved for this revision; read the script and audit, then approve first")
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

    # ------------------------------------------------------------------
    def run_chapter(self, book_id: str, chapter_id: str, *, upto: str = "script", llm_checks: bool = True) -> dict[str, Any]:
        """Everything that does not need a human or hours of GPU, up to ``upto``.

        ``upto="render"`` approves on its own authority and starts the real
        Chatterbox render: only ask for that when nobody is going to read first.
        """
        order = ["plan", "glossary", "script", "approve", "render"]
        if upto not in order:
            raise ValueError(f"upto must be one of {order}")
        stop = order.index(upto)
        names = {"plan": "plan", "glossary": "lexicon", "script": "script en audit",
                 "approve": "goedkeuren", "render": "Chatterbox render"}

        def step(name: str) -> None:
            self.emit("run", "step", book_id=book_id, chapter=chapter_id, index=order.index(name) + 1, total=stop + 1,
                      label=names[name])

        out: dict[str, Any] = {}
        if stop >= 0:
            step("plan")
            out["plan"] = self.plan(book_id, chapter_id)
        if stop >= 1:
            step("glossary")
            out["glossary"] = self.glossary(book_id, chapter_id)
        if stop >= 2:
            step("script")
            out["script"], out["audit"] = self.script(book_id, chapter_id, llm_checks=llm_checks)
        if stop >= 3:
            step("approve")
            out["approve"] = self.approve(book_id, chapter_id, force=upto == "render" and not out["audit"].passed)
        if stop >= 4:
            step("render")
            out["render"] = self.render(book_id, chapter_id)[0]
        return out

    def run_book(self, book_id: str, *, upto: str = "script", chapters: list[str] | None = None, llm_checks: bool = True) -> dict[str, dict]:
        book = self.load_book(book_id)
        results = {}
        todo = [ch for ch in book.episode_chapters() if not chapters or ch.id in chapters]
        for n, ch in enumerate(todo, start=1):
            self.emit("book", "chapter", book_id=book_id, chapter=ch.id, index=n, total=len(todo), label=f"{ch.id} {ch.title}")
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
                manifest = RenderManifest.load_or_none(paths.manifest(ch.id))
                chapters.append({
                    "id": ch.id, "title": ch.title, "pages": list(ch.pages), "chars": ch.char_count,
                    "plan": paths.plan(ch.id).is_file(),
                    "script": script is not None, "script_revision": script.revision if script else None,
                    "audit": audit is not None, "audit_passed": audit.passed if audit else None,
                    "audit_blocking": len(audit.blocking()) if audit else None,
                    "approved": self.is_approved(book_id, ch.id),
                    "final": paths.out_audio(ch.id).is_file(),
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
