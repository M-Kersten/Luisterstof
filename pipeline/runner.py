"""Orchestration: runs stages per book and chapter, writes artifacts, emits events.

The CLI and the web app both drive this class. Every stage reads its inputs
from disk and writes its output to disk, so any stage can be re-run alone.
"""

from __future__ import annotations

import logging
import shutil
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
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
from pipeline.timings import PassTiming, new_run_id, now_iso
from pipeline.timings import append as append_timing

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
        self.run_id = new_run_id()  # groups the timings of everything this pipeline does

    # ------------------------------------------------------------ timings
    @contextmanager
    def _timed(self, book_id: str, chapter_id: str | None, pass_name: str, label: str = "") -> Iterator[None]:
        """Records how long a pass took in the book's timings.jsonl, also when it fails or is stopped."""
        started_at, t0, calls = now_iso(), time.monotonic(), self._llm_usage()[0]
        status = "done"
        try:
            yield
        except Cancelled:
            status = "stopped"
            raise
        except BaseException:
            status = "failed"
            raise
        finally:
            self._record(book_id, chapter_id, pass_name, label, status, started_at, time.monotonic() - t0,
                         self._llm_usage()[0] - calls)

    def _record(self, book_id: str, chapter_id: str | None, pass_name: str, label: str, status: str,
                started_at: str, seconds: float, llm_calls: int = 0) -> None:
        entry = PassTiming(run=self.run_id, book_id=book_id, chapter=chapter_id, pass_name=pass_name, label=label,
                           status=status, started_at=started_at, finished_at=now_iso(), seconds=round(seconds, 1),
                           llm_calls=max(0, llm_calls))
        try:
            append_timing(self.paths(book_id).timings, entry)
        except OSError as exc:  # a full disk must not end the run
            log.warning("could not write timings: %s", exc)
        if self.on_event:  # straight to the listener: emit() would refuse after a Stop, and this must still land
            try:
                self.on_event("timing", status, {**entry.model_dump(), "title": entry.title})
            except Exception:  # noqa: BLE001
                pass

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
        with self._timed(book_id, None, "ingest"):
            return self._ingest(pdf_path, book_id, method=method, title=title)

    def _ingest(self, pdf_path: Path | str, book_id: str, *, method: str = "auto", title: str | None = None) -> Book:
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
        with self._timed(book_id, chapter_id, "plan"):
            return self._plan(book_id, chapter_id)

    def _plan(self, book_id: str, chapter_id: str) -> ContentPlan:
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
        with self._timed(book_id, chapter_id, "glossary"):
            return self._glossary(book_id, chapter_id)

    def _glossary(self, book_id: str, chapter_id: str) -> Glossary:
        from pipeline.plan.thorough import propose_lexicon_thorough

        book = self.load_book(book_id)
        glossary = self.load_glossary(book_id)
        self.emit("glossary", "start", book_id=book_id, chapter=chapter_id)
        before = self._llm_usage()
        chapter = book.chapter(chapter_id)
        entries = propose_lexicon_thorough(chapter, glossary, self.llm, part_chars=self.settings.plan_part_chars,
                                           progress=lambda kind, data: self.emit("glossary", kind, chapter=chapter_id, **data))
        added = glossary.merge(entries)
        if chapter_id not in glossary.chapters:
            glossary.chapters.append(chapter_id)
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
        with self._timed(book_id, chapter_id, "script"):
            script = writer.write(plan, book, glossary, previous_plan=previous_plan,
                                  next_chapter_title=nxt.title if nxt else None)
        script.save(self.paths(book_id).script(chapter_id))
        self.emit("script", "done", book_id=book_id, chapter=chapter_id, lines=sum(len(s.lines) for s in script.segments),
                  estimated_minutes=round(script.estimated_seconds(self.settings.chars_per_second) / 60, 1))
        audit = self.audit(book_id, chapter_id, script=script, llm_checks=llm_checks, label="na het schrijven")
        tried = [(script, audit)]
        rounds = 0
        while rounds < rounds_left and self._needs_rewrite(audit):
            rounds += 1
            self.emit("script", "revise", book_id=book_id, chapter=chapter_id, round=rounds,
                      blocking=len(audit.blocking()), naturalness=len(naturalness_issues(audit)))
            with self._timed(book_id, chapter_id, "revise", f"ronde {rounds}"):
                script = writer.revise(script, audit, plan, glossary, also_naturalness=self.settings.quality_mode)
            script.save(self.paths(book_id).script(chapter_id))
            audit = self.audit(book_id, chapter_id, script=script, llm_checks=llm_checks, label=f"na ronde {rounds}")
            tried.append((script, audit))
        script, audit = self._keep_best(book_id, chapter_id, tried)
        if self.settings.script_polish:
            script, audit = self._polish(writer, book_id, chapter_id, script, audit, plan, glossary, llm_checks)
        self.continuity(book_id, chapter_id, script=script)
        return script, audit

    def _polish(self, writer: Writer, book_id: str, chapter_id: str, script: Script, audit: AuditResult,
                plan: ContentPlan, glossary: Glossary, llm_checks: bool) -> tuple[Script, AuditResult]:
        """The dialogue edit, then the audit again. The edit stays unless it added blocking issues: a script that
        sounds better with as few problems wins, one that broke a fact goes back to the version before."""
        self.emit("script", "polish", book_id=book_id, chapter=chapter_id, segments=len(script.segments))
        with self._timed(book_id, chapter_id, "polish"):
            polished = writer.polish(script, audit, plan, glossary)
        polished.save(self.paths(book_id).script(chapter_id))
        polished_audit = self.audit(book_id, chapter_id, script=polished, llm_checks=llm_checks, label="na de tafellezing")
        if len(polished_audit.blocking()) <= len(audit.blocking()):
            self.emit("script", "polished", book_id=book_id, chapter=chapter_id, revision=polished.revision,
                      blocking=len(polished_audit.blocking()))
            return polished, polished_audit
        return self._keep_best(book_id, chapter_id, [(script, audit), (polished, polished_audit)])

    def _keep_best(self, book_id: str, chapter_id: str, tried: list[tuple[Script, AuditResult]]) -> tuple[Script, AuditResult]:
        """A rewrite round can make a script worse: a model told a line is unsupported may start hedging
        everything. The revision with the best audit wins (fewest blocking issues, then fewest dialogue-shape
        warnings, then fewest warnings; the later one on a tie). If that is an earlier one, it is saved again
        as the newest revision, so the file and its audit stay in step."""
        def score(item: tuple[int, tuple[Script, AuditResult]]) -> tuple:
            i, (_, a) = item
            return (len(a.blocking()), len(naturalness_issues(a)), len(a.warnings()), -i)

        best_i, (best, best_audit) = min(enumerate(tried), key=score)
        last, last_audit = tried[-1]
        if best_i == len(tried) - 1:
            return last, last_audit
        restored = best.model_copy(deep=True)
        restored.revision = last.revision + 1
        restored.save(self.paths(book_id).script(chapter_id))
        audit = best_audit.model_copy(deep=True)
        audit.script_revision = restored.revision
        audit.save(self.paths(book_id).audit(chapter_id))
        self.emit("script", "kept_best", book_id=book_id, chapter=chapter_id, from_revision=best.revision,
                  revision=restored.revision, blocking=len(audit.blocking()), last_blocking=len(last_audit.blocking()))
        return restored, audit

    def _needs_rewrite(self, audit: AuditResult) -> bool:
        """Whether another rewrite round is warranted.

        A blocking issue always is. In quality mode a script that passes but speaks flat is not done
        either: interruptions and reactions have to be written in, and a run that may take all night
        should spend some of it on the part the listener actually hears.
        """
        if not audit.passed:
            return True
        return self.settings.quality_mode and bool(naturalness_issues(audit))

    def audit(self, book_id: str, chapter_id: str, *, script: Script | None = None, llm_checks: bool = True,
              label: str = "") -> AuditResult:
        with self._timed(book_id, chapter_id, "audit", label):
            return self._audit(book_id, chapter_id, script=script, llm_checks=llm_checks)

    def _audit(self, book_id: str, chapter_id: str, *, script: Script | None = None, llm_checks: bool = True) -> AuditResult:
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
        with self._timed(book_id, chapter_id, "continuity"):
            return self._continuity(book_id, chapter_id, script=script)

    def _continuity(self, book_id: str, chapter_id: str, *, script: Script | None = None):
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
        with self._timed(book_id, chapter_id, "render"):
            return self._render(book_id, chapter_id, force=force, synth=synth, transcriber=transcriber, aligner=aligner)

    def _render(self, book_id: str, chapter_id: str, *, force: bool = False, synth: Synth | None = None,
                transcriber=None, aligner=None) -> tuple[Path, Path, RenderManifest]:
        if not force and not self.is_approved(book_id, chapter_id):
            raise StageError("script is not approved for this revision; read the script and audit, then approve first")
        script = self.load_script(book_id, chapter_id)
        glossary = self.load_glossary(book_id)
        self.release_llm()
        self.emit("render", "start", book_id=book_id, chapter=chapter_id, fake=self.fake_audio)
        if not self.fake_audio:
            from pipeline.audio.gpu import busy_warning

            warning = busy_warning(self.settings.chatterbox_workers)
            if warning:
                log.warning(warning)
                self.emit("render", "note", book_id=book_id, chapter=chapter_id, message=warning)
        result = render_final(script, glossary, self.cast, self.settings, self.paths(book_id), synth=synth,
                              transcriber=transcriber, aligner=aligner, fake=self.fake_audio,
                              on_event=lambda kind, data: self.emit("render", kind, **data))
        self.emit("render", "done", book_id=book_id, chapter=chapter_id, audio=str(result[0]),
                  flagged=[t.turn_id for t in result[2].flagged_turns()])
        return result

    # ------------------------------------------------------------------
    def run_chapter(self, book_id: str, chapter_id: str, *, upto: str = "script", llm_checks: bool = True,
                    resume: bool = False) -> dict[str, Any]:
        """Everything that does not need a human or hours of GPU, up to ``upto``.

        ``upto="render"`` approves on its own authority and starts the real
        Chatterbox render: only ask for that when nobody is going to read first.

        ``resume`` skips the stages whose output is already on disk and still
        current (see ``_done``), so a run that died in chapter 7 picks up there.
        A script edited by hand is never rewritten: at most it is audited again.
        """
        order = ["plan", "glossary", "script", "approve", "render"]
        if upto not in order:
            raise ValueError(f"upto must be one of {order}")
        stop = order.index(upto)
        names = {"plan": "plan", "glossary": "lexicon", "script": "script en audit",
                 "approve": "goedkeuren", "render": "Chatterbox render"}
        done = self._done(book_id, chapter_id) if resume else {}

        def step(name: str) -> bool:
            """Announce the step; False when resuming and it is already done."""
            self.emit("run", "step", book_id=book_id, chapter=chapter_id, index=order.index(name) + 1, total=stop + 1,
                      label=names[name])
            if done.get(name):
                self.emit(name, "skipped", book_id=book_id, chapter=chapter_id, reason=done[name])
                self._record(book_id, chapter_id, name, done[name], "skipped", now_iso(), 0.0)
                return False
            return True

        out: dict[str, Any] = {}
        if stop >= 0 and step("plan"):
            out["plan"] = self.plan(book_id, chapter_id)
            done = {}  # a new plan makes everything after it stale
        if stop >= 1 and step("glossary"):
            out["glossary"] = self.glossary(book_id, chapter_id)
        if stop >= 2:
            if step("script"):
                out["script"], out["audit"] = self.script(book_id, chapter_id, llm_checks=llm_checks)
                done = {}
            elif done.get("audit_stale"):
                self.emit("audit", "resume", book_id=book_id, chapter=chapter_id, reason=done["audit_stale"])
                out["audit"] = self.audit(book_id, chapter_id, llm_checks=llm_checks)
            else:
                out["audit"] = AuditResult.load_or_none(self.paths(book_id).audit(chapter_id))
        if stop >= 3 and step("approve"):
            audit = out.get("audit")
            out["approve"] = self.approve(book_id, chapter_id, force=upto == "render" and not (audit and audit.passed))
        if stop >= 4 and step("render"):
            out["render"] = self.render(book_id, chapter_id)[0]
        return out

    def _done(self, book_id: str, chapter_id: str) -> dict[str, str]:
        """Which stages of a chapter can be skipped, with the reason. An artifact counts only when it is at
        least as new as what it was made from (book.json -> plan -> script -> final audio), so a re-ingested
        book or a new plan redoes everything after it."""
        paths = self.paths(book_id)

        def mtime(path: Path) -> float | None:
            return path.stat().st_mtime if path.is_file() else None

        book_t, plan_t, script_t = mtime(paths.book_json), mtime(paths.plan(chapter_id)), mtime(paths.script(chapter_id))
        done: dict[str, str] = {}
        if plan_t is None or book_t is None or plan_t < book_t:
            return done
        done["plan"] = "plan staat er al"
        glossary = Glossary.load_or_none(paths.glossary_json)
        glossary_t = mtime(paths.glossary_json)
        if glossary and chapter_id in glossary.chapters and glossary_t is not None and glossary_t >= book_t:
            done["glossary"] = "lexicon voor dit hoofdstuk staat er al"
        if script_t is None or script_t < plan_t:
            return done
        done["script"] = "script staat er al"
        script = Script.load_or_none(paths.script(chapter_id))
        audit = AuditResult.load_or_none(paths.audit(chapter_id))
        if script is None or audit is None or audit.script_revision != script.revision:
            done["audit_stale"] = "de audit hoort niet bij deze scriptrevisie"
            return done
        if self.is_approved(book_id, chapter_id):
            done["approve"] = "deze revisie is al goedgekeurd"
        audio_t = mtime(paths.out_audio(chapter_id))
        if done.get("approve") and audio_t is not None and audio_t >= script_t:
            done["render"] = "aflevering is al gerenderd"
        return done

    def run_book(self, book_id: str, *, upto: str = "script", chapters: list[str] | None = None, llm_checks: bool = True,
                 resume: bool = True) -> dict[str, dict]:
        """Every chapter up to ``upto``. A chapter that fails does not stop the others: the run goes on and
        ends with a StageError naming the failed chapters. With ``resume`` (the default) finished work is
        skipped, so running the book again after a failure only does what is missing."""
        book = self.load_book(book_id)
        results: dict[str, dict] = {}
        failed: dict[str, str] = {}
        todo = [ch for ch in book.episode_chapters() if not chapters or ch.id in chapters]
        for n, ch in enumerate(todo, start=1):
            self.emit("book", "chapter", book_id=book_id, chapter=ch.id, index=n, total=len(todo), label=f"{ch.id} {ch.title}")
            try:
                with self._timed(book_id, ch.id, "chapter"):
                    results[ch.id] = self.run_chapter(book_id, ch.id, upto=upto, llm_checks=llm_checks, resume=resume)
            except Exception as exc:  # noqa: BLE001 - Cancelled is a BaseException and still ends the run
                log.exception("chapter %s failed; going on with the next one", ch.id)
                failed[ch.id] = str(exc)
                self.emit("book", "chapter_failed", book_id=book_id, chapter=ch.id, error=str(exc)[:500])
        if failed:
            listing = "; ".join(f"{cid}: {err[:160]}" for cid, err in failed.items())
            raise StageError(f"{len(failed)} of {len(todo)} chapters failed ({listing}). The other chapters are done; "
                             "run the book again and only what is missing is made.")
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
