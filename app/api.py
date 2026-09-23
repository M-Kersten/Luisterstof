"""HTTP API. Same interaction model as the recorder project: upload, watch stages stream, inspect artifacts."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.jobs import JobManager, JobStore
from pipeline.config import Settings
from pipeline.models import (
    Approval,
    AuditResult,
    ContentPlan,
    EpisodeTranscript,
    Glossary,
    RenderManifest,
    Script,
)
from pipeline.paths import validate_id
from pipeline.runner import Pipeline, StageError
from pipeline.script.cast import continuity_text, load_cast, load_continuity

STATIC_DIR = Path(__file__).parent / "static"


class ElevenRequest(BaseModel):
    blocks: list[str]


class ApproveRequest(BaseModel):
    note: str | None = None
    force: bool = False


class RunRequest(BaseModel):
    stage: str = "script"  # plan | glossary | script | audit | draft | render | all
    upto: str = "draft"
    llm_checks: bool = True
    force: bool = False


def create_app(settings: Settings | None = None, *, fake_llm: bool | None = None, fake_audio: bool | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    fake_llm = os.environ.get("STUDIEPODCAST_FAKE_LLM") == "1" if fake_llm is None else fake_llm
    fake_audio = os.environ.get("STUDIEPODCAST_FAKE_AUDIO") == "1" if fake_audio is None else fake_audio

    store = JobStore(settings.data_dir / "jobs.sqlite")
    store.requeue_stale()
    cast = load_cast(settings.cast_dir)

    def factory(on_event):
        return Pipeline(settings, fake_llm=fake_llm, fake_audio=fake_audio, cast=cast, on_event=on_event)

    manager = JobManager(store, factory)
    reader = Pipeline(settings, fake_llm=fake_llm, fake_audio=fake_audio, cast=cast)  # read-only helper, no LLM calls

    app = FastAPI(title="Studiepodcast", version="0.1.0")
    app.state.settings = settings
    app.state.manager = manager
    app.state.store = store

    def book_or_404(book_id: str):
        try:
            validate_id(book_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if not reader.paths(book_id).root.exists():
            raise HTTPException(404, f"unknown book {book_id}")
        return reader.paths(book_id)

    # ------------------------------------------------------------------ books
    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "fake_llm": fake_llm, "fake_audio": fake_audio, "model": settings.llm_model}

    @app.get("/api/books")
    def list_books() -> list[dict[str, Any]]:
        return [reader.status(b) for b in reader.books()]

    @app.post("/api/books")
    async def upload_book(file: UploadFile = File(...), book_id: str = Form(""), method: str = Form("auto"),  # noqa: B008
                          auto: bool = Form(True), upto: str = Form("draft")) -> dict[str, Any]:
        name = book_id.strip() or re.sub(r"[^A-Za-z0-9_-]+", "-", Path(file.filename or "boek").stem).strip("-").lower()[:40] or "boek"
        try:
            validate_id(name)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        paths = reader.paths(name).ensure()
        payload = await file.read()
        if not payload.startswith(b"%PDF"):
            raise HTTPException(400, "not a PDF")
        paths.source_pdf.write_bytes(payload)

        def work(p: Pipeline):
            p.ingest(paths.source_pdf, name, method=method)
            if auto:
                return {ch: {k: str(v) for k, v in out.items() if k in ("draft", "render")} for ch, out in p.run_book(name, upto=upto).items()}
            return {"ingested": True}

        job = manager.submit(name, None, "pipeline" if auto else "ingest", work)
        return {"book_id": name, "job": job}

    @app.get("/api/books/{book_id}")
    def book_status(book_id: str) -> dict[str, Any]:
        book_or_404(book_id)
        return reader.status(book_id)

    @app.get("/api/books/{book_id}/book")
    def book_json(book_id: str) -> Any:
        paths = book_or_404(book_id)
        if not paths.book_json.is_file():
            raise HTTPException(404, "not ingested yet")
        return FileResponse(paths.book_json, media_type="application/json")

    @app.get("/api/books/{book_id}/artifacts")
    def artifacts(book_id: str) -> list[dict[str, Any]]:
        return book_or_404(book_id).list_artifacts()

    @app.get("/api/books/{book_id}/artifacts/{relative:path}")
    def artifact(book_id: str, relative: str):
        paths = book_or_404(book_id)
        try:
            target = paths.resolve_artifact(relative)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if not target.is_file():
            raise HTTPException(404, "no such artifact")
        return FileResponse(target)

    @app.get("/api/books/{book_id}/glossary")
    def get_glossary(book_id: str) -> Any:
        book_or_404(book_id)
        return reader.load_glossary(book_id).model_dump()

    @app.put("/api/books/{book_id}/glossary")
    def put_glossary(book_id: str, glossary: Glossary) -> Any:
        book_or_404(book_id)
        glossary.book_id = book_id
        return reader.save_glossary(book_id, glossary).model_dump()

    # --------------------------------------------------------------- chapters
    @app.get("/api/books/{book_id}/chapters/{chapter_id}")
    def chapter(book_id: str, chapter_id: str) -> dict[str, Any]:
        paths = book_or_404(book_id)
        try:
            validate_id(chapter_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        plan = ContentPlan.load_or_none(paths.plan(chapter_id))
        script = Script.load_or_none(paths.script(chapter_id))
        audit = AuditResult.load_or_none(paths.audit(chapter_id))
        approval = Approval.load_or_none(paths.approval(chapter_id))
        manifest = RenderManifest.load_or_none(paths.manifest(chapter_id, "final"))
        draft_tr = EpisodeTranscript.load_or_none(paths.transcript(chapter_id, "draft"))
        final_tr = EpisodeTranscript.load_or_none(paths.transcript(chapter_id, "final"))
        blocks = []
        if script is not None:
            from pipeline.audio.chunker import chunk_script

            blocks = [b.__dict__ for b in chunk_script(script)]
        manifest_quality = None
        if manifest is not None:
            verified, verifiable = manifest.verification_coverage()
            manifest_quality = {"verified_turns": verified, "verifiable_turns": verifiable,
                                "alignment_fallback_turns": manifest.alignment_fallback_count()}
        return {
            "book_id": book_id,
            "chapter_id": chapter_id,
            "plan": plan.model_dump() if plan else None,
            "script": script.model_dump() if script else None,
            "audit": audit.model_dump() if audit else None,
            "approval": approval.model_dump() if approval else None,
            "approved": reader.is_approved(book_id, chapter_id),
            "manifest": manifest.model_dump() if manifest else None,
            "manifest_quality": manifest_quality,
            "draft": {"audio": _rel(paths, paths.out_audio(chapter_id, "draft")), "transcript": draft_tr.model_dump() if draft_tr else None},
            "final": {"audio": _rel(paths, paths.out_audio(chapter_id, "final")), "transcript": final_tr.model_dump() if final_tr else None},
            "blocks": blocks,
        }

    @app.put("/api/books/{book_id}/chapters/{chapter_id}/script")
    def put_script(book_id: str, chapter_id: str, script: Script) -> dict[str, Any]:
        book_or_404(book_id)
        script.episode_id = chapter_id
        saved = reader.save_script(book_id, chapter_id, script)
        return {"revision": saved.revision, "approved": False}


    @app.post("/api/books/{book_id}/chapters/{chapter_id}/run")
    def run_stage(book_id: str, chapter_id: str, req: RunRequest) -> dict[str, Any]:
        book_or_404(book_id)
        stage = req.stage

        def work(p: Pipeline):
            if stage == "plan":
                return p.plan(book_id, chapter_id)
            if stage == "glossary":
                return p.glossary(book_id, chapter_id)
            if stage == "script":
                return p.script(book_id, chapter_id, llm_checks=req.llm_checks)[1]
            if stage == "audit":
                return p.audit(book_id, chapter_id, llm_checks=req.llm_checks)
            if stage == "draft":
                return p.draft(book_id, chapter_id)[0]
            if stage == "render":
                return p.render(book_id, chapter_id, force=req.force)[0]
            if stage == "all":
                return {k: str(v) if isinstance(v, Path) else True for k, v in p.run_chapter(book_id, chapter_id, upto=req.upto, llm_checks=req.llm_checks).items()}
            raise StageError(f"unknown stage {stage}")

        return manager.submit(book_id, chapter_id, stage, work)


    @app.post("/api/books/{book_id}/chapters/{chapter_id}/approve")
    def approve(book_id: str, chapter_id: str, req: ApproveRequest) -> dict[str, Any]:
        book_or_404(book_id)
        try:
            approval = reader.approve(book_id, chapter_id, note=req.note, force=req.force)
        except StageError as exc:
            raise HTTPException(409, str(exc)) from exc
        return approval.model_dump()


    @app.post("/api/books/{book_id}/chapters/{chapter_id}/eleven")
    def eleven(book_id: str, chapter_id: str, req: ElevenRequest) -> dict[str, Any]:
        book_or_404(book_id)
        blocks = list(req.blocks)

        def work(p: Pipeline):
            p.eleven(book_id, chapter_id, blocks)
            return p.render(book_id, chapter_id, force=True)[0]

        return manager.submit(book_id, chapter_id, "eleven", work)

    # ------------------------------------------------------------------- jobs
    @app.get("/api/jobs")
    def jobs(book_id: str | None = None) -> list[dict[str, Any]]:
        return store.list(book_id)

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str) -> dict[str, Any]:
        try:
            return store.get(job_id)
        except KeyError as exc:
            raise HTTPException(404, "unknown job") from exc

    @app.get("/api/jobs/{job_id}/events")
    def job_events(job_id: str):
        try:
            store.get(job_id)
        except KeyError as exc:
            raise HTTPException(404, "unknown job") from exc
        return StreamingResponse(manager.stream(job_id), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/cast")
    def cast_info() -> dict[str, Any]:
        return {"cast": cast.model_dump(), "continuity": continuity_text(load_continuity(settings.cast_dir))}

    if STATIC_DIR.is_dir():
        app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
    return app


def _rel(paths, target: Path) -> str | None:
    if not target.is_file():
        return None
    return f"/api/books/{paths.book_id}/artifacts/{target.relative_to(paths.root).as_posix()}"


app = create_app()
