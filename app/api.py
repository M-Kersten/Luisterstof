"""HTTP API. Same interaction model as the recorder project: upload, watch stages stream, inspect artifacts."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
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
from pipeline.voicelab import MAX_TEXT as VOICE_MAX_TEXT
from pipeline.voicelab import PARAMS as VOICE_PARAMS
from pipeline.voicelab import VoiceLab, find_speaker, voice_info
from pipeline.voicelab import clean_params as clean_voice_params
from pipeline.voicelab import save_params as save_voice_params

VOICELAB_JOBS = "stemlab"  # the jobs table wants a book id; voice-lab takes file under this one

STATIC_DIR = Path(__file__).parent / "static"


class ApproveRequest(BaseModel):
    note: str | None = None
    force: bool = False


class BookRunRequest(BaseModel):
    upto: str = "script"  # plan | glossary | script (render would approve without a human reading first)
    quality: bool = False
    fresh: bool = False  # redo what is already done


class VoiceTryRequest(BaseModel):
    text: str
    exaggeration: float
    cfg_weight: float
    temperature: float
    speech_rate: float
    seed: int | None = None


class VoiceParams(BaseModel):
    exaggeration: float
    cfg_weight: float
    temperature: float
    speech_rate: float


class RunRequest(BaseModel):
    stage: str = "script"  # plan | glossary | script | audit | render | all
    upto: str = "script"
    llm_checks: bool = True
    force: bool = False
    # Quality over speed for this one run: more plan review rounds, more script rewrite rounds and a
    # longer per-call ceiling. A chapter written like this can take hours. See Settings.with_quality.
    quality: bool = False


def create_app(settings: Settings | None = None, *, fake_llm: bool | None = None, fake_audio: bool | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    fake_llm = os.environ.get("STUDIEPODCAST_FAKE_LLM") == "1" if fake_llm is None else fake_llm
    fake_audio = os.environ.get("STUDIEPODCAST_FAKE_AUDIO") == "1" if fake_audio is None else fake_audio

    store = JobStore(settings.data_dir / "jobs.sqlite")
    store.requeue_stale()
    # The cast is read again after the voice lab saves a voice's settings, so the next job uses them.
    casts = {"cast": load_cast(settings.cast_dir)}

    def factory(on_event, job_settings: Settings | None = None):
        return Pipeline(job_settings or settings, fake_llm=fake_llm, fake_audio=fake_audio, cast=casts["cast"],
                        on_event=on_event)

    manager = JobManager(store, factory)
    reader = Pipeline(settings, fake_llm=fake_llm, fake_audio=fake_audio, cast=casts["cast"])  # read-only helper, no LLM calls
    lab = VoiceLab(settings, fake_audio=fake_audio)

    # The interactive API docs load their JavaScript from a CDN; offline mode serves none of it.
    app = FastAPI(title="Studiepodcast", version="0.1.0",
                  docs_url=None if settings.offline else "/docs", redoc_url=None if settings.offline else "/redoc")
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
        from pipeline.tags import known_tags

        return {"ok": True, "fake_llm": fake_llm, "fake_audio": fake_audio, "model": settings.opencode_model,
                "llm_backend": settings.llm_backend, "offline": settings.offline,
                "quality_mode": settings.quality_mode, "writing_passes": settings.writing_passes(),
                "tags": known_tags()}

    @app.get("/api/engines")
    def engines() -> dict[str, Any]:
        """For the status chips in the web app: can OpenCode be reached and is Chatterbox installed. Cheap on
        purpose: it asks OpenCode for its model list and does not wake the model; every job still checks that."""
        import importlib.util

        if fake_llm or settings.llm_backend == "fake":
            writer = {"state": "test", "message": "nep-model (testmodus)"}
        else:
            try:
                from pipeline.opencode_llm import OpenCodeLLM

                llm = OpenCodeLLM(settings)
                try:
                    ok, message = llm.check()
                finally:
                    llm.client.close()
            except Exception as exc:  # noqa: BLE001 - offline check, discovery: shown as the reason
                ok, message = False, str(exc)
            writer = {"state": "reachable" if ok else "down", "message": message}
        if fake_audio:
            voice = {"state": "test", "message": "nep-audio (testmodus)"}
        elif importlib.util.find_spec("chatterbox") is None:
            voice = {"state": "down", "message": "chatterbox is not installed in this environment; see `studiepodcast doctor`"}
        else:
            voice = {"state": "ready", "message": "chatterbox is installed; `studiepodcast doctor` checks the weights"}
        return {"writer": writer, "voice": voice}

    @app.get("/api/books")
    def list_books() -> list[dict[str, Any]]:
        return [reader.status(b) for b in reader.books()]

    @app.post("/api/books")
    async def upload_book(file: UploadFile = File(...), book_id: str = Form(""), method: str = Form("auto"),  # noqa: B008
                          auto: bool = Form(True), upto: str = Form("script"), quality: bool = Form(False)) -> dict[str, Any]:
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
            if auto:
                p.check_writer()  # a model that is not running is started now, not found missing at chapter one
            p.ingest(paths.source_pdf, name, method=method)
            if auto:
                return {ch: {k: str(v) for k, v in out.items() if k == "render"} for ch, out in p.run_book(name, upto=upto).items()}
            return {"ingested": True}

        job = manager.submit(name, None, "pipeline" if auto else "ingest", work,
                             settings=settings.with_quality(True) if quality else None)
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

    @app.post("/api/books/{book_id}/run")
    def run_book(book_id: str, req: BookRunRequest) -> dict[str, Any]:
        """Every chapter up to ``upto``, picking up where an earlier run stopped. The PDF is not read again."""
        paths = book_or_404(book_id)
        if not paths.book_json.is_file():
            raise HTTPException(409, "the book has not been read in yet")
        if req.upto not in ("plan", "glossary", "script"):
            raise HTTPException(400, "upto must be plan, glossary or script")

        def work(p: Pipeline):
            p.check_writer()
            out = p.run_book(book_id, upto=req.upto, resume=not req.fresh)
            return {ch: True for ch in out}

        return manager.submit(book_id, None, "pipeline", work, settings=settings.with_quality(True) if req.quality else None)

    @app.get("/api/books/{book_id}/timings")
    def book_timings(book_id: str) -> dict[str, Any]:
        """Start, end and duration of every pass, newest run last, plus the latest run per chapter."""
        from pipeline.timings import load, summary

        entries = load(book_or_404(book_id).timings)
        return {"entries": [{**e.model_dump(), "title": e.title} for e in entries], "summary": summary(entries)}

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
        manifest = RenderManifest.load_or_none(paths.manifest(chapter_id))
        transcript = EpisodeTranscript.load_or_none(paths.transcript(chapter_id))
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
            "final": {"audio": _rel(paths, paths.out_audio(chapter_id)), "transcript": transcript.model_dump() if transcript else None},
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
            if stage in ("plan", "glossary", "script", "audit", "all"):
                p.check_writer()
            if stage == "plan":
                return p.plan(book_id, chapter_id)
            if stage == "glossary":
                return p.glossary(book_id, chapter_id)
            if stage == "script":
                return p.script(book_id, chapter_id, llm_checks=req.llm_checks)[1]
            if stage == "audit":
                return p.audit(book_id, chapter_id, llm_checks=req.llm_checks)
            if stage == "render":
                lab.release()
                return p.render(book_id, chapter_id, force=req.force)[0]
            if stage == "all":
                if req.upto == "render":
                    lab.release()
                return {k: str(v) if isinstance(v, Path) else True for k, v in p.run_chapter(book_id, chapter_id, upto=req.upto, llm_checks=req.llm_checks).items()}
            raise StageError(f"unknown stage {stage}")

        return manager.submit(book_id, chapter_id, stage, work,
                              settings=settings.with_quality(True) if req.quality else None)


    @app.post("/api/books/{book_id}/chapters/{chapter_id}/approve")
    def approve(book_id: str, chapter_id: str, req: ApproveRequest) -> dict[str, Any]:
        book_or_404(book_id)
        try:
            approval = reader.approve(book_id, chapter_id, note=req.note, force=req.force)
        except StageError as exc:
            raise HTTPException(409, str(exc)) from exc
        return approval.model_dump()

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

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> dict[str, Any]:
        try:
            return manager.cancel(job_id)
        except KeyError as exc:
            raise HTTPException(404, "unknown job") from exc

    @app.get("/api/jobs/{job_id}/events")
    def job_events(job_id: str, after: int = 0, last_event_id: str | None = Header(None)):  # noqa: B008
        try:
            store.get(job_id)
        except KeyError as exc:
            raise HTTPException(404, "unknown job") from exc
        resume = int(last_event_id) if last_event_id and last_event_id.isdigit() else after
        return StreamingResponse(manager.stream(job_id, after=resume), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/cast")
    def cast_info() -> dict[str, Any]:
        return {"cast": casts["cast"].model_dump(), "continuity": continuity_text(load_continuity(settings.cast_dir))}

    # ------------------------------------------------------------- voice lab
    def speaker_or_404(speaker_id: str):
        try:
            return find_speaker(casts["cast"], speaker_id)[1]
        except KeyError as exc:
            raise HTTPException(404, f"unknown voice {speaker_id}") from exc

    @app.get("/api/voices")
    def voices() -> dict[str, Any]:
        return {"voices": voice_info(casts["cast"], settings.cast_dir), "params": VOICE_PARAMS,
                "max_text": VOICE_MAX_TEXT, "model_loaded": lab.loaded(), "fake_audio": fake_audio}

    @app.get("/api/voices/{speaker_id}/ref")
    def voice_ref(speaker_id: str):
        sp = speaker_or_404(speaker_id)
        path = settings.cast_dir / sp.voice_ref if sp.voice_ref else None
        if not path or not path.is_file():
            raise HTTPException(404, "no reference clip")
        return FileResponse(path)

    @app.get("/api/voices/{speaker_id}/takes")
    def voice_takes(speaker_id: str) -> list[dict[str, Any]]:
        speaker_or_404(speaker_id)
        return [{**t, "audio": f"/api/voices/{speaker_id}/takes/{t['id']}.wav"} for t in lab.takes(speaker_id)]

    @app.get("/api/voices/{speaker_id}/takes/{take_id}.wav")
    def voice_take_audio(speaker_id: str, take_id: str):
        speaker_or_404(speaker_id)
        try:
            return FileResponse(lab.take_path(speaker_id, take_id), media_type="audio/wav")
        except KeyError as exc:
            raise HTTPException(404, "unknown take") from exc

    @app.delete("/api/voices/{speaker_id}/takes/{take_id}")
    def voice_take_delete(speaker_id: str, take_id: str) -> dict[str, Any]:
        speaker_or_404(speaker_id)
        try:
            lab.delete(speaker_id, take_id)
        except KeyError as exc:
            raise HTTPException(404, "unknown take") from exc
        return {"deleted": take_id}

    @app.post("/api/voices/{speaker_id}/try")
    def voice_try(speaker_id: str, req: VoiceTryRequest) -> dict[str, Any]:
        sp = speaker_or_404(speaker_id)
        try:
            params = clean_voice_params(req.model_dump())
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        ref = settings.cast_dir / sp.voice_ref if sp.voice_ref else None
        if not fake_audio and not (ref and ref.is_file()):
            raise HTTPException(409, f"{sp.name} heeft nog geen referentieclip: zet hem in cast/{sp.voice_ref or 'refs/'}.")

        def work(p: Pipeline):
            return lab.generate(sp, req.text, params, req.seed, on_event=p.on_event)

        return manager.submit(VOICELAB_JOBS, speaker_id, "voice", work)

    @app.put("/api/voices/{speaker_id}/params")
    def voice_save(speaker_id: str, req: VoiceParams) -> dict[str, Any]:
        speaker_or_404(speaker_id)
        try:
            params = clean_voice_params(req.model_dump())
            path = save_voice_params(settings.cast_dir, speaker_id, params)
        except (ValueError, KeyError) as exc:
            raise HTTPException(400, f"niet opgeslagen: {exc}") from exc
        casts["cast"] = load_cast(settings.cast_dir)
        reader.cast = casts["cast"]
        return {"saved": params, "file": path.name}

    @app.post("/api/voices/release")
    def voice_release() -> dict[str, Any]:
        lab.release()
        return {"model_loaded": lab.loaded()}

    if STATIC_DIR.is_dir():
        app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
    return app


def _rel(paths, target: Path) -> str | None:
    if not target.is_file():
        return None
    return f"/api/books/{paths.book_id}/artifacts/{target.relative_to(paths.root).as_posix()}"


app = create_app()
