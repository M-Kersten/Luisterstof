"""Job queue and state machine on SQLite, with a single background worker.

Stages run one at a time on purpose: the GPU is shared and the frontier API
calls for one episode already run in parallel where it matters.
"""

from __future__ import annotations

import json
import queue
import sqlite3
import threading
import time
import traceback
import uuid
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

TERMINAL = {"done", "failed", "cancelled"}


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


class JobStore:
    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, book_id TEXT, chapter_id TEXT, stage TEXT, status TEXT,
                    created_at TEXT, updated_at TEXT, error TEXT, result TEXT
                );
                CREATE TABLE IF NOT EXISTS events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT, ts TEXT, stage TEXT, status TEXT, data TEXT
                );
                CREATE INDEX IF NOT EXISTS events_job ON events(job_id, seq);
                """
            )
            self._conn.commit()

    def create(self, book_id: str, chapter_id: str | None, stage: str) -> dict[str, Any]:
        job_id = uuid.uuid4().hex[:12]
        now = _now()
        with self._lock:
            self._conn.execute(
                "INSERT INTO jobs (id, book_id, chapter_id, stage, status, created_at, updated_at) VALUES (?,?,?,?,?,?,?)",
                (job_id, book_id, chapter_id, stage, "queued", now, now),
            )
            self._conn.commit()
        return self.get(job_id)

    def update(self, job_id: str, status: str, *, error: str | None = None, result: Any = None) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE jobs SET status=?, updated_at=?, error=COALESCE(?, error), result=COALESCE(?, result) WHERE id=?",
                (status, _now(), error, json.dumps(result, default=str) if result is not None else None, job_id),
            )
            self._conn.commit()

    def get(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return self._row(row)

    def list(self, book_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            if book_id:
                rows = self._conn.execute("SELECT * FROM jobs WHERE book_id=? ORDER BY created_at DESC LIMIT ?", (book_id, limit)).fetchall()
            else:
                rows = self._conn.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [self._row(r) for r in rows]

    def add_event(self, job_id: str, stage: str, status: str, data: dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute("INSERT INTO events (job_id, ts, stage, status, data) VALUES (?,?,?,?,?)",
                               (job_id, _now(), stage, status, json.dumps(data, default=str, ensure_ascii=False)))
            self._conn.commit()

    def events_since(self, job_id: str, after_seq: int = 0, limit: int = 500) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM events WHERE job_id=? AND seq>? ORDER BY seq LIMIT ?",
                                      (job_id, after_seq, limit)).fetchall()
        out = []
        for r in rows:
            out.append({"seq": r["seq"], "ts": r["ts"], "stage": r["stage"], "status": r["status"],
                        "data": json.loads(r["data"] or "{}")})
        return out

    def requeue_stale(self) -> int:
        """Jobs left 'running' by a crashed process become failed on startup."""
        with self._lock:
            cur = self._conn.execute("UPDATE jobs SET status='failed', error='worker restarted', updated_at=? WHERE status IN ('running','queued')", (_now(),))
            self._conn.commit()
            return cur.rowcount

    @staticmethod
    def _row(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        if d.get("result"):
            try:
                d["result"] = json.loads(d["result"])
            except json.JSONDecodeError:
                pass
        return d


class JobManager:
    """Runs submitted callables on one worker thread, streaming their events."""

    def __init__(self, store: JobStore, pipeline_factory: Callable[[Callable[[str, str, dict], None]], Any]):
        self.store = store
        self.pipeline_factory = pipeline_factory
        self._queue: queue.Queue[tuple[str, Callable[[Any], Any]] | None] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="studiepodcast-worker", daemon=True)
        self._started = False

    def start(self) -> None:
        if not self._started:
            self._started = True
            self._thread.start()

    def stop(self) -> None:
        if self._started:
            self._queue.put(None)

    def submit(self, book_id: str, chapter_id: str | None, stage: str, fn: Callable[[Any], Any]) -> dict[str, Any]:
        job = self.store.create(book_id, chapter_id, stage)
        self._queue.put((job["id"], fn))
        self.start()
        return job

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            job_id, fn = item
            self.store.update(job_id, "running")

            def on_event(stage: str, status: str, data: dict, job_id=job_id) -> None:
                self.store.add_event(job_id, stage, status, data)

            try:
                pipeline = self.pipeline_factory(on_event)
                result = fn(pipeline)
                self.store.update(job_id, "done", result=_summarise(result))
            except Exception as exc:  # noqa: BLE001 - surfaced to the UI
                self.store.add_event(job_id, "job", "error", {"error": str(exc), "trace": traceback.format_exc()[-2000:]})
                self.store.update(job_id, "failed", error=str(exc))

    def stream(self, job_id: str, poll_s: float = 0.4, timeout_s: float = 3600) -> Iterator[str]:
        """Server-sent events: every stage event, then a final job status event."""
        last = 0
        started = time.monotonic()
        yield f"event: job\ndata: {json.dumps(self.store.get(job_id))}\n\n"
        while True:
            for ev in self.store.events_since(job_id, last):
                last = ev["seq"]
                yield f"event: stage\ndata: {json.dumps(ev, ensure_ascii=False)}\n\n"
            job = self.store.get(job_id)
            if job["status"] in TERMINAL:
                for ev in self.store.events_since(job_id, last):
                    last = ev["seq"]
                    yield f"event: stage\ndata: {json.dumps(ev, ensure_ascii=False)}\n\n"
                yield f"event: job\ndata: {json.dumps(job)}\n\n"
                return
            if time.monotonic() - started > timeout_s:
                yield "event: timeout\ndata: {}\n\n"
                return
            time.sleep(poll_s)


def _summarise(result: Any) -> Any:
    if result is None:
        return None
    if isinstance(result, (str, int, float, bool)):
        return result
    if isinstance(result, Path):
        return str(result)
    if isinstance(result, dict):
        return {k: _summarise(v) for k, v in result.items()}
    if isinstance(result, (list, tuple)):
        return [_summarise(v) for v in result]
    if hasattr(result, "model_dump"):
        d = result.model_dump()
        return {k: v for k, v in d.items() if not isinstance(v, (list, dict))}
    return str(result)
