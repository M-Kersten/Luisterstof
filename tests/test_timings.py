"""Every pass leaves its start, end and duration behind, so a night's run can be read back in the morning."""

import pytest

from pipeline.timings import fmt_seconds, load, summary


def test_a_chapter_run_records_every_pass_in_order(ingested):
    p = ingested
    p.run_book("demo", upto="script", chapters=["ch01"])
    entries = load(p.paths("demo").timings)
    passes = [(e.pass_name, e.label) for e in entries if e.chapter == "ch01"]
    assert passes == [("plan", ""), ("glossary", ""), ("script", ""), ("audit", "na het schrijven"), ("polish", ""),
                      ("audit", "na de tafellezing"), ("continuity", ""), ("chapter", "")]
    assert ("ingest", None) == (entries[0].pass_name, entries[0].chapter)
    assert len({e.run for e in entries}) == 1 and all(e.status == "done" for e in entries)
    assert all(e.started_at <= e.finished_at and e.seconds >= 0 for e in entries)
    assert next(e for e in entries if e.pass_name == "plan").llm_calls > 0

    report = summary(entries)
    chapter = report["chapters"]["ch01"]
    assert [x["pass_name"] for x in chapter["passes"]][0] == "plan" and chapter["seconds"] >= 0
    assert report["latest_run"]["started_at"] <= report["latest_run"]["finished_at"]


def test_a_later_run_is_the_one_the_summary_shows_and_failures_are_recorded(ingested):
    p = ingested
    p.run_book("demo", upto="plan", chapters=["ch01"])
    from pipeline.runner import Pipeline

    again = Pipeline(p.settings, llm=p.llm, fake_audio=True)  # a new job: a new run id
    original = again.llm.handlers["lexicon"]
    again.llm.on("lexicon", lambda req: (_ for _ in ()).throw(RuntimeError("model weg")))
    with pytest.raises(Exception, match="model weg"):
        again.run_book("demo", upto="glossary", chapters=["ch01"])
    again.llm.on("lexicon", original)
    entries = load(p.paths("demo").timings)
    latest = summary(entries)["chapters"]["ch01"]
    assert latest["run"] == again.run_id
    statuses = {(x["pass_name"], x["status"]) for x in latest["passes"]}
    assert ("plan", "skipped") in statuses and ("glossary", "failed") in statuses
    assert any("model weg" in x["error"] for x in latest["passes"] if x["status"] == "failed")  # the why, for the UI


def test_durations_read_like_a_clock():
    assert fmt_seconds(42) == "42 s" and fmt_seconds(75) == "1 min 15 s" and fmt_seconds(3 * 3600 + 125) == "3 u 02 min"


def test_the_laptop_is_kept_awake_on_windows_and_released_after(monkeypatch):
    import ctypes
    import sys

    from pipeline import awake

    calls = []

    class Kernel32:
        def SetThreadExecutionState(self, flags):  # noqa: N802 - the Windows name
            calls.append(flags)
            return 1

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(ctypes, "windll", type("WinDLL", (), {"kernel32": Kernel32()})(), raising=False)
    with awake.keep_awake() as on:
        assert on and calls == [awake.ES_CONTINUOUS | awake.ES_SYSTEM_REQUIRED]
    assert calls[-1] == awake.ES_CONTINUOUS  # sleep allowed again

    monkeypatch.setattr(sys, "platform", "linux")
    with awake.keep_awake() as on:
        assert on is False


def test_the_book_can_be_carried_on_from_the_web_app(settings, sample_pdf):
    import time

    from fastapi.testclient import TestClient

    from app.api import create_app

    app = create_app(settings, fake_llm=True, fake_audio=True)
    with TestClient(app) as client:
        with open(sample_pdf, "rb") as fh:
            job = client.post("/api/books", files={"file": ("s.pdf", fh, "application/pdf")},
                              data={"book_id": "demo", "auto": "false"}).json()["job"]

        def wait(job_id):
            for _ in range(600):
                j = client.get(f"/api/jobs/{job_id}").json()
                if j["status"] in ("done", "failed", "cancelled"):
                    return j
                time.sleep(0.1)
            raise AssertionError("job did not finish")

        assert wait(job["id"])["status"] == "done"
        assert client.post("/api/books/demo/run", json={"upto": "render"}).status_code == 400
        run = client.post("/api/books/demo/run", json={"upto": "plan"}).json()
        assert wait(run["id"])["status"] == "done"
        timings = client.get("/api/books/demo/timings").json()
        assert set(timings["summary"]["chapters"]) == {"ch01", "ch02", "ch03"}
        assert any(e["pass_name"] == "plan" and e["title"] == "plan" for e in timings["entries"])
        again = client.post("/api/books/demo/run", json={"upto": "plan"}).json()  # carries on: nothing left to do
        assert wait(again["id"])["status"] == "done"
        latest = client.get("/api/books/demo/timings").json()["summary"]["chapters"]["ch01"]["passes"]
        assert {p["status"] for p in latest if p["pass_name"] == "plan"} == {"skipped"}
