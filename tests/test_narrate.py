"""What a run says while it works: plain Dutch per event, and the Meekijken endpoint that tells a run back."""

import time

from fastapi.testclient import TestClient

from app.api import create_app
from pipeline.narrate import duration, entry, say


def test_events_read_as_sentences():
    chapter = say("book", "chapter", {"chapter": "ch03", "index": 3, "total": 9, "label": "ch03 De stressreactie"})
    assert chapter == {"text": "Hoofdstuk 3 van 9: De stressreactie.", "kind": "step"}

    plan = say("plan", "done", {"claims": 15, "needs_expert": True, "warnings": []})
    assert plan["kind"] == "result" and "15 beweringen" in plan["text"] and "gast" in plan["text"]

    failed = say("audit", "done", {"passed": False, "blocking": 4, "warnings": 1, "coverage_missing": ["c3"]})
    assert failed["kind"] == "problem"
    assert failed["text"] == "Controle vond 4 problemen en 1 tip. 1 bewering uit het plan komt nog niet aan bod."
    assert say("audit", "done", {"passed": True, "blocking": 0, "warnings": 0})["text"] == "Controle geslaagd."

    revise = say("script", "revise", {"round": 2, "blocking": 1, "naturalness": 3})
    assert revise["text"].startswith("Herschrijfronde 2: 1 probleem en 3 opmerkingen over hoe het klinkt")

    written = say("script", "done", {"lines": 60, "estimated_minutes": 14.5})
    assert written["text"].endswith("60 regels, ongeveer 14,5 minuten.")  # a Dutch decimal comma

    call = say("llm", "call", {"task": "script_segment", "typical": 95})
    assert call["kind"] == "detail" and "een blok van het gesprek schrijven" in call["text"]
    assert "1 min 35 s" in call["text"]

    check = say("audit", "segment", {"index": 2, "total": 6, "label": "bronnencontrole: Blok 1"})
    assert check["text"] == "Controle (bronnen), blok 2 van 6: Blok 1."

    done = say("timing", "done", {"pass_name": "chapter", "seconds": 4380})
    assert done == {"text": "Hoofdstuk klaar na 1 u 13 min.", "kind": "result"}
    assert say("timing", "done", {"pass_name": "plan", "seconds": 12}) is None  # the plan says its own result
    assert say("run", "step", {"index": 1, "total": 3, "label": "plan"}) is None
    assert say("nothing", "here", {}) is None
    assert duration(0) == "0 s" and duration(125) == "2 min 05 s" and duration(3600) == "1 u 00 min"


def test_entry_keeps_what_the_page_needs():
    ev = {"seq": 7, "ts": "2026-11-16T22:14:00+00:00", "stage": "script", "status": "written",
          "data": {"chapter": "ch01", "index": 2, "total": 6, "label": "Blok 1", "lines": 6,
                   "preview": [{"speaker": "Tessa", "text": "Stel je voor."}]}}
    out = entry(ev)
    assert out["chapter"] == "ch01" and out["seq"] == 7 and out["kind"] == "detail"
    assert out["text"] == "Blok 2 van 6 staat er: 6 regels." and out["preview"][0]["speaker"] == "Tessa"
    assert entry({"stage": "run", "status": "step", "data": {}}) is None


def test_a_run_tells_itself(pipeline, sample_pdf):
    events = []
    pipeline.on_event = lambda stage, status, data: events.append((stage, status, data))
    pipeline.ingest(sample_pdf, "demo")
    pipeline.run_book("demo", upto="script")
    said = [(s, st, d["say"]) for s, st, d in events if "say" in d]
    kinds = {k for _, _, x in said for k in [x["kind"]]}
    assert {"step", "result", "detail"} <= kinds
    # every pass that starts or finishes says so
    for stage, status in [("ingest", "done"), ("plan", "start"), ("plan", "done"), ("script", "start"),
                          ("script", "done"), ("audit", "done"), ("book", "chapter")]:
        assert any(s == stage and st == status for s, st, _ in said), (stage, status)
    # each question to the model is announced with what it is for, before it is asked
    calls = [d for s, st, d in events if s == "llm" and st == "call"]
    assert calls and all(c["what"] and c["task"] for c in calls)
    # each block that is written comes with its first lines and who says them
    written = [d for s, st, d in events if s == "script" and st == "written"]
    assert written and all(w["preview"] and w["preview"][0]["speaker"] in ("Tessa", "Joris") for w in written
                           if w["preview"])


def test_activity_endpoint_tells_the_run_back(settings, sample_pdf):
    app = create_app(settings, fake_llm=True, fake_audio=True)
    with TestClient(app) as client:
        assert client.get("/api/books/nope/activity").status_code == 404
        with open(sample_pdf, "rb") as fh:
            r = client.post("/api/books", files={"file": ("sample.pdf", fh, "application/pdf")},
                            data={"book_id": "demo", "auto": "true", "upto": "script"})
        job_id = r.json()["job"]["id"]
        deadline = time.time() + 120
        while client.get(f"/api/jobs/{job_id}").json()["status"] not in ("done", "failed"):
            assert time.time() < deadline
            time.sleep(0.2)

        a = client.get("/api/books/demo/activity").json()
        assert a["job"]["id"] == job_id and a["jobs"][0]["id"] == job_id
        texts = [e["text"] for e in a["entries"]]
        assert texts[0].startswith("Het boek inlezen")
        assert any(t.startswith("Hoofdstuk 1 van ") for t in texts)
        assert any(e["kind"] == "result" and e["chapter"] for e in a["entries"])
        assert a["last_seq"] >= max(e["seq"] for e in a["entries"])
        assert a["typical_chapter_seconds"] is None or a["typical_chapter_seconds"] > 0

        light = client.get("/api/books/demo/activity", params={"entries": "false"}).json()
        assert "entries" not in light and light["job"]["id"] == job_id
        assert client.get("/api/books/demo/activity", params={"job": "nope"}).status_code == 404
