import time

from fastapi.testclient import TestClient

from app.api import create_app


def _wait(client, job_id, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "failed"):
            return job
        time.sleep(0.2)
    raise AssertionError("job did not finish")


def test_upload_run_edit_approve(settings, sample_pdf):
    app = create_app(settings, fake_llm=True, fake_audio=True)
    with TestClient(app) as client:
        assert client.get("/api/health").json()["fake_llm"] is True
        assert client.get("/").status_code == 200

        bad = client.post("/api/books", files={"file": ("x.pdf", b"not a pdf", "application/pdf")}, data={"book_id": "bad"})
        assert bad.status_code == 400

        with open(sample_pdf, "rb") as fh:
            r = client.post("/api/books", files={"file": ("sample.pdf", fh, "application/pdf")},
                            data={"book_id": "demo", "auto": "true", "upto": "draft"})
        assert r.status_code == 200, r.text
        job = _wait(client, r.json()["job"]["id"])
        assert job["status"] == "done", job

        # SSE stream replays events and terminates
        with client.stream("GET", f"/api/jobs/{job['id']}/events") as resp:
            body = b"".join(resp.iter_bytes()).decode()
        assert "event: stage" in body and '"stage": "ingest"' in body and body.rstrip().endswith("}")

        status = client.get("/api/books/demo").json()
        assert status["ingested"] and len(status["chapters"]) == 3 and all(c["draft"] for c in status["chapters"])
        assert client.get("/api/books/demo/artifacts/../../secret").status_code in (400, 404)
        artifacts = client.get("/api/books/demo/artifacts").json()
        assert any(a["path"] == "scripts/ch01.script.json" for a in artifacts)
        assert client.get("/api/books/demo/artifacts/scripts/ch01.script.json").status_code == 200

        chapter = client.get("/api/books/demo/chapters/ch01").json()
        assert chapter["audit"]["passed"] and chapter["draft"]["audio"] and not chapter["approved"] and chapter["blocks"]

        script = chapter["script"]
        script["segments"][0]["lines"][0]["text"] = "Nee, dat accepteer ik gewoon niet."
        saved = client.put("/api/books/demo/chapters/ch01/script", json=script).json()
        assert saved["revision"] == 2
        assert client.post("/api/books/demo/chapters/ch01/approve", json={}).status_code == 200  # re-audits, then approves
        assert client.get("/api/books/demo/chapters/ch01").json()["approved"]

        glossary = client.get("/api/books/demo/glossary").json()
        glossary["entries"].append({"surface": "overfitting", "kind": "loanword_en", "spoken": "overfitting", "lock": True})
        assert client.put("/api/books/demo/glossary", json=glossary).status_code == 200
        assert any(e["surface"] == "overfitting" for e in client.get("/api/books/demo/glossary").json()["entries"])

        r = client.post("/api/books/demo/chapters/ch01/run", json={"stage": "render"})
        job = _wait(client, r.json()["id"])
        assert job["status"] == "done", job
        chapter = client.get("/api/books/demo/chapters/ch01").json()
        assert chapter["final"]["audio"] and chapter["final"]["transcript"]["blocks"]
        assert client.get(chapter["final"]["audio"]).status_code == 200

        r = client.post("/api/books/demo/chapters/ch99/run", json={"stage": "plan"})
        assert _wait(client, r.json()["id"])["status"] == "failed"
        assert client.get("/api/books/nope").status_code == 404
