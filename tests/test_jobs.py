"""Stopping jobs: queued ones are dropped, running ones stop at the next checkpoint, streams resume."""

import threading
import time

import httpx
import pytest
from pydantic import BaseModel

from app.jobs import JobManager, JobStore
from pipeline.cancel import Cancelled
from pipeline.config import Settings
from pipeline.llm import LLMRequest
from pipeline.opencode_llm import OpenCodeLLM
from pipeline.runner import Pipeline


class Verdict(BaseModel):
    ok: bool


def _wait(store, job_id, statuses=("done", "failed", "cancelled"), timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = store.get(job_id)
        if job["status"] in statuses:
            return job
        time.sleep(0.05)
    raise AssertionError(f"job stayed {store.get(job_id)['status']}")


def test_stop_drops_a_queued_job_and_halts_a_running_one(tmp_path, settings):
    store = JobStore(tmp_path / "jobs.sqlite")
    manager = JobManager(store, lambda on_event: Pipeline(settings, fake_llm=True, fake_audio=True, on_event=on_event))
    release = threading.Event()

    def endless(p):
        for i in range(10_000):
            p.emit("plan", "section", index=i + 1, total=10_000)
            release.set()
            time.sleep(0.01)
        return "never"

    running = manager.submit("demo", "ch01", "plan", endless)
    queued = manager.submit("demo", "ch01", "script", lambda p: "ran anyway")
    assert release.wait(10)
    assert manager.cancel(queued["id"])["status"] == "cancelled"
    manager.cancel(running["id"])
    assert _wait(store, running["id"])["status"] == "cancelled"
    events = store.events_since(running["id"])
    assert [(e["stage"], e["status"]) for e in events][-1] == ("job", "cancelled")
    assert any(e["status"] == "stopping" for e in events)
    time.sleep(0.3)
    assert store.get(queued["id"])["status"] == "cancelled"  # the worker skipped it

    after = events[-3]["seq"]
    replay = "".join(manager.stream(running["id"], after=after))
    assert replay.count("event: stage") == 2 and f"id: {events[-1]['seq']}" in replay


def _opencode() -> OpenCodeLLM:
    return OpenCodeLLM(Settings(llm_backend="opencode", opencode_url="http://127.0.0.1:4096",
                                opencode_password="geheim"))


def test_a_model_call_in_flight_stops_when_the_job_is_cancelled():
    llm = _opencode()

    def reply(request):
        llm.abort()  # the job was cancelled while the model was still busy
        return httpx.Response(200, json={"data": {"text": '{"ok": true}'}})

    llm.client = httpx.Client(transport=httpx.MockTransport(reply), base_url="http://127.0.0.1:4096")
    with pytest.raises(Cancelled):
        llm._post("Maak een plan")
    with pytest.raises(Cancelled):  # and it takes no new calls
        llm.generate(LLMRequest(task="support", system="s", user="u", schema=Verdict))


def test_token_progress_is_reported_after_every_answer():
    llm = _opencode()
    llm.client = httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json={"data": {"text": '{"ok": true}'}})),
        base_url="http://127.0.0.1:4096")
    beats = []
    llm.on_tokens = lambda task, tokens, seconds: beats.append((task, tokens))
    llm.generate(LLMRequest(task="lexicon", system="s", user="u", schema=Verdict))
    assert beats == [("lexicon", beats[-1][1])] and beats[-1][1] > 0


def test_a_stopped_pipeline_raises_at_its_next_event(settings):
    p = Pipeline(settings, fake_llm=True, fake_audio=True)
    p.emit("plan", "start")
    p.abort()
    with pytest.raises(Cancelled):
        p.emit("plan", "section")
