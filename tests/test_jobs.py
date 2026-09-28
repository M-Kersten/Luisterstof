"""Stopping jobs: queued ones are dropped, running ones stop at the next checkpoint, streams resume."""

import threading
import time

import httpx
import pytest
from pydantic import BaseModel

from app.jobs import JobManager, JobStore
from pipeline import local_llm
from pipeline.cancel import Cancelled
from pipeline.config import Settings
from pipeline.llm import LLMRequest
from pipeline.local_llm import LocalLLM
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


def test_a_local_model_call_stops_between_tokens():
    llm = LocalLLM(Settings().with_(llm_backend="local"))

    def body():
        yield b'{"message": {"content": "{\\"verdict\\""}, "done": false}\n'
        llm.abort()
        yield b'{"message": {"content": ": \\"ok\\"}"}, "done": false}\n'
        yield b'{"message": {"content": ""}, "done": true, "done_reason": "stop"}\n'

    llm.client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=body())),
                              base_url="http://localhost:11434")
    with pytest.raises(Cancelled):
        llm._post({"model": "m", "messages": []}, "plan_section")
    with pytest.raises(Cancelled):  # and it takes no new calls
        llm.generate(LLMRequest(task="support", system="s", user="u", schema=Verdict))


def test_heartbeat_reports_tokens_while_the_model_writes(monkeypatch):
    monkeypatch.setattr(local_llm, "HEARTBEAT_S", 0.0)
    llm = LocalLLM(Settings().with_(llm_backend="local"))
    lines = [b'{"message": {"content": "x"}, "done": false}\n'] * 5 + [b'{"message": {}, "done": true, "done_reason": "stop", "eval_count": 5}\n']
    llm.client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=b"".join(lines))),
                              base_url="http://localhost:11434")
    beats = []
    llm.on_tokens = lambda task, tokens, seconds: beats.append((task, tokens))
    content, finish, usage, _ = llm._post({"model": "m", "messages": []}, "lexicon")
    assert content == "xxxxx" and finish == "stop" and usage.output_tokens == 5
    assert beats and beats[-1] == ("lexicon", 6)


def test_a_stopped_pipeline_raises_at_its_next_event(settings):
    p = Pipeline(settings, fake_llm=True, fake_audio=True)
    p.emit("plan", "start")
    p.abort()
    with pytest.raises(Cancelled):
        p.emit("plan", "section")
