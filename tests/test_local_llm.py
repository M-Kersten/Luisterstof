"""Local LLM backend and offline mode: nothing leaves the local network."""

import base64
import json
import re
import socket

import httpx
import pytest
from pydantic import BaseModel

from pipeline import fake_handlers as fh
from pipeline.config import Settings
from pipeline.ingest.figures import FigureDescription
from pipeline.ingest.structure import StructureGuess
from pipeline.llm import AnthropicLLM, LLMError, LLMRequest, LLMTruncated, image_block, make_llm, text_block
from pipeline.local_llm import LocalLLM, inline_schema
from pipeline.offline import HF_OFFLINE_ENV, OfflineViolation, apply_env, check_local_url
from pipeline.plan.content_plan import PlanOut
from pipeline.plan.glossary import LexiconOut
from pipeline.runner import Pipeline
from pipeline.script.audit import LintOut, SupportOut
from pipeline.script.continuity import ContinuityOut
from pipeline.script.writer import PerformanceSegmentOut, SegmentOut

SCHEMAS = (PlanOut, LexiconOut, StructureGuess, FigureDescription, SegmentOut, PerformanceSegmentOut, SupportOut,
           LintOut, ContinuityOut)
HANDLERS = {"PlanOut": fh.fake_plan, "LexiconOut": fh.fake_lexicon, "StructureGuess": fh.fake_structure,
            "FigureDescription": fh.fake_figure_caption, "SegmentOut": fh.fake_script_segment,
            "PerformanceSegmentOut": fh.fake_script_scene, "SupportOut": fh.fake_support, "LintOut": fh.fake_lint,
            "ContinuityOut": fh.fake_continuity}


class Answer(BaseModel):
    verdict: str
    score: int


def _local(settings=None, **overrides) -> Settings:
    return (settings or Settings()).with_(llm_backend="local", **overrides)


class FakeServer:
    """Stands in for Ollama (/api/*) or an OpenAI-compatible server; records every request."""

    def __init__(self, replies=None, models=("gemma3:27b",)):
        self.replies = list(replies or [])
        self.models = list(models)
        self.requests: list[tuple[str, str, dict]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        self.requests.append((request.method, request.url.path, body))
        path = request.url.path
        if path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": m} for m in self.models]})
        if path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": m} for m in self.models]})
        if path == "/api/generate":
            return httpx.Response(200, json={"done": True})
        content, finish = self.replies.pop(0) if self.replies else (self._canned(body, path), "stop")
        if path == "/api/chat":
            return httpx.Response(200, json={"message": {"role": "assistant", "content": content}, "done_reason": finish,
                                             "prompt_eval_count": 100, "eval_count": 20})
        return httpx.Response(200, json={"choices": [{"message": {"content": content}, "finish_reason": finish}],
                                         "usage": {"prompt_tokens": 100, "completion_tokens": 20}})

    @staticmethod
    def _canned(body, path) -> str:
        schema = body["format"] if path == "/api/chat" else body["response_format"]["json_schema"]["schema"]
        system, user = body["messages"][0]["content"], body["messages"][1]["content"]
        segment = re.search(r"\(type (\w+)\)", user)
        request = LLMRequest(task=schema["title"], system=system, user=user, schema=None,
                             metadata={"segment": segment.group(1)} if segment else {})
        result = HANDLERS[schema["title"]](request)
        return json.dumps(result.model_dump() if hasattr(result, "model_dump") else result, ensure_ascii=False)

    def chats(self):
        return [b for _, p, b in self.requests if p in ("/api/chat", "/v1/chat/completions")]


def _request(**kw) -> LLMRequest:
    return LLMRequest(task=kw.pop("task", "support"), system=kw.pop("system", "Je bent een beoordelaar."),
                      user=kw.pop("user", "Beoordeel dit."), schema=kw.pop("schema", Answer), **kw)


def test_every_pipeline_schema_inlines_without_references():
    for schema in SCHEMAS:
        text = json.dumps(inline_schema(schema.model_json_schema()))
        assert "$ref" not in text and "$defs" not in text, schema.__name__


def test_ollama_request_carries_schema_context_and_images_and_parses_a_messy_answer():
    png = base64.b64encode(b"\x89PNG fake").decode()
    server = FakeServer([('<think>eerst nadenken</think>\n```json\n{"verdict": "ok", "score": 4}\n```', "stop")])
    llm = LocalLLM(_local(), transport=httpx.MockTransport(server))
    result = llm.generate(_request(task="figure_caption", system=[text_block("Regels.", cache=True)],
                                   user=[image_block(png), text_block("Beschrijf de figuur.")]))
    assert result == Answer(verdict="ok", score=4)
    method, path, body = server.requests[0]
    assert (method, path, body["model"], body["stream"]) == ("POST", "/api/chat", "gemma3:27b", False)
    assert body["options"]["num_ctx"] == 32768 and body["options"]["temperature"] == 0.2
    assert body["format"]["title"] == "Answer" and "cache_control" not in json.dumps(body)
    assert "Antwoordformaat" in body["messages"][0]["content"] and '"score"' in body["messages"][0]["content"]
    assert body["messages"][1] == {"role": "user", "content": "Beschrijf de figuur.", "images": [png]}
    assert llm.usage.calls == 1 and llm.usage.input_tokens == 100


def test_openai_compatible_request_shape():
    png = base64.b64encode(b"img").decode()
    server = FakeServer([('{"verdict": "ja", "score": 2}', "stop")])
    llm = LocalLLM(_local(local_llm_api="openai", local_llm_url="http://127.0.0.1:8080"), transport=httpx.MockTransport(server))
    assert llm.generate(_request(task="script_segment", user=[text_block("Schrijf."), image_block(png)])).score == 2
    _, path, body = server.requests[0]
    assert path == "/v1/chat/completions" and body["temperature"] == 0.8 and body["max_tokens"] == 8192
    assert body["response_format"]["json_schema"]["name"] == "Answer"
    assert body["messages"][1]["content"][1] == {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png}"}}


def test_invalid_answer_goes_back_once_with_the_error():
    server = FakeServer([("{not json", "stop"), ('{"verdict": "ok", "score": 1}', "stop")])
    llm = LocalLLM(_local(), transport=httpx.MockTransport(server))
    assert llm.generate(_request()).score == 1
    second = server.chats()[1]["messages"]
    assert second[-2]["role"] == "assistant" and "ongeldig" in second[-1]["content"]
    stubborn = FakeServer([("nope", "stop"), ('{"verdict": 3}', "stop")])
    with pytest.raises(LLMError, match="no valid Answer"):
        LocalLLM(_local(), transport=httpx.MockTransport(stubborn)).generate(_request())


def test_output_limit_and_unreachable_server_raise_clear_errors():
    truncated = FakeServer([('{"verdict": "o', "length")])
    with pytest.raises(LLMTruncated, match="LOCAL_LLM_MAX_TOKENS"):
        LocalLLM(_local(), transport=httpx.MockTransport(truncated)).generate(_request())

    def down(request):
        raise httpx.ConnectError("connection refused")

    with pytest.raises(LLMError, match="not reachable"):
        LocalLLM(_local(), transport=httpx.MockTransport(down)).generate(_request())


def test_a_model_stuck_in_a_loop_gets_one_retry_and_then_a_specific_error():
    loop = '{"items": [' + ",".join(f'"{i}. Een JSON-object met de betekenis van de sectie"' for i in range(200))
    healed = FakeServer([(loop, "length"), ('{"verdict": "ok", "score": 3}', "stop")])
    assert LocalLLM(_local(), transport=httpx.MockTransport(healed)).generate(_request()).score == 3
    retry = healed.chats()[1]
    assert "herhalen" in retry["messages"][1]["content"] and retry["options"]["seed"] != healed.chats()[0]["options"]["seed"]
    stuck = FakeServer([(loop, "length"), (loop, "length")])
    with pytest.raises(LLMTruncated, match="stuck repeating"):
        LocalLLM(_local(), transport=httpx.MockTransport(stuck)).generate(_request())


def test_a_prompt_that_does_not_fit_is_refused_before_it_is_silently_truncated():
    server = FakeServer()
    llm = LocalLLM(_local(local_llm_context=4096), transport=httpx.MockTransport(server))
    with pytest.raises(LLMError, match=r"LOCAL_LLM_CONTEXT=\d+"):
        llm.generate(_request(user="woord " * 4000))
    assert server.requests == []  # nothing was sent


def test_release_unloads_and_check_finds_missing_models():
    server = FakeServer(models=["qwen3:32b"])
    llm = LocalLLM(_local(), transport=httpx.MockTransport(server))
    ok, detail = llm.check()
    assert not ok and "ollama pull gemma3:27b" in detail
    llm.release()
    assert ("POST", "/api/generate", {"model": "gemma3:27b", "keep_alive": 0}) in server.requests
    ok, _ = LocalLLM(_local(), transport=httpx.MockTransport(FakeServer())).check()
    assert ok


def test_settings_and_backend_selection(monkeypatch):
    for key in ("STUDIEPODCAST_LLM_BACKEND", "LOCAL_LLM_URL", "LOCAL_LLM_MODEL", "LOCAL_LLM_VISION", "LOCAL_LLM_CONTEXT"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("STUDIEPODCAST_OFFLINE", "1")
    monkeypatch.setenv("LOCAL_LLM_VISION", "0")
    monkeypatch.setenv("LOCAL_LLM_CONTEXT", "65536")
    s = Settings.from_env(dotenv=None)
    assert (s.offline, s.llm_backend, s.local_llm_vision, s.local_llm_context) == (True, "local", False, 65536)
    assert isinstance(make_llm(s), LocalLLM)
    with pytest.raises(OfflineViolation):
        make_llm(s.with_(llm_backend="anthropic"))
    with pytest.raises(OfflineViolation):
        AnthropicLLM(s, client=object())
    with pytest.raises(ValueError):
        make_llm(Settings(llm_backend="openrouter"))


def test_offline_only_accepts_local_addresses(monkeypatch):
    for url in ("http://localhost:11434", "http://127.0.0.1:8080", "http://192.168.1.20:11434", "http://10.0.0.5",
                "http://[::1]:11434"):
        assert check_local_url(url)
    with pytest.raises(OfflineViolation):
        check_local_url("http://8.8.8.8:11434")
    real = socket.getaddrinfo

    def fake_dns(host, *args, **kwargs):
        table = {"gpu-box.lan": "192.168.1.30", "llm.example.com": "93.184.216.34"}
        if host in table:
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (table[host], 0))]
        return real(host, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", fake_dns)
    assert check_local_url("http://gpu-box.lan:11434") == ["192.168.1.30"]
    with pytest.raises(OfflineViolation, match="not on the local network"):
        LocalLLM(Settings(offline=True, llm_backend="local", local_llm_url="http://llm.example.com"))


def test_offline_blocks_elevenlabs_and_switches_hugging_face_to_cache_only(monkeypatch):
    from pipeline.audio.synth import ElevenLabsDialogue

    for key in HF_OFFLINE_ENV:
        monkeypatch.delenv(key, raising=False)
    offline = Settings(offline=True, elevenlabs_api_key="secret")
    with pytest.raises(OfflineViolation):
        ElevenLabsDialogue(offline)
    apply_env(Settings())
    assert "HF_HUB_OFFLINE" not in __import__("os").environ
    apply_env(offline)
    import os

    assert all(os.environ[k] == v for k, v in HF_OFFLINE_ENV.items())


def test_offline_web_app_serves_no_cdn_docs(tmp_path, cast_dir, monkeypatch):
    from fastapi.testclient import TestClient

    from app.api import create_app

    for key in HF_OFFLINE_ENV:
        monkeypatch.delenv(key, raising=False)
    settings = Settings(data_dir=tmp_path / "data", cast_dir=cast_dir, offline=True, llm_backend="local")
    client = TestClient(create_app(settings, fake_llm=True, fake_audio=True))
    assert client.get("/docs").status_code == 404
    assert client.get("/api/health").json()["offline"] is True


def test_whole_script_chain_runs_on_the_local_backend(tmp_path, cast_dir, sample_pdf, monkeypatch):
    """Ingest (with a figure caption), plan, lexicon, script, audit, continuity and render, over the wire format."""
    for key in HF_OFFLINE_ENV:
        monkeypatch.delenv(key, raising=False)
    settings = _local(Settings(data_dir=tmp_path / "data", cast_dir=cast_dir, target_minutes=25))
    server = FakeServer()
    p = Pipeline(settings, llm=LocalLLM(settings, transport=httpx.MockTransport(server)), fake_audio=True)
    p.ingest(sample_pdf, "demo")
    out = p.run_chapter("demo", "ch01", upto="render")
    assert out["render"].is_file() and out["script"].segments
    titles = {b["format"]["title"] for b in server.chats()}
    assert {"PlanOut", "LexiconOut", "SegmentOut", "SupportOut", "ContinuityOut"} <= titles
    assert all(b["options"]["num_ctx"] == settings.local_llm_context for b in server.chats())
    unload = [i for i, (_, path, _) in enumerate(server.requests) if path == "/api/generate"]
    assert unload and unload[-1] > max(i for i, (_, path, _) in enumerate(server.requests) if path == "/api/chat")


def test_without_vision_figures_are_not_sent_to_the_model(tmp_path, cast_dir, sample_pdf, monkeypatch):
    import pipeline.runner as runner
    from pipeline.ingest.figures import LLMCaptioner

    captioners = []
    original = runner.ingest_book
    monkeypatch.setattr(runner, "ingest_book", lambda *a, **kw: (captioners.append(kw["captioner"]), original(*a, **kw))[1])
    for vision in (True, False):
        settings = _local(Settings(data_dir=tmp_path / f"data{vision}", cast_dir=cast_dir), local_llm_vision=vision)
        events = []
        p = Pipeline(settings, llm=LocalLLM(settings, transport=httpx.MockTransport(FakeServer())),
                     on_event=lambda s, st, d, ev=events: ev.append((s, st, d)))
        p.ingest(sample_pdf, "demo")
        skipped = any(st == "note" and "captions skipped" in d.get("message", "") for _, st, d in events)
        assert skipped is (not vision)
    assert isinstance(captioners[0], LLMCaptioner) and captioners[1] is None
