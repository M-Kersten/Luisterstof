"""The OpenCode backend: the model runs at home, the pipeline only hands over a prompt."""

import base64
import json
import re
import sys
import threading
import time
import types
from pathlib import Path

import httpx
import pytest
from pydantic import BaseModel

from pipeline import fake_handlers as fh
from pipeline import opencode_llm
from pipeline.cancel import Cancelled
from pipeline.config import Settings
from pipeline.llm import LLMError, LLMRequest, LLMTruncated, LLMUnavailable, make_llm, text_block
from pipeline.offline import OfflineViolation
from pipeline.opencode_llm import FALLBACK_URL, TRANSIENT_RETRIES, OpenCodeLLM, discover, split_model
from pipeline.runner import Pipeline

GENERATE = "/api/experimental/generate/text"


@pytest.fixture(autouse=True)
def _no_backoff_sleep(monkeypatch):
    """The retry ladder doubles from RETRY_WAIT_S; tests take the waits out."""
    monkeypatch.setattr(opencode_llm, "RETRY_WAIT_S", 0.0)
HANDLERS = {"LexiconOut": fh.fake_lexicon, "SegmentOut": fh.fake_script_segment,
            "PerformanceSegmentOut": fh.fake_script_scene, "SupportOut": fh.fake_support,
            "ContinuityOut": fh.fake_continuity, "SectionPlanOut": fh.fake_plan_section,
            "QuoteFixOut": fh.fake_plan_quotes, "PlanMergeOut": fh.fake_plan_merge,
            "PlanReviewOut": fh.fake_plan_review, "LintOut": fh.fake_lint, "StructureGuess": fh.fake_structure}


class Answer(BaseModel):
    verdict: str
    score: int


def _settings(**overrides) -> Settings:
    base = dict(llm_backend="opencode", opencode_url="http://127.0.0.1:4096", opencode_password="geheim")
    return Settings(**{**base, **overrides})


def _request(**kw) -> LLMRequest:
    return LLMRequest(task=kw.pop("task", "support"), system=kw.pop("system", "Je bent een beoordelaar."),
                      user=kw.pop("user", "Beoordeel dit."), schema=kw.pop("schema", Answer), **kw)


def schema_in(prompt: str) -> dict:
    """The JSON schema the backend wrote into the prompt, between the instruction and the user text."""
    return split_prompt(prompt)[2]


def split_prompt(prompt: str) -> tuple[str, str, dict]:
    """Back into (system, user, schema): the backend joins them with the Antwoordformaat instruction."""
    marker = prompt.index("## Antwoordformaat")
    start = prompt.index("{", marker)
    schema, end = json.JSONDecoder().raw_decode(prompt, start)
    return prompt[:marker].rstrip(), prompt[end:].strip(), schema


class FakeServer:
    """Stands in for the OpenCode server's HTTP API; records every request it receives."""

    def __init__(self, replies=(), models=("strata/strata",), status=None):
        self.replies = list(replies)
        self.models = list(models)
        self.status = status  # answer this status for every call instead of a real reply
        self.requests: list[tuple[str, str, dict, dict]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        self.requests.append((request.method, request.url.path, dict(request.headers), body))
        if self.status:
            return httpx.Response(self.status, json={"error": "nee"})
        if request.url.path == "/api/model":
            return httpx.Response(200, json={"data": [
                {"id": m, "providerID": m.split("/")[0], "modelID": m.split("/")[-1]} for m in self.models]})
        if request.url.path != GENERATE:
            return httpx.Response(404, json={"error": f"no such endpoint: {request.url.path}"})
        text = self.replies.pop(0) if self.replies else self._canned(body["prompt"])
        if isinstance(text, tuple):  # (status, body): the server stumbling instead of answering
            return httpx.Response(text[0], json=text[1])
        return httpx.Response(200, json={"data": {"text": text}})

    @staticmethod
    def _canned(prompt: str) -> str:
        system, user, schema = split_prompt(prompt)
        segment = re.search(r"\(type (\w+)\)", user)
        request = LLMRequest(task=schema["title"], system=system, user=user, schema=None,
                             metadata={"segment": segment.group(1)} if segment else {})
        result = HANDLERS[schema["title"]](request)
        return json.dumps(result.model_dump() if hasattr(result, "model_dump") else result, ensure_ascii=False)

    def prompts(self) -> list[str]:
        return [b["prompt"] for _, path, _, b in self.requests if path == GENERATE]

    def bodies(self) -> list[dict]:
        return [b for _, path, _, b in self.requests if path == GENERATE]


def test_one_prompt_and_a_model_reference_is_all_the_endpoint_gets():
    server = FakeServer(['```json\n{"verdict": "ok", "score": 4}\n```'])
    llm = OpenCodeLLM(_settings(), transport=httpx.MockTransport(server))
    result = llm.generate(_request(system=[text_block("Je bent een beoordelaar.")],
                                   user=[text_block("Beoordeel dit.")]))
    assert result == Answer(verdict="ok", score=4)

    method, path, headers, body = server.requests[0]
    assert (method, path) == ("POST", GENERATE)
    assert set(body) == {"prompt", "model"} and body["model"] == {"providerID": "strata", "id": "strata"}
    assert headers["authorization"] == "Basic " + base64.b64encode(b"opencode:geheim").decode()
    prompt = body["prompt"]
    assert prompt.startswith("Je bent een beoordelaar.") and prompt.rstrip().endswith("Beoordeel dit.")
    assert schema_in(prompt)["title"] == "Answer" and '"score"' in prompt
    assert "cache_control" not in prompt  # Anthropic-only keys are dropped on the way in
    assert llm.usage.calls == 1 and llm.usage.input_tokens > 0  # the endpoint reports no usage, so we estimate


def test_a_variant_and_a_bare_model_name_reach_the_server_as_openencode_wants_them():
    server = FakeServer(['{"verdict": "ok", "score": 1}'])
    llm = OpenCodeLLM(_settings(opencode_model="strata/strata#high"), transport=httpx.MockTransport(server))
    assert llm.model_ref == "strata/strata#high"
    assert llm.generate(_request()).score == 1
    assert server.bodies()[0]["model"] == {"providerID": "strata", "id": "strata", "variant": "high"}

    assert split_model("strata/strata") == ("strata", "strata", None)
    assert split_model("ollama/qwen3:32b#thinking") == ("ollama", "qwen3:32b", "thinking")
    assert split_model("gemma") == ("gemma", "gemma", None)  # the local single-model provider spells it like that


def test_the_prompt_is_plain_text_and_a_stray_image_block_is_never_sent():
    server = FakeServer(['{"verdict": "ok", "score": 4}'])
    llm = OpenCodeLLM(_settings(), transport=httpx.MockTransport(server))
    llm.generate(_request(user=[text_block("Beschrijf."), {"type": "image", "source": {"data": "aW1n"}}]))
    prompt = server.prompts()[0]
    assert prompt.rstrip().endswith("Beschrijf.") and "aW1n" not in prompt


def test_an_invalid_answer_comes_back_once_with_the_error_attached():
    server = FakeServer(["{not json", '{"verdict": "ok", "score": 1}'])
    assert OpenCodeLLM(_settings(), transport=httpx.MockTransport(server)).generate(_request()).score == 1
    first, second = server.prompts()
    assert "ongeldig" in second and "{not json" in second and second.startswith(first)

    stubborn = FakeServer(["nope", '{"verdict": 3}'])
    with pytest.raises(LLMError, match="no valid Answer"):
        OpenCodeLLM(_settings(), transport=httpx.MockTransport(stubborn)).generate(_request())
    assert len(stubborn.prompts()) == 2


def test_a_model_stuck_in_a_loop_gets_one_nudge_and_then_says_so():
    loop = '{"items": [' + ",".join(f'"{i}. Een JSON-object met de betekenis van de sectie"' for i in range(200))
    healed = FakeServer([loop, '{"verdict": "ok", "score": 3}'])
    assert OpenCodeLLM(_settings(), transport=httpx.MockTransport(healed)).generate(_request()).score == 3
    assert "herhalen" in healed.prompts()[1]

    stuck = FakeServer([loop, loop])
    with pytest.raises(LLMTruncated, match="stuck repeating"):
        OpenCodeLLM(_settings(), transport=httpx.MockTransport(stuck)).generate(_request())
    assert len(stuck.prompts()) == 2  # one retry, no hammering a model that is clearly looping


def test_the_three_ways_this_backend_breaks_are_named_specifically():
    with pytest.raises(LLMError, match=r"rejected the password \(401\).*OPENCODE_PASSWORD"):
        OpenCodeLLM(_settings(), transport=httpx.MockTransport(FakeServer(status=401))).generate(_request())
    with pytest.raises(LLMError, match="does not know model strata/strata"):
        OpenCodeLLM(_settings(), transport=httpx.MockTransport(FakeServer(status=404))).generate(_request())

    def down(request):
        raise httpx.ConnectError("connection refused")

    with pytest.raises(LLMError, match=r"not reachable.*Is the OpenCode app running"):
        OpenCodeLLM(_settings(), transport=httpx.MockTransport(down)).generate(_request())


DROPPED_STREAM = (502, {"_tag": "ProviderRequestError", "providerID": "strata", "reason": "provider",
                        "message": "OpenAI Chat stream ended without finish_reason", "status": 200})


def test_a_provider_that_drops_a_stream_is_retried_and_the_job_never_notices():
    """The real failure this ladder was written for: the local runner ends the stream without
    finish_reason, OpenCode wraps it in a 502, and one answer of a many-hour run must not die."""
    flaky = FakeServer([DROPPED_STREAM, DROPPED_STREAM, '{"verdict": "ok", "score": 4}'])
    llm = OpenCodeLLM(_settings(), transport=httpx.MockTransport(flaky))
    assert llm.generate(_request()).score == 4
    assert len(flaky.bodies()) == 3  # the two dropped streams were retried on the spot
    assert all(b["prompt"] == flaky.bodies()[0]["prompt"] for b in flaky.bodies())  # same question again


def test_a_provider_that_stumbles_under_a_different_name_is_still_retried():
    flaky = FakeServer([(503, {"error": "Server Error"}), '{"verdict": "ok", "score": 2}'])
    assert OpenCodeLLM(_settings(), transport=httpx.MockTransport(flaky)).generate(_request()).score == 2


def test_a_provider_that_keeps_stumbling_gives_up_with_its_original_error():
    down = FakeServer([DROPPED_STREAM] * 99)
    with pytest.raises(LLMError, match=r"answered 502.*stream ended without finish_reason.*also after 5 retries"):
        OpenCodeLLM(_settings(), transport=httpx.MockTransport(down)).generate(_request())
    assert len(down.bodies()) == TRANSIENT_RETRIES + 1


def test_a_stop_during_the_retry_wait_is_a_stop_not_a_retry():
    calls: list = []

    def stumbling(request):
        calls.append(request)
        llm.abort()  # the user pressed Stop while the call was in flight
        return httpx.Response(DROPPED_STREAM[0], json=DROPPED_STREAM[1])

    llm = OpenCodeLLM(_settings(), transport=httpx.MockTransport(stumbling))
    with pytest.raises(Cancelled):
        llm.generate(_request())
    assert len(calls) == 1  # aborted before the ladder could try again


def test_a_wrong_password_is_never_retried():
    stubborn = FakeServer(status=401)
    with pytest.raises(LLMError, match="rejected the password"):
        OpenCodeLLM(_settings(), transport=httpx.MockTransport(stubborn)).generate(_request())
    assert len(stubborn.requests) == 1  # the answer is never the model's fault here


def test_abort_ends_the_call_in_flight_and_refuses_the_next_one():
    holder: dict[str, OpenCodeLLM] = {}
    sent: list[str] = []

    def answering(request):
        sent.append(str(request.url))
        holder["llm"].abort()  # the job was cancelled while the model was still busy
        return httpx.Response(200, json={"data": {"text": '{"verdict": "ok", "score": 1}'}})

    llm = OpenCodeLLM(_settings(), transport=httpx.MockTransport(answering))
    holder["llm"] = llm
    with pytest.raises(Cancelled):
        llm.generate(_request())
    with pytest.raises(Cancelled):
        llm.generate(_request())
    assert len(sent) == 1  # the second call never reached the server


def test_stop_does_not_wait_for_a_long_answer():
    release = threading.Event()

    def thinking_for_minutes(request):
        release.wait(30)  # a long chapter: the server is still writing
        return httpx.Response(200, json={"data": {"text": '{"verdict": "ok", "score": 1}'}})

    llm = OpenCodeLLM(_settings(), transport=httpx.MockTransport(thinking_for_minutes))
    threading.Timer(0.3, llm.abort).start()
    started = time.monotonic()
    try:
        with pytest.raises(Cancelled):
            llm.generate(_request())
        assert time.monotonic() - started < 2.0  # the job ends now, not when the answer arrives
    finally:
        release.set()  # let the abandoned call finish


def test_check_says_whether_the_server_and_the_model_are_there():
    ok, detail = OpenCodeLLM(_settings(), transport=httpx.MockTransport(FakeServer())).check()
    assert ok and "strata/strata listed" in detail

    ok, detail = OpenCodeLLM(_settings(opencode_model="strata/lokaal-8b"),
                             transport=httpx.MockTransport(FakeServer(models=["strata/strata", "ollama/qwen3"]))).check()
    assert not ok and "strata/lokaal-8b is not listed" in detail and "ollama/qwen3" in detail

    ok, detail = OpenCodeLLM(_settings(), transport=httpx.MockTransport(FakeServer(models=[]))).check()
    assert not ok and "lists no models" in detail

    def down(request):
        raise httpx.ConnectError("connection refused")

    ok, detail = OpenCodeLLM(_settings(), transport=httpx.MockTransport(down)).check()
    assert not ok and "not reachable at http://127.0.0.1:4096" in detail


def test_discovery_walks_env_then_service_file_then_the_desktop_logs(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    service = home / ".config" / "opencode" / "service.json"
    service.parent.mkdir(parents=True)
    service.write_text(json.dumps({"url": "http://127.0.0.1:28894", "password": "uit-bestand"}), encoding="utf-8")
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv("OPENCODE_URL", raising=False)

    appdata = tmp_path / "appdata"
    log = appdata / "ai.opencode.desktop" / "logs" / "2026-11-01" / "main.log"
    log.parent.mkdir(parents=True)
    log.write_text("info: server listening url: 'http://127.0.0.1:28894'\n", encoding="utf-8")
    monkeypatch.setenv("APPDATA", str(appdata))

    assert discover(Settings(llm_backend="opencode")) == ("http://127.0.0.1:28894", "uit-bestand")
    assert discover(Settings(llm_backend="opencode", opencode_url="http://127.0.0.1:5555/")) \
        == ("http://127.0.0.1:5555", "uit-bestand")  # OPENCODE_URL wins, the password still comes from the file

    service.unlink()
    assert discover(Settings(llm_backend="opencode")) == ("http://127.0.0.1:28894", None)  # from the desktop log
    monkeypatch.delenv("APPDATA")
    assert discover(Settings(llm_backend="opencode")) == (FALLBACK_URL, None)  # what `opencode serve` listens on


def test_settings_pick_the_backend_and_offline_keeps_the_server_on_this_machine(monkeypatch):
    for key in ("OPENCODE_URL", "OPENCODE_PASSWORD", "OPENCODE_MODEL", "OPENCODE_TIMEOUT",
                "STUDIEPODCAST_LLM_BACKEND"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("STUDIEPODCAST_LLM_BACKEND", "opencode")
    monkeypatch.setenv("OPENCODE_URL", "http://127.0.0.1:28894")
    monkeypatch.setenv("OPENCODE_MODEL", "strata/strata#high")
    monkeypatch.setenv("OPENCODE_TIMEOUT", "600")
    s = Settings.from_env(dotenv=None)
    assert (s.llm_backend, s.opencode_url, s.opencode_model, s.opencode_timeout_s) \
        == ("opencode", "http://127.0.0.1:28894", "strata/strata#high", 600.0)
    assert isinstance(make_llm(s), OpenCodeLLM)
    assert s.profile()["llm_model"] == "strata/strata#high"

    with pytest.raises(OfflineViolation, match="OPENCODE_URL"):
        OpenCodeLLM(s.with_(offline=True, opencode_url="http://8.8.8.8:4096"))  # offline means offline
    OpenCodeLLM(s.with_(offline=True))  # 127.0.0.1 is fine: nothing leaves this machine
    with pytest.raises(ValueError):
        make_llm(Settings(llm_backend="openrouter"))


def test_whole_script_chain_runs_over_the_opencode_endpoint(tmp_path, cast_dir, sample_pdf):
    """Plan, lexicon, script, audit and continuity through one text endpoint, with captions skipped."""
    settings = Settings(data_dir=tmp_path / "data", cast_dir=cast_dir, target_minutes=25, llm_backend="opencode",
                        opencode_url="http://127.0.0.1:4096", opencode_password="geheim")
    server = FakeServer()
    p = Pipeline(settings, llm=OpenCodeLLM(settings, transport=httpx.MockTransport(server)), fake_audio=True)
    p.ingest(sample_pdf, "demo")
    out = p.run_chapter("demo", "ch01", upto="render")
    assert out["render"].is_file() and out["script"].segments

    titles = {schema_in(prompt)["title"] for prompt in server.prompts()}
    assert {"SectionPlanOut", "QuoteFixOut", "PlanMergeOut", "PlanReviewOut", "LexiconOut", "SegmentOut",
            "SupportOut", "ContinuityOut"} <= titles
    assert all(set(body) == {"prompt", "model"} for body in server.bodies())
    assert all(body["model"] == {"providerID": "strata", "id": "strata"} for body in server.bodies())


def test_the_doctor_reports_the_opencode_server(capsys):
    from pipeline.cli import _doctor_llm

    _doctor_llm(_settings(opencode_url="http://127.0.0.1:1"))  # nothing listens there
    printed = capsys.readouterr().out
    assert "OpenCode (http://127.0.0.1:1)" in printed and "strata/strata" in printed
    assert "not reachable" in printed


# ---------------------------------------------------------------------------
# the Chatterbox checkpoint (V3 is newer than the chatterbox-tts on PyPI knows)
# ---------------------------------------------------------------------------

def test_the_chatterbox_checkpoint_files_are_named():
    from pipeline.audio.synth import T3_CHECKPOINT, chatterbox_files, chatterbox_required

    assert T3_CHECKPOINT == "t3_mtl23ls_v3.safetensors"
    files = chatterbox_files()
    assert files[1] == "t3_mtl23ls_v3.safetensors"
    assert {"ve.pt", "s3gen.pt", "grapheme_mtl_merged_expanded_v1.json", "conds.pt"} <= set(files)
    assert "t3_mtl23ls_v2.safetensors" not in files  # only the T3 weights differ between the two
    # the doctor only calls the load complete when these are there
    assert chatterbox_required() == ["ve.pt", "t3_mtl23ls_v3.safetensors", "s3gen.pt",
                                     "grapheme_mtl_merged_expanded_v1.json"]


def test_the_checkpoint_is_part_of_the_render_cache_key(tmp_path):
    """Switching to V3 has to re-render: same text and seed, different weights, different audio."""
    from pipeline.audio.cache import RenderCache
    from pipeline.audio.synth import CACHE_TAG, ChatterboxSynth, NullSynth, VoiceSpec
    from pipeline.audio.takes import render_with_takes

    assert CACHE_TAG == "t3=t3_mtl23ls_v3.safetensors"
    assert ChatterboxSynth.cache_tag == CACHE_TAG  # every real chatterbox render carries it

    class Tagged(NullSynth):
        def __init__(self, tag):
            super().__init__()
            self.name = "chatterbox"
            self.cache_tag = tag

    text = "Een kans is een getal tussen nul en een dat uitdrukt hoe waarschijnlijk een gebeurtenis is."
    cache = RenderCache(tmp_path / "cache")
    voice = VoiceSpec(speaker_id="tessa")

    def run(tag):
        return render_with_takes(text, voice, Tagged(tag), cache, transcriber=None, exaggeration=0.5,
                                 n_takes=1, seed_base=9000)

    v2 = run("t3=t3_mtl23ls_v2.safetensors")
    v3 = run("t3=t3_mtl23ls_v3.safetensors")
    v2_again = run("t3=t3_mtl23ls_v2.safetensors")

    assert not v3.from_cache and v2.take_key != v3.take_key
    assert v2_again.from_cache and v2_again.take_key == v2.take_key
    assert cache.get(v3.take_key)[1]["model_tag"] == "t3=t3_mtl23ls_v3.safetensors"


def test_v3_loads_through_our_own_loader_because_the_pypi_package_only_knows_v2(tmp_path, monkeypatch):
    """chatterbox-tts 0.1.7 hardcodes V2, so from_local is mirrored in the repo to reach the V3 weights."""
    from pipeline.audio import synth

    calls: list[dict] = []

    @classmethod
    def from_pretrained(cls, device):  # exactly the 0.1.7 signature: no t3_model
        calls.append({"from_pretrained": device})
        return "package-model"

    package = types.ModuleType("chatterbox")
    mtl = types.ModuleType("chatterbox.mtl_tts")
    mtl.ChatterboxMultilingualTTS = type("ChatterboxMultilingualTTS", (), {"from_pretrained": from_pretrained})
    mtl.REPO_ID = "ResembleAI/chatterbox"
    mtl.T3 = mtl.T3Config = mtl.S3Gen = mtl.MTLTokenizer = mtl.VoiceEncoder = mtl.Conditionals = object
    mtl.load_safetensors = lambda path: {}
    mtl.snapshot_download = lambda **kw: str(tmp_path)
    package.mtl_tts = mtl
    monkeypatch.setitem(sys.modules, "chatterbox", package)
    monkeypatch.setitem(sys.modules, "chatterbox.mtl_tts", mtl)

    monkeypatch.setattr(synth, "_multilingual_from_local",
                        lambda ckpt_dir, device: calls.append({"local": (str(ckpt_dir), device)}) or "v3-model")
    monkeypatch.setenv("CHATTERBOX_CKPT_DIR", str(tmp_path))
    assert synth.load_multilingual_model("cuda") == "v3-model"
    assert calls[-1]["local"] == (str(Path(tmp_path)), "cuda")

    def broken(ckpt_dir, device):
        raise RuntimeError("shape mismatch")

    monkeypatch.setattr(synth, "_multilingual_from_local", broken)
    with pytest.raises(RuntimeError, match="t3_mtl23ls_v3.safetensors"):
        synth.load_multilingual_model("cuda")  # no silent fallback: one episode, one voice

    # a chatterbox that does accept t3_model may load V3 itself
    @classmethod
    def from_pretrained_v3(cls, device, t3_model=None):
        calls.append({"from_pretrained": (device, t3_model)})
        return "new-package-model"

    mtl.ChatterboxMultilingualTTS = type("ChatterboxMultilingualTTS", (), {"from_pretrained": from_pretrained_v3})
    assert synth.load_multilingual_model("cuda") == "new-package-model"
    assert calls[-1]["from_pretrained"] == ("cuda", synth.T3_CHECKPOINT)


def test_v3_text_vocabulary_is_read_from_the_checkpoint(tmp_path, monkeypatch):
    """V3 has 2454 text tokens, the 0.1.7 package builds 2352: the loader sizes T3 from the weights."""
    from pipeline.audio import synth

    built: dict = {}

    class Cfg:
        text_tokens_dict_size = 2352

        @classmethod
        def multilingual(cls):
            return cls()

    class FakeT3:
        def __init__(self, config):
            built["vocab"] = config.text_tokens_dict_size

        def load_state_dict(self, state):
            assert state["text_emb.weight"].shape[0] == built["vocab"]

        def to(self, device):
            return self

        def eval(self):
            return self

    class Weight:
        shape = (2454, 1024)

    class Plain:
        def __init__(self, *a, **kw):
            pass

        def load_state_dict(self, *a, **kw):
            pass

        def to(self, *a, **kw):
            return self

        def eval(self):
            return self

    torch = types.ModuleType("torch")
    torch.device = lambda name: name
    torch.load = lambda *a, **kw: {}
    mtl = types.ModuleType("chatterbox.mtl_tts")
    mtl.T3, mtl.T3Config = FakeT3, Cfg
    mtl.VoiceEncoder = mtl.S3Gen = Plain
    mtl.MTLTokenizer = Plain
    mtl.Conditionals = Plain
    mtl.ChatterboxMultilingualTTS = lambda *a, **kw: "v3-model"
    mtl.load_safetensors = lambda path: {"text_emb.weight": Weight()}
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "chatterbox", types.ModuleType("chatterbox"))
    monkeypatch.setitem(sys.modules, "chatterbox.mtl_tts", mtl)

    assert synth._multilingual_from_local(tmp_path, "cuda") == "v3-model"
    assert built["vocab"] == 2454


STRATA_DOWN = (502, {"_tag": "ProviderRequestError", "providerID": "strata", "reason": "provider",
                     "message": "ECONNREFUSED: Unable to connect. Is the computer able to access the url?"})


class SleepyStrata:
    """OpenCode with a model server that only starts on a first prompt in a session, like the app does."""

    def __init__(self, v2=True, starts=True):
        self.v2, self.starts, self.up = v2, starts, False
        self.requests: list[tuple[str, str, dict]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        path = request.url.path
        self.requests.append((request.method, path, body))
        prefix = "/api/session" if self.v2 else "/session"
        if path == GENERATE:
            if not self.up:
                return httpx.Response(STRATA_DOWN[0], json=STRATA_DOWN[1])
            return httpx.Response(200, json={"data": {"text": '{"verdict": "ok", "score": 1}'}})
        if request.method == "POST" and path == prefix:
            return httpx.Response(200, json={"id": "ses_1", "title": body.get("title")})
        if request.method == "POST" and path in (f"{prefix}/ses_1/prompt", f"{prefix}/ses_1/prompt_async"):
            self.up = self.starts  # the first prompt is what starts the model server
            return httpx.Response(200 if self.v2 else 204)
        if request.method == "DELETE" and path == f"{prefix}/ses_1":
            return httpx.Response(204)
        return httpx.Response(404, json={"error": f"no such route: {request.method} {path}"})

    def calls(self) -> list[tuple[str, str]]:
        return [(m, p) for m, p, _ in self.requests if p != GENERATE]


@pytest.fixture
def no_wake_wait(monkeypatch):
    monkeypatch.setattr(opencode_llm, "WAKE_POLL_S", 0.0)


def test_a_model_that_is_not_running_is_started_like_the_app_does(no_wake_wait):
    server = SleepyStrata()
    llm = OpenCodeLLM(_settings(), transport=httpx.MockTransport(server))
    heard: list[str] = []
    llm.on_status = heard.append
    assert llm.generate(_request()) == Answer(verdict="ok", score=1)
    assert server.calls() == [("POST", "/api/session"), ("POST", "/api/session/ses_1/prompt"), ("DELETE", "/api/session/ses_1")]
    prompt = next(b for m, p, b in server.requests if p.endswith("/prompt"))
    assert prompt["model"] == {"providerID": "strata", "modelID": "strata"} and prompt["parts"][0]["type"] == "text"
    assert heard and "wordt gestart" in heard[0] and "antwoordt" in heard[-1]


def test_the_older_session_routes_work_too(no_wake_wait):
    server = SleepyStrata(v2=False)
    llm = OpenCodeLLM(_settings(), transport=httpx.MockTransport(server))
    assert llm.ping() >= 0
    assert ("POST", "/session/ses_1/prompt_async") in server.calls() and ("DELETE", "/session/ses_1") in server.calls()


def test_a_model_that_will_not_start_is_reported_with_where_it_should_be(no_wake_wait, tmp_path, monkeypatch):
    config = tmp_path / "opencode"
    config.mkdir()
    (config / "opencode.jsonc").write_text("""{
      // Strata, started by hand or by a plugin
      "provider": {"strata": {"npm": "@ai-sdk/openai-compatible", "options": {"baseURL": "http://127.0.0.1:8080/v1",},},},
    }""", encoding="utf-8")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    server = SleepyStrata(starts=False)
    llm = OpenCodeLLM(_settings(opencode_wake_timeout_s=0.0), transport=httpx.MockTransport(server))
    with pytest.raises(LLMUnavailable, match=r"provider 'strata' at http://127\.0\.0\.1:8080/v1 \(from opencode\.json\)"):
        llm.ping()
    with pytest.raises(LLMUnavailable, match="ECONNREFUSED"):
        llm.generate(_request())  # within the cooldown: no second wake, the ladder ends in the same clear error
    assert sum(1 for m, p in server.calls() if p == "/api/session" and m == "POST") == 1


def test_jsonc_comments_go_and_urls_stay():
    from pipeline.opencode_llm import _strip_jsonc

    text = '{"a": "http://x//y", /* weg */ "b": [1, 2,], // weg\n "c": "/* blijft */"}'
    assert json.loads(_strip_jsonc(text)) == {"a": "http://x//y", "b": [1, 2], "c": "/* blijft */"}
