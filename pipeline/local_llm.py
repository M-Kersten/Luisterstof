"""A language model on your own machine or network, over HTTP.

Two wire formats:

* ``ollama``: Ollama's native ``/api/chat``. Structured output goes in ``format``
  (a JSON schema, enforced with a grammar) and ``options.num_ctx`` sets the
  context window on every request. Ollama's OpenAI-compatible endpoint cannot
  set it and silently drops the start of any prompt longer than the server
  default, which these prompts (rules, cast, plan, a whole chapter) exceed.
* ``openai``: ``/v1/chat/completions`` with ``response_format: json_schema``,
  for llama.cpp's server, vLLM, LM Studio and similar. Their context window is
  fixed when the server starts; LOCAL_LLM_CONTEXT has to match it.

The schema is also written into the system prompt, so a server that ignores the
format constraint is still told what to produce. Every answer is validated, and
an invalid one goes back to the model once together with the error.
"""

from __future__ import annotations

import json
import logging
import re
import time
from types import SimpleNamespace
from typing import Any

import httpx
from pydantic import BaseModel

from pipeline.config import Settings
from pipeline.llm import LLMError, LLMRequest, LLMTruncated, Usage, _as_blocks, parse_json_into
from pipeline.offline import check_local_url, on_this_machine

log = logging.getLogger(__name__)

CHARS_PER_TOKEN = 3.2  # conservative for Dutch: overestimating fails loudly, underestimating truncates silently
TOKENS_PER_IMAGE = 1024
CONTEXT_MARGIN = 256
RETRIES = 1
WRITING_TASKS = frozenset({"script_segment", "script_scene"})
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
LOOP_NUDGE = ("\n\nLet op: een eerder antwoord op deze vraag bleef hetzelfde herhalen tot het werd afgekapt. "
              "Houd het beknopt, herhaal niets en sluit elke lijst zodra de inhoud op is.")

SCHEMA_INSTRUCTION = (
    "\n\n## Antwoordformaat\nAntwoord uitsluitend met één JSON-object dat voldoet aan dit JSON-schema. "
    "Geen uitleg, geen markdown, geen tekst ervoor of erna.\n"
)


def inline_schema(schema: dict) -> dict:
    """Resolve pydantic's ``$defs``/``$ref``: not every server's schema-to-grammar step follows references."""
    defs = schema.get("$defs", {})

    def resolve(node: Any, seen: frozenset[str]) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                name = node["$ref"].rsplit("/", 1)[-1]
                if name in seen:
                    raise ValueError(f"recursive schema through {name}")
                target = resolve(defs[name], seen | {name})
                return {**target, **{k: resolve(v, seen) for k, v in node.items() if k != "$ref"}}
            return {k: resolve(v, seen) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [resolve(x, seen) for x in node]
        return node

    return resolve(schema, frozenset())


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1


def _text_and_images(content) -> tuple[str, list[tuple[str, str]]]:
    texts: list[str] = []
    images: list[tuple[str, str]] = []
    for block in _as_blocks(content):
        if block.get("type") == "image":
            source = block.get("source", {})
            images.append((source.get("media_type", "image/png"), source.get("data", "")))
        elif block.get("type") == "text" and block.get("text"):
            texts.append(block["text"])  # cache_control and other Anthropic-only keys are dropped here
    return "\n\n".join(texts), images


def looping(text: str, window: int = 2500, repeats: int = 6) -> bool:
    """True when the tail of an answer is one fragment over and over (numbers ignored)."""
    tail = re.sub(r"\d+", "#", text[-window:])
    parts = [p.strip(' \t"[],{}') for p in re.split(r"[\n,]", tail)]
    parts = [p for p in parts if len(p) >= 25]  # short key/value pairs repeat in any healthy answer
    if len(parts) < repeats:
        return False
    return max(parts.count(p) for p in set(parts)) >= repeats


def clean_answer(text: str) -> str:
    """Drop the reasoning block some models (Qwen3, R1 distills) put before the answer."""
    return _THINK.sub("", text or "").strip()


def _with_nudge(content):
    """Append the anti-repetition note to the user text (a string, or the first part of a multimodal list)."""
    if isinstance(content, str):
        return content + LOOP_NUDGE
    first, *rest = content
    return [{**first, "text": first.get("text", "") + LOOP_NUDGE}, *rest]


_THINK_ON = {"on", "true", "1", "yes", "ja"}
_THINK_OFF = {"off", "false", "0", "no", "nee"}
_THINK_LEVELS = {"low", "medium", "high"}


def think_value(raw: str) -> bool | str | None:
    """LOCAL_LLM_THINK -> what to send: None = leave it to the model, True/False, or a level."""
    raw = (raw or "").strip().casefold()
    if not raw:
        return None
    if raw in _THINK_ON:
        return True
    if raw in _THINK_OFF:
        return False
    if raw in _THINK_LEVELS:
        return raw
    raise ValueError(f"LOCAL_LLM_THINK must be on, off, low, medium or high, not {raw!r}")


class LocalLLM:
    """Same contract as AnthropicLLM: a request with a pydantic schema in, a validated instance out."""

    name = "local"

    def __init__(self, settings: Settings, *, transport: httpx.BaseTransport | None = None):
        if settings.local_llm_api not in ("ollama", "openai"):
            raise ValueError(f"LOCAL_LLM_API must be 'ollama' or 'openai', not {settings.local_llm_api!r}")
        if settings.offline:
            check_local_url(settings.local_llm_url)
        self.settings = settings
        self.usage = Usage()
        self.api = settings.local_llm_api
        self.model = settings.local_llm_model
        self.base_url = settings.local_llm_url.rstrip("/")
        self.think = think_value(settings.local_llm_think)
        self._capabilities: set[str] | None | bool = False  # False = not looked up yet
        headers = {"Authorization": f"Bearer {settings.local_llm_api_key}"} if settings.local_llm_api_key else {}
        # trust_env=False: traffic for a local server never goes through an HTTP(S)_PROXY from the environment.
        self.client = httpx.Client(base_url=self.base_url, headers=headers, transport=transport, trust_env=False,
                                   timeout=httpx.Timeout(settings.local_llm_timeout_s, connect=10.0))

    # ------------------------------------------------------------------ model facts
    def capabilities(self) -> set[str] | None:
        """What Ollama says the model can do ("vision", "thinking", ...); None when unknown."""
        if self._capabilities is False:
            self._capabilities = None
            if self.api == "ollama":
                try:
                    response = self.client.post("/api/show", json={"model": self.model}, timeout=15.0)
                    data = response.json() if response.status_code == 200 else {}
                    if isinstance(data.get("capabilities"), list):
                        self._capabilities = set(data["capabilities"])
                except (httpx.HTTPError, ValueError):
                    pass
        return self._capabilities  # type: ignore[return-value]

    @property
    def supports_images(self) -> bool:
        if not self.settings.local_llm_vision:
            return False
        caps = self.capabilities()
        return caps is None or "vision" in caps

    # ------------------------------------------------------------------ request
    def _messages(self, request: LLMRequest, schema: dict) -> tuple[list[dict], int]:
        system_text, system_images = _text_and_images(request.system)
        user_text, images = _text_and_images(request.user)
        images = system_images + images
        if images and not self.supports_images:
            raise LLMError(f"task {request.task} sends an image, but LOCAL_LLM_VISION is off for {self.model}")
        system = system_text + SCHEMA_INSTRUCTION + json.dumps(schema, ensure_ascii=False)
        user: dict[str, Any] = {"role": "user", "content": user_text}
        if images and self.api == "ollama":
            user["images"] = [data for _, data in images]
        elif images:
            user["content"] = [{"type": "text", "text": user_text}] + [
                {"type": "image_url", "image_url": {"url": f"data:{media};base64,{data}"}} for media, data in images]
        return [{"role": "system", "content": system}, user], len(images)

    def _output_budget(self, request: LLMRequest, messages: list[dict], n_images: int) -> int:
        chars = sum(len(m["content"]) if isinstance(m["content"], str)
                    else sum(len(p.get("text", "")) for p in m["content"]) for m in messages)
        prompt = int(chars / CHARS_PER_TOKEN) + 1 + n_images * TOKENS_PER_IMAGE
        wanted = request.max_tokens or self.settings.local_llm_max_tokens
        room = self.settings.local_llm_context - prompt - CONTEXT_MARGIN
        if room < min(wanted, 1024):
            suggestion = 1 << (prompt + wanted + CONTEXT_MARGIN - 1).bit_length()
            server = " and start the server with at least that context" if self.api == "openai" else ""
            raise LLMError(
                f"task {request.task}: the prompt is about {prompt} tokens and the answer needs room too, but "
                f"LOCAL_LLM_CONTEXT is {self.settings.local_llm_context}. Set LOCAL_LLM_CONTEXT={suggestion}{server}. "
                "(Sending it anyway would cut off the start of the prompt without any error.)")
        return min(wanted, room)

    def _body(self, request: LLMRequest, messages: list[dict], schema: dict, max_tokens: int, attempt: int) -> dict:
        temperature = 0.8 if request.task in WRITING_TASKS else 0.2 + 0.2 * attempt
        if self.api == "ollama":
            body = {"model": self.model, "messages": messages, "stream": False, "format": schema, "keep_alive": "10m",
                    "options": {"num_ctx": self.settings.local_llm_context, "num_predict": max_tokens,
                                "temperature": temperature, "seed": 1000 + attempt}}
            if self.think is not None:
                body["think"] = self.think
            return body
        body = {"model": self.model, "messages": messages, "stream": False, "max_tokens": max_tokens,
                "temperature": temperature, "seed": 1000 + attempt,
                "response_format": {"type": "json_schema",
                                    "json_schema": {"name": request.schema.__name__, "schema": schema}}}
        if isinstance(self.think, bool):
            body["chat_template_kwargs"] = {"enable_thinking": self.think}  # llama.cpp, vLLM (Qwen-style templates)
        elif self.think:
            body["reasoning_effort"] = self.think
        return body

    def _post(self, body: dict) -> tuple[str, str | None, SimpleNamespace, int]:
        path = "/api/chat" if self.api == "ollama" else "/v1/chat/completions"
        try:
            response = self.client.post(path, json=body)
        except httpx.HTTPError as exc:
            raise LLMError(f"local LLM at {self.base_url} not reachable ({type(exc).__name__}: {exc}). "
                           "Is the server running? `studiepodcast doctor` checks it.") from exc
        if response.status_code >= 400:
            if "does not support thinking" in response.text:
                raise LLMError(f"{self.model} can't reason step by step, but LOCAL_LLM_THINK={self.settings.local_llm_think} "
                               "asks for it: unset LOCAL_LLM_THINK or set it to off")
            raise LLMError(f"local LLM at {self.base_url} answered {response.status_code}: {response.text[:500]}")
        data = response.json()
        if self.api == "ollama":
            message = data.get("message") or {}
            usage = SimpleNamespace(input_tokens=data.get("prompt_eval_count", 0), output_tokens=data.get("eval_count", 0))
            return message.get("content") or "", data.get("done_reason"), usage, len(message.get("thinking") or "")
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        u = data.get("usage") or {}
        usage = SimpleNamespace(input_tokens=u.get("prompt_tokens", 0), output_tokens=u.get("completion_tokens", 0))
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        return message.get("content") or "", choice.get("finish_reason"), usage, len(reasoning)

    def generate(self, request: LLMRequest) -> BaseModel:
        schema = inline_schema(request.schema.model_json_schema())
        messages, n_images = self._messages(request, schema)
        error: LLMError | None = None
        for attempt in range(RETRIES + 1):
            max_tokens = self._output_budget(request, messages, n_images)
            started = time.monotonic()
            log.info("llm %s task=%s model=%s attempt=%d", self.name, request.task, self.model, attempt + 1)
            content, finish, usage, thought = self._post(self._body(request, messages, schema, max_tokens, attempt))
            self.usage.add(usage, time.monotonic() - started)
            if finish == "length":
                if thought and not clean_answer(content):
                    raise LLMTruncated(
                        f"task {request.task}: {self.model} spent the whole {max_tokens}-token budget reasoning before "
                        "it wrote an answer. Raise LOCAL_LLM_MAX_TOKENS (and LOCAL_LLM_CONTEXT to make room), or set "
                        "LOCAL_LLM_THINK=off for faster, shorter answers.")
                stuck = looping(content)
                if stuck and attempt < RETRIES:
                    log.warning("local model repeated itself on %s until the limit; retrying once", request.task)
                    messages = [messages[0], {**messages[1], "content": _with_nudge(messages[1]["content"])}]
                    continue
                if stuck:
                    raise LLMTruncated(f"task {request.task}: {self.model} got stuck repeating itself until the "
                                       f"{max_tokens}-token limit, also on the retry. That is the model, not the limit: "
                                       "use a larger model (LOCAL_LLM_MODEL).")
                raise LLMTruncated(f"task {request.task} hit the output limit of {max_tokens} tokens; "
                                   "raise LOCAL_LLM_MAX_TOKENS (and LOCAL_LLM_CONTEXT if needed)")
            answer = clean_answer(content)
            try:
                return parse_json_into(answer, request.schema)
            except LLMError as exc:
                error = exc
                log.warning("local answer for %s did not match %s (attempt %d): %s",
                            request.task, request.schema.__name__, attempt + 1, str(exc)[:300])
                messages = messages + [
                    {"role": "assistant", "content": answer[:6000]},
                    {"role": "user", "content": f"Dat antwoord is ongeldig: {str(exc)[:800]}\n"
                                                "Geef opnieuw uitsluitend één geldig JSON-object volgens het schema."},
                ]
        raise LLMError(f"local model {self.model} gave no valid {request.schema.__name__} for {request.task}: {error}")

    # ------------------------------------------------------------------ housekeeping
    def release(self) -> None:
        """Free the GPU before rendering, when the model runs on this machine. A server on another laptop keeps
        its model loaded: it may be writing the next chapter for someone else, and reloading costs minutes."""
        if self.api != "ollama" or not on_this_machine(self.base_url):
            return
        try:
            self.client.post("/api/generate", json={"model": self.model, "keep_alive": 0}, timeout=30.0)
        except httpx.HTTPError as exc:
            log.info("could not unload %s: %s", self.model, exc)

    def check(self) -> tuple[bool, str]:
        """(usable, explanation) for `studiepodcast doctor`: server reachable and the model available."""
        try:
            if self.api == "ollama":
                response = self.client.get("/api/tags", timeout=10.0)
                response.raise_for_status()
                names = [m.get("name") or m.get("model") or "" for m in response.json().get("models", [])]
                wanted = self.model if ":" in self.model else f"{self.model}:latest"
                if wanted in names:
                    return True, "reachable, model present"
                return False, f"reachable, but {self.model} is not pulled: run `ollama pull {self.model}` (present: {', '.join(names) or 'none'})"
            response = self.client.get("/v1/models", timeout=10.0)
            response.raise_for_status()
            entries = response.json().get("data", [])
            ids = [e.get("id", "") for e in entries]
            if self.model not in ids:
                return True, (f"reachable; serves {', '.join(ids) or 'nothing listed'}, not {self.model} "
                              "(fine for llama.cpp, which ignores the name; vLLM and LM Studio need it to match)")
            served = next((e.get("max_model_len") for e in entries if e.get("id") == self.model), None)
            if served and served < self.settings.local_llm_context:
                return False, f"reachable, but the server context is {served}, below LOCAL_LLM_CONTEXT={self.settings.local_llm_context}"
            return True, "reachable, model present" + (f" (server context {served})" if served else "")
        except httpx.HTTPError as exc:
            return False, f"not reachable at {self.base_url} ({type(exc).__name__}: {exc})"
