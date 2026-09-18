"""LLM access for the script side of the pipeline.

Every stage that needs a model goes through ``LLM.generate`` with a pydantic
schema, so the stages never see raw text. Two implementations:

* ``AnthropicLLM`` calls the Claude API with structured outputs, adaptive
  thinking, prompt caching on the stable system prefix and server-side
  refusal fallbacks.
* ``FakeLLM`` answers from canned handlers and records every request. Tests
  and the CLI ``--fake-llm`` flag use it.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from pipeline.config import Settings

log = logging.getLogger(__name__)

SchemaT = TypeVar("SchemaT", bound=BaseModel)

ContentBlocks = str | list[dict[str, Any]]


class LLMError(RuntimeError):
    pass


class LLMRefusal(LLMError):
    pass


class LLMTruncated(LLMError):
    pass


@dataclass
class LLMRequest:
    task: str
    system: ContentBlocks
    user: ContentBlocks
    schema: type[BaseModel]
    max_tokens: int | None = None
    effort: str | None = None
    cache_system: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    seconds: float = 0.0

    def add(self, usage: Any, seconds: float) -> None:
        self.calls += 1
        self.seconds += seconds
        if usage is None:
            return
        self.input_tokens += getattr(usage, "input_tokens", 0) or 0
        self.output_tokens += getattr(usage, "output_tokens", 0) or 0
        self.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0
        self.cache_write_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "seconds": round(self.seconds, 1),
        }


class LLM(Protocol):
    name: str
    usage: Usage

    def generate(self, request: LLMRequest) -> BaseModel: ...


def text_block(text: str, cache: bool = False) -> dict[str, Any]:
    block: dict[str, Any] = {"type": "text", "text": text}
    if cache:
        block["cache_control"] = {"type": "ephemeral"}
    return block


def image_block(data_b64: str, media_type: str = "image/png") -> dict[str, Any]:
    return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data_b64}}


def _as_blocks(content: ContentBlocks) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [text_block(content)]
    return [dict(b) for b in content]


class AnthropicLLM:
    """Claude via the official SDK. Structured output is the only output mode."""

    name = "anthropic"

    FALLBACK_BETA = "server-side-fallback-2026-07-01"

    def __init__(self, settings: Settings, client: Any | None = None, *, fallbacks: bool | None = None):
        self.settings = settings
        self.usage = Usage()
        if client is None:
            import anthropic

            client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.client = client
        self.model = settings.llm_model
        if fallbacks is None:
            fallbacks = self.model.startswith(("claude-opus-5", "claude-fable", "claude-mythos"))
        self.fallbacks = fallbacks

    def _system_blocks(self, request: LLMRequest) -> list[dict[str, Any]]:
        blocks = _as_blocks(request.system)
        if request.cache_system and blocks:
            blocks[-1] = {**blocks[-1], "cache_control": {"type": "ephemeral"}}
        return blocks

    def generate(self, request: LLMRequest) -> BaseModel:
        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=request.max_tokens or self.settings.llm_max_tokens,
            system=self._system_blocks(request),
            messages=[{"role": "user", "content": _as_blocks(request.user)}],
            output_format=request.schema,
            thinking={"type": "adaptive"},
            output_config={"effort": request.effort or self.settings.llm_effort},
        )
        if self.fallbacks:
            kwargs["betas"] = [self.FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
        started = time.monotonic()
        log.info("llm %s task=%s model=%s", self.name, request.task, self.model)
        with self.client.beta.messages.stream(**kwargs) as stream:
            message = stream.get_final_message()
        self.usage.add(getattr(message, "usage", None), time.monotonic() - started)

        if message.stop_reason == "refusal":
            details = getattr(message, "stop_details", None)
            raise LLMRefusal(f"model refused task {request.task}: {getattr(details, 'explanation', '')}")
        if message.stop_reason == "max_tokens":
            raise LLMTruncated(f"task {request.task} hit max_tokens={kwargs['max_tokens']}")

        parsed = getattr(message, "parsed_output", None)
        if parsed is not None:
            return parsed
        text = "".join(getattr(b, "text", "") for b in message.content if getattr(b, "type", "") == "text")
        return parse_json_into(text, request.schema)


def parse_json_into(text: str, schema: type[SchemaT]) -> SchemaT:
    """Parse model text into the schema, tolerating fenced JSON."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    try:
        return schema.model_validate_json(cleaned)
    except ValidationError as exc:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start >= 0 and end > start:
            try:
                return schema.model_validate(json.loads(cleaned[start : end + 1]))
            except (ValidationError, json.JSONDecodeError):
                pass
        raise LLMError(f"could not parse model output as {schema.__name__}: {exc}") from exc


Handler = Callable[[LLMRequest], BaseModel | dict[str, Any]]


class FakeLLM:
    """Deterministic stand-in. Handlers are keyed by ``request.task``."""

    name = "fake"

    def __init__(self, handlers: dict[str, Handler] | None = None, default: Handler | None = None):
        self.handlers = dict(handlers or {})
        self.default = default
        self.calls: list[LLMRequest] = []
        self.usage = Usage()

    def on(self, task: str, handler: Handler) -> FakeLLM:
        self.handlers[task] = handler
        return self

    def generate(self, request: LLMRequest) -> BaseModel:
        self.calls.append(request)
        self.usage.add(None, 0.0)
        handler = self.handlers.get(request.task, self.default)
        if handler is None:
            raise LLMError(f"FakeLLM has no handler for task {request.task!r}")
        result = handler(request)
        if isinstance(result, BaseModel):
            if not isinstance(result, request.schema):
                result = request.schema.model_validate(result.model_dump())
            return result
        return request.schema.model_validate(result)


def make_llm(settings: Settings, fake: bool = False) -> LLM:
    if fake:
        from pipeline.fake_handlers import default_fake_llm

        return default_fake_llm()
    return AnthropicLLM(settings)
