"""LLM access for the script side of the pipeline.

Every stage that needs a model goes through ``LLM.generate`` with a pydantic
schema, so the stages never see raw text. Two implementations:

* ``OpenCodeLLM`` (pipeline/opencode_llm.py) asks a running OpenCode server to
  generate with the model configured in its own opencode.json. This is the
  normal path: the model runs on your own machine and book text stays there.
* ``FakeLLM`` answers from canned handlers and records every request. Tests
  and the CLI ``--fake-llm`` flag use it.

The OpenCode endpoint takes a prompt and returns text, so the JSON schema
travels inside the prompt (``SCHEMA_INSTRUCTION``, ``inline_schema``) and the
answer is cleaned of thinking blocks (``clean_answer``), checked for loops
(``looping``) and parsed here.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from pipeline.config import Settings

log = logging.getLogger(__name__)

SchemaT = TypeVar("SchemaT", bound=BaseModel)

ContentBlocks = str | list[dict[str, Any] | str]


class LLMError(RuntimeError):
    pass


class LLMTruncated(LLMError):
    pass


@dataclass
class LLMRequest:
    task: str
    system: ContentBlocks
    user: ContentBlocks
    schema: type[BaseModel]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0

    def add(self, usage: Any, seconds: float) -> None:
        self.calls += 1
        self.seconds += seconds
        if usage is None:
            return
        self.input_tokens += getattr(usage, "input_tokens", 0) or 0
        self.output_tokens += getattr(usage, "output_tokens", 0) or 0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "seconds": round(self.seconds, 1),
        }


class LLM(Protocol):
    name: str
    usage: Usage

    def generate(self, request: LLMRequest) -> BaseModel: ...


def text_block(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


# ---------------------------------------------------------------------------
# prompt and answer helpers, shared by the OpenCode backend
# ---------------------------------------------------------------------------

CHARS_PER_TOKEN = 3.2  # conservative for Dutch: overestimating fails loudly, underestimating truncates silently
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

SCHEMA_INSTRUCTION = (
    "\n\n## Antwoordformaat\nAntwoord uitsluitend met één JSON-object dat voldoet aan dit JSON-schema. "
    "Geen uitleg, geen markdown, geen tekst ervoor of erna.\n"
)


def inline_schema(schema: dict) -> dict:
    """Resolve pydantic's ``$defs``/``$ref``: not every model follows references when it writes JSON by hand."""
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


def content_text(content: ContentBlocks) -> str:
    """The prompt text of a request: content blocks are text blocks, joined in order."""
    texts: list[str] = []
    blocks = [text_block(content)] if isinstance(content, str) else content
    for block in blocks:
        if isinstance(block, str):
            if block:
                texts.append(block)
        elif block.get("type") == "text" and block.get("text"):
            texts.append(block["text"])
    return "\n\n".join(texts)


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
    if fake or settings.llm_backend == "fake":
        from pipeline.fake_handlers import default_fake_llm

        return default_fake_llm()
    from pipeline.opencode_llm import OpenCodeLLM

    return OpenCodeLLM(settings)
