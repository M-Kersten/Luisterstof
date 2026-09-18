from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from anthropic.lib._parse._transform import transform_schema
from pydantic import BaseModel

from pipeline.config import Settings
from pipeline.ingest.figures import FigureDescription
from pipeline.ingest.structure import StructureGuess
from pipeline.llm import (
    AnthropicLLM,
    FakeLLM,
    LLMRefusal,
    LLMRequest,
    LLMTruncated,
    make_llm,
    parse_json_into,
)
from pipeline.plan.content_plan import PlanOut
from pipeline.plan.glossary import LexiconOut
from pipeline.script.audit import LintOut, SupportOut
from pipeline.script.continuity import ContinuityOut
from pipeline.script.writer import SegmentOut


@pytest.mark.parametrize("model", [PlanOut, LexiconOut, SegmentOut, SupportOut, LintOut, ContinuityOut, StructureGuess, FigureDescription])
def test_output_schemas_are_structured_output_compatible(model):
    schema = transform_schema(model)
    assert schema["type"] == "object" and schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])


class Answer(BaseModel):
    text: str
    n: int


class StubClient:
    def __init__(self, message):
        self.message = message
        self.kwargs = None
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._stream))

    @contextmanager
    def _stream(self, **kwargs):
        self.kwargs = kwargs
        yield SimpleNamespace(get_final_message=lambda: self.message)


def _message(stop="end_turn", parsed=None, text=""):
    return SimpleNamespace(stop_reason=stop, parsed_output=parsed, stop_details=SimpleNamespace(explanation="nope"),
                           content=[SimpleNamespace(type="text", text=text)],
                           usage=SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=3, cache_creation_input_tokens=0))


def test_request_assembly_and_parsed_output():
    settings = Settings(llm_model="claude-opus-5", llm_effort="high", llm_max_tokens=1234)
    client = StubClient(_message(parsed=Answer(text="ok", n=1)))
    llm = AnthropicLLM(settings, client=client)
    req = LLMRequest(task="t", system=["stable", "chapter text"], user="vraag", schema=Answer, effort="low")
    out = llm.generate(req)
    assert out == Answer(text="ok", n=1)
    k = client.kwargs
    assert k["model"] == "claude-opus-5" and k["max_tokens"] == 1234 and k["output_format"] is Answer
    assert k["thinking"] == {"type": "adaptive"} and k["output_config"] == {"effort": "low"}
    assert k["betas"] == [AnthropicLLM.FALLBACK_BETA] and k["fallbacks"] == "default"
    assert k["system"][-1]["cache_control"] == {"type": "ephemeral"} and "cache_control" not in k["system"][0]
    assert k["messages"] == [{"role": "user", "content": [{"type": "text", "text": "vraag"}]}]
    assert llm.usage.calls == 1 and llm.usage.cache_read_tokens == 3


def test_fallbacks_off_for_other_models_and_text_fallback():
    settings = Settings(llm_model="claude-sonnet-4-6")
    client = StubClient(_message(text='```json\n{"text": "x", "n": 2}\n```'))
    out = AnthropicLLM(settings, client=client).generate(LLMRequest(task="t", system="s", user="u", schema=Answer, cache_system=False))
    assert out.n == 2 and "fallbacks" not in client.kwargs and "cache_control" not in client.kwargs["system"][0]


def test_refusal_and_truncation():
    settings = Settings()
    with pytest.raises(LLMRefusal):
        AnthropicLLM(settings, client=StubClient(_message(stop="refusal"))).generate(LLMRequest(task="t", system="s", user="u", schema=Answer))
    with pytest.raises(LLMTruncated):
        AnthropicLLM(settings, client=StubClient(_message(stop="max_tokens"))).generate(LLMRequest(task="t", system="s", user="u", schema=Answer))


def test_parse_json_into_and_fake():
    assert parse_json_into('Here: {"text": "a", "n": 3} done', Answer).n == 3
    fake = FakeLLM({"t": lambda req: {"text": "f", "n": 0}})
    assert fake.generate(LLMRequest(task="t", system="s", user="u", schema=Answer)).text == "f" and fake.calls
    assert isinstance(make_llm(Settings(), fake=True), FakeLLM)
