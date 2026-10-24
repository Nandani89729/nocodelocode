"""Copilot LLM providers: message mapping (pure) and streaming with fake clients."""

from types import SimpleNamespace

import httpx
import pytest

from api.services.copilot import llm
from api.services.copilot.contracts import (
    COPILOT_MODEL,
    ToolCall,
    ToolSpec,
    TranscriptMessage,
)

TOOLS = [
    ToolSpec(
        name="get_workflow_code",
        description="Read code",
        input_schema={"type": "object", "properties": {}},
    )
]


def transcript():
    return [
        TranscriptMessage(role="user", content="Edit it"),
        TranscriptMessage(
            role="assistant",
            content="Reading.",
            tool_calls=[
                ToolCall(id="t1", name="get_workflow_code", input={"id": 5}),
                ToolCall(id="t2", name="list_tools", input={}),
            ],
        ),
        TranscriptMessage(
            role="tool", content='{"code": "x"}', tool_call_id="t1", tool_name="g"
        ),
        TranscriptMessage(
            role="tool", content="boom", tool_call_id="t2", is_error=True
        ),
        TranscriptMessage(role="user", content="And then?"),
        TranscriptMessage(role="assistant", content=""),  # empty: dropped
    ]


# ---------------------------------------------------------------------------
# Mapping
# ---------------------------------------------------------------------------


def test_anthropic_messages_mapping():
    out = llm.to_anthropic_messages(transcript())
    assert [m["role"] for m in out] == ["user", "assistant", "user"]
    assert out[1]["content"] == [
        {"type": "text", "text": "Reading."},
        {
            "type": "tool_use",
            "id": "t1",
            "name": "get_workflow_code",
            "input": {"id": 5},
        },
        {"type": "tool_use", "id": "t2", "name": "list_tools", "input": {}},
    ]
    # Tool results first, then the next user text, in one user message.
    assert out[2]["content"] == [
        {
            "type": "tool_result",
            "tool_use_id": "t1",
            "content": '{"code": "x"}',
            "is_error": False,
        },
        {
            "type": "tool_result",
            "tool_use_id": "t2",
            "content": "boom",
            "is_error": True,
        },
        {"type": "text", "text": "And then?"},
    ]


def test_anthropic_replay_replaces_assistant_blocks():
    blocks = [
        {"type": "thinking", "thinking": "hmm", "signature": "sig"},
        {"type": "tool_use", "id": "t1", "name": "get_workflow_code", "input": {}},
    ]
    out = llm.to_anthropic_messages(transcript(), {"t1": blocks})
    assert out[1]["content"] == blocks


def test_anthropic_system_and_tools():
    system = llm.to_anthropic_system(["stable", "", "session"])
    assert system == [
        {"type": "text", "text": "stable", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "session"},
    ]
    assert llm.to_anthropic_tools(TOOLS) == [
        {
            "name": "get_workflow_code",
            "description": "Read code",
            "input_schema": {"type": "object", "properties": {}},
        }
    ]


def _block(**kw):
    return SimpleNamespace(**kw)


def _final_message(stop_reason="tool_use"):
    return SimpleNamespace(
        content=[
            _block(type="thinking", thinking="plan", signature="sig"),
            _block(type="redacted_thinking", data="enc"),
            _block(type="text", text="Let me look."),
            _block(type="tool_use", id="t9", name="get_workflow_code", input={"a": 1}),
        ],
        stop_reason=stop_reason,
        usage=SimpleNamespace(
            input_tokens=10, output_tokens=5, cache_read_input_tokens=3
        ),
    )


def test_anthropic_turn_keeps_thinking_for_replay():
    turn = llm.anthropic_turn(_final_message())
    assert turn.text == "Let me look."
    assert turn.tool_calls == [
        ToolCall(id="t9", name="get_workflow_code", input={"a": 1})
    ]
    assert turn.stop_reason == llm.STOP_TOOL_USE
    assert turn.usage == {
        "input_tokens": 10,
        "output_tokens": 5,
        "cache_read_input_tokens": 3,
    }
    assert turn.replay == [
        {"type": "thinking", "thinking": "plan", "signature": "sig"},
        {"type": "redacted_thinking", "data": "enc"},
        {"type": "text", "text": "Let me look."},
        {
            "type": "tool_use",
            "id": "t9",
            "name": "get_workflow_code",
            "input": {"a": 1},
        },
    ]


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("refusal", llm.STOP_REFUSAL),
        ("max_tokens", llm.STOP_MAX_TOKENS),
        ("model_context_window_exceeded", llm.STOP_MAX_TOKENS),
        ("end_turn", llm.STOP_END_TURN),
    ],
)
def test_anthropic_stop_reasons(raw, expected):
    assert llm.anthropic_turn(_final_message(raw)).stop_reason == expected


def test_openai_messages_mapping():
    out = llm.to_openai_messages(["stable", "session"], transcript())
    assert out[0] == {"role": "developer", "content": "stable\n\nsession"}
    assert out[1] == {"role": "user", "content": "Edit it"}
    assert out[2] == {
        "role": "assistant",
        "content": "Reading.",
        "tool_calls": [
            {
                "id": "t1",
                "type": "function",
                "function": {"name": "get_workflow_code", "arguments": '{"id": 5}'},
            },
            {
                "id": "t2",
                "type": "function",
                "function": {"name": "list_tools", "arguments": "{}"},
            },
        ],
    }
    assert out[3] == {"role": "tool", "tool_call_id": "t1", "content": '{"code": "x"}'}
    assert out[4] == {"role": "tool", "tool_call_id": "t2", "content": "ERROR: boom"}
    assert out[5] == {"role": "user", "content": "And then?"}
    assert len(out) == 6
    assert llm.to_openai_tools(TOOLS)[0]["function"]["parameters"] == {
        "type": "object",
        "properties": {},
    }


# ---------------------------------------------------------------------------
# Anthropic streaming
# ---------------------------------------------------------------------------


class FakeAnthropicStream:
    def __init__(self, texts, final, error=None):
        self.texts, self.final, self.error = texts, final, error

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def __aiter__(self):
        for text in self.texts:
            yield SimpleNamespace(type="text", text=text)
        if self.error is not None:
            raise self.error

    async def get_final_message(self):
        return self.final


class FakeAnthropicClient:
    def __init__(self, streams):
        self.streams = list(streams)
        self.kwargs = []
        self.messages = SimpleNamespace(stream=self._stream)

    def _stream(self, **kwargs):
        self.kwargs.append(kwargs)
        item = self.streams.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _status_error(cls, status):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status, request=request)
    return cls("err", response=response, body=None)


async def _collect(provider, messages=None):
    return [
        item
        async for item in provider.stream(
            system=["stable", "session"],
            messages=messages or [TranscriptMessage(role="user", content="hi")],
            tools=TOOLS,
            replay={},
        )
    ]


async def test_anthropic_stream_request_and_events():
    client = FakeAnthropicClient(
        [FakeAnthropicStream(["Let ", "me look."], _final_message())]
    )
    items = await _collect(llm.AnthropicCopilotLLM(client))
    assert items[:2] == ["Let ", "me look."]
    assert isinstance(items[2], llm.ModelTurn) and items[2].tool_calls[0].id == "t9"
    kwargs = client.kwargs[0]
    assert kwargs["model"] == COPILOT_MODEL
    assert kwargs["thinking"] == {"type": "adaptive"}
    assert kwargs["max_tokens"] == llm.MAX_OUTPUT_TOKENS
    assert kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert kwargs["tools"][0]["name"] == "get_workflow_code"


async def test_anthropic_retries_before_any_text(monkeypatch):
    import anthropic

    async def no_sleep(_):
        return None

    monkeypatch.setattr(llm.asyncio, "sleep", no_sleep)
    client = FakeAnthropicClient(
        [
            _status_error(anthropic.RateLimitError, 429),
            _status_error(anthropic.InternalServerError, 529),
            FakeAnthropicStream(["ok"], _final_message("end_turn")),
        ]
    )
    items = await _collect(llm.AnthropicCopilotLLM(client))
    assert items[0] == "ok" and len(client.kwargs) == 3


async def test_anthropic_no_retry_after_text_was_streamed(monkeypatch):
    import anthropic

    client = FakeAnthropicClient(
        [
            FakeAnthropicStream(
                ["partial"],
                None,
                error=_status_error(anthropic.InternalServerError, 500),
            )
        ]
    )
    with pytest.raises(llm.CopilotLLMError) as info:
        await _collect(llm.AnthropicCopilotLLM(client))
    assert "dropped" in str(info.value) and len(client.kwargs) == 1


async def test_anthropic_bad_request_is_not_retried():
    import anthropic

    client = FakeAnthropicClient([_status_error(anthropic.BadRequestError, 400)])
    with pytest.raises(llm.CopilotLLMError) as info:
        await _collect(llm.AnthropicCopilotLLM(client))
    assert str(info.value) == "The AI service rejected the request."


# ---------------------------------------------------------------------------
# OpenAI streaming
# ---------------------------------------------------------------------------


def _chunk(content=None, tool_calls=None, finish=None, usage=None, refusal=None):
    choices = []
    if content is not None or tool_calls or finish or refusal:
        choices = [
            SimpleNamespace(
                delta=SimpleNamespace(
                    content=content, tool_calls=tool_calls, refusal=refusal
                ),
                finish_reason=finish,
            )
        ]
    return SimpleNamespace(choices=choices, usage=usage)


def _tc(index, id_=None, name=None, args=None):
    return SimpleNamespace(
        index=index, id=id_, function=SimpleNamespace(name=name, arguments=args)
    )


class FakeOpenAIClient:
    def __init__(self, chunks):
        self.chunks = chunks
        self.kwargs = None

        async def create(**kwargs):
            self.kwargs = kwargs

            async def gen():
                for c in self.chunks:
                    yield c

            return gen()

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


async def test_openai_stream_accumulates_tool_calls(monkeypatch):
    monkeypatch.delenv("DOCTOR_OPENAI_MODEL", raising=False)
    client = FakeOpenAIClient(
        [
            _chunk(content="Checking"),
            _chunk(tool_calls=[_tc(0, "call_a", "get_workflow_code", '{"wor')]),
            _chunk(tool_calls=[_tc(0, args='kflow_id": 5}')]),
            _chunk(tool_calls=[_tc(1, "call_b", "list_tools", "")]),
            _chunk(finish="tool_calls"),
            _chunk(
                usage=SimpleNamespace(
                    prompt_tokens=7,
                    completion_tokens=2,
                    prompt_tokens_details=SimpleNamespace(cached_tokens=4),
                )
            ),
        ]
    )
    items = await _collect(llm.OpenAICopilotLLM(client))
    assert items[0] == "Checking"
    turn = items[-1]
    assert turn.stop_reason == llm.STOP_TOOL_USE
    assert turn.tool_calls == [
        ToolCall(id="call_a", name="get_workflow_code", input={"workflow_id": 5}),
        ToolCall(id="call_b", name="list_tools", input={}),
    ]
    assert turn.usage == {
        "input_tokens": 7,
        "output_tokens": 2,
        "cache_read_input_tokens": 4,
    }
    assert client.kwargs["model"] == "gpt-5.5"
    assert client.kwargs["messages"][0]["role"] == "developer"
    assert client.kwargs["tools"][0]["type"] == "function"


@pytest.mark.parametrize(
    "chunk,expected",
    [
        (_chunk(finish="length"), llm.STOP_MAX_TOKENS),
        (_chunk(refusal="no", finish="stop"), llm.STOP_REFUSAL),
        (_chunk(finish="stop"), llm.STOP_END_TURN),
    ],
)
async def test_openai_stop_reasons(chunk, expected):
    items = await _collect(llm.OpenAICopilotLLM(FakeOpenAIClient([chunk])))
    assert items[-1].stop_reason == expected


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def test_get_provider_follows_doctor_switch(monkeypatch):
    llm.set_provider_for_tests(None)
    monkeypatch.delenv("DOCTOR_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(llm.CopilotLLMUnavailable):
        llm.get_provider()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    assert isinstance(llm.get_provider(), llm.AnthropicCopilotLLM)
    monkeypatch.setenv("DOCTOR_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    assert isinstance(llm.get_provider(), llm.OpenAICopilotLLM)
