"""LLM providers for the Agent Copilot loop.

One interface for the agent loop (``CopilotLLM.stream``): it streams text
deltas (``str``) and finishes with one ``ModelTurn`` carrying the tool calls,
a normalised stop reason and usage. Claude is the default; OpenAI is the
stand-in behind the Doctor's provider switch (``DOCTOR_LLM_PROVIDER=openai``).

The key check, provider switch and retry policy are shared with the Doctor
(``api/services/workflow/doctor/llm.py``). The message mappers are pure
functions so they can be tested without a network.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional, Protocol, Sequence

from loguru import logger

from api.services.copilot.contracts import (
    COPILOT_MODEL,
    ToolCall,
    ToolSpec,
    TranscriptMessage,
)
from api.services.workflow.doctor.llm import (
    DEFAULT_OPENAI_MODEL,
    DOCTOR_LLM_PROVIDER_ENV,
    RATE_LIMIT_RETRIES,
    _retry_delay,
    anthropic_key_configured,
)

#: Output budget per model call (thinking + text + tool input).
MAX_OUTPUT_TOKENS = 16000
#: Attempts for connection errors and 5xx before giving up.
DEFAULT_RETRIES = 3

#: Normalised stop reasons (``ModelTurn.stop_reason``).
STOP_END_TURN = "end_turn"
STOP_TOOL_USE = "tool_use"
STOP_MAX_TOKENS = "max_tokens"
STOP_REFUSAL = "refusal"

#: Shown to the model in place of an empty tool result (the API rejects
#: empty content blocks).
EMPTY_TOOL_RESULT = "(no output)"


class CopilotLLMError(Exception):
    """The model call failed. ``str(exc)`` is safe to show to the user;
    provider details are logged, never put in the message."""


class CopilotLLMUnavailable(CopilotLLMError):
    """No key configured for the selected provider."""


@dataclass
class ModelTurn:
    """One completed model call."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = STOP_END_TURN
    usage: dict[str, int] = field(default_factory=dict)
    #: Provider-specific content to resend verbatim for this assistant message
    #: (Anthropic: full content blocks incl. thinking, which the API requires
    #: on the last assistant message of a tool-use loop). Kept in memory for
    #: the current turn only; the persisted transcript stores text + calls.
    replay: Any = None


class CopilotLLM(Protocol):
    def stream(
        self,
        *,
        system: Sequence[str],
        messages: list[TranscriptMessage],
        tools: list[ToolSpec],
        replay: dict[str, Any],
    ) -> AsyncIterator[str | ModelTurn]:
        """Yield text deltas, then exactly one ``ModelTurn``.

        ``system`` parts: the first is stable (cacheable across sessions),
        later parts are session-specific. ``replay`` maps an assistant
        message's first tool-call id to ``ModelTurn.replay`` from earlier
        calls in this turn.
        """
        ...


def _provider_name() -> str:
    # Same switch as the Doctor (doctor.llm._provider).
    return (os.environ.get(DOCTOR_LLM_PROVIDER_ENV) or "anthropic").strip().lower()


def _usage_dict(input_tokens: Any, output_tokens: Any, cache_read: Any) -> dict:
    return {
        "input_tokens": input_tokens or 0,
        "output_tokens": output_tokens or 0,
        "cache_read_input_tokens": cache_read or 0,
    }


def _transient_status(status_code: int) -> bool:
    """5xx/529 overloads, 429, and errors sent mid-stream (the SSE ``error``
    event arrives on an HTTP 200 response) are worth a retry; other 4xx are
    request problems."""
    return status_code >= 500 or status_code == 429 or status_code < 400


async def _backoff(provider: str, attempt: int, limit: int, exc: Exception) -> None:
    if attempt >= limit:
        logger.error(f"Copilot {provider} call failed after {attempt} attempts: {exc}")
        raise CopilotLLMError(
            "The AI service is unavailable right now. Please try again in a minute."
        ) from exc
    delay = _retry_delay(attempt - 1, exc)
    logger.warning(
        f"Copilot {provider} call retry {attempt}/{limit} in {delay}s: {exc}"
    )
    await asyncio.sleep(delay)


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


def to_anthropic_tools(tools: list[ToolSpec]) -> list[dict]:
    return [
        {"name": t.name, "description": t.description, "input_schema": t.input_schema}
        for t in tools
    ]


def to_anthropic_system(system: Sequence[str]) -> list[dict]:
    """System blocks; the stable first block carries the cache breakpoint so
    tools + base prompt are reused across sessions."""
    blocks: list[dict] = []
    for index, part in enumerate(p for p in system if p):
        block: dict = {"type": "text", "text": part}
        if index == 0:
            block["cache_control"] = {"type": "ephemeral"}
        blocks.append(block)
    return blocks


def anthropic_replay_blocks(content: Sequence[Any]) -> list[dict]:
    """Final-message content blocks as request params, verbatim where the
    API requires it (thinking signatures), whitelisted fields otherwise."""
    blocks: list[dict] = []
    for block in content:
        kind = getattr(block, "type", None)
        if kind == "thinking":
            blocks.append(
                {
                    "type": "thinking",
                    "thinking": block.thinking,
                    "signature": block.signature,
                }
            )
        elif kind == "redacted_thinking":
            blocks.append({"type": "redacted_thinking", "data": block.data})
        elif kind == "text":
            if block.text:
                blocks.append({"type": "text", "text": block.text})
        elif kind == "tool_use":
            blocks.append(
                {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": dict(block.input or {}),
                }
            )
    return blocks


def to_anthropic_messages(
    messages: list[TranscriptMessage], replay: Optional[dict[str, Any]] = None
) -> list[dict]:
    """Transcript -> Anthropic messages.

    Tool results become ``tool_result`` blocks in a user message; consecutive
    same-role messages are merged (tool results first, as the API expects).
    """
    replay = replay or {}
    out: list[dict] = []

    def push(role: str, blocks: list[dict]) -> None:
        if not blocks:
            return
        if out and out[-1]["role"] == role:
            out[-1]["content"].extend(blocks)
        else:
            out.append({"role": role, "content": list(blocks)})

    for message in messages:
        if message.role == "user":
            if message.content:
                push("user", [{"type": "text", "text": message.content}])
        elif message.role == "assistant":
            key = message.tool_calls[0].id if message.tool_calls else None
            if key is not None and key in replay:
                push("assistant", list(replay[key]))
                continue
            blocks: list[dict] = []
            if message.content:
                blocks.append({"type": "text", "text": message.content})
            blocks.extend(
                {"type": "tool_use", "id": c.id, "name": c.name, "input": c.input}
                for c in message.tool_calls
            )
            push("assistant", blocks)
        elif message.role == "tool":
            push(
                "user",
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": message.tool_call_id,
                        "content": message.content or EMPTY_TOOL_RESULT,
                        "is_error": message.is_error,
                    }
                ],
            )
    return out


_ANTHROPIC_STOP = {
    "end_turn": STOP_END_TURN,
    "stop_sequence": STOP_END_TURN,
    "pause_turn": STOP_END_TURN,
    "tool_use": STOP_TOOL_USE,
    "max_tokens": STOP_MAX_TOKENS,
    "model_context_window_exceeded": STOP_MAX_TOKENS,
    "refusal": STOP_REFUSAL,
}


def anthropic_turn(message: Any) -> ModelTurn:
    """Final Anthropic message -> ModelTurn."""
    text = "".join(
        b.text for b in message.content if getattr(b, "type", None) == "text"
    )
    calls = [
        ToolCall(id=b.id, name=b.name, input=dict(b.input or {}))
        for b in message.content
        if getattr(b, "type", None) == "tool_use"
    ]
    usage = getattr(message, "usage", None)
    return ModelTurn(
        text=text,
        tool_calls=calls,
        stop_reason=_ANTHROPIC_STOP.get(message.stop_reason or "", STOP_END_TURN),
        usage=_usage_dict(
            getattr(usage, "input_tokens", 0),
            getattr(usage, "output_tokens", 0),
            getattr(usage, "cache_read_input_tokens", 0),
        ),
        replay=anthropic_replay_blocks(message.content) if calls else None,
    )


_anthropic_client: Any = None


def _get_anthropic_client() -> Any:
    global _anthropic_client
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise CopilotLLMUnavailable("The Copilot is not configured on this server.")
    if _anthropic_client is None:
        import anthropic

        # Retries are ours (Retry-After aware, more patience on 429).
        _anthropic_client = anthropic.AsyncAnthropic(max_retries=0)
    return _anthropic_client


class AnthropicCopilotLLM:
    def __init__(self, client: Any = None, *, retries: int = DEFAULT_RETRIES):
        self._client = client
        self._retries = retries

    async def stream(
        self,
        *,
        system: Sequence[str],
        messages: list[TranscriptMessage],
        tools: list[ToolSpec],
        replay: dict[str, Any],
    ) -> AsyncIterator[str | ModelTurn]:
        import anthropic

        client = self._client or _get_anthropic_client()
        kwargs: dict[str, Any] = {
            "model": COPILOT_MODEL,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "thinking": {"type": "adaptive"},
            "system": to_anthropic_system(system),
            "messages": to_anthropic_messages(messages, replay),
            # Automatic breakpoint on the last block: each loop iteration
            # re-reads the growing transcript from cache.
            "cache_control": {"type": "ephemeral"},
        }
        if tools:
            kwargs["tools"] = to_anthropic_tools(tools)

        attempt = 0
        while True:
            emitted = False
            try:
                async with client.messages.stream(**kwargs) as stream:
                    async for event in stream:
                        if getattr(event, "type", None) == "text" and event.text:
                            emitted = True
                            yield event.text
                    final = await stream.get_final_message()
            except anthropic.RateLimitError as exc:
                limit, error = max(self._retries, RATE_LIMIT_RETRIES), exc
            except anthropic.APIConnectionError as exc:
                limit, error = self._retries, exc
            except anthropic.APIStatusError as exc:
                if not _transient_status(exc.status_code):
                    logger.error(f"Copilot Claude request rejected: {exc}")
                    raise CopilotLLMError(
                        "The AI service rejected the request."
                    ) from exc
                limit, error = self._retries, exc
            else:
                yield anthropic_turn(final)
                return
            # Text already reached the user: a retry would duplicate it.
            if emitted:
                logger.error(f"Copilot Claude stream broke mid-reply: {error}")
                raise CopilotLLMError(
                    "The AI service dropped the connection. Please try again."
                ) from error
            attempt += 1
            await _backoff("Claude", attempt, limit, error)


# ---------------------------------------------------------------------------
# OpenAI (stand-in)
# ---------------------------------------------------------------------------


def to_openai_tools(tools: list[ToolSpec]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            },
        }
        for t in tools
    ]


def to_openai_messages(
    system: Sequence[str], messages: list[TranscriptMessage]
) -> list[dict]:
    """Transcript -> chat.completions messages (developer system message).

    OpenAI has no tool-error flag, so failed results are prefixed ``ERROR:``.
    """
    out: list[dict] = []
    system_text = "\n\n".join(p for p in system if p)
    if system_text:
        out.append({"role": "developer", "content": system_text})
    for message in messages:
        if message.role == "user":
            if message.content:
                out.append({"role": "user", "content": message.content})
        elif message.role == "assistant":
            if not message.content and not message.tool_calls:
                continue
            entry: dict[str, Any] = {
                "role": "assistant",
                "content": message.content or None,
            }
            if message.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {
                            "name": c.name,
                            "arguments": json.dumps(c.input, ensure_ascii=False),
                        },
                    }
                    for c in message.tool_calls
                ]
            out.append(entry)
        elif message.role == "tool":
            content = message.content or EMPTY_TOOL_RESULT
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": message.tool_call_id,
                    "content": f"ERROR: {content}" if message.is_error else content,
                }
            )
    return out


_OPENAI_STOP = {
    "stop": STOP_END_TURN,
    "tool_calls": STOP_TOOL_USE,
    "function_call": STOP_TOOL_USE,
    "length": STOP_MAX_TOKENS,
    "content_filter": STOP_REFUSAL,
}


def _parse_arguments(raw: str, name: str) -> dict:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(f"Copilot OpenAI: unparsable arguments for {name}")
        return {}
    return parsed if isinstance(parsed, dict) else {}


_openai_client: Any = None


def _get_openai_client() -> Any:
    global _openai_client
    if not os.environ.get("OPENAI_API_KEY"):
        raise CopilotLLMUnavailable("The Copilot is not configured on this server.")
    if _openai_client is None:
        import openai

        _openai_client = openai.AsyncOpenAI(max_retries=0)
    return _openai_client


class OpenAICopilotLLM:
    def __init__(self, client: Any = None, *, retries: int = DEFAULT_RETRIES):
        self._client = client
        self._retries = retries

    async def stream(
        self,
        *,
        system: Sequence[str],
        messages: list[TranscriptMessage],
        tools: list[ToolSpec],
        replay: dict[str, Any],
    ) -> AsyncIterator[str | ModelTurn]:
        import openai

        client = self._client or _get_openai_client()
        kwargs: dict[str, Any] = {
            "model": os.environ.get("DOCTOR_OPENAI_MODEL") or DEFAULT_OPENAI_MODEL,
            "messages": to_openai_messages(system, messages),
            "max_completion_tokens": MAX_OUTPUT_TOKENS,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            kwargs["tools"] = to_openai_tools(tools)

        attempt = 0
        while True:
            emitted = False
            text_parts: list[str] = []
            calls: dict[int, dict] = {}
            finish: Optional[str] = None
            refused = False
            usage: Any = None
            try:
                stream = await client.chat.completions.create(**kwargs)
                async for chunk in stream:
                    if getattr(chunk, "usage", None):
                        usage = chunk.usage
                    if not chunk.choices:
                        continue
                    choice = chunk.choices[0]
                    delta = choice.delta
                    if delta is not None:
                        if delta.content:
                            emitted = True
                            text_parts.append(delta.content)
                            yield delta.content
                        if getattr(delta, "refusal", None):
                            refused = True
                        for tc in delta.tool_calls or []:
                            slot = calls.setdefault(
                                tc.index, {"id": "", "name": "", "arguments": ""}
                            )
                            if tc.id:
                                slot["id"] = tc.id
                            fn = tc.function
                            if fn is not None:
                                if fn.name:
                                    slot["name"] += fn.name
                                if fn.arguments:
                                    slot["arguments"] += fn.arguments
                    if choice.finish_reason:
                        finish = choice.finish_reason
            except openai.RateLimitError as exc:
                limit, error = max(self._retries, RATE_LIMIT_RETRIES), exc
            except openai.APIConnectionError as exc:
                limit, error = self._retries, exc
            except openai.APIStatusError as exc:
                if not _transient_status(exc.status_code):
                    logger.error(f"Copilot OpenAI request rejected: {exc}")
                    raise CopilotLLMError(
                        "The AI service rejected the request."
                    ) from exc
                limit, error = self._retries, exc
            else:
                details = getattr(usage, "prompt_tokens_details", None)
                stop = STOP_REFUSAL if refused else _OPENAI_STOP.get(finish or "")
                tool_calls = [
                    ToolCall(
                        id=slot["id"] or f"call_{index}",
                        name=slot["name"],
                        input=_parse_arguments(slot["arguments"], slot["name"]),
                    )
                    for index, slot in sorted(calls.items())
                ]
                yield ModelTurn(
                    text="".join(text_parts),
                    tool_calls=tool_calls,
                    stop_reason=stop
                    or (STOP_TOOL_USE if tool_calls else STOP_END_TURN),
                    usage=_usage_dict(
                        getattr(usage, "prompt_tokens", 0),
                        getattr(usage, "completion_tokens", 0),
                        getattr(details, "cached_tokens", 0),
                    ),
                )
                return
            if emitted:
                logger.error(f"Copilot OpenAI stream broke mid-reply: {error}")
                raise CopilotLLMError(
                    "The AI service dropped the connection. Please try again."
                ) from error
            attempt += 1
            await _backoff("OpenAI", attempt, limit, error)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

_provider_override: Optional[CopilotLLM] = None


def set_provider_for_tests(provider: Optional[CopilotLLM]) -> None:
    """Inject a fake provider (tests only); None restores the default."""
    global _provider_override
    _provider_override = provider


def get_provider() -> CopilotLLM:
    """The configured provider; raises CopilotLLMUnavailable without a key."""
    if _provider_override is not None:
        return _provider_override
    if not anthropic_key_configured():
        raise CopilotLLMUnavailable("The Copilot is not configured on this server.")
    if _provider_name() == "openai":
        return OpenAICopilotLLM()
    return AnthropicCopilotLLM()
