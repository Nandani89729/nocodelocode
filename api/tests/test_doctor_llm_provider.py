"""Doctor LLM provider switch: Claude by default, OpenAI only when opted in."""

from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from api.services.workflow.doctor import llm


class Verdict(BaseModel):
    passed: bool


def test_default_provider_needs_anthropic_key(monkeypatch):
    monkeypatch.delenv(llm.DOCTOR_LLM_PROVIDER_ENV, raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert llm.anthropic_key_configured() is False
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    assert llm.anthropic_key_configured() is True


def test_openai_provider_needs_openai_key(monkeypatch):
    monkeypatch.setenv(llm.DOCTOR_LLM_PROVIDER_ENV, "openai")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert llm.anthropic_key_configured() is False
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert llm.anthropic_key_configured() is True


def _fake_openai(parsed, *, refusal=None, finish_reason="stop"):
    calls = []

    async def parse(**kwargs):
        calls.append(kwargs)
        message = SimpleNamespace(parsed=parsed, refusal=refusal)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=2,
                prompt_tokens_details=SimpleNamespace(cached_tokens=4),
            ),
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(parse=parse))
    )
    return client, calls


@pytest.mark.asyncio
async def test_openai_provider_routes_and_maps_usage(monkeypatch):
    monkeypatch.setenv(llm.DOCTOR_LLM_PROVIDER_ENV, "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("DOCTOR_OPENAI_MODEL", raising=False)
    client, calls = _fake_openai(Verdict(passed=True))
    monkeypatch.setattr(llm, "_openai_client", client)

    parsed, usage = await llm.claude_structured(
        system="sys", user_content="data", output_model=Verdict, effort="medium"
    )

    assert parsed.passed is True
    assert usage == {
        "input_tokens": 10,
        "output_tokens": 2,
        "cache_read_input_tokens": 4,
    }
    assert calls[0]["model"] == llm.DEFAULT_OPENAI_MODEL
    assert calls[0]["response_format"] is Verdict
    assert calls[0]["messages"][0] == {"role": "developer", "content": "sys"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"refusal": "no"}, "refusal"),
        ({"finish_reason": "length"}, "truncated"),
    ],
)
async def test_openai_provider_raises_same_errors(monkeypatch, kwargs, message):
    monkeypatch.setenv(llm.DOCTOR_LLM_PROVIDER_ENV, "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    client, _ = _fake_openai(Verdict(passed=True), **kwargs)
    monkeypatch.setattr(llm, "_openai_client", client)

    with pytest.raises(llm.DoctorLLMError, match=message):
        await llm.claude_structured(system="s", user_content="u", output_model=Verdict)


def test_retry_delay_honours_retry_after_with_cap():
    def exc_with(headers):
        return SimpleNamespace(response=SimpleNamespace(headers=headers))

    assert llm._retry_delay(0, exc_with({"retry-after": "6"})) == 6
    assert llm._retry_delay(3, exc_with({"retry-after": "1"})) == 8  # backoff wins
    assert llm._retry_delay(0, exc_with({"retry-after": "999"})) == 60  # capped
    assert llm._retry_delay(1, None) == 2


@pytest.mark.asyncio
async def test_rate_limits_get_extra_patience(monkeypatch):
    import httpx
    import openai

    monkeypatch.setenv(llm.DOCTOR_LLM_PROVIDER_ENV, "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    sleeps: list[float] = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(llm.asyncio, "sleep", fake_sleep)
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    limited = openai.RateLimitError(
        "slow down",
        response=httpx.Response(429, headers={"retry-after": "6"}, request=request),
        body=None,
    )
    outcomes = [limited] * 4
    ok_client, _ = _fake_openai(Verdict(passed=True))

    async def parse(**kwargs):
        if outcomes:
            raise outcomes.pop()
        return await ok_client.chat.completions.parse(**kwargs)

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(parse=parse))
    )
    monkeypatch.setattr(llm, "_openai_client", client)

    parsed, _ = await llm.claude_structured(
        system="s", user_content="u", output_model=Verdict, retries=3
    )

    assert parsed.passed is True  # 4 rate limits > retries=3, still succeeds
    assert sleeps == [6, 6, 6, 8]
