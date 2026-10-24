"""Claude client wrapper for the Agent Doctor (Layer 1, clustering, rules, judge).

One place for: key resolution, structured output, prompt caching, retries,
refusal handling and usage metering. Callers pass a frozen system prompt
(cached) and volatile user content; output is a validated Pydantic model.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Optional, TypeVar

from loguru import logger
from pydantic import BaseModel

from api.services.workflow.doctor.contracts import DOCTOR_MODEL

T = TypeVar("T", bound=BaseModel)


class DoctorLLMUnavailable(Exception):
    """No Doctor LLM key configured (deployment-level ANTHROPIC_API_KEY)."""


class DoctorLLMError(Exception):
    """Claude call failed after retries, refused, or was truncated."""


#: Temporary stand-in provider for deployments without an Anthropic key.
#: Set DOCTOR_LLM_PROVIDER=openai (uses OPENAI_API_KEY and
#: DOCTOR_OPENAI_MODEL). Claude stays the default and the calibrated choice:
#: the judge eval gate was set for Claude, so re-run it after switching.
DOCTOR_LLM_PROVIDER_ENV = "DOCTOR_LLM_PROVIDER"
DEFAULT_OPENAI_MODEL = "gpt-5.5"


def _provider() -> str:
    return (os.environ.get(DOCTOR_LLM_PROVIDER_ENV) or "anthropic").strip().lower()


def anthropic_key_configured() -> bool:
    """True when the configured Doctor LLM provider has its key."""
    if _provider() == "openai":
        return bool(os.environ.get("OPENAI_API_KEY"))
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


#: Rate-limit waits honour the provider's Retry-After, capped here.
MAX_RETRY_WAIT_SECONDS = 60
#: Low-tier keys (e.g. 10 requests/min) need more patience than 5xx errors.
RATE_LIMIT_RETRIES = 8


def _retry_delay(attempt: int, exc: Optional[Exception]) -> float:
    """Backoff 1, 2, 4... s, or the provider's Retry-After when it is longer."""
    delay = float(2**attempt)
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    try:
        delay = max(delay, float(headers.get("retry-after") or 0))
    except (TypeError, ValueError):
        pass
    return min(delay, MAX_RETRY_WAIT_SECONDS)


_client: Any = None


def _get_client() -> Any:
    global _client
    if not anthropic_key_configured():
        raise DoctorLLMUnavailable("ANTHROPIC_API_KEY is not set")
    if _client is None:
        import anthropic

        _client = anthropic.AsyncAnthropic()
    return _client


def set_client_for_tests(client: Any) -> None:
    """Inject a fake client (tests only)."""
    global _client
    _client = client


async def claude_structured(
    *,
    system: str,
    user_content: str,
    output_model: type[T],
    max_tokens: int = 16000,
    effort: str = "high",
    retries: int = 3,
) -> tuple[T, dict]:
    """Run one structured-output call. Returns (parsed_model, usage_dict).

    The system prompt is marked cacheable so repeated batches (Layer 1) reuse
    the stable prefix; keep volatile data in ``user_content`` only.
    """
    if _provider() == "openai":
        return await _openai_structured(
            system=system,
            user_content=user_content,
            output_model=output_model,
            max_tokens=max_tokens,
            effort=effort,
            retries=retries,
        )
    import anthropic

    client = _get_client()
    system_blocks = [
        {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
    ]
    last_exc: Optional[Exception] = None
    attempt = 0
    while True:
        try:
            response = await client.messages.parse(
                model=DOCTOR_MODEL,
                max_tokens=max_tokens,
                system=system_blocks,
                messages=[{"role": "user", "content": user_content}],
                output_format=output_model,
                output_config={"effort": effort},
            )
        except anthropic.RateLimitError as exc:
            last_exc = exc
            limit = max(retries, RATE_LIMIT_RETRIES)
        except anthropic.APIConnectionError as exc:
            last_exc = exc
            limit = retries
        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500:
                last_exc = exc
                limit = retries
            else:
                raise DoctorLLMError(f"Claude request rejected: {exc}") from exc
        else:
            if response.stop_reason == "refusal":
                raise DoctorLLMError("Claude declined the request (refusal)")
            if response.stop_reason == "max_tokens":
                raise DoctorLLMError("Claude output truncated (max_tokens)")
            parsed = response.parsed_output
            if parsed is None:
                raise DoctorLLMError("Claude returned no parsable output")
            usage = getattr(response, "usage", None)
            usage_dict = {
                "input_tokens": getattr(usage, "input_tokens", 0) or 0,
                "output_tokens": getattr(usage, "output_tokens", 0) or 0,
                "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0)
                or 0,
            }
            return parsed, usage_dict
        attempt += 1
        if attempt >= limit:
            raise DoctorLLMError(
                f"Claude call failed after {attempt} attempts: {last_exc}"
            )
        delay = _retry_delay(attempt - 1, last_exc)
        logger.warning(
            f"Doctor Claude call retry {attempt}/{limit} in {delay}s: {last_exc}"
        )
        await asyncio.sleep(delay)


_openai_client: Any = None


async def _openai_structured(
    *,
    system: str,
    user_content: str,
    output_model: type[T],
    max_tokens: int,
    effort: str,
    retries: int,
) -> tuple[T, dict]:
    """OpenAI equivalent of ``claude_structured`` (same contract and errors)."""
    global _openai_client
    import openai

    if not os.environ.get("OPENAI_API_KEY"):
        raise DoctorLLMUnavailable("OPENAI_API_KEY is not set")
    if _openai_client is None:
        _openai_client = openai.AsyncOpenAI()
    model = os.environ.get("DOCTOR_OPENAI_MODEL") or DEFAULT_OPENAI_MODEL

    last_exc: Optional[Exception] = None
    attempt = 0
    while True:
        try:
            response = await _openai_client.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "developer", "content": system},
                    {"role": "user", "content": user_content},
                ],
                response_format=output_model,
                max_completion_tokens=max_tokens,
                reasoning_effort=effort,
            )
        except openai.RateLimitError as exc:
            last_exc = exc
            limit = max(retries, RATE_LIMIT_RETRIES)
        except openai.APIConnectionError as exc:
            last_exc = exc
            limit = retries
        except openai.APIStatusError as exc:
            if exc.status_code >= 500:
                last_exc = exc
                limit = retries
            else:
                raise DoctorLLMError(f"OpenAI request rejected: {exc}") from exc
        else:
            choice = response.choices[0]
            if choice.message.refusal:
                raise DoctorLLMError("OpenAI declined the request (refusal)")
            if choice.finish_reason == "length":
                raise DoctorLLMError("OpenAI output truncated (max tokens)")
            parsed = choice.message.parsed
            if parsed is None:
                raise DoctorLLMError("OpenAI returned no parsable output")
            usage = response.usage
            details = getattr(usage, "prompt_tokens_details", None)
            return parsed, {
                "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
                "cache_read_input_tokens": getattr(details, "cached_tokens", 0) or 0,
            }
        attempt += 1
        if attempt >= limit:
            raise DoctorLLMError(
                f"OpenAI call failed after {attempt} attempts: {last_exc}"
            )
        delay = _retry_delay(attempt - 1, last_exc)
        logger.warning(
            f"Doctor OpenAI call retry {attempt}/{limit} in {delay}s: {last_exc}"
        )
        await asyncio.sleep(delay)
