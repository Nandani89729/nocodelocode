"""Agent Copilot loop: one user message in, a stream of CopilotEvents out.

``run_turn`` appends the user message, then alternates model calls (text
streamed as ``text_delta``) and tool calls (``tool_call`` / ``tool_result``,
plus ``plan`` / ``proposal`` when a tool produced one) until the model stops
calling tools or the step cap is hit. The transcript is persisted after
every step so a crash mid-turn keeps the history.

Every turn ends with exactly one terminal event: ``message_done`` or
``error``. Error messages are plain sentences; details go to the log.
"""

from __future__ import annotations

import json
from contextlib import aclosing
from datetime import UTC, datetime
from typing import Any, AsyncIterator, Literal, Optional

from loguru import logger
from pydantic import ValidationError

from api.db import db_client
from api.services.copilot import llm
from api.services.copilot.brief import brief_prompt_text, brief_response_from_model
from api.services.copilot.contracts import (
    MAX_PROPOSAL_FIX_ROUNDS,
    MAX_TOOL_ITERATIONS_PER_TURN,
    MAX_TOOL_RESULT_CHARS,
    MAX_TRANSCRIPT_MESSAGES,
    USAGE_CONTEXT_COPILOT,
    CopilotEvent,
    CopilotEventType,
    ToolCall,
    ToolContext,
    ToolOutcome,
    TranscriptMessage,
)
from api.services.copilot.prompt import (
    PROPOSE_CHANGES_TOOL,
    build_system_prompt,
)

TITLE_MAX_CHARS = 60

#: ToolContext.mode values.
CopilotMode = Literal["chat", "self_improve"]
SELF_IMPROVE_MODE = "self_improve"
SESSION_KIND_CHAT = "chat"
SELF_IMPROVE_SESSION_TITLE = "Self-improve"

#: Result recorded for a tool call that never got one (e.g. the worker died
#: between the model's call and the tool's result).
INTERRUPTED_TOOL_RESULT = "This tool call was interrupted before it returned."

GENERIC_ERROR = "Something went wrong while answering. Please try again."
REFUSAL_ERROR = "The assistant declined to continue with this request."
TRUNCATED_ERROR = (
    "The reply was cut off because it got too long. Ask me to continue, "
    "or narrow the request."
)
STEP_CAP_MESSAGE = (
    f"I stopped after {MAX_TOOL_ITERATIONS_PER_TURN} steps without finishing. "
    "Tell me to continue, or narrow the request."
)
FIX_LIMIT_ERROR = (
    f"Fix round limit reached ({MAX_PROPOSAL_FIX_ROUNDS}). Do not propose "
    "again in this turn: explain the remaining validation errors to the user."
)


def _event(kind: CopilotEventType, **data: Any) -> CopilotEvent:
    return CopilotEvent(type=kind, data=data)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def session_title(text: str) -> str:
    """First line-ish of the first user message, at most TITLE_MAX_CHARS."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= TITLE_MAX_CHARS:
        return collapsed
    return collapsed[: TITLE_MAX_CHARS - 1].rstrip() + "…"


def load_transcript(raw: Any) -> list[TranscriptMessage]:
    messages: list[TranscriptMessage] = []
    for item in raw or []:
        try:
            messages.append(TranscriptMessage.model_validate(item))
        except ValidationError:
            logger.warning("Copilot: skipping malformed transcript entry")
    return messages


def tool_result_text(outcome: ToolOutcome) -> str:
    """What the model sees for a tool result, capped at MAX_TOOL_RESULT_CHARS."""
    content = outcome.content
    if content is None:
        text = outcome.summary or ("ok" if outcome.ok else "failed")
    elif isinstance(content, str):
        text = content
    else:
        text = json.dumps(content, ensure_ascii=False, default=str)
    if len(text) > MAX_TOOL_RESULT_CHARS:
        dropped = len(text) - MAX_TOOL_RESULT_CHARS
        text = (
            text[:MAX_TOOL_RESULT_CHARS]
            + f"\n…[truncated {dropped} characters; request a narrower result]"
        )
    return text


def repair_tool_pairs(messages: list[TranscriptMessage]) -> list[TranscriptMessage]:
    """Make every tool call have a result directly after its assistant
    message, and drop tool results that answer no call (both are API errors).
    """
    out: list[TranscriptMessage] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        index += 1
        if message.role == "tool":
            continue  # orphan: not directly after an assistant call
        out.append(message)
        if message.role != "assistant" or not message.tool_calls:
            continue
        expected = {c.id for c in message.tool_calls}
        answered: set[str] = set()
        while index < len(messages) and messages[index].role == "tool":
            result = messages[index]
            index += 1
            if result.tool_call_id in expected and result.tool_call_id not in answered:
                answered.add(result.tool_call_id)
                out.append(result)
        for call in message.tool_calls:
            if call.id not in answered:
                out.append(
                    TranscriptMessage(
                        role="tool",
                        content=INTERRUPTED_TOOL_RESULT,
                        tool_call_id=call.id,
                        tool_name=call.name,
                        is_error=True,
                    )
                )
    return out


def trim_transcript(
    messages: list[TranscriptMessage], limit: int = MAX_TRANSCRIPT_MESSAGES
) -> list[TranscriptMessage]:
    """Keep the first user message plus the newest messages, at most
    ``limit`` in total. The kept tail never starts with a tool result (its
    tool call would be gone)."""
    if len(messages) <= limit:
        return list(messages)
    first_user = next((i for i, m in enumerate(messages) if m.role == "user"), None)
    if first_user is None:
        head: list[TranscriptMessage] = []
        tail_start = len(messages) - limit
    else:
        head = [messages[first_user]]
        tail_start = max(first_user + 1, len(messages) - (limit - 1))
    while tail_start < len(messages) and messages[tail_start].role == "tool":
        tail_start += 1
    return head + list(messages[tail_start:])


def messages_for_model(messages: list[TranscriptMessage]) -> list[TranscriptMessage]:
    return trim_transcript(repair_tool_pairs(messages))


def describe_call(call: ToolCall) -> str:
    """Short human line for the tool_call chip, e.g. 'Get workflow code'."""
    return call.name.replace("_", " ").strip().capitalize() or "Tool"


async def _workflow_name(workflow_id: Optional[int], organization_id: int) -> Any:
    if workflow_id is None:
        return None
    try:
        workflow = await db_client.get_workflow(
            workflow_id, organization_id=organization_id
        )
    except Exception as exc:
        logger.warning(f"Copilot: workflow name lookup failed: {exc}")
        return None
    return getattr(workflow, "name", None)


def _add_usage(total: dict[str, int], usage: dict[str, int]) -> None:
    for key, value in (usage or {}).items():
        total[key] = total.get(key, 0) + (value or 0)


async def _brief_text(workflow_id: Optional[int], organization_id: int) -> str:
    """The workflow's brief for the session context (best effort: a failed
    lookup reads as "no brief" rather than failing the turn)."""
    if workflow_id is None:
        return ""
    try:
        row = await db_client.get_workflow_brief(workflow_id, organization_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Copilot: brief lookup failed: {exc!r}")
        return brief_prompt_text(None)
    if row is None:
        return brief_prompt_text(None)
    return brief_prompt_text(brief_response_from_model(workflow_id, row))


async def run_turn(
    *,
    session_uuid: str,
    organization_id: int,
    user: Any,
    text: str,
    mode: CopilotMode = "chat",
    iteration_id: Optional[int] = None,
    report_uuid: Optional[str] = None,
) -> AsyncIterator[CopilotEvent]:
    """Run one user message through the copilot; yields CopilotEvents.

    ``mode="self_improve"`` (M2b RI-4/O4/O5) runs a Self-improve round on the
    workflow's self-improve session: no present_plan, read-only brief,
    get_doctor_report bound to ``report_uuid``, one proposal per round
    (stored with ``iteration_id``), and the self_improve prompt section. The
    model sees only this round's messages (earlier rounds stay in the
    stored transcript). Chat turns refuse self-improve sessions (RI-1).
    """
    usage: dict[str, int] = {}
    try:
        session = await db_client.get_copilot_session(session_uuid, organization_id)
        if session is None:
            yield _event(CopilotEventType.error, message="Session not found.")
            return
        kind = getattr(session, "kind", None) or SESSION_KIND_CHAT
        if mode == SELF_IMPROVE_MODE:
            if (
                kind != SELF_IMPROVE_MODE
                or session.workflow_id is None
                or iteration_id is None
                or not report_uuid
            ):
                yield _event(
                    CopilotEventType.error,
                    message=(
                        "A Self-improve round needs the workflow's self-improve "
                        "session, a round and its Doctor report."
                    ),
                )
                return
        elif kind != SESSION_KIND_CHAT:
            yield _event(
                CopilotEventType.error,
                message="This session is managed by Self-improve.",
            )
            return

        from api.services.copilot import tools

        transcript = load_transcript(session.transcript)

        async def save(**fields: Any) -> None:
            await db_client.update_copilot_session(
                session_uuid,
                organization_id,
                transcript=[m.model_dump(mode="json") for m in transcript],
                **fields,
            )

        # Self-improve rounds show the model only this round's messages.
        turn_start = len(transcript) if mode == SELF_IMPROVE_MODE else 0
        transcript.append(
            TranscriptMessage(role="user", content=text, created_at=_now())
        )
        if session.title:
            await save()
        elif mode == SELF_IMPROVE_MODE:
            await save(title=SELF_IMPROVE_SESSION_TITLE)
        else:
            await save(title=session_title(text))

        ctx = ToolContext(
            session_uuid=session_uuid,
            organization_id=organization_id,
            user_id=user.id,
            workflow_id=session.workflow_id,
            mode=mode,
            iteration_id=iteration_id if mode == SELF_IMPROVE_MODE else None,
            report_uuid=report_uuid if mode == SELF_IMPROVE_MODE else None,
        )
        specs = tools.get_tool_specs(ctx)
        system = build_system_prompt(
            workflow_id=session.workflow_id,
            workflow_name=await _workflow_name(session.workflow_id, organization_id),
            mode=mode,
            brief_text=await _brief_text(session.workflow_id, organization_id),
        )
        provider = llm.get_provider()
        # Anthropic thinking blocks of this turn's tool-use messages, keyed by
        # the message's first tool-call id (in memory only).
        replay: dict[str, Any] = {}

        for _ in range(MAX_TOOL_ITERATIONS_PER_TURN):
            turn: Optional[llm.ModelTurn] = None
            # aclosing: a client disconnect closes the provider's HTTP stream
            # now rather than whenever the generator is garbage-collected.
            async with aclosing(
                provider.stream(
                    system=system,
                    messages=messages_for_model(transcript[turn_start:]),
                    tools=specs,
                    replay=replay,
                )
            ) as stream:
                async for item in stream:
                    if isinstance(item, str):
                        yield _event(CopilotEventType.text_delta, text=item)
                    else:
                        turn = item
            if turn is None:
                raise llm.CopilotLLMError("The AI service returned no response.")
            _add_usage(usage, turn.usage)

            # A truncated or refused reply may carry a half-written tool call;
            # keep only its text so the transcript stays valid.
            final = turn.stop_reason in (llm.STOP_MAX_TOKENS, llm.STOP_REFUSAL)
            calls = [] if final else turn.tool_calls
            if turn.text or calls:
                transcript.append(
                    TranscriptMessage(
                        role="assistant",
                        content=turn.text,
                        tool_calls=calls,
                        created_at=_now(),
                    )
                )
                await save()
            if calls and turn.replay is not None:
                replay[calls[0].id] = turn.replay

            if turn.stop_reason == llm.STOP_REFUSAL:
                yield _event(CopilotEventType.error, message=REFUSAL_ERROR)
                return
            if turn.stop_reason == llm.STOP_MAX_TOKENS:
                yield _event(CopilotEventType.error, message=TRUNCATED_ERROR)
                return
            if not calls:
                yield _event(CopilotEventType.message_done)
                return

            for call in calls:
                yield _event(
                    CopilotEventType.tool_call,
                    id=call.id,
                    name=call.name,
                    summary=describe_call(call),
                )
                outcome = await _execute(tools, call, ctx)
                if call.name == PROPOSE_CHANGES_TOOL and not outcome.ok:
                    ctx.fix_rounds_used += 1
                yield _event(
                    CopilotEventType.tool_result,
                    id=call.id,
                    name=call.name,
                    ok=outcome.ok,
                    summary=outcome.summary,
                )
                transcript.append(
                    TranscriptMessage(
                        role="tool",
                        content=tool_result_text(outcome),
                        tool_call_id=call.id,
                        tool_name=call.name,
                        is_error=not outcome.ok,
                        created_at=_now(),
                    )
                )
                if outcome.plan is not None:
                    plan = outcome.plan.model_dump(mode="json")
                    await save(plan=plan)
                    yield _event(CopilotEventType.plan, plan=plan)
                else:
                    await save()
                if outcome.proposal is not None:
                    yield _event(
                        CopilotEventType.proposal,
                        proposal=outcome.proposal.model_dump(mode="json"),
                    )
                if outcome.brief_change is not None:
                    yield _event(
                        CopilotEventType.brief_change,
                        brief_change=outcome.brief_change.model_dump(mode="json"),
                    )

        # Step cap: the last model call still wanted tools.
        transcript.append(
            TranscriptMessage(
                role="assistant", content=STEP_CAP_MESSAGE, created_at=_now()
            )
        )
        await save()
        yield _event(CopilotEventType.text_delta, text=STEP_CAP_MESSAGE)
        yield _event(CopilotEventType.message_done)
    except llm.CopilotLLMError as exc:
        logger.warning(f"Copilot turn {session_uuid}: LLM error: {exc!r}")
        yield _event(CopilotEventType.error, message=str(exc) or GENERIC_ERROR)
    except Exception:
        logger.exception(f"Copilot turn {session_uuid} failed")
        yield _event(CopilotEventType.error, message=GENERIC_ERROR)
    finally:
        if usage:
            logger.info(
                f"Copilot usage session={session_uuid} org={organization_id} "
                f"usage_context={USAGE_CONTEXT_COPILOT} {usage}"
            )


async def _execute(tools: Any, call: ToolCall, ctx: ToolContext) -> ToolOutcome:
    """Run one tool; never raises (tool failures become ok=False results)."""
    if (
        call.name == PROPOSE_CHANGES_TOOL
        and ctx.fix_rounds_used > MAX_PROPOSAL_FIX_ROUNDS
    ):
        return ToolOutcome(
            ok=False, content={"error": FIX_LIMIT_ERROR}, summary="Fix limit reached"
        )
    try:
        return await tools.execute_tool(call.name, call.input, ctx)
    except Exception:
        logger.exception(f"Copilot tool {call.name} raised")
        return ToolOutcome(
            ok=False,
            content={"error": "The tool failed unexpectedly."},
            summary="Tool failed",
        )


# ─── Self-improve rounds (called by api/services/self_improve) ───────────


class SelfImproveTurnError(Exception):
    """The round's copilot turn failed before producing a proposal."""


async def run_self_improve_turn(
    *,
    session_uuid: str,
    organization_id: int,
    user: Any,
    iteration_id: int,
    report_uuid: str,
    text: str,
) -> tuple[Optional[str], str]:
    """Run one Self-improve copilot turn to completion.

    Returns ``(proposal_uuid or None, summary_text)``: the round's proposal
    (at most one, stored with ``iteration_id``) and the model's closing
    paragraph. Idempotent for retries (O5): when the round already has a
    proposal, no model call is made and that proposal is returned with its
    summary. Raises SelfImproveTurnError when the turn errors without a
    proposal; an error after the proposal was stored still returns it.
    """
    existing = await db_client.get_proposal_for_iteration(iteration_id, organization_id)
    if existing is not None:
        return str(existing.proposal_uuid), existing.summary or ""

    proposal_uuid: Optional[str] = None
    error: Optional[str] = None
    current = ""  # text of the current model step
    last_text = ""  # text of the last step that had any
    async for event in run_turn(
        session_uuid=session_uuid,
        organization_id=organization_id,
        user=user,
        text=text,
        mode=SELF_IMPROVE_MODE,
        iteration_id=iteration_id,
        report_uuid=report_uuid,
    ):
        if event.type == CopilotEventType.text_delta:
            current += event.data.get("text", "")
        elif event.type in (CopilotEventType.tool_call, CopilotEventType.tool_result):
            if current.strip():
                last_text = current
            current = ""
        elif event.type == CopilotEventType.proposal:
            proposal_uuid = (event.data.get("proposal") or {}).get("proposal_uuid")
        elif event.type == CopilotEventType.error:
            error = event.data.get("message") or GENERIC_ERROR
    summary = (current.strip() and current or last_text).strip()

    if proposal_uuid is None:
        # A retried round whose model saw "already proposed" (or a lost
        # event) still has its stored proposal.
        stored = await db_client.get_proposal_for_iteration(
            iteration_id, organization_id
        )
        if stored is not None:
            proposal_uuid = str(stored.proposal_uuid)
    if error and proposal_uuid is None:
        raise SelfImproveTurnError(error)
    return proposal_uuid, summary
