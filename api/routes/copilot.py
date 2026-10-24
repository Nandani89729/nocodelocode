"""Agent Copilot routes (milestone 2).

Registered in ``api/routes/main.py``. Paths (under ``/api/v1``):

- ``POST /copilot/sessions``                         create (per workflow, or build-new)
- ``GET  /copilot/sessions/{session_uuid}``          history, proposals, last plan
- ``GET  /workflow/{workflow_id}/copilot/sessions``  the caller's sessions, latest first
- ``POST /copilot/sessions/{session_uuid}/messages`` one turn, as text/event-stream
- ``POST /copilot/proposals/{proposal_uuid}/apply``  save accepted changes as a draft
- ``POST /copilot/proposals/{proposal_uuid}/discard``
- ``GET  /copilot/proposals/{proposal_uuid}``        one proposal (e.g. a round's)
- ``GET  /workflow/{workflow_id}/brief``             brief + pending change
- ``PUT  /workflow/{workflow_id}/brief``             user edit (409 brief_version_conflict)
- ``POST /workflow/{workflow_id}/brief/changes/{change_uuid}/confirm``  (409 brief_change_stale)
- ``POST /workflow/{workflow_id}/brief/changes/{change_uuid}/dismiss``
- ``POST /workflow/{workflow_id}/brief/draft``       draft from chats -> pending change

Every route is gated like the Doctor: the deployment key (503) and the
org's AI opt-in (403). All lookups are scoped to
``user.selected_organization_id``.
"""

from __future__ import annotations

import json
import time
from typing import Any, AsyncIterator, Awaitable

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from api.db import db_client
from api.services.auth.depends import get_user
from api.services.copilot import agent
from api.services.copilot import brief as brief_service
from api.services.copilot.contracts import (
    ApplyProposalRequest,
    ApplyProposalResponse,
    BriefResponse,
    CopilotEvent,
    CreateSessionRequest,
    DisplayMessage,
    Plan,
    Proposal,
    SendMessageRequest,
    SessionResponse,
    UpdateBriefRequest,
)
from api.services.workflow.doctor.analysis import resolve_ai_access

router = APIRouter(tags=["copilot"])

LIST_LIMIT = 20
#: Tool messages are shown as chips; their content is only a preview.
TOOL_PREVIEW_CHARS = 300

NO_KEY_DETAIL = "The Copilot is not configured on this server"
NOT_OPTED_IN_DETAIL = "AI features are not enabled for this organization"


async def _require_ai(organization_id: int) -> None:
    enabled, reason = await resolve_ai_access(organization_id)
    if enabled:
        return
    if reason == "no_key":
        raise HTTPException(status_code=503, detail=NO_KEY_DETAIL)
    raise HTTPException(status_code=403, detail=NOT_OPTED_IN_DETAIL)


class _RunningTurns:
    """Sessions with a turn in flight, so two tabs cannot interleave one
    transcript.

    In-process only: with several API workers a second request routed to
    another worker is not blocked (the transcript then takes last-writer-
    wins). Entries expire after TTL_SECONDS so a stream that never started
    (its ``finally`` never ran) cannot lock a session forever.
    """

    TTL_SECONDS = 30 * 60

    def __init__(self) -> None:
        self._started: dict[str, float] = {}

    def acquire(self, session_uuid: str) -> bool:
        now = time.monotonic()
        started = self._started.get(session_uuid)
        if started is not None and now - started < self.TTL_SECONDS:
            return False
        self._started[session_uuid] = now
        return True

    def release(self, session_uuid: str) -> None:
        self._started.pop(session_uuid, None)


running_turns = _RunningTurns()


def _iso(value: Any) -> str:
    return value.isoformat() if hasattr(value, "isoformat") else str(value or "")


def display_messages(transcript: Any) -> list[DisplayMessage]:
    """Transcript -> what the UI renders (tool messages become chips)."""
    out: list[DisplayMessage] = []
    for message in agent.load_transcript(transcript):
        if message.role == "tool":
            content = message.content
            if len(content) > TOOL_PREVIEW_CHARS:
                content = content[: TOOL_PREVIEW_CHARS - 1] + "…"
            out.append(
                DisplayMessage(
                    role="tool",
                    content=content,
                    tool_name=message.tool_name,
                    is_error=message.is_error,
                    created_at=message.created_at,
                )
            )
        elif message.content:
            # Assistant messages that only call tools have no text to show.
            out.append(
                DisplayMessage(
                    role=message.role,
                    content=message.content,
                    created_at=message.created_at,
                )
            )
    return out


async def _session_response(
    session: Any, organization_id: int, *, detail: bool = True
) -> SessionResponse:
    messages: list[DisplayMessage] = []
    proposals: list[Proposal] = []
    plan = None
    if detail:
        from api.services.copilot.proposals import proposal_from_model

        messages = display_messages(session.transcript)
        proposals = [
            proposal_from_model(model, session_uuid=str(session.session_uuid))
            for model in await db_client.list_copilot_proposals(
                session.id, organization_id
            )
        ]
        if session.plan:
            plan = Plan.model_validate(session.plan)
    return SessionResponse(
        session_uuid=str(session.session_uuid),
        workflow_id=session.workflow_id,
        title=session.title,
        messages=messages,
        proposals=proposals,
        plan=plan,
        created_at=_iso(session.created_at),
        updated_at=_iso(session.updated_at),
    )


async def _session_or_404(
    session_uuid: str, organization_id: int, *, chat_only: bool = False
) -> Any:
    session = await db_client.get_copilot_session(session_uuid, organization_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Copilot session not found")
    if chat_only and (getattr(session, "kind", None) or "chat") != "chat":
        # Self-improve sessions are driven by rounds, never by chat (RI-1).
        raise HTTPException(status_code=404, detail="Copilot session not found")
    return session


def sse_frame(event: CopilotEvent) -> str:
    payload = json.dumps(event.data, ensure_ascii=False, default=str)
    return f"event: {event.type.value}\ndata: {payload}\n\n"


@router.post(
    "/copilot/sessions",
    response_model=SessionResponse,
    summary="Start a Copilot chat for a workflow (or for building a new one)",
)
async def create_copilot_session(
    request: CreateSessionRequest,
    user=Depends(get_user),
) -> SessionResponse:
    organization_id = user.selected_organization_id
    await _require_ai(organization_id)
    if request.workflow_id is not None:
        workflow = await db_client.get_workflow(
            request.workflow_id, organization_id=organization_id
        )
        if workflow is None:
            raise HTTPException(status_code=404, detail="Workflow not found")
    session = await db_client.create_copilot_session(
        organization_id=organization_id,
        user_id=user.id,
        workflow_id=request.workflow_id,
    )
    return await _session_response(session, organization_id)


@router.get(
    "/copilot/sessions/{session_uuid}",
    response_model=SessionResponse,
    summary="Get a Copilot chat with its history, proposals and plan",
)
async def get_copilot_session(
    session_uuid: str,
    user=Depends(get_user),
) -> SessionResponse:
    organization_id = user.selected_organization_id
    await _require_ai(organization_id)
    session = await _session_or_404(session_uuid, organization_id)
    return await _session_response(session, organization_id)


@router.get(
    "/workflow/{workflow_id}/copilot/sessions",
    response_model=list[SessionResponse],
    summary="List the caller's Copilot chats for a workflow (latest first)",
)
async def list_copilot_sessions(
    workflow_id: int,
    user=Depends(get_user),
) -> list[SessionResponse]:
    organization_id = user.selected_organization_id
    await _require_ai(organization_id)
    workflow = await db_client.get_workflow(
        workflow_id, organization_id=organization_id
    )
    if workflow is None:
        raise HTTPException(status_code=404, detail="Workflow not found")
    sessions = await db_client.list_copilot_sessions(
        organization_id, workflow_id=workflow_id, user_id=user.id, limit=LIST_LIMIT
    )
    # Light list: history, proposals and plan come from the detail route.
    return [
        await _session_response(session, organization_id, detail=False)
        for session in sessions
    ]


@router.post(
    "/copilot/sessions/{session_uuid}/messages",
    summary="Send a message; streams the Copilot's reply as server-sent events",
    response_class=StreamingResponse,
)
async def send_copilot_message(
    session_uuid: str,
    request: SendMessageRequest,
    user=Depends(get_user),
) -> StreamingResponse:
    organization_id = user.selected_organization_id
    await _require_ai(organization_id)
    await _session_or_404(session_uuid, organization_id, chat_only=True)
    if not running_turns.acquire(session_uuid):
        raise HTTPException(
            status_code=409, detail="The Copilot is still answering in this chat"
        )

    async def stream() -> AsyncIterator[str]:
        # A client disconnect cancels this generator: the turn stops at the
        # next await, and the transcript keeps every completed step.
        try:
            async for event in agent.run_turn(
                session_uuid=session_uuid,
                organization_id=organization_id,
                user=user,
                text=request.text,
            ):
                yield sse_frame(event)
        finally:
            running_turns.release(session_uuid)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post(
    "/copilot/proposals/{proposal_uuid}/apply",
    response_model=ApplyProposalResponse,
    summary="Save the accepted changes of a proposal as a draft",
)
async def apply_copilot_proposal(
    proposal_uuid: str,
    request: ApplyProposalRequest,
    user=Depends(get_user),
) -> ApplyProposalResponse:
    from api.services.copilot.proposals import apply_proposal

    organization_id = user.selected_organization_id
    await _require_ai(organization_id)
    return await apply_proposal(
        proposal_uuid=proposal_uuid,
        organization_id=organization_id,
        user=user,
        request=request,
    )


@router.post(
    "/copilot/proposals/{proposal_uuid}/discard",
    response_model=Proposal,
    summary="Discard a proposal",
)
async def discard_copilot_proposal(
    proposal_uuid: str,
    user=Depends(get_user),
) -> Proposal:
    from api.services.copilot.proposals import discard_proposal

    organization_id = user.selected_organization_id
    await _require_ai(organization_id)
    proposal = await discard_proposal(
        proposal_uuid=proposal_uuid, organization_id=organization_id
    )
    if proposal is None:
        raise HTTPException(status_code=404, detail="Proposal not found")
    return proposal


@router.get(
    "/copilot/proposals/{proposal_uuid}",
    response_model=Proposal,
    summary="Get a proposal (e.g. a Self-improve round's proposal)",
)
async def get_copilot_proposal(
    proposal_uuid: str,
    user=Depends(get_user),
) -> Proposal:
    from api.services.copilot.proposals import proposal_from_model

    organization_id = user.selected_organization_id
    await _require_ai(organization_id)
    model = await db_client.get_copilot_proposal(proposal_uuid, organization_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Proposal not found")
    return proposal_from_model(model)


# ---------------------------------------------------------------------------
# Workflow brief (North Star, M2b R15/D2, O8, DS-6, DS-9)
# ---------------------------------------------------------------------------

BRIEF_VERSION_CONFLICT = "brief_version_conflict"
BRIEF_CHANGE_STALE = "brief_change_stale"
BRIEF_CHANGE_NOT_PENDING = "brief_change_not_pending"


async def _brief_call(coro: Awaitable[Any]) -> Any:
    """Run a brief service call, mapping its errors to HTTP."""
    try:
        return await coro
    except brief_service.BriefNotFound:
        raise HTTPException(status_code=404, detail="Workflow not found")
    except brief_service.BriefChangeNotFound:
        raise HTTPException(status_code=404, detail="Brief change not found")
    except brief_service.BriefVersionConflict:
        raise HTTPException(status_code=409, detail=BRIEF_VERSION_CONFLICT)
    except brief_service.BriefChangeStale:
        raise HTTPException(status_code=409, detail=BRIEF_CHANGE_STALE)
    except brief_service.BriefChangeNotPending:
        raise HTTPException(status_code=409, detail=BRIEF_CHANGE_NOT_PENDING)


@router.get(
    "/workflow/{workflow_id}/brief",
    response_model=BriefResponse,
    summary="Get the workflow's brief and any pending Copilot change to it",
)
async def get_workflow_brief(
    workflow_id: int,
    user=Depends(get_user),
) -> BriefResponse:
    organization_id = user.selected_organization_id
    await _require_ai(organization_id)
    return await _brief_call(brief_service.get_brief(workflow_id, organization_id))


@router.put(
    "/workflow/{workflow_id}/brief",
    response_model=BriefResponse,
    summary="Save the brief (409 brief_version_conflict when it changed)",
)
async def update_workflow_brief(
    workflow_id: int,
    request: UpdateBriefRequest,
    user=Depends(get_user),
) -> BriefResponse:
    organization_id = user.selected_organization_id
    await _require_ai(organization_id)
    return await _brief_call(
        brief_service.save_brief(
            workflow_id=workflow_id,
            organization_id=organization_id,
            brief=request.brief,
            expected_version=request.expected_version,
        )
    )


@router.post(
    "/workflow/{workflow_id}/brief/changes/{change_uuid}/confirm",
    response_model=BriefResponse,
    summary="Confirm a Copilot-proposed brief change (409 brief_change_stale)",
)
async def confirm_workflow_brief_change(
    workflow_id: int,
    change_uuid: str,
    user=Depends(get_user),
) -> BriefResponse:
    organization_id = user.selected_organization_id
    await _require_ai(organization_id)
    return await _brief_call(
        brief_service.confirm_brief_change(
            workflow_id=workflow_id,
            organization_id=organization_id,
            change_uuid=change_uuid,
        )
    )


@router.post(
    "/workflow/{workflow_id}/brief/changes/{change_uuid}/dismiss",
    response_model=BriefResponse,
    summary="Dismiss a Copilot-proposed brief change",
)
async def dismiss_workflow_brief_change(
    workflow_id: int,
    change_uuid: str,
    user=Depends(get_user),
) -> BriefResponse:
    organization_id = user.selected_organization_id
    await _require_ai(organization_id)
    return await _brief_call(
        brief_service.dismiss_brief_change(
            workflow_id=workflow_id,
            organization_id=organization_id,
            change_uuid=change_uuid,
        )
    )


@router.post(
    "/workflow/{workflow_id}/brief/draft",
    response_model=brief_service.BriefDraftResponse,
    summary="Draft a brief from the workflow's Copilot chats (pending change)",
)
async def draft_workflow_brief(
    workflow_id: int,
    user=Depends(get_user),
) -> brief_service.BriefDraftResponse:
    from api.services.workflow.doctor.llm import DoctorLLMError, DoctorLLMUnavailable

    organization_id = user.selected_organization_id
    await _require_ai(organization_id)
    try:
        change = await _brief_call(
            brief_service.draft_brief_from_chats(
                workflow_id=workflow_id, organization_id=organization_id
            )
        )
    except brief_service.BriefDraftUnavailable as exc:
        raise HTTPException(status_code=422, detail=exc.reason)
    except DoctorLLMUnavailable:
        raise HTTPException(status_code=503, detail=NO_KEY_DETAIL)
    except DoctorLLMError:
        raise HTTPException(
            status_code=502,
            detail="The AI service could not draft a brief. Try again.",
        )
    return brief_service.BriefDraftResponse(brief_change=change)
