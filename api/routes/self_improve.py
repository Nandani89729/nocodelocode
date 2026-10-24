"""Self-improve routes (milestone 2b).

Register with ``router.include_router(self_improve_router)`` in
``api/routes/main.py``. Paths (under ``/api/v1``):

- ``GET  /workflow/{workflow_id}/self-improve``             status + settings
- ``PUT  /workflow/{workflow_id}/self-improve``             turn on/off, set N
- ``POST /workflow/{workflow_id}/self-improve/run``         run a round now
- ``POST /workflow/{workflow_id}/self-improve/resume``      clear a D6 pause
- ``GET  /workflow/{workflow_id}/self-improve/iterations``  rounds, newest first

Every handler first checks that the workflow belongs to the caller's
selected organization (404 otherwise); all DB calls are org-scoped.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger

from api.db import db_client
from api.services.auth.depends import get_user
from api.services.self_improve.contracts import (
    ALLOWED_EVERY_N_CALLS,
    IterationListResponse,
    RunNowResponse,
    SelfImproveState,
    SelfImproveStatus,
    UpdateSelfImproveRequest,
)
from api.services.self_improve.counting import ai_access_ok, claim_and_enqueue
from api.services.self_improve.status import (
    build_status,
    iteration_to_response,
    iteration_with_problems,
)

router = APIRouter(tags=["self-improve"])

ITERATIONS_LIMIT = 20

#: RunNowResponse.reason for each state that blocks a round.
_BLOCKED_REASONS = {
    SelfImproveState.no_key: "no_key",
    SelfImproveState.ai_disabled: "ai_disabled",
    SelfImproveState.off: "off",
    SelfImproveState.paused: "paused",
    SelfImproveState.waiting: "waiting_for_review",
    SelfImproveState.running: "running",
}


async def _require_workflow(workflow_id: int, organization_id: int) -> Any:
    workflow = await db_client.get_workflow(
        workflow_id, organization_id=organization_id
    )
    if workflow is None:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return workflow


async def _try_start_round(workflow_id: int, organization_id: int) -> None:
    """Start a round if the window is already full (DS-10, Resume)."""
    try:
        if await ai_access_ok(organization_id):
            await claim_and_enqueue(workflow_id, organization_id)
    except Exception as exc:
        logger.error(
            f"Self-improve: could not start a round for workflow {workflow_id}: {exc}"
        )


@router.get(
    "/workflow/{workflow_id}/self-improve",
    response_model=SelfImproveStatus,
    summary="Self-improve settings and progress for a workflow",
)
async def get_self_improve_status(workflow_id: int, user=Depends(get_user)):
    organization_id = user.selected_organization_id
    await _require_workflow(workflow_id, organization_id)
    return await build_status(workflow_id, organization_id)


@router.put(
    "/workflow/{workflow_id}/self-improve",
    response_model=SelfImproveStatus,
    summary="Turn Self-improve on or off and set how often it runs",
)
async def update_self_improve_settings(
    workflow_id: int, request: UpdateSelfImproveRequest, user=Depends(get_user)
):
    organization_id = user.selected_organization_id
    await _require_workflow(workflow_id, organization_id)
    if request.every_n_calls not in ALLOWED_EVERY_N_CALLS:
        raise HTTPException(
            status_code=422,
            detail=f"every_n_calls must be one of {list(ALLOWED_EVERY_N_CALLS)}",
        )
    try:
        await db_client.upsert_self_improve_settings(
            workflow_id=workflow_id,
            organization_id=organization_id,
            enabled=request.enabled,
            every_n_calls=request.every_n_calls,
            user_id=user.id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if request.enabled:
        # DS-10: counted calls are kept, so a smaller N may start a round now.
        await _try_start_round(workflow_id, organization_id)
    return await build_status(workflow_id, organization_id)


@router.post(
    "/workflow/{workflow_id}/self-improve/run",
    response_model=RunNowResponse,
    summary="Start a Self-improve round now on the calls collected so far",
)
async def run_self_improve_now(workflow_id: int, user=Depends(get_user)):
    organization_id = user.selected_organization_id
    await _require_workflow(workflow_id, organization_id)
    status = await build_status(workflow_id, organization_id)
    reason = _BLOCKED_REASONS.get(status.state)
    if reason is not None:
        return RunNowResponse(
            started=False, iteration=status.current_iteration, reason=reason
        )
    if status.collected == 0:
        return RunNowResponse(started=False, reason="no_calls")

    iteration = await claim_and_enqueue(workflow_id, organization_id, force=True)
    if iteration is None:
        # Lost a race (a completion job started the round, or the calls were
        # taken): report what blocks it now.
        status = await build_status(workflow_id, organization_id)
        return RunNowResponse(
            started=False,
            iteration=status.current_iteration,
            reason=_BLOCKED_REASONS.get(status.state) or "no_calls",
        )
    return RunNowResponse(started=True, iteration=iteration_to_response(iteration))


@router.post(
    "/workflow/{workflow_id}/self-improve/resume",
    response_model=SelfImproveStatus,
    summary="Resume Self-improve after it paused on failed rounds",
)
async def resume_self_improve(workflow_id: int, user=Depends(get_user)):
    organization_id = user.selected_organization_id
    await _require_workflow(workflow_id, organization_id)
    resumed = await db_client.resume_self_improve(workflow_id, organization_id)
    if resumed is None:
        raise HTTPException(
            status_code=404, detail="Self-improve is not set up for this workflow"
        )
    await _try_start_round(workflow_id, organization_id)
    return await build_status(workflow_id, organization_id)


@router.get(
    "/workflow/{workflow_id}/self-improve/iterations",
    response_model=IterationListResponse,
    summary="Self-improve rounds of a workflow, newest first",
)
async def list_self_improve_iterations(workflow_id: int, user=Depends(get_user)):
    organization_id = user.selected_organization_id
    await _require_workflow(workflow_id, organization_id)
    rounds = await db_client.list_iterations(
        workflow_id, organization_id, limit=ITERATIONS_LIMIT
    )
    return IterationListResponse(
        iterations=[await iteration_with_problems(it) for it in rounds]
    )
