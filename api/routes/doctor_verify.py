"""Agent Doctor: propose → verify → save routes, plus tester "Regenerate ×10".

Not registered here: the integrator mounts ``router`` in ``api/routes``.
Every lookup is scoped to ``user.selected_organization_id``.
"""

from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from pydantic import BaseModel

from api.db import db_client
from api.db.models import UserModel
from api.services.auth.depends import get_user_with_selected_organization
from api.services.workflow.doctor.contracts import (
    RegenerateGroup,
    RegenerateRequest,
    RegenerateResponse,
    ReportResult,
    VerificationResult,
    VerificationStatus,
)
from api.services.workflow.doctor.llm import (
    DoctorLLMError,
    DoctorLLMUnavailable,
    anthropic_key_configured,
)
from api.services.workflow.doctor.verify import (
    ProposalError,
    VerifyError,
    discard_candidate,
    get_working_definition,
    group_replies,
    mark_verifying,
    propose_fix,
    regenerate,
    save_verified,
)
from api.tasks.function_names import FunctionNames

router = APIRouter(tags=["doctor"])


class VerificationResponse(VerificationResult):
    """VerificationResult plus the scenario id the endpoints are keyed by."""

    scenario_id: int | None = None


class SaveVerificationRequest(BaseModel):
    #: working copy's version_number (int) or definition hash (str)
    expected_draft_version: int | str | None = None


class SaveVerificationResponse(BaseModel):
    saved: bool
    workflow_id: int | None = None
    version_number: int | None = None
    verified: bool = False
    verification_status: str | None = None


def _response(scenario: Any) -> VerificationResponse:
    return VerificationResponse(
        **VerificationResult.model_validate(scenario.verification or {}).model_dump(),
        scenario_id=scenario.id,
    )


def _require_claude() -> None:
    if not anthropic_key_configured():
        raise HTTPException(
            status_code=503, detail="Agent Doctor AI is not configured on this server"
        )


async def _scenario_or_404(scenario_id: int, organization_id: int) -> Any:
    scenario = await db_client.get_regression_scenario(scenario_id, organization_id)
    if scenario is None:
        raise HTTPException(status_code=404, detail="Verification not found")
    return scenario


@router.post(
    "/doctor/reports/{report_uuid}/clusters/{cluster_id}/propose",
    response_model=VerificationResponse,
)
async def propose_doctor_fix(
    report_uuid: str,
    cluster_id: str,
    user: UserModel = Depends(get_user_with_selected_organization),
) -> VerificationResponse:
    """Claude writes a minimal fix for one cluster and creates the candidate
    plus its regression scenario(s). Returns ``awaiting_start`` with a cost
    estimate, or ``not_text_verifiable`` (D29)."""
    organization_id = user.selected_organization_id
    report = await db_client.get_analysis_report(report_uuid, organization_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")
    # A5: enforce key + opt-in server-side, not only in the UI.
    _require_claude()
    if not ReportResult.model_validate(report.result or {}).ai_enabled:
        raise HTTPException(
            status_code=403, detail="AI analysis is not enabled for this organization"
        )
    try:
        result, scenario_id = await propose_fix(
            report=report, cluster_id=cluster_id, organization_id=organization_id
        )
    except VerifyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ProposalError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except DoctorLLMUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e))
    except DoctorLLMError as e:
        raise HTTPException(status_code=502, detail=str(e))
    return VerificationResponse(**result.model_dump(), scenario_id=scenario_id)


@router.post(
    "/doctor/verifications/{scenario_id}/start",
    response_model=VerificationResponse,
)
async def start_doctor_verification(
    scenario_id: int,
    user: UserModel = Depends(get_user_with_selected_organization),
) -> VerificationResponse:
    """Enqueue the ×10 baseline/candidate replay (also used to re-verify)."""
    organization_id = user.selected_organization_id
    scenario = await _scenario_or_404(scenario_id, organization_id)
    status = VerificationResult.model_validate(scenario.verification or {}).status
    if status == VerificationStatus.verifying:
        raise HTTPException(status_code=409, detail="Verification already running")
    if status == VerificationStatus.not_text_verifiable:
        raise HTTPException(
            status_code=400,
            detail="This fix can't be checked by a text replay; verify on a real call",
        )
    if not scenario.candidate_definition_id:
        raise HTTPException(status_code=410, detail="The fix candidate was removed")
    _require_claude()

    from api.tasks.arq import enqueue_job

    await mark_verifying(scenario, organization_id)
    await enqueue_job(
        FunctionNames.VERIFY_DOCTOR_FIX,
        scenario_id,
        organization_id,
        _job_id=f"doctor-verify-{scenario_id}-{uuid4().hex[:8]}",
    )
    return _response(scenario)


@router.get(
    "/doctor/verifications/{scenario_id}",
    response_model=VerificationResponse,
)
async def get_doctor_verification(
    scenario_id: int,
    user: UserModel = Depends(get_user_with_selected_organization),
) -> VerificationResponse:
    scenario = await _scenario_or_404(scenario_id, user.selected_organization_id)
    return _response(scenario)


@router.post(
    "/doctor/verifications/{scenario_id}/save",
    response_model=SaveVerificationResponse,
)
async def save_doctor_verification(
    scenario_id: int,
    request: SaveVerificationRequest,
    user: UserModel = Depends(get_user_with_selected_organization),
) -> SaveVerificationResponse:
    """Save the tested candidate as a draft. 409 ``changed_since_test`` (hash
    mismatch, re-verify) or ``draft_conflict`` (optimistic lock)."""
    organization_id = user.selected_organization_id
    await _scenario_or_404(scenario_id, organization_id)
    result = await save_verified(
        scenario_id=scenario_id,
        organization_id=organization_id,
        expected_draft_version=request.expected_draft_version,
    )
    if not result.get("saved"):
        code = result.get("error_code")
        status_code, detail = {
            "not_found": (404, "Not found"),
            "candidate_missing": (410, result.get("error")),
            # The UI keys its "Re-verify" / "Review their changes" states on
            # these phrases (design spec 3A).
            "changed_since_test": (
                409,
                "This version changed since it was tested. Re-verify it.",
            ),
            "draft_conflict": (
                409,
                "draft_conflict: someone edited this draft since you opened it.",
            ),
            "not_savable": (400, result.get("error")),
            "validation_error": (422, result.get("error")),
        }.get(code, (400, result.get("error")))
        raise HTTPException(status_code=status_code, detail=detail)
    return SaveVerificationResponse(**result)


@router.post(
    "/doctor/verifications/{scenario_id}/discard",
    response_model=VerificationResponse,
)
async def discard_doctor_verification(
    scenario_id: int,
    user: UserModel = Depends(get_user_with_selected_organization),
) -> VerificationResponse:
    organization_id = user.selected_organization_id
    scenario = await _scenario_or_404(scenario_id, organization_id)
    await discard_candidate(scenario_id=scenario_id, organization_id=organization_id)
    scenario = await _scenario_or_404(scenario_id, organization_id)
    return _response(scenario)


@router.post(
    "/workflow/{workflow_id}/text-chat/regenerate",
    response_model=RegenerateResponse,
)
async def regenerate_text_chat_turn(
    workflow_id: int,
    request: RegenerateRequest,
    user: UserModel = Depends(get_user_with_selected_organization),
) -> RegenerateResponse:
    """Re-roll one recorded turn n× on the workflow's current working copy
    and group the replies by frequency (D13). Each re-roll is a dry-run
    ``doctor_replay`` run."""
    organization_id = user.selected_organization_id
    workflow = await db_client.get_workflow(
        workflow_id, organization_id=organization_id
    )
    if workflow is None:
        raise HTTPException(status_code=404, detail="Workflow not found")
    run = await db_client.get_workflow_run(
        request.run_id, organization_id=organization_id
    )
    if run is None or run.workflow_id != workflow_id:
        raise HTTPException(status_code=404, detail="Run not found")
    definition = await get_working_definition(workflow)
    if definition is None:
        raise HTTPException(status_code=400, detail="Workflow has no definition")
    try:
        samples = await regenerate(
            run.id,
            request.turn,
            definition,
            request.n,
            organization_id=organization_id,
            user_id=getattr(workflow, "user_id", None),
            label="tester",
        )
    except VerifyError as e:
        raise HTTPException(status_code=402, detail=str(e))
    except Exception as e:
        logger.exception(f"Regenerate failed for run {run.id} turn {request.turn}")
        raise HTTPException(status_code=500, detail=f"Regenerate failed: {e}")
    groups = [RegenerateGroup(**g) for g in group_replies(samples)]
    return RegenerateResponse(n=len(samples), groups=groups)
